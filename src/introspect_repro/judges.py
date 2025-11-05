import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Literal, Optional

try:
    from . import activate_local_venv, load_project_env
except ImportError:
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        load_dotenv = None  # type: ignore[assignment]

    module_dir = Path(__file__).resolve().parent

    def _fallback_activate_local_venv() -> None:
        python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        for parent in (module_dir, *module_dir.parents):
            venv_dir = parent / ".venv"
            if not venv_dir.exists():
                continue
            candidates = [
                venv_dir / "lib" / python_version / "site-packages",
                venv_dir / "Lib" / "site-packages",
            ]
            for site_dir in candidates:
                if site_dir.exists():
                    if str(site_dir) not in sys.path:
                        sys.path.append(str(site_dir))
                    os.environ.setdefault("VIRTUAL_ENV", str(venv_dir))
                    return

    HF_TOKEN_ENV_KEYS = (
        "HUGGINGFACEHUB_API_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HF_TOKEN",
        "HF_API_TOKEN",
    )

    def _normalise_hf_token_env() -> None:
        token = next((os.environ.get(k) for k in HF_TOKEN_ENV_KEYS if os.environ.get(k)), None)
        if not token:
            return
        for key in HF_TOKEN_ENV_KEYS:
            if not os.environ.get(key):
                os.environ[key] = token

    for parent in (module_dir, *module_dir.parents):
        env_path = parent / ".env"
        if env_path.exists():
            if load_dotenv is not None:
                load_dotenv(env_path, override=False)
            else:
                for line in env_path.read_text().splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and (key not in os.environ or not os.environ.get(key)):
                        os.environ[key] = value
            break

    _normalise_hf_token_env()

    _fallback_activate_local_venv()
else:
    activate_local_venv()
    load_project_env()

from .prompts import (
    COHERENCE_PROMPT, THINKING_ABOUT_PROMPT,
    AFFIRMATIVE_PROMPT, AFFIRMATIVE_CORRECT_ID_PROMPT,
    APOLOGY_GRADER_PROMPT, JudgeConfig
)

def _anthropic_chat(messages, model, temperature=0.0, max_tokens=256):
    from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    # Convert to Anthropic schema (single user message)
    content = ""
    for m in messages:
        role = m.get("role")
        if role == "user":
            content += f"{HUMAN_PROMPT} {m['content']}\n"
        elif role == "assistant":
            content += f"{AI_PROMPT} {m['content']}\n"
    resp = client.messages.create(model=model, max_tokens=max_tokens,
                                  temperature=temperature,
                                  messages=[{"role":"user","content":content}])
    return resp.content[0].text

def _openai_chat(messages, model, temperature=0.0, max_tokens=256):
    from openai import BadRequestError, RateLimitError, OpenAI

    if model.lower().startswith("gpt-5") and abs(temperature - 1.0) > 1e-6:
        warnings.warn(
            f"OpenAI model '{model}' requires temperature=1.0; overriding provided value {temperature}.",
            RuntimeWarning,
        )
        temperature = 1.0

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    user_text = messages[-1]["content"]

    def _handle_openai_error(err):
        """Extract error message and code from OpenAI error."""
        message = str(err)
        error_code = None
        error_type = None
        try:
            body = getattr(err, "body", None)
            if isinstance(body, dict):
                error_info = body.get("error", {})
                message = error_info.get("message", message)
                error_code = error_info.get("code")
                error_type = error_info.get("type")
        except Exception:
            pass
        return message, error_code, error_type

    def _make_responses_api_call(with_json_format=True):
        """Make a call to the Responses API with retry logic."""
        response_kwargs = {
            "model": model,
            "input": [{"role": "user", "content": user_text}],
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if with_json_format:
            response_kwargs["response_format"] = {"type": "json_object"}
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                resp = client.responses.create(**response_kwargs)
                return resp.output_text
            except RateLimitError as err:
                message, error_code, error_type = _handle_openai_error(err)
                # Check if it's a quota error (non-retryable)
                if error_type == "insufficient_quota" or "quota" in message.lower():
                    raise RuntimeError(
                        f"OpenAI API quota exceeded. Please check your billing and plan details.\n"
                        f"Error: {message}\n"
                        f"For more information, see: https://platform.openai.com/docs/guides/error-codes/api-errors"
                    ) from err
                # Rate limit error - retry with exponential backoff
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    warnings.warn(
                        f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})",
                        RuntimeWarning,
                    )
                    time.sleep(delay)
                else:
                    raise RuntimeError(
                        f"OpenAI API rate limit exceeded after {max_retries} attempts.\n"
                        f"Error: {message}"
                    ) from err
            except TypeError:
                # Older client versions do not support response_format; retry without it.
                if with_json_format:
                    return _make_responses_api_call(with_json_format=False)
                raise
            except BadRequestError as err:
                param = getattr(err, "param", None)
                message, _, _ = _handle_openai_error(err)
                # Fall back to Chat Completions if the model does not support Responses.
                if (param or "").lower() not in {"response_format", "max_output_tokens"} and "response_format" not in message:
                    raise
                # Intentional fall-through to try Chat Completions API
                break
        return None

    def _make_chat_completions_api_call(use_max_completion_tokens=False):
        """Make a call to the Chat Completions API with retry logic."""
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if use_max_completion_tokens:
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = max_tokens
                
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            except RateLimitError as err:
                message, error_code, error_type = _handle_openai_error(err)
                # Check if it's a quota error (non-retryable)
                if error_type == "insufficient_quota" or "quota" in message.lower():
                    raise RuntimeError(
                        f"OpenAI API quota exceeded. Please check your billing and plan details.\n"
                        f"Error: {message}\n"
                        f"For more information, see: https://platform.openai.com/docs/guides/error-codes/api-errors"
                    ) from err
                # Rate limit error - retry with exponential backoff
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    warnings.warn(
                        f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})",
                        RuntimeWarning,
                    )
                    time.sleep(delay)
                else:
                    raise RuntimeError(
                        f"OpenAI API rate limit exceeded after {max_retries} attempts.\n"
                        f"Error: {message}"
                    ) from err
            except BadRequestError as err:
                param = getattr(err, "param", None)
                message, _, _ = _handle_openai_error(err)
                if (param or "").lower() != "max_tokens" and "max_tokens" not in message:
                    raise
                # Try with max_completion_tokens instead
                if not use_max_completion_tokens:
                    return _make_chat_completions_api_call(use_max_completion_tokens=True)
                raise

    # Try Responses API first (for gpt-5 family)
    result = _make_responses_api_call()
    if result is not None:
        return result

    # Fall back to Chat Completions API
    return _make_chat_completions_api_call()

def _openrouter_chat(messages, model, temperature=0.0, max_tokens=256):
    import httpx, os
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
        "HTTP-Referer": "https://localhost",
        "X-Title": "introspection-repro"
    }
    data = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    
    max_retries = 5
    base_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=120) as client:
                r = client.post("https://openrouter.ai/api/v1/chat/completions",
                                headers=headers, json=data)
                r.raise_for_status()
                j = r.json()
                return j["choices"][0]["message"]["content"]
        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout, 
                httpx.NetworkError, httpx.ConnectTimeout) as err:
            # Network errors - retry with exponential backoff
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                warnings.warn(
                    f"OpenRouter network error (attempt {attempt + 1}/{max_retries}): {type(err).__name__}. "
                    f"Retrying in {delay:.1f}s...",
                    RuntimeWarning,
                )
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"OpenRouter API request failed after {max_retries} attempts due to network error.\n"
                    f"Error: {type(err).__name__}: {str(err)}"
                ) from err
        except httpx.HTTPStatusError as err:
            # HTTP error status codes (4xx, 5xx) - check if retryable
            status_code = err.response.status_code if hasattr(err, 'response') else None
            if status_code and status_code >= 500 and attempt < max_retries - 1:
                # Server errors (5xx) are retryable
                delay = base_delay * (2 ** attempt)
                warnings.warn(
                    f"OpenRouter server error {status_code} (attempt {attempt + 1}/{max_retries}). "
                    f"Retrying in {delay:.1f}s...",
                    RuntimeWarning,
                )
                time.sleep(delay)
            else:
                # Client errors (4xx) or final attempt - don't retry
                raise RuntimeError(
                    f"OpenRouter API returned error status {status_code}.\n"
                    f"Error: {str(err)}"
                ) from err

def _call_llm(provider, model, user_text, temperature=0.0, max_tokens=256):
    messages = [{"role":"user","content": user_text}]
    if provider == "anthropic":
        return _anthropic_chat(messages, model, temperature, max_tokens)
    elif provider == "openai":
        return _openai_chat(messages, model, temperature, max_tokens)
    elif provider == "openrouter":
        return _openrouter_chat(messages, model, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")

_TRUE_TOKENS = {"yes", "y", "true", "1"}
_FALSE_TOKENS = {"no", "n", "false", "0"}


def _token_to_bool(token: str) -> Optional[bool]:
    tok = token.strip().lower()
    if tok in _TRUE_TOKENS:
        return True
    if tok in _FALSE_TOKENS:
        return False
    return None


def _extract_bool_from_json(text: str) -> Optional[bool]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    candidates = []
    if isinstance(payload, dict):
        keys = ["answer", "final", "result", "response", "output"]
        for key in keys:
            if key in payload:
                candidates.append(payload[key])
        # Flatten nested dicts/lists lightly
        for value in payload.values():
            if isinstance(value, dict):
                for key in keys:
                    if key in value:
                        candidates.append(value[key])
    elif isinstance(payload, list):
        candidates.extend(payload)
    for candidate in candidates:
        if isinstance(candidate, str):
            b = _token_to_bool(candidate)
            if b is not None:
                return b
    return None


def _yes_no_to_bool(text: str) -> Optional[bool]:
    if not text:
        return None

    # Try JSON-parsed formats first
    json_bool = _extract_bool_from_json(text)
    if json_bool is not None:
        return json_bool

    stripped_lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    for line in reversed(stripped_lines):
        # common patterns like "Final answer: YES"
        if ":" in line:
            _, candidate = line.rsplit(":", 1)
            b = _token_to_bool(candidate)
            if b is not None:
                return b
        tokens = re.findall(r"\b(?:yes|no|y|n|true|false|1|0)\b", line.lower())
        if tokens:
            tok = tokens[-1]
            if tok in _TRUE_TOKENS:
                return True
            if tok in _FALSE_TOKENS:
                return False

    # Fallback: look for last YES/NO token anywhere
    matches = re.findall(r"\b(yes|no)\b", text.lower())
    if matches:
        return matches[-1] == "yes"
    return None

class Judge:
    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg

    def _ask(self, prompt: str) -> Optional[bool]:
        out = _call_llm(self.cfg.provider, self.cfg.model, prompt,
                        temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens)
        return _yes_no_to_bool(out)

    def grade_coherence(self, prompt: str, response: str) -> Optional[bool]:
        return self._ask(COHERENCE_PROMPT.format(prompt=prompt, response=response))

    def grade_thinking_about(self, question: str, response: str, word: str) -> Optional[bool]:
        return self._ask(THINKING_ABOUT_PROMPT.format(question=question, response=response, word=word.lower()))

    def grade_affirmative(self, prompt: str, response: str) -> Optional[bool]:
        return self._ask(AFFIRMATIVE_PROMPT.format(prompt=prompt, response=response))

    def grade_affirmative_correct_id(self, question: str, response: str, word: str) -> Optional[bool]:
        return self._ask(AFFIRMATIVE_CORRECT_ID_PROMPT.format(question=question, response=response, word=word.lower()))

    def grade_intent(self, response: str, word: str) -> Optional[bool]:
        # Returns True if the model *claimed it intended* to say the word (i.e., NO apology)
        return self._ask(APOLOGY_GRADER_PROMPT.format(response=response, word=word.lower()))
