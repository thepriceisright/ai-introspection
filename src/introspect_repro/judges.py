import json
import os
import time
from pathlib import Path
from typing import Literal, Optional

try:
    from . import load_project_env
except ImportError:
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        load_dotenv = None  # type: ignore[assignment]

    module_dir = Path(__file__).resolve().parent
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
                    os.environ.setdefault(key, value)
            break
else:
    load_project_env()

from .prompts import (
    COHERENCE_PROMPT, THINKING_ABOUT_PROMPT,
    AFFIRMATIVE_PROMPT, AFFIRMATIVE_CORRECT_ID_PROMPT,
    APOLOGY_GRADER_PROMPT, JudgeConfig
)

def _anthropic_chat(messages, model, temperature=0.0, max_tokens=128):
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

def _openai_chat(messages, model, temperature=0.0, max_tokens=128):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(model=model, messages=messages,
                                          temperature=temperature, max_tokens=max_tokens)
    return resp.choices[0].message.content

def _openrouter_chat(messages, model, temperature=0.0, max_tokens=128):
    import httpx, os
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
        "HTTP-Referer": "https://localhost",
        "X-Title": "introspection-repro"
    }
    data = {"model": model, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens}
    with httpx.Client(timeout=120) as client:
        r = client.post("https://openrouter.ai/api/v1/chat/completions",
                        headers=headers, json=data)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]

def _call_llm(provider, model, user_text, temperature=0.0, max_tokens=128):
    messages = [{"role":"user","content": user_text}]
    if provider == "anthropic":
        return _anthropic_chat(messages, model, temperature, max_tokens)
    elif provider == "openai":
        return _openai_chat(messages, model, temperature, max_tokens)
    elif provider == "openrouter":
        return _openrouter_chat(messages, model, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")

def _yes_no_to_bool(text: str) -> Optional[bool]:
    if not text: return None
    t = text.lower()
    if "yes" in t and not "no" in t[:10]: return True
    if t.strip().endswith("yes"): return True
    if "no" in t: return False
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
