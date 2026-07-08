import re
import time

from config import settings


class LLMProvider:

    def __init__(self, provider: str | None = None, model: str | None = None):
        self._provider = provider or settings.llm.provider
        self._model = model or settings.llm.model

    def generate(self, prompt: str) -> str:
        provider = self._provider.lower()

        if provider == "ollama":
            return self._ollama(prompt)

        if provider == "groq":
            return self._groq(prompt)

        if provider == "gemini":
            return self._gemini(prompt)

        raise ValueError(
            f"Unknown provider: {provider}"
        )

    def _ollama(self, prompt: str) -> str:
        from ollama import Client

        # Keep provider timeout aligned with evaluator RunConfig (900s) so
        # long metric prompts do not fail at the HTTP client layer first.
        client = Client(host=settings.llm.ollama_host, timeout=900)

        # Prepend /no_think to suppress qwen3-family thinking tokens at generation time,
        # which dramatically reduces latency on CPU for long evaluation prompts.
        response = client.chat(
            model=self._model,
            messages=[{"role": "user", "content": "/no_think\n\n" + prompt}],
            think=False,
        )

        content = response.message.content or ""
        # Fallback: strip any residual </think> delimiter that Ollama didn't remove
        if "</think>" in content:
            content = content.split("</think>", 1)[-1]
        return content.strip()

    def _groq(self, prompt: str) -> str:
        from groq import Groq, RateLimitError

        client = Groq(
            api_key=settings.llm.groq_api_key
        )

        # Groq's free tier enforces a hard tokens-per-minute cap. A large
        # evaluation run (dozens of questions x several RAGAS metrics) will
        # exceed it partway through — back off and retry instead of letting
        # the whole run die on the first 429.
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                response = client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                )
                # Pace requests so a sequential run of many calls (e.g. a full
                # RAGAS evaluation) doesn't front-load enough tokens to trip
                # the per-minute cap before the first 429 ever fires.
                time.sleep(2)
                return response.choices[0].message.content
            except RateLimitError as e:
                if attempt == max_attempts - 1:
                    raise
                match = re.search(r"try again in ([\d.]+)s", str(e))
                wait_seconds = float(match.group(1)) if match else 2 ** attempt
                time.sleep(wait_seconds + 0.5)

        raise RuntimeError("unreachable")  # pragma: no cover

    def _gemini(self, prompt: str) -> str:
        import google.generativeai as genai
        from google.api_core.exceptions import ResourceExhausted

        genai.configure(
            api_key=settings.llm.gemini_api_key
        )

        model = genai.GenerativeModel(
            self._model
        )

        # Gemini's free tier enforces per-minute and per-day request/token caps.
        # Back off and retry on 429s instead of letting the whole eval run die.
        max_attempts = 6
        for attempt in range(max_attempts):
            try:
                response = model.generate_content(prompt)
                return response.text
            except ResourceExhausted as e:
                if attempt == max_attempts - 1:
                    raise
                match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(e))
                wait_seconds = float(match.group(1)) if match else 2 ** attempt
                time.sleep(wait_seconds + 0.5)

        raise RuntimeError("unreachable")  # pragma: no cover