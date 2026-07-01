from config import settings


class LLMProvider:

    def generate(self, prompt: str) -> str:

        # DEBUG
        # print("PROVIDER =", settings.llm.provider)
        # print("MODEL =", settings.llm.model)

        provider = settings.llm.provider.lower()

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
            model=settings.llm.model,
            messages=[{"role": "user", "content": "/no_think\n\n" + prompt}],
            think=False,
        )

        content = response.message.content or ""
        # Fallback: strip any residual </think> delimiter that Ollama didn't remove
        if "</think>" in content:
            content = content.split("</think>", 1)[-1]
        return content.strip()

    def _groq(self, prompt: str) -> str:
        from groq import Groq

        client = Groq(
            api_key=settings.llm.groq_api_key
        )

        response = client.chat.completions.create(
            model=settings.llm.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response.choices[0].message.content

    def _gemini(self, prompt: str) -> str:
        import google.generativeai as genai

        genai.configure(
            api_key=settings.llm.gemini_api_key
        )

        model = genai.GenerativeModel(
            settings.llm.model
        )

        response = model.generate_content(prompt)

        return response.text