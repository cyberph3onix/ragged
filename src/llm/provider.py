print("Loaded provider.py")

from src.config import settings


class LLMProvider:

    def generate(self, prompt: str) -> str:

        print("PROVIDER =", settings.llm.provider)
        print("MODEL =", settings.llm.model)

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
        from ollama import chat

        response = chat(
            model=settings.llm.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        return response["message"]["content"]

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