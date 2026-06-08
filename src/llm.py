from openai import OpenAI

from src.config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

SYSTEM_PROMPT = """You are an employee handbook assistant.

Answer ONLY using the supplied context.

If the answer is not available in the context, respond:
"I cannot find that information in the handbook."
"""


def build_prompt(context: str, question: str) -> str:
    return f"""Context:
{context}

Question:
{question}
"""


class GroqLLM:
    def __init__(self, api_key: str | None = GROQ_API_KEY, model: str = GROQ_MODEL):
        if not api_key or api_key == "your_groq_api_key":
            raise RuntimeError("Set GROQ_API_KEY in .env before generating answers.")
        self.client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        self.model = model

    def answer(self, question: str, context: str) -> str:
        return self.generate(question=question, context=context)["answer"]

    def generate(self, question: str, context: str) -> dict[str, object]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(context, question)},
            ],
            temperature=0,
        )
        usage = response.usage
        return {
            "answer": response.choices[0].message.content or "",
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }
