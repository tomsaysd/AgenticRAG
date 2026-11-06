from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
question = input("Enter a question about Agentic RAG: ").strip()
if not question:
    raise SystemExit("Question cannot be empty.")

response = client.responses.create(
    model="gpt-6-luna",
    input=(
        "You are a research assistant. Answer this question about Agentic RAG "
        "from your general model knowledge, without using external documents. "
        "State uncertainty and do not invent citations.\n\n"
        f"Question: {question}"
    ),
)
print("\n========== BASE LLM RESPONSE ==========\n")
print(response.output_text)
