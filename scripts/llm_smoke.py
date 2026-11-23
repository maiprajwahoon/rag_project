import sys

from langchain_google_genai import ChatGoogleGenerativeAI

from rag_project.config import get_settings

model = sys.argv[1]
key = get_settings().llm_api_key.get_secret_value()
print("key loaded:", bool(key), "| model:", model)
llm = ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0)
print("reply:", llm.invoke("Reply with exactly: OK").content)
