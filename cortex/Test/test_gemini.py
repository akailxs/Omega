import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

print("Supported embedding models:")
for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(m.name)

# try simple embed
try:
    res = genai.embed_content(model="models/text-embedding-004", content="hello")
    print("004 success, single")
except Exception as e:
    print("004 single failed:", type(e).__name__)

try:
    res = genai.embed_content(model="models/embedding-001", content="hello")
    print("001 success, single")
except Exception as e:
    print("001 single failed:", type(e).__name__)
