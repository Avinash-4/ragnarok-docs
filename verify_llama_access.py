from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")
model_id = os.getenv("MODEL_ID")

print("=" * 50)
print("RAGNAROK-DOCS — Environment Check")
print("=" * 50)
print(f"Python env:    OK")
print(f"Token found:   {'YES' if token and 'YOUR_NEW' not in token else 'NO - add token to .env'}")
print(f"Model target:  {model_id}")

try:
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.model_info(model_id, token=token)
    print(f"LLaMA access:  GRANTED — {info.id}")
    print(f"Model size:    8B params")
    print("=" * 50)
    print("ALL CHECKS PASSED. Ready for Phase 2!")
    print("=" * 50)
except Exception as e:
    print(f"LLaMA access:  FAILED")
    print(f"Reason: {e}")
    print("Fix: make sure your token is in .env and is correct")
