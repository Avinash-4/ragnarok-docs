"""
Runtime state shared across the app.
model_mode controls which LLM is used per request.
  "instruct"        — meta-llama/Llama-3.1-8B-Instruct via HF Inference API
  "ragnarok_tuned"  — avinashkongara4/llama3-ragnarok-merged via HF Endpoint
  "local_ragnarok"  — local machine running via ngrok tunnel
"""

model_mode: str = "instruct"
