from dotenv import load_dotenv
import os

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
MERGED_MODEL_ID = os.getenv("MERGED_MODEL_ID", "avinashkongara4/llama3-ragnarok-merged")
RAGNAROK_ENDPOINT = os.getenv("RAGNAROK_ENDPOINT")
LOCAL_RAGNAROK_ENDPOINT = os.getenv("LOCAL_RAGNAROK_ENDPOINT")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")


def build_rag_prompt(question: str, context: str) -> str:
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a document assistant. You ONLY answer using the CONTEXT below.
STRICT RULES — you must follow all of these:
1. If the answer is not explicitly stated in the CONTEXT, respond EXACTLY with: "I could not find this information in the uploaded documents."
2. Do NOT use your training knowledge. Do NOT guess. Do NOT infer from general knowledge.
3. Every answer must cite the exact file name and page number from the CONTEXT.
4. If the CONTEXT does not mention the topic at all, say you could not find it — even if you know the answer from training.
<|eot_id|><|start_header_id|>user<|end_header_id|>

CONTEXT FROM UPLOADED DOCUMENTS:
{context}

QUESTION: {question}

Remember: only answer from the CONTEXT above. If it is not there, say you could not find it.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def generate_answer_hf_api(question: str, context: str, model_id: str = None) -> str:
    """Call HF serverless Inference API (for instruct model)."""
    from huggingface_hub import InferenceClient

    target = model_id or MODEL_ID
    print(f"Calling HF API: {target}")

    client = InferenceClient(token=HUGGINGFACE_TOKEN)
    result = client.chat_completion(
        model=target,
        messages=[
            {
                "role": "system",
                "content": "You are a document assistant. Answer ONLY from the provided context. If the answer is not in the context, say 'I could not find this information in the uploaded documents.' Do NOT use training knowledge. Always cite the file name and page number."
            },
            {
                "role": "user",
                "content": f"CONTEXT FROM UPLOADED DOCUMENTS:\n{context}\n\nQUESTION: {question}\n\nRemember: only answer from the context above. If it is not there, say you could not find it."
            }
        ],
        max_tokens=512,
        temperature=0.1
    )
    return result.choices[0].message.content.strip()


def generate_answer_endpoint(question: str, context: str) -> str:
    """Call a dedicated HF Inference Endpoint with raw text-generation format."""
    import httpx

    prompt = build_rag_prompt(question, context)
    print(f"Calling endpoint: {RAGNAROK_ENDPOINT}")

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_TOKEN}",
        "Content-Type": "application/json"
    }

    # Try chat messages format first (TGI chat template)
    response = httpx.post(
        f"{RAGNAROK_ENDPOINT.rstrip('/')}/v1/chat/completions",
        headers=headers,
        json={
            "model": "tgi",
            "messages": [
                {"role": "system", "content": "You are a document assistant. Answer ONLY from the provided context. If the answer is not in the context, say 'I could not find this information in the uploaded documents.' Do NOT use training knowledge. Always cite the file name and page number."},
                {"role": "user", "content": f"CONTEXT FROM UPLOADED DOCUMENTS:\n{context}\n\nQUESTION: {question}\n\nRemember: only answer from the context above. If it is not there, say you could not find it."}
            ],
            "max_tokens": 512,
            "temperature": 0.1
        },
        timeout=60
    )

    # If chat endpoint not found, fall back to raw text-generation format
    if response.status_code == 404:
        response = httpx.post(
            RAGNAROK_ENDPOINT.rstrip('/'),
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 512, "temperature": 0.1, "do_sample": True, "return_full_text": False}
            },
            timeout=60
        )

    if not response.is_success:
        print(f"Endpoint error {response.status_code}: {response.text}")
    response.raise_for_status()
    data = response.json()

    # Chat completions format: {"choices": [{"message": {"content": "..."}}]}
    if "choices" in data:
        return data["choices"][0]["message"]["content"].strip()
    # TGI text-generation format: [{"generated_text": "..."}]
    if isinstance(data, list):
        return data[0].get("generated_text", "").strip()
    return data.get("generated_text", str(data)).strip()


def generate_answer_local_ragnarok(question: str) -> dict:
    """
    Forward the question to the local Ragnarok pipeline running via ngrok.
    The local server handles its own retrieval + generation and returns
    {"answer": "...", "sources": [...], "chunks_searched": N}.
    """
    import httpx

    url = f"{LOCAL_RAGNAROK_ENDPOINT.rstrip('/')}/query"
    print(f"Calling local Ragnarok: {url}")

    response = httpx.post(
        url,
        headers={
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true"
        },
        json={"question": question, "top_k": 5},
        timeout=120
    )

    if not response.is_success:
        print(f"Local Ragnarok error {response.status_code}: {response.text}")
    response.raise_for_status()
    return response.json()
