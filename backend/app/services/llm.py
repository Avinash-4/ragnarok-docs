from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
import os
import torch

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Global variable to cache the model so we only load it once
_model_pipeline = None


def load_llm():
    """
    Loads LLaMA 3.1 8B Instruct model from HuggingFace.

    Uses 4-bit quantization (load_in_4bit=True) so the 8B parameter
    model fits in less RAM. Without this, it needs ~16GB RAM.
    With 4-bit quantization, it needs only ~5GB RAM.

    The model is cached globally so it only loads once per server start.
    Loading takes ~2-3 minutes the first time (downloads ~4GB of weights).

    Returns:
        HuggingFace text generation pipeline
    """
    global _model_pipeline

    # Return cached model if already loaded
    if _model_pipeline is not None:
        print("Using cached LLaMA model")
        return _model_pipeline

    print(f"Loading LLaMA model: {MODEL_ID}")
    print("This takes 2-3 minutes on first load...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HUGGINGFACE_TOKEN
    )

    # Detect if GPU is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HUGGINGFACE_TOKEN,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True
    )

    _model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    print("LLaMA model loaded successfully!")
    return _model_pipeline


def build_rag_prompt(question: str, context: str) -> str:
    """
    Builds the prompt that gets sent to LLaMA.

    The prompt instructs LLaMA to ONLY answer from the provided
    context and to cite which sources it used. This prevents
    hallucination — the model making things up.

    Args:
        question: The user's question
        context: The retrieved chunks from ChromaDB

    Returns:
        Formatted prompt string ready for LLaMA
    """
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful document assistant for the ragnarok-docs system.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say "I could not find this information in the uploaded documents."
Always mention which source (file name and page number) your answer comes from.
Be concise and accurate.
<|eot_id|><|start_header_id|>user<|end_header_id|>

CONTEXT FROM DOCUMENTS:
{context}

QUESTION: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    return prompt


def generate_answer(question: str, context: str) -> str:
    """
    The main function — takes a question and context chunks,
    builds the RAG prompt, sends it to LLaMA 3, and returns
    the generated answer.

    This is the GENERATION step in RAG.

    Args:
        question: User's question
        context: Formatted context string from retriever.py

    Returns:
        LLaMA's answer as a string
    """
    llm = load_llm()

    prompt = build_rag_prompt(question, context)

    print(f"Generating answer for: '{question}'")

    # Generate the response
    outputs = llm(prompt)

    # Extract just the assistant's reply from the full output
    full_output = outputs[0]["generated_text"]

    # Split on the assistant header to get only the answer part
    answer = full_output.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    answer = answer.strip()

    return answer


def generate_answer_hf_api(question: str, context: str) -> str:
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=HUGGINGFACE_TOKEN)

    result = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful document assistant. Answer using only the provided context. Always cite the source file and page number."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=512,
        temperature=0.1
    )

    answer = result.choices[0].message.content
    return answer.strip()
    