# 🗡️ Ragnarok-Docs - LLaMA 3.1 Fine-Tuning Journal

> I wrote this journal during the actual process of fine-tuning LLaMA 3.1 8B for my Ragnarok-Docs project. Every problem listed here is something I personally hit, debugged, and fixed. Nothing was added after the fact. This is the real story of how the training went.

---

## Table of Contents

- [What I Was Trying to Do](#what-i-was-trying-to-do)
- [Tools I Used](#tools-i-used)
- [Hardware : The RAM Problem That Stopped Me](#hardware--the-ram-problem-that-stopped-me)
- [Concepts I Learned Along the Way](#concepts-i-learned-along-the-way)
- [My Three Training Runs](#my-three-training-runs)
- [Every Problem I Hit and How I Fixed It](#every-problem-i-hit-and-how-i-fixed-it)
- [Dataset Comparison](#dataset-comparison)
- [What I Learned](#what-i-learned)
- [What I Would Do Differently](#what-i-would-do-differently)
- [Final Model Details](#final-model-details)

---

## What I Was Trying to Do

I wanted to fine-tune Meta's LLaMA 3.1 8B Instruct model so it becomes better at answering questions from documents. The base LLaMA model is already smart but it answers from its training knowledge. I needed it to read a specific document and answer only from what is in that document with citations.

The method I used is called QLoRA. Instead of retraining all 8 billion parameters which would take days and cost thousands of dollars, QLoRA adds small trainable layers on top of the frozen model. I only trained 41.9 million out of 8 billion parameters just 0.52% of the model. The result is a small adapter file that sits on top of LLaMA and changes its behavior.

---

## Tools I Used

### Google Colab

I used Google Colab because fine-tuning LLaMA 3.1 8B needs a powerful GPU and I do not own one. Colab runs in the browser and gives GPU access through Google's cloud servers.

**I started with the free tier.** This gave me a T4 GPU with 14.6GB of VRAM and 12.7GB of RAM. I thought this would be enough since LLaMA in 4-bit only needs 5.3GB of VRAM. I was wrong and I will explain exactly why below.

**I eventually upgraded to Colab Pro ($10/month).** This gave me an A100 GPU with 39.5GB VRAM and 83.5GB RAM. The model loaded in 10 seconds with no issues. I cancelled the subscription after training completed.

**The painful part about Colab free tier:**
- Every time my session crashed I lost everything model in memory, variables, packages
- I had to reinstall all packages from scratch every restart
- The session would disconnect after 90 minutes of inactivity
- There is a 12 hour maximum session limit

---

### HuggingFace

I used HuggingFace for three things:

**1. Getting LLaMA 3.1** : The model lives on HuggingFace at `meta-llama/Llama-3.1-8B-Instruct`. I had to fill out a form to request access from Meta. It was approved in about an hour. I needed a HuggingFace token to download it.

**2. Getting the training datasets** : Both SQuAD v2 and Natural Questions are on HuggingFace. I downloaded them with one line of code:
```python
dataset = load_dataset("sentence-transformers/natural-questions")
```

**3. Hosting my trained adapter** : After training I pushed my LoRA adapter to HuggingFace so it is permanently stored and publicly accessible. Anyone with LLaMA 3.1 access can load my adapter.

---

### GitHub

My project code lives at `https://github.com/Avinash-4/ragnarok-docs`. I used GitHub for all the application code, the FastAPI backend, the RAG pipeline, the training notebooks.

What I do NOT store on GitHub:
- Model weights (too large, use HuggingFace instead)
- Training data (too large, use HuggingFace datasets)
- The ChromaDB vector database (local only)

GitHub is for code. HuggingFace is for models and datasets. This separation matters.

---

### Google Drive

I used Google Drive to save training checkpoints between Colab sessions. Every 50 training steps the trainer automatically saved the LoRA adapter to Drive. This way if my session crashed I could resume from the last checkpoint instead of starting over.

The problem I hit: Google Drive free tier is 15GB. The LLaMA model cache alone needs 16GB. I could not store the full model cache in Drive, only the small checkpoints (about 80-200MB each).

---

### Python Libraries

Getting the right package versions was one of the biggest headaches of this project. Here is what I ended up with that actually worked:

```
transformers==4.44.0
trl==0.8.6
peft==0.11.1
bitsandbytes>=0.49.0
torch==2.10.0+cu128  (pre-installed by Colab, do not touch)
triton               (UNINSTALLED - caused conflicts)
```

The version conflicts I hit are documented in detail below.

---

## Hardware: The RAM Problem That Stopped Me

This was the biggest blocker I faced and I want to explain it clearly because it confused me for a long time.

**What I thought:** LLaMA 3.1 8B in 4-bit needs 5.3GB of VRAM. The T4 has 14.6GB. So it should fit with room to spare.

**What actually happened:** My session crashed every single time at exactly 67% of model loading.

**What I eventually understood:** There is a difference between VRAM (GPU memory) and RAM (CPU memory). They are completely separate.

Loading the model happens in two stages:

```
Stage 1: Read from disk into CPU RAM
The model files on disk are 16GB in fp16 format
To load them, the CPU needs to temporarily hold ~13GB in RAM
T4 Colab RAM: 12.7GB  ←  NOT ENOUGH. Crash at 67%.

Stage 2: Compress and move to GPU
After compression to 4-bit: 5.3GB
Move to GPU VRAM: 5.3GB out of 14.6GB  ←  This would have been fine
```

The T4 never made it to Stage 2. It always crashed in Stage 1.

I tried many things to fix this:
- `low_cpu_mem_usage=True` slowed loading down but still crashed
- Storing model cache on Google Drive : still needed RAM for the conversion
- Clearing all other processes : only freed about 200MB, not enough
- Restarting multiple times : same crash every time

**The fix:** Upgrading to Colab Pro with A100 (83.5GB RAM). The model loaded in 10 seconds. Peak RAM usage during loading was about 16GB out of 83.5GB available.

---

## Concepts I Learned Along the Way

### Loss

Loss measures how wrong the model's predictions are. Lower is better.

```
Loss 2.0  →  Random guessing, knows nothing
Loss 1.0  →  Starting to learn patterns
Loss 0.7  →  Performing well
Loss 0.3  →  Performing very well
```

### Training Loss vs Validation Loss

I split my data into two groups:

- **Training set** : the examples the model learns from (5,000-20,000 examples)
- **Validation set** : examples held back that the model never trains on (500-1,000 examples)

Every 50-100 steps I checked both:

**Training loss** tells me how well the model is doing on what it is learning.

**Validation loss** tells me how well it generalizes to new data it has never seen.

**The gap between them is the most important thing to watch.** A small gap means the model is genuinely learning. A large gap means it is memorizing.

I learned this the hard way with my V1 training run:

```
Step 700:
Training loss:   0.189  ← model thinks it's a genius
Validation loss: 1.927  ← actually terrible on new data
Gap:             1.738  ← this is overfitting
```

### Epochs

One epoch means the model has seen every training example exactly once.

```
Epoch 0.24  →  Seen 24% of the training data
Epoch 1.0   →  Seen every example once, one full pass done
Epoch 2.0   →  Seen every example twice
Epoch 3.0   →  Seen every example three times
```

I used 3 epochs in my V1 run and got severe overfitting. The model saw each of my 5,000 examples three times and memorized them.

I switched to 1 epoch in V2 and the overfitting disappeared.

The shorter answers in SQuAD v2 made this worse: the model memorized them faster than longer answers would allow.

### Overfitting

Overfitting is when a model memorizes training data instead of learning general patterns.

**The analogy that helped me understand it:** Imagine a student who memorizes the exact practice test questions word for word. They score 100% on the practice test but fail the real exam because they never actually learned the concepts they just memorized the specific questions.

That is exactly what happened in my V1 run. The model memorized 5,000 short answers like "France" and "1981" and got a perfect training score but completely failed on new examples it had never seen.

**How I detected it:** Watched the gap between training loss and validation loss.

**How I fixed it:** Switched to a dataset with longer, more complex answers that are harder to memorize.

### bf16 vs fp16

Both are 16-bit number formats but they work differently.

fp16 can only represent numbers up to 65,504. In a deep neural network with 32 layers of matrix multiplications, numbers can easily exceed this limit. When they do they become infinity, then NaN (not a number), then the loss becomes NaN.

bf16 can represent numbers up to 3.4×10³⁸ same range as 32-bit floats. It never overflows.

The A100 GPU was specifically designed to work with bf16. When I switched to loading and training both in bf16, the NaN loss problem disappeared completely.

---

## My Three Training Runs

### V1 : First Attempt with SQuAD v2

**What I used:**
- Dataset: Stanford Question Answering Dataset v2
- 5,000 training examples
- 3 epochs
- Learning rate: 2e-4
- Dropout: 0.05

**What I saw:**

| Step | Training Loss | Validation Loss | Gap |
|---|---|---|---|
| 100 | 1.342 | 1.537 | 0.195 |
| 200 | 0.908 | 1.601 | 0.693 |
| 300 | 0.585 | 1.728 | 1.143 |
| 400 | 0.411 | 1.800 | 1.389 |
| 500 | 0.364 | 1.775 | 1.411 |
| 600 | 0.239 | 1.935 | 1.696 |
| 700 | 0.189 | 1.927 | 1.738 |

**I stopped it at step 712.** The validation loss was rising while training loss was collapsing. Classic overfitting.

**What I diagnosed:** I ran a dataset check and found:

```
Average answer length:  2.1 words
Minimum answer length:  1 word
Maximum answer length:  5 words
```

The answers were things like "France", "1981", "in the late 1990s". LLaMA already knew these facts. It memorized all 5,000 answers within a few hundred steps and had nothing left to learn.

**What I saved:** `llama3-ragnarok-debug-v1` : kept it as a record.

---

### V2 : Fixing the Overfitting with Natural Questions

**What I changed:**

| Setting | V1 | V2 | My reason |
|---|---|---|---|
| Dataset | SQuAD v2 | Natural Questions | Longer answers |
| Avg answer length | 2.1 words | 26.9 words | Much harder to memorize |
| Epochs | 3 | 1 | Stop before memorization happens |
| Dropout | 0.05 | 0.10 | More regularization |
| Learning rate | 2e-4 | 1e-4 | More careful learning |
| Model precision | fp16 | bf16 | A100 native, no NaN |
| Min answer filter | None | 8 words | Remove trivially short answers |

**What I saw:**

| Step | Training Loss | Validation Loss | Gap |
|---|---|---|---|
| 50 | 1.036 | 1.023 | 0.013 |
| 100 | 0.972 | 0.996 | 0.024 |
| 150 | 1.010 | 0.988 | 0.022 |
| 200 | 0.972 | 0.984 | 0.012 |

**That gap of 0.012 made me very happy.** Compare to V1's gap of 1.738 at the same point. The model was genuinely learning, not memorizing.

**Final validation loss: ~0.98**

Not bad but I wanted to push it lower. The model had only seen 5,000 examples once. I decided to scale up.

**What I saved:** `llama3-ragnarok-v2` : downloaded to my laptop immediately after training.

---
### V3 — Scaling Up (Final Model)

**What I changed from V2:**

| Setting | V2 | V3 | My reason |
|---|---|---|---|
| Training examples | 5,000 | 20,000 | 4x more data |
| Epochs | 1 | 2 | More passes through richer data |

**Results:**

| Step | Training Loss | Validation Loss | Gap |
|---|---|---|---|
| 100 | 1.097 | 0.998 | 0.099 |
| 500 | 0.962 | 0.940 | 0.022 |
| 1000 | 0.932 | 0.918 | 0.014 |
| 1300 | 0.845 | 0.910 | 0.065 |
| 1500 | 0.814 | 0.909 | 0.095 |
| 1700 | 0.824 | 0.907 | 0.083 |
| 2000 | 0.818 | 0.904 | 0.086 |
| 2500 | 0.829 | 0.903 | 0.074 |

**Final training loss: 0.829**

**Final validation loss: 0.903**

**Best checkpoint: Step 2500 : loaded automatically by `load_best_model_at_end=True`**

**What the numbers tell me:**

The gap between training and validation loss stayed between 0.014 and 0.095 throughout training far healthier than V1's gap of 1.738. Validation loss consistently improved from 0.998 down to 0.903. The slight gap widening between steps 1300-1500 was when the second epoch started normal behavior and it stabilized back down by step 2500.

Compared to V2's final validation loss of 0.98, V3 achieved 0.903 an 8% improvement from using 4x more data and 2 epochs.

**Published at:** `huggingface.co/avinashkongara4/llama3-ragnarok-nq-adapter`

**Adapter size:** 83.9 MB

---

## Every Problem I Hit and How I Fixed It

### Problem 1 : Model Crashed at 67% on Free Colab

I tried to load LLaMA 3.1 8B on the free T4 GPU. The session crashed every single time at exactly 67% of model loading.

**What I tried that did not work:**
- `low_cpu_mem_usage=True` - made loading slower, still crashed
- Clearing GPU cache - freed VRAM which was not the problem
- Mounting Google Drive for cache - model still needed RAM for the conversion
- Restarting multiple times - same result every time

**What actually fixed it:** Got the A100 with 83.5GB RAM. Model loaded successfully.

**What I learned:** VRAM and RAM are completely different. My model needed 5.3GB VRAM but 13GB RAM during loading. The free T4 only had 12.7GB RAM total.

---

### Problem 2 : Loss Was NaN Before Training Even Started

After I got the A100 I ran a check before training and saw `Loss: nan`. This confused me because the model had loaded fine.

**What I figured out:** I had loaded the model in `torch.float16` but set `bf16=True` in training. These two precisions have different number ranges. During the forward pass some values exceeded fp16's maximum (65,504) and became infinity, then NaN.

**What fixed it:**
```python
# Load model in bf16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16  ← changed from float16
)

# Train in bf16
training_args = TrainingArguments(
    bf16=True,   ← matches model dtype
    fp16=False   ← explicitly off
)
```

Matching the model and training precision eliminated the NaN completely.

**I now always run this check before training:**
```python
model.eval()
with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
print(f"Loss: {outputs.loss.item()}")  # If this is nan, do not start training
```

---

### Problem 3 : Training Loss Showed Exactly 0.000000

This was the most confusing problem. Training started, ran for 100 steps, and the loss showed 0.0 exactly. Not a small number, exactly zero.

**What I found:** Two separate issues caused this at different times.

First time: I was using `DataCollatorForCompletionOnlyLM` which only computes loss on the answer portion. My answers were 2.1 words on average: 8 tokens out of 223. The model already knew these 8-token answers perfectly from pre-training. Loss immediately went to 0.

Second time: NaN loss was being silently reported as 0.0 by the trainer.

**What fixed it:** Removed the data collator and computed loss on the full sequence. Fixed the dtype mismatch to eliminate NaN.

---

### Problem 4 : Severe Overfitting on SQuAD v2

Training loss 0.189, validation loss 1.927. The model was memorizing, not learning.

**How I diagnosed it:** I ran this check on the dataset:
```python
answer_lengths = []
for i in range(100):
    text = train_dataset[i]['text']
    answer = text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
    answer = answer.replace("<|eot_id|>", "").strip()
    answer_lengths.append(len(answer.split()))

print(f"Average: {sum(answer_lengths)/len(answer_lengths):.1f} words")
# Output: Average: 2.1 words
```

2.1 words. That was the problem. The model memorized thousands of one and two word answers instantly.

**What fixed it:** Switched to Natural Questions (26.9 word average) with a minimum answer length filter of 8 words.

---

### Problem 5 : Package Version Conflicts

This was the most time consuming problem I faced. Here are the specific conflicts I hit:

**Conflict 1 : triton versions:**
- `torch 2.10.0` needed `triton>=3.x`
- `bitsandbytes 0.43.x` needed `triton==2.3.1`
- These cannot both be installed
- **My fix:** Uninstall triton completely. Use `bitsandbytes>=0.49.0` which does not need triton.

**Conflict 2 : transformers and trl:**
- `transformers 5.x` removed `tokenizer` argument from the base Trainer class
- `trl 0.8.6` SFTTrainer still passes `tokenizer` internally
- **My fix:** Downgrade to `transformers==4.44.0` which still supports the argument.

**Conflict 3 : bitsandbytes and CUDA:**
- `bitsandbytes 0.43.x` does not support CUDA 12.8
- Colab A100 runs CUDA 12.8
- **My fix:** Use `bitsandbytes>=0.49.0` which added CUDA 12.8 support.

**Conflict 4 : peft and bitsandbytes:**
- `peft 0.11.1` expected `memory_efficient_backward` attribute in bitsandbytes MatmulLtState
- `bitsandbytes 0.49.x` removed that attribute
- This made 8-bit loading impossible
- **My fix:** Switched from 8-bit to 4-bit quantization which uses a completely different code path and does not have this conflict.

**Final working versions:**
```
transformers==4.44.0
trl==0.8.6
peft==0.11.1
bitsandbytes>=0.49.0
triton → uninstalled
```

---

### Problem 6 : Colab Session Crashes Losing All Progress

This happened multiple times on the free T4. The session would crash and everything in memory was gone model, variables, training progress, installed packages.

**What I did to protect against this:**

1. Saved datasets to Google Drive so I did not have to redownload:
```python
train_dataset.save_to_disk("/content/drive/MyDrive/train_v3")
val_dataset.save_to_disk("/content/drive/MyDrive/val_v3")
```

2. Saved checkpoints to Google Drive every 50 steps:
```python
output_dir="/content/drive/MyDrive/ragnarok-checkpoints"
save_steps=50
```

3. Pasted the keep-alive JavaScript in the browser console to prevent idle timeout.

4. Set a phone alarm every 45 minutes to come back and interact with the page.

**Ultimate fix:** The A100 on Colab Pro was much more stable. I completed the full training.

---

## Dataset Comparison

| Feature | SQuAD v2 | Natural Questions |
|---|---|---|
| Who made it | Stanford University | Google Research |
| Total examples | 130,000 | 100,231 |
| How many I used | 5,000 | 5,000 → 20,000 |
| Average answer length | 2.1 words | 26.9 words |
| Type of answers | Short extracted phrases | Full explanatory sentences |
| Real user questions | No written by crowd workers | Yes real Google searches |
| Has unanswerable questions | Yes | No |
| Led to overfitting | Yes severe | No |
| Good for RAG training | Partially | Yes |

**Why I switched:** SQuAD v2 answers are things like "France", "1981", "researchers at Google". These are so short that the model memorized all 5,000 of them in about 300 training steps and had nothing left to learn.

Natural Questions has answers like full paragraphs explaining who did what and when. Much harder to memorize. Forces the model to actually understand how to read a document and construct an answer.

---

## What I Learned

**Lesson 1 : Diagnose before you fix**

When I saw 0.0 loss I immediately started changing settings randomly. I should have run a forward pass first to understand what was actually broken. One diagnostic check would have saved hours.

**Lesson 2 : Dataset quality matters more than size**

5,000 examples with 26-word answers trained better than 5,000 examples with 2-word answers. The model needs meaningful content to learn from.

**Lesson 3 : Match precision throughout**

Loading in fp16 and training in bf16 caused NaN. Always match the model dtype with the training dtype. On A100 use bf16 for both.

**Lesson 4 : Check RAM not just VRAM**

I kept thinking the T4 had enough VRAM (14.6GB) so the model should load. The model only needs 5.3GB VRAM. But loading it requires 13GB of RAM which the T4 does not have. Always check peak RAM requirements, not just final GPU memory.

**Lesson 5 : Watch the gap between train and val loss**

The absolute loss number matters less than the gap between training and validation loss. V1 had training loss 0.189 - sounds great. But the validation gap was 1.738 - terrible. V2 had training loss 0.972 - sounds worse. But the gap was 0.012 - much healthier.

**Lesson 6 : One epoch is often enough**

I used 3 epochs in V1 and got severe overfitting. One epoch in V2 eliminated it. For instruction fine-tuning on diverse data with long answers one epoch frequently gives good results without memorization.

**Lesson 7 : Push to HuggingFace the moment training finishes**

GPU memory and Colab disk are temporary. HuggingFace is permanent. I lost training progress multiple times to session crashes. Now the first thing I do after seeing "Training Complete" is run save and push.

**Lesson 8 : Pin package versions from day one**

Version conflicts caused more debugging time than any other issue. The ML ecosystem changes constantly. Pin exact versions in a requirements file and never upgrade mid-project.

---

## What I Would Do Differently

1. Use the A100 from day one the free T4 was never going to work for LLaMA 3.1 8B
2. Run the loss verification check before every training attempt
3. Check dataset statistics (average answer length) before training starts
4. Use Natural Questions from the beginning better dataset for this task
5. Pin all package versions before writing any training code
6. Set up Google Drive checkpoints before the first training step
7. Load and train both in bf16 on A100, would have avoided all NaN issues

---

## Final Model Details

| Detail | Value |
|---|---|
| Base model | meta-llama/Llama-3.1-8B-Instruct |
| Fine-tuning method | QLoRA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target layers | q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj |
| Parameters trained | 41,943,040 out of 8,072,204,288 (0.52%) |
| Dataset | Google Natural Questions |
| Training examples | 20,000 |
| Validation examples | 1,000 |
| Epochs | 2 |
| Learning rate | 1e-4 |
| Dropout | 0.10 |
| GPU | NVIDIA A100-SXM4-40GB |
| Precision | bfloat16 |
| Final training loss | TBD |
| Final validation loss | TBD |
| Adapter file size | ~80MB |
| Published at | huggingface.co/avinashkongara4/llama3-ragnarok-nq-adapter |

---

*I wrote this during the actual training process. Every number, every error, every fix is real. I kept notes as things happened so I would remember exactly what worked and what did not.*