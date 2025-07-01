# 🧠 LLaMA 3.1–8B: Tensor Parallel JAX Implementation (Draft PR)

This draft PR adds a tensor-parallel JAX implementation of Meta’s LLaMA 3.1–8B model using a 1×4 device mesh. The code supports both sharded and unsharded execution and matches Hugging Face’s PyTorch reference implementation.

---

## ✅ Setup Instructions

### 🌿 Branch for This Implementation
```
All changes for this draft PR are in the branch:

Llama3.1-8B-paralel

Clone the repository and checkout the branch:
git checkout Llama3.1-8B-paralel

cd Llama3.1-8B-Jax-Paralel
```

### 📦 Install Python Dependencies
```
Make sure you're using Python ≥3.10 (tested on 3.12):

pip install -r requirements.txt
```


### 🌿 Hugging Face Login
```
You must log into Hugging Face to download the LLaMA 3.1 weights.

pip install huggingface_hub
huggingface-cli login

    Make sure you've requested access to the Meta LLaMA 3 model: https://huggingface.co/meta-llama
```

### 🌿 Download and Structure Model Files
```

huggingface-cli download meta-llama/Llama-3.1-8B original/tokenizer.model --local-dir sw/llama3.1-8B/original
huggingface-cli download meta-llama/Llama-3.1-8B original/consolidated.00.pth --local-dir sw/llama3.1-8B
huggingface-cli download meta-llama/Llama-3.1-8B original/params.json --local-dir sw/llama3.1-8B

mv llama3.1-8B/consolidated.00.pth sw/llama3.1-8B/8B/
mv llama3.1-8B/params.json sw/llama3.1-8B/8B/

Final structure:

llama3.1-8B/
├── 8B/
│   ├── consolidated.00.pth
│   └── params.json
└── original/
    └── tokenizer.model
```


### ▶️ Running the Scripts
```
You can run any of the available generation scripts using:

python3 jax_llama/generate_jax.py
python3 jax_llama/generate_jax_paralel.py
python3 hf_llama/generate_hf.py

    generate_jax_paralel.py: Runs the sharded tensor-parallel JAX model (1×4 mesh).

    generate_jax.py: Runs the unsharded JAX model.

    generate_hf.py: Runs the Hugging Face PyTorch reference model.


✅ All three scripts produce identical outputs for the same input prompt (up to floating point precision).
```

