import sys
from llama3_tokenizer import Tokenizer as LLaMA3Tokenizer

def decode_tokens_file():
    tokens_file_path = "/root/tt/Llama3.1-8B-Jax-Paralel/new_branch/jax_llama/out_tokens_jax_unsharded.txt"
    tokenizer_path = "/root/tt/Llama3.1-8B-Jax-Paralel/new_branch/llama3.1-8B/original/tokenizer.model"
    # Initialize tokenizer
    tokenizer = LLaMA3Tokenizer(model_path=tokenizer_path)

    with open(tokens_file_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        token_ids = list(map(int, line.strip().split()))
        decoded = tokenizer.decode(token_ids)
        print(f"[Sample {i}]")
        print("Token IDs:", token_ids)
        print("Decoded text:", decoded)
        print("=" * 50)

if __name__ == "__main__":

    decode_tokens_file()
