import os
import jax
import jax.numpy as jnp
import numpy as np
import fire
from flax.core.frozen_dict import freeze
from model import FlaxLLaMAForCausalLM
from convert_weights import convert_llama_weights
from llama3_tokenizer import Tokenizer as LLaMA3Tokenizer
from generation import LLaMA  # your class is here
from pathlib import Path
ROOT = Path(__file__).parent.parent  # This gives you the path to "project/"


def jax_load(ckpt_dir: str, tokenizer_path: str, max_seq_length: int = 2048, n_layers: int = 32) -> LLaMA:
    print("🔧 Loading tokenizer and weights...")
    tokenizer = LLaMA3Tokenizer(tokenizer_path)
    print(n_layers, max_seq_length)
    jax_params, jax_config = convert_llama_weights(
        ckpt_dir=ckpt_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_length,
        n_layers=n_layers
    )
    jax_params = freeze(jax.tree.map(jnp.asarray, jax_params))
    model = FlaxLLaMAForCausalLM(config=jax_config, _do_init=False)
    llama = LLaMA(params=jax_params, model=model, tokenizer=tokenizer)
    
    return llama

def main(
    ckpt_dir = str(ROOT / "llama3.1-8B/8B"),
    tokenizer_path =  str(ROOT / "llama3.1-8B/original/tokenizer.model"),
    prompt = (
        "What is the name of the largest planet in our solar system?"
    ),
    max_gen_len: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    n_layers: int = 16,
    max_seq_length: int = 1024
):
    print("🚀 Loading LLaMA...")
    llama = jax_load(ckpt_dir, tokenizer_path, max_seq_length=max_seq_length, n_layers=n_layers)

    print("✍️ Generating...")
    results = llama.generate_from_str(
        [prompt],
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        do_sample=False,  # greedy decoding

    )
    np.savetxt("jax_output.txt", results, fmt="%d")
if __name__ == "__main__":
    fire.Fire(main)