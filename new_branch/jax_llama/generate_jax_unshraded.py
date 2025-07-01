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
    ckpt_dir: str = "/root/tt/Llama3.1-8B-Jax-Paralel/new_branch/llama3.1-8B/8B",
    tokenizer_path: str = "/root/tt/Llama3.1-8B-Jax-Paralel/new_branch/llama3.1-8B/original/tokenizer.model",
    prompt = (
        "What is the name of the largest planet in our solar system?"
    ),
    max_gen_len: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    n_layers: int = 16,
    max_seq_length: int = 1024
):
    
    #example prompts
    #In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. What percentage of the entire students enrolled in hip-hop dance?
    #A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
    #A bumper car rink has 12 red cars. They have 2 fewer green cars than they have red cars. They have 3 times the number of blue cars as they have green cars. The rink also has yellow cars.  If the rink has 75 cars in total how many yellow cars do they have?
    
    #answers
    # 60
    # 3
    # 23

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
    np.savetxt("out_tokens_jax_unsharded.txt", results, fmt="%d")
if __name__ == "__main__":
    fire.Fire(main)