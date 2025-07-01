import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax
import jax.numpy as jnp
import numpy as np
import fire
from flax.core.frozen_dict import freeze
from model import FlaxLLaMAForCausalLM
from convert_weights import convert_llama_weights
from llama3_tokenizer import Tokenizer as LLaMA3Tokenizer
from generation import LLaMA  # your class is here
from jax.sharding import Mesh
import gc

def jax_load(ckpt_dir: str, tokenizer_path: str, mesh, max_seq_length: int = 2048, n_layers: int = 32) -> LLaMA:
    print("🔧 Loading tokenizer and weights...")
    tokenizer = LLaMA3Tokenizer(tokenizer_path)

    params_np, jax_config = convert_llama_weights(
        ckpt_dir=ckpt_dir,
        tokenizer=tokenizer,
        max_seq_len=max_seq_length,
        n_layers=n_layers        
    )
    jax_params = freeze(jax.tree.map(jnp.asarray, params_np))

    del params_np
    gc.collect()    

    model = FlaxLLaMAForCausalLM(config=jax_config, _do_init=False)
    llama = LLaMA(params=jax_params, model=model, tokenizer=tokenizer, mesh=mesh)
    del jax_params
    gc.collect()
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
    # Define mesh
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('mp',))    
    print("✅ Mesh initialized:", mesh)

    print("🚀 Loading LLaMA...")
    llama = jax_load(ckpt_dir, tokenizer_path, mesh, max_seq_length=max_seq_length, n_layers=n_layers)

    print("✍️ Generating...")
    with mesh:
        results = llama.generate_from_str(
            [prompt],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            do_sample=False
        )
    del llama
    gc.collect()
    print("✅ Generation complete.")
    np.savetxt("output_jax.txt", results, fmt="%d")
    #for i, r in enumerate(results):
    #    print(f"\n🧾 Prompt {i + 1}: {prompt}")
    #    print("🧠 Output:", r)

if __name__ == "__main__":
    fire.Fire(main)
