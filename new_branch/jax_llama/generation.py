import jax.numpy as jnp
from model import FlaxLLaMAForCausalLM
from llama3_tokenizer import Tokenizer as LLaMA3Tokenizer
from transformers.generation import GenerationConfig
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import PyTree
from flax import struct
from typing import List, Optional


class LLaMA(struct.PyTreeNode):
    params: PyTree
    model: FlaxLLaMAForCausalLM = struct.field(pytree_node=False)
    tokenizer: LLaMA3Tokenizer = struct.field(pytree_node=False)
    mesh: Optional[Mesh] = struct.field(pytree_node=False, default=None)

    def generate(self, tokens: jnp.ndarray, attention_mask: jnp.ndarray, max_gen_len: int, temperature: float = 0.0, top_p: float = 1, do_sample: bool = False) -> jnp.ndarray:
        generations = self.model.generate(
            input_ids=tokens, 
            attention_mask=attention_mask, 
            params=self.params, 
            generation_config=GenerationConfig(
                num_beams=1, 
                do_sample=do_sample, 
                max_length=max_gen_len+tokens.shape[1], 
                pad_token_id=self.tokenizer.eos_id, 
                eos_token_id=self.tokenizer.eos_id, 
                temperature=temperature, 
                top_p=top_p,
            )
        )
        return generations.sequences
    
    def generate_from_str(self, prompts: List[str], max_gen_len: int, temperature: float = 0.0, top_p: float = 1, do_sample: bool = False) -> jnp.ndarray:
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False,  allowed_special="all", disallowed_special=()) for x in prompts]


        max_prompt_size = max([len(t) for t in prompt_tokens])

        tokens = jnp.full((len(prompts), max_prompt_size), self.tokenizer.pad_id).astype(jnp.int32)
        for i, t in enumerate(prompt_tokens):
            tokens = tokens.at[i, -len(t):].set(t) # left pad
        attention_mask = (tokens != self.tokenizer.eos_id).astype(jnp.int32)
                
        out_tokens = self.generate(tokens, attention_mask, max_gen_len, temperature, top_p, do_sample=False)
        
        return out_tokens