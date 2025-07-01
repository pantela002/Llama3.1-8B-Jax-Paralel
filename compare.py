def compare_token_lines(file_path1,filepath2):
    with open(file_path1, "r") as f:
        lines1 = f.readlines()
    with open(file_path1, "r") as f:
        lines2 = f.readlines()

    tokens_1 = list(map(int, lines1[0].strip().split()))
    tokens_2 = list(map(int, lines2[0].strip().split()))

    len_1 = len(tokens_1)
    len_2 = len(tokens_2)
    min_len = min(len_1, len_2)

    same = sum(t1 == t2 for t1, t2 in zip(tokens_1, tokens_2))
    diff = min_len - same

    print(f"Length of line 1: {len_1}")
    print(f"Length of line 2: {len_2}")
    print(f"Compared first {min_len} tokens.")
    print(f"Matching tokens: {same}")
    print(f"Differing tokens: {diff}")
    if len_1 != len_2:
        print(f"⚠️ Note: Token counts differ, comparison done up to shortest length.")

hf = "hf_output.txt"
jax = "jax_output.txt"
jax_paralel = "jax_output_paralel.txt"
print("-----------------------------------HF VS JAX-----------------------------------")
compare_token_lines(hf,jax)
print("-----------------------------------HF VS JAX PARALEL-----------------------------------")
compare_token_lines(jax_paralel,hf)

