def compare_token_lines(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 2:
        print("File must contain at least two lines.")
        return

    tokens_1 = list(map(int, lines[0].strip().split()))
    tokens_2 = list(map(int, lines[1].strip().split()))

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

# Example usage:
compare_token_lines("/root/tt/Llama3.1-8B-Jax-Paralel/new_branch/hf_llama/merged_tokens_hf.txt")
