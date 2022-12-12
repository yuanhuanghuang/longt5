option_keys = sorted([
    key for key in examples
    if key.startswith("option_")
])
input_strs = []
target_strs = []

for i in range(len(examples[option_keys[0]])):
    # There are all 6 <eos>
    # I can set only 4 <eos>
    all_options = "".join([f" <eos> choice {j}: {examples[option_key][i]}" for j, option_key in enumerate(option_keys)])
    input_str = f"{all_options} question: {examples['query'][i]}  context: {examples['context'][i]}"
    target_str = f"{examples['label'][i]}"
    input_strs.append(input_str)
    target_strs.append(target_str)