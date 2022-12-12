import json
import pandas as pd
import numpy as np
phases = ['train','validation']
for phase in phases:

    input_str = [] #source
    target_str = [] #target
    data = pd.read_json(phase + '.jsonl',lines = True)
    cols = list(data.columns)
    eos = ' <extra_id_99> '

    for i in range(len(data['context'])):
        this_text = ''
        question = ''
        all_options = ''
        cont = ''
        for col in cols[:-1]:
            if 'option' in col:
                all_options = all_options + eos + col + ': ' + data.iloc[i][col].replace('\n', '').replace('\r', '')
            if 'query' in col:
                question = 'question: ' + data.iloc[i][col].replace('\n', '').replace('\r', '') + ' '
            if 'context' in col:
                cont = data.iloc[i][col].replace('\n', '').replace('\r', '')
        this_text = f"{all_options} {question}  context: {cont}"
        input_str.append(this_text)
        target_str.append(str(data.iloc[i]['label']))

    name1 = phase + '_source.txt'
    name11 = phase + '_source.json'
    name2 = phase + '_target.txt'
    name22 = phase + '_target.json'
    df1 = pd.DataFrame(input_str,columns=['source'])
    df2 = pd.DataFrame(target_str,columns=['target'])
    df1.to_csv(name1, sep='\n', index=False,header=False)
    df2.to_csv(name2, sep='\n', index=False,header=False)

'''

option_keys = sorted([
    key for key in examples
    if key.startswith("option_")
])
input_strs = []
target_strs = []

for i in range(len(examples[option_keys[0]])):
    # There are all 6 <eos>
    # I can set only 4 <eos>
    all_options = "".join([f" <extra_id_99> choice {j}: {examples[option_key][i]}" for j, option_key in enumerate(option_keys)])
    input_str = f"{all_options} question: {examples['query'][i]}  context: {examples['context'][i]}"
    target_str = f"{examples['label'][i]}"
    input_strs.append(input_str)
    target_strs.append(target_str)
'''