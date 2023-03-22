import pandas as pd
import numpy as np
import os
import json
import sys

acc = []
checkpoints = [14000, 28000, 42000, 56000, 70000, 84000, 98000, 112000, 126000]

for checkpoint in checkpoints: 
    f = open('/home/ubuntu/stanford-babylm/blimp/results/blimp_eval_xsmall-' + str(checkpoint) + '.txt')
    root = "/home/ubuntu/stanford-babylm/blimp/data"
    result_dict = {'Task':[], "Value":[], "Stderr":[]}
    f.readline()
    f.readline()
    for line in f:
        cur_line = line.strip().split('|')
        if len(cur_line) == 8:
            _,task, _, _, value,_,stderr,_ = cur_line
            # '', 'Task', 'Version', 'Metric', 'Value', '   ', 'Stderr', ''
            task_name = task[6:].strip()
            try:
                float_value = float(value)
            except ValueError:
                continue
            result_dict["Task"].append(task_name)
            result_dict["Value"].append(float(value))
            result_dict["Stderr"].append(float(stderr))
            # data_file = open(os.path.join(root, task_name+".jsonl"))
            # for l in data_file:
            #     l = json.loads(l)
            #     linguistics_term = l["linguistics_term"]
            #     result_dict["linguistics_term"].append(linguistics_term)
            #     break
    
    df = pd.DataFrame.from_dict(result_dict)
    # df1 = df.groupby('linguistics_term')['Value'].mean()
    acc.append(df['Value'].mean())
print(acc)