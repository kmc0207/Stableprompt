#!/bin/bash
datasets=(
    'sst2'
    'mrpc'
    'rte'
    'mnli'
    'qnli'
    'snli'
)

# Loop through each main dataset and run the script
for dataset in "${datasets[@]}"
do
    python stableprompt_tc.py --dataset $dataset --epoch 100 --update_term 5
done