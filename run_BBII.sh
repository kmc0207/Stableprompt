#!/bin/bash
datasets=(
    'causal_judgment'
    'disambiguation_qa'
    'epistemic_reasoning'
    'hyperbaton'
    'implicatures'
    'logical_fallacy_detection'
    'movie_recommendation'
    'navigate'
    'presuppositions_as_nli'
    'ruin_names'
    'snarks'
    'sports_understanding'
    'winowhy'
)

# Loop through each main dataset and run the script
for dataset in "${datasets[@]}"
do
    python stableprompt_bbii_tc.py --dataset $dataset --epoch 30 --update_term 5
done

datasets=(
    'dyck_languages'
    'gender_inclusive_sentences_german'
    'object_counting'
    'operators'
    'tense'
    'word_sorting'
    'linguistics_puzzles'
)

# Loop through each main dataset and run the script
for dataset in "${datasets[@]}"
do
    python stableprompt_bbii_tg.py --dataset $dataset --epoch 30 --update_term 5
done