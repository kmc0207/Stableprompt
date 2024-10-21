#!/bin/bash
datasets=(
    'antonyms'
    'word_in_context'
    'rhymes'
    'num_to_verbal'
    'cause_and_effect'
    'larger_animal'
    'second_word_letter'
    'taxonomy_animal'
    'negation'
    'common_concept'
    'diff'
    'translation_en-es'
    'orthography_starts_with'
    'sentiment'
    'informal_to_formal'
    'sum'
    'singular_to_plural'
    'active_to_passive'
    'translation_en-de'
    'sentence_similarity'
    'translation_en-fr'
    'letters_list'
    'first_word_letter'
    'synonyms'
)

# Loop through each main dataset and run the script
for dataset in "${datasets[@]}"
do
    python stableprompt_ii.py --dataset $dataset --epoch 30 --update_term 5
done