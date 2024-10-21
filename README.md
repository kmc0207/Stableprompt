Official Code Implementation of StablePrompt

To reproduce the experiments in paper, following below sequences.

Setting : 
docker run pytorch:latest
pip install -r requirements.txt
git clone https://github.com/keirp/automatic_prompt_engineer.git


Experiment 4.1. Few-shot Text classification
./run_fewshot.sh


Experiment 4.2 Induction Task
./run_BBII.sh
./run_II.sh


Experiment 4.3 Question Answering
./run_QA.sh




