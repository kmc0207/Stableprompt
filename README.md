# Official Code Implementation of StablePrompt

To reproduce the experiments in the paper, follow the sequences below:

## Setting Up the Environment
```bash
docker run pytorch:latest
pip install -r requirements.txt
git clone https://github.com/keirp/automatic_prompt_engineer.git
```
## Experiment 4.1: Few-shot Text Classification
```bash
./run_fewshot.sh
```

## Experiment 4.2: Induction Task
```bash
./run_BBII.sh
./run_II.sh
```

## Experiment 4.3: Question Answering
```bash
./run_QA.sh
```
