<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Official Code Implementation of StablePrompt</title>
</head>
<body>
    <h1>Official Code Implementation of StablePrompt</h1>

    <p>To reproduce the experiments in the paper, follow the sequences below:</p>

    <h2>Setting Up the Environment</h2>
    <pre><code>
docker run pytorch:latest
pip install -r requirements.txt
git clone https://github.com/keirp/automatic_prompt_engineer.git
    </code></pre>

    <h2>Experiment 4.1: Few-shot Text Classification</h2>
    <pre><code>
./run_fewshot.sh
    </code></pre>

    <h2>Experiment 4.2: Induction Task</h2>
    <pre><code>
./run_BBII.sh
./run_II.sh
    </code></pre>

    <h2>Experiment 4.3: Question Answering</h2>
    <pre><code>
./run_QA.sh
    </code></pre>
</body>
</html>
