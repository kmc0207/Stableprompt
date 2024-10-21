import os
import json
import random
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import pandas as pd
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader,TensorDataset
from datasets import load_dataset
from tqdm.auto import tqdm
import argparse
from PIL import Image
import os
from peft import LoraConfig
import warnings
import numpy as np
import wandb
import copy
from collections import deque
from transformers import ViltProcessor, ViltForQuestionAnswering
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
import random
import torch
import heapq
import torch.nn.functional as F
from collections import Counter
import string
from datasets import load_metric
from collections import Counter
from typing import List
import re
induce_data_path = os.path.join(os.path.dirname(__file__), 'automatic_prompt_engineer/experiments/data/instruction_induction/raw/induce/')
eval_data_path = os.path.join(os.path.dirname(__file__), 'automatic_prompt_engineer/experiments/data/instruction_induction/raw/execute/')
annotation_data_path = os.path.join(os.path.dirname(__file__), 'automatic_prompt_engineer/experiments/data/instruction_induction/annotations/')

# Get a list of tasks (by looking at the names of the files in the induced directory)
tasks = [f.split('.')[0] for f in os.listdir(induce_data_path)]
TASK_TO_METRIC = {'common_concept': 'f1', 'informal_to_formal': 'f1', 'orthography_starts_with': 'es',
                  'taxonomy_animal': 'es', 'synonyms': 'contains',
                    'dyck_languages': 'f1',
                    'gender_inclusive_sentences_german': 'f1',
                    'object_counting': 'f1',
                    'operators': 'f1',
                    'tense': 'f1',
                    'word_sorting': 'f1',
                    'word_unscrambling': 'f1',
                    'linguistics_puzzles': 'f1',}
default_metric = 'em'

def load_data(type, task):
    base_path = induce_data_path if type == 'induce' else eval_data_path
    path = base_path + task + '.json'
    with open(path, 'r') as f:
        data = json.load(f)

    examples = data['examples']
    num_examples = len(examples)

    inputs, outputs = [], []
    for i in range(num_examples):
        data = examples[str(i + 1)]
        if task == 'cause_and_effect':
            cause, effect = data['cause'], data['effect']
            # Pick an order randomly
            if random.random() < 0.5:
                input_ = f'Sentence 1: {cause} Sentence 2: {effect}'
            else:
                input_ = f'Sentence 1: {effect} Sentence 2: {cause}'
            output_ = cause
        elif task == 'common_concept':
            items = data['items']
            # Make comma separated list of items
            input_ = ', '.join(items[:-1])
            output_ = data['all_common_concepts']
        elif task == 'rhymes':
            input_, output_ = data['input'], data['other_rhymes']
        elif 'translation' in task:
            input_, output_ = data['input'], data['possible_translations']
        else:
            input_, output_ = data['input'], [data['output']]
        if isinstance(output_, list):
            output_ = output_[0]
        if isinstance(input_, list):
            input_ = input_[0]
        inputs.append(input_)
        outputs.append(output_)
    return inputs, outputs

def load_annotation(task):
    path = annotation_data_path + task + '.json'
    with open(path, 'r') as f:
        data = json.load(f)
    annotations = data['annotations']
    return annotations[0]





def normalize_prediction(prediction, lowercase=True):
    prediction = prediction.replace(' and ', ' ')
    prediction = prediction.replace('Sentence 1:', ' ')
    prediction = prediction.replace('Sentence 2:', ' ')
    prediction = prediction.strip()
    prediction = prediction.split("\n")[0]
    prediction = prediction.split(".")[0]

    if lowercase:
        prediction = prediction.lower()

    # remove punctuation
    prediction = prediction.replace('-', ' ')
    prediction = prediction.translate(
        str.maketrans('', '', string.punctuation))

    return prediction




def load_ii_data(task):
    train_inputs, train_outputs = load_data('induce', task)
    test_inputs, test_outputs = load_data('execute', task)
    if len(train_inputs) > 32:
        train_ds = Dataset.from_dict({'text': train_inputs[:32], 'label': train_outputs[:32]})
        validation_ds = Dataset.from_dict({'text': train_inputs[64:96], 'label': train_outputs[64:96]})
    else:
        train_ds = Dataset.from_dict({'text': train_inputs, 'label': train_outputs})
        validation_ds = train_ds
    test_ds = Dataset.from_dict({'text': test_inputs, 'label': test_outputs})
    return train_ds, test_ds, validation_ds


def get_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_prediction(
        prediction, lowercase=True).split()
    ground_truth_tokens = normalize_prediction(
        ground_truth, lowercase=True).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_em_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True)
    return prediction_normalized == ground_truth_normalized


def get_exact_set_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(
        prediction, lowercase=True).split()
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True).split()
    return int(set(prediction_normalized) == set(ground_truth_normalized))


def get_contains_score(prediction, ground_truth):
    prediction_normalized = normalize_prediction(prediction, lowercase=True)
    ground_truth_normalized = normalize_prediction(
        ground_truth, lowercase=True)
    if re.search(r'\b({0})\b'.format(ground_truth_normalized), prediction_normalized):
        return 1
    else:
        return 0

def _format_prompts(prompts,inputs):
    return [prompt + '\n' + 'Input : ' + inputss + '\nOutput : ' for prompt,inputss in zip(prompts,inputs)]

def _format_prompt(prompt,inputs):
    template = "{prompt} Input : {sentence_1} Output : "
    return [template.format(prompt=prompt,sentence_1=inputss) for inputss in inputs]

def _format_prompt_tta(prompt,inputs):
    template = "{prompt} Current Input : {sentence_1} \n \nRewritten Instruction : "
    return [template.format(prompt=prompt,sentence_1=inputss) for inputss in inputs]

def _get_only_generated(outputs,trigger):
    #print(outputs,trigger)
    new_output = []
    for output in outputs:
        if trigger in output:
            new_output.append(output.split(trigger)[1].strip())
        else:
            print(output)
            new_output.append(output)
    #print(outputs)
    return new_output

def _get_generated_text(inputs,outputs):
    new_outputs = []
    for i in range(len(inputs)):
        current_input = inputs[i]
        current_output = outputs[i]
        length = len(current_input)
        if len(current_output) > length:
            new_outputs.append(current_output[length:])
        else:
            new_outputs.append(current_output)
    return new_outputs


def ii_tta_evaluation(
    dataset,
    agent_model,
    agent_tokenizer,
    target_model,
    target_tokenizer,
    device,
    meta_prompt,
    generation_kwargs,
    prompt_generation_kwargs,
    task,
    batch_size= 8,
    
):
    total = 0
    scores = 0
    check_manual = False
    if 'Instruction : ' in meta_prompt:
        check_manual = True
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    for batch in tqdm(dataloader):
        inputs = batch['text']
        labels = batch['label']
        meta_prompted_inputs = _format_prompt_tta(meta_prompt,inputs)
        input_ids = agent_tokenizer(meta_prompted_inputs,
                                    return_tensors='pt',
                                    padding=True,
                                    truncation=True,
                                    max_length=512,).input_ids.to(device)
        with torch.no_grad():
            outputs= agent_model.generate(input_ids,
                                          **prompt_generation_kwargs)
        generated_texts = target_tokenizer.batch_decode(outputs,
                                                         skip_special_tokens=True)
        generated_texts = _get_generated_text(meta_prompted_inputs,generated_texts)
        prompted_inputs = _format_prompts(generated_texts,inputs)
        prompted_input_ids = target_tokenizer(prompted_inputs,
                                             return_tensors='pt', 
                                             padding=True, 
                                             truncation=True, 
                                             max_length=512).input_ids.to(device)
        with torch.no_grad():
            outputs = target_model.generate(prompted_input_ids, **generation_kwargs)
        generated_texts_ = target_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_texts = _get_generated_text(prompted_inputs,generated_texts_)
        metric = TASK_TO_METRIC.get(task, default_metric)
        for i in range(len(inputs)):
            prediction = generated_texts[i]
            ground_truth = labels[i]
            if metric == 'f1':
                score = get_f1_score(prediction, ground_truth)
            elif metric == 'em':
                score = get_em_score(prediction, ground_truth)
            elif metric == 'es':
                score = get_exact_set_score(prediction, ground_truth)
            elif metric == 'contains':
                score = get_contains_score(prediction, ground_truth)
            else:
                raise ValueError(f'Invalid metric {metric}')     
            scores += score
            total += 1
        print('Generated Text : ',generated_texts_[-1])
        print('-------------------')
        print('Prediction : ',prediction)
        print('-------------------')
        print('Ground Truth : ',ground_truth)
        print('-------------------')
        print('Score : ',score)
        print('-------------------')
    return scores / total
        
    
def ii_tta_evaluation_test(
    dataset,
    agent_model,
    agent_tokenizer,
    target_model,
    target_tokenizer,
    device,
    meta_prompt,
    generation_kwargs,
    prompt_generation_kwargs,
    task,
    batch_size= 8,
    
):
    total = 0
    scores = 0
    check_manual = False
    if 'Instruction : ' in meta_prompt:
        check_manual = True
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    for batch in tqdm(dataloader):
        inputs = batch['text']
        labels = batch['label']
        prompted_inputs = _format_prompts([meta_prompt for i in range(len(inputs))],inputs)
        #print(prompted_inputs)
        prompted_input_ids = target_tokenizer(prompted_inputs,
                                             return_tensors='pt', 
                                             padding=True, 
                                             truncation=True, 
                                             max_length=512).input_ids.to(device)
        with torch.no_grad():
            outputs = target_model.generate(prompted_input_ids, **generation_kwargs)
        generated_texts_ = target_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_texts = _get_generated_text(prompted_inputs,generated_texts_)
        #print(generated_texts)
        #print(generated_texts_)
        metric = TASK_TO_METRIC.get(task, default_metric)
        for i in range(len(inputs)):
            prediction = generated_texts[i]
            ground_truth = labels[i]
            if metric == 'f1':
                score = get_f1_score(prediction, ground_truth)
            elif metric == 'em':
                score = get_em_score(prediction, ground_truth)
            elif metric == 'es':
                score = get_exact_set_score(prediction, ground_truth)
            elif metric == 'contains':
                score = get_contains_score(prediction, ground_truth)
            else:
                raise ValueError(f'Invalid metric {metric}')     
            scores += score
            total += 1
        print('Generated Text : ',generated_texts_[-1])
        print('-------------------')
        print('Prediction : ',prediction)
        print('-------------------')
        print('Ground Truth : ',ground_truth)
        print('-------------------')
        print('Score : ',score)
        print('-------------------')
    return scores / total
        
    
    
    
    
    
def evaluation_ii_batch(
    prompt,
    dataset,
    target_model,
    target_tokenizer,
    device,
    meta_prompt,
    generation_kwargs,
    task,
    batch_size= 8,
    
):
    total = 0
    scores = 0
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=False)
    for batch in tqdm(dataloader):
        inputs = batch['text']
        labels = batch['label']
        prompts = [prompt] * len(inputs)
        prompted_inputs = _format_prompts(prompts,inputs)
        prompted_input_ids = target_tokenizer(prompted_inputs,
                                             return_tensors='pt', 
                                             padding=True, 
                                             truncation=True, 
                                             max_length=512).input_ids.to(device)
        with torch.no_grad():
            outputs = target_model.generate(prompted_input_ids, **generation_kwargs)
        generated_texts_ = target_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_texts = _get_only_generated(generated_texts_, 'Output : ')
        metric = TASK_TO_METRIC.get(task, default_metric)
        for i in range(len(inputs)):
            prediction = generated_texts[i]
            ground_truth = labels[i]
            if metric == 'f1':
                score = get_f1_score(prediction, ground_truth)
            elif metric == 'em':
                score = get_em_score(prediction, ground_truth)
            elif metric == 'es':
                score = get_exact_set_score(prediction, ground_truth)
            elif metric == 'contains':
                score = get_contains_score(prediction, ground_truth)
            else:
                raise ValueError(f'Invalid metric {metric}')     
            scores += score
            total += 1
        print('\nPrompt : \n',prompt)
        print('-------------------')        
        print('\nGiven Input : \n',inputs[-1])
        print('-------------------')          
        print('\nTemplate : \n',prompted_inputs[-1])
        print('-------------------')   
        print('\nGenerated Text : \n',generated_texts_[-1])
        print('-------------------')
        print('\nPrediction : \n',prediction)
        print('-------------------')        
        print('\nGround Truth : \n',ground_truth)
        print('-------------------')        
        print('\nScore : \n',score)
        print('-------------------')  
            

    return scores / total

def evaluation_ii(
    prompts,
    dataset,
    model,
    tokenizer,
    device,
    task,
    generation_kwargs=None,
    show=False,
    must_show=False,
):
    dataloader = DataLoader(dataset,batch_size=1,shuffle=False)
    rewardss = []
    accs = []
    if generation_kwargs is None:
        generation_kwargs = {
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 30,
        "min_length": -1,
        }
    with torch.no_grad():
        for prompt in prompts:
            loss = 0
            acc = 0
            total = 0
            scores = 0
            reward =0
            for batch in dataloader:
                inputs = batch['text']
                labels = batch['label']
                template = prompt + '\nInput : ' + inputs[0] + '\nOutput : '
                prompt_encoded = tokenizer(template, return_tensors='pt').to(device)
                label_encoded = tokenizer(labels[0], return_tensors='pt')
                length_label = label_encoded['input_ids']
                #print(' LNEGTH  : ',len(length_label[0]))
                #print(length_label.size())
                outputs = model.generate(**prompt_encoded,**generation_kwargs)
                prediction_ = tokenizer.decode(outputs[0],skip_special_tokens=True)
                prediction = _get_generated_text([template], [prediction_])[0]
                ground_truth = labels[0]
                rewards = get_f1_score(prediction,ground_truth)
                metric = TASK_TO_METRIC.get(task, default_metric)
                if metric == 'f1':
                    score = get_f1_score(prediction, ground_truth)
                elif metric == 'em':
                    score = get_em_score(prediction, ground_truth)
                elif metric == 'es':
                    score = get_exact_set_score(prediction, ground_truth)
                elif metric == 'contains':
                    score = get_contains_score(prediction, ground_truth)
                else:
                    raise ValueError(f'Invalid metric {metric}')
                             
                reward += rewards
                scores += score
                total += 1
            acc = scores / total
            accs.append(acc)
            rewardss.append(reward)
            if must_show==True:
                print('\nPrompt : \n',prompt)
                print('-------------------')        
                print('\nGiven Input : \n',inputs[0])
                print('-------------------')        
                print('\nTemplate : \n' ,template)
                print('-------------------')        
                print('\nGenerated Text : \n',prediction_)
                print('-------------------')
                print('\nPrediction : \n',prediction)
                print('-------------------')        
                print('\nGround Truth : \n',ground_truth)
                print('-------------------')        
                print('\nScore : \n',score)
                print('-------------------')        
                print('\nReward : \n',rewards)
                print('-------------------')   
    return rewardss, accs

def got_example_ii(dataset,shot=5):
    examples = ''
    for i in range(shot):
        idx = random.randint(0,len(dataset)-1)
        example = dataset[idx]
        #print('Input : ',example['text'])
        #print('Output : ',example['label'])
        a = 'Input : ' + example['text'] + '\nOutput : ' + example['label']
        examples += a + '\n'
    return examples