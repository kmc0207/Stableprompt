from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import json


qa_dict= ['A','B','C','D']
openbookdict = {'A':0,'B':1,'C':2,'D':3}
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        """
        texts: 평가할 텍스트의 리스트
        labels: 각 텍스트에 해당하는 레이블의 리스트
        """
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        """데이터셋의 총 크기를 반환합니다."""
        return len(self.texts)
    
    def __getitem__(self, idx):
        """주어진 인덱스에 해당하는 텍스트와 레이블을 반환합니다."""
        text = self.texts[idx]
        label = self.labels[idx]
        return {"text": text, "label": label}





def load_bigbench_causal_judgment():
    with open('./automatic_prompt_engineer/data/bigbench-ii/causal_judgment/task.json', 'r' ) as f:
        json_data = json.load(f)
    text_input = []
    labels = []
    for data in json_data['examples']:
        text_input.append(data['input'])
        label = data['target_scores']
        if label['Yes'] == 1 :
            labels.append(1)
        else:
            labels.append(0)
    train_dataset = TextDataset(text_input[:30], labels[:30])
    train_labels = labels[:30]
    test_dataset = TextDataset(text_input[30:], labels[30:])
    test_labels = labels[30:]
    return train_dataset,train_labels,test_dataset,test_labels

def load_bigbench_sports_understanding():
    with open('./automatic_prompt_engineer/data/bigbench-ii/sports_understanding/task.json', 'r' ) as f:
        json_data = json.load(f)
    text_input = []
    labels = []
    for data in json_data['examples']:
        text_input.append(data['input'])
        label = data['target_scores']
        if label['implausible'] == 1 :
            labels.append(1)
        else:
            labels.append(0)
    train_dataset = TextDataset(text_input[:30], labels[:30])
    train_labels = labels[:30]
    test_dataset = TextDataset(text_input[30:], labels[30:])
    test_labels = labels[30:]
    return train_dataset,train_labels,test_dataset,test_labels

def choice_to_sentence(choices):
    sentence = ''
    for idx,choice in enumerate(choices):
        sentence += f'{qa_dict[idx]} : {choice} \n'
    return sentence

def load_openbookqa():
    train_sentences = load_dataset('allenai/openbookqa','additional',split='validation')
    train_labels = train_sentences['answerKey']
    test_sentences = load_dataset('allenai/openbookqa','additional',split='test')
    test_labels = train_sentences['answerKey']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    
    for item in train_sentences:
        #print(item)
        item['text'] = 'question : ' + item['question_stem'] + ' choices : ' + choice_to_sentence(item['choices']['text']) + 'Hint :' + item['fact1']
        item['label'] = openbookdict[item['answerKey']]
    for item in test_sentences:
        item['text'] = 'question : ' + item['question_stem'] + ' choices : ' + choice_to_sentence(item['choices']['text']) + 'Hint :' + item['fact1']
        item['label'] = openbookdict[item['answerKey']]
    return train_sentences, train_labels, test_sentences, test_labels, train_sentences, train_labels

def load_mmlu(dataset_name):
    #dataset_name = dataset_name.split('-')[1]
    train_sentences = load_dataset('cais/mmlu',dataset_name,split='dev')
    train_labels = train_sentences['answer']
    test_sentences = load_dataset('cais/mmlu',dataset_name,split='test')
    test_labels = test_sentences['answer']
    validation_sentences = load_dataset('cais/mmlu',dataset_name,split='validation')
    validation_labels = validation_sentences['answer']
    #train_sentences['label'] = train_sentences['answer']
    #test_sentences['label'] = test_sentences['answer']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    validation_sentences = [sentence for sentence in validation_sentences]
    for item in train_sentences:
        item['text'] = 'question : ' + item['question'] + ' choices : ' + choice_to_sentence(item['choices'])
        item['label'] = item['answer']
    for item in test_sentences:
        item['text'] = 'question : ' + item['question'] + ' choices : ' + choice_to_sentence(item['choices'])
        item['label'] = item['answer']
    for item in validation_sentences:
        item['text'] = 'question : ' + item['question'] + ' choices : ' + choice_to_sentence(item['choices'])
        item['label'] = item['answer']
    return train_sentences, train_labels, test_sentences, test_labels, validation_sentences, validation_labels

def load_squad():
    train_sentences = load_dataset('rajpurkar/squad',split='train')
    train_labels = train_sentences['answers']
    test_sentences = load_dataset('rajpurkar/squad',split='validation')
    test_labels = test_sentences['answers']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    for item in train_sentences:
        item['text'] = 'Hint :' + item['context'] + ' \nQuestion : ' + item['question'] + '\n'
        item['label'] = item['answers']
    for item in test_sentences:
        item['text'] = 'Hint :' + item['context'] + ' \nQuestion : ' + item['question'] + '\n'
        item['label'] = item['answers']
    return train_sentences, train_labels, test_sentences, test_labels



def load_bigbench(dataset_name):
    with open('./automatic_prompt_engineer/data/bigbench-ii/'+dataset_name+'/task.json', 'r' ) as f:
        json_data = json.load(f)
    metrics = json_data['preferred_score']
    text_input = []
    labels = []
    correct_answer = []
    no_choice_text = []
    verbalizer = []
    if metrics == 'multiple_choice_grade':
        for data in json_data['examples']:
            text = 'Input : ' + data['input'] + '\n Choices : \n'
            no_choice_text.append(text)
            choice_dict = data['target_scores']
            for i,key in enumerate(choice_dict.keys()):
                #print(choice_dict[key],i)
                if choice_dict[key] > 0.9:
                    labels.append(i)
                    correct_answer.append(key)
                text += qa_dict[i] + ' : ' + key + '\n'
            text_input.append(text)
        max_label = max(labels)
        verbalizer = {}
        for i in range(int(max_label)+1):
            verbalizer[i] = qa_dict[i]
    elif metrics == 'exact_str_match':
        for data in json_data['examples']:
            text = 'Input : ' + data['input']
            no_choice_text.append(text)
            #print(data['target'])
            if isinstance(data['target'],list):
                if dataset_name == 'object_counting':
                    labels.append(data['target'][1])
                    correct_answer.append(data['target'][1])
                else:
                    labels.append(data['target'][0])
                    correct_answer.append(data['target'][0])
            else:
                labels.append(data['target'])
                correct_answer.append(data['target'])
            text_input.append(text)
            
    train_dataset = TextDataset(text_input[:30], labels[:30])
    test_dataset = TextDataset(text_input[30:], labels[30:])
    task_prefix = json_data['task_prefix']
    return metrics,train_dataset,test_dataset,verbalizer,task_prefix

def load_qa_econometrics():
    train_sentences = load_dataset('cais/mmlu','econometrics',split='test')
    train_labels = train_sentences['answer']
    test_sentences = load_dataset('cais/mmlu','econometrics',split='test')
    test_labels = test_sentences['answer']
    #train_sentences['label'] = train_sentences['answer']
    #test_sentences['label'] = test_sentences['answer']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    for item in train_sentences:
        item['text'] = 'question : ' + item['question'] + ' choices : ' + str(item['choices'])
        item['label'] = item['answer']
    for item in test_sentences:
        item['text'] = 'question : ' + item['question'] + ' choices : ' + str(item['choices'])
        item['label'] = item['answer']
    return train_sentences, train_labels, test_sentences, test_labels
    


def load_bigbench_snarks():
    with open('./automatic_prompt_engineer/data/bigbench-ii/snarks/task.json', 'r' ) as f:
        json_data = json.load(f)
    text_input = []
    labels = []
    for data in json_data['examples']:
        text_input.append(data['input'])
        label = data['target_scores']
        if label['(b)'] == 1 :
            labels.append(1)
        else:
            labels.append(0)
    train_dataset = TextDataset(text_input[:30], labels[:30])
    train_labels = labels[:30]
    test_dataset = TextDataset(text_input[30:], labels[30:])
    test_labels = labels[30:]
    return train_dataset,train_labels,test_dataset,test_labels


def load_bigbench_presuppositions_as_nli():
    with open('./automatic_prompt_engineer/data/bigbench-ii/presuppositions_as_nli/task.json', 'r' ) as f:
        json_data = json.load(f)
    text_input = []
    labels = []
    for data in json_data['examples']:
        text_input.append(data['input'])
        label = data['target_scores']
        keys = label.keys()
        for num,key in enumerate(keys):
            if label[key] == 1 :
                labels.append(num)
    train_dataset = TextDataset(text_input[:30], labels[:30])
    train_labels = labels[:30]
    test_dataset = TextDataset(text_input[30:], labels[30:])
    test_labels = labels[30:]
    return train_dataset,train_labels,test_dataset,test_labels

def load_bigbench_implicatures():
    with open('./automatic_prompt_engineer/data/bigbench-ii/implicatures/task.json', 'r' ) as f:
        json_data = json.load(f)
    text_input = []
    labels = []
    for data in json_data['examples']:
        text_input.append(data['input'])
        label = data['target_scores']
        if label['yes'] > 0.5 :
            labels.append(1)
        else:
            labels.append(0)
    train_dataset = TextDataset(text_input[:30], labels[:30])
    train_labels = labels[:30]
    test_dataset = TextDataset(text_input[30:], labels[30:])
    test_labels = labels[30:]
    return train_dataset,train_labels,test_dataset,test_labels

def load_bigbench_navigate():
    with open('./automatic_prompt_engineer/data/bigbench-ii/navigate/task.json', 'r' ) as f:
        json_data = json.load(f)
    text_input = []
    labels = []
    for data in json_data['examples']:
        text_input.append(data['input'])
        label = data['target_scores']
        if label['False'] > 0.5 :
            labels.append(1)
        else:
            labels.append(0)
    train_dataset = TextDataset(text_input[:30], labels[:30])
    train_labels = labels[:30]
    test_dataset = TextDataset(text_input[30:], labels[30:])
    test_labels = labels[30:]
    return train_dataset,train_labels,test_dataset,test_labels


def load_bigbench_epistemic_reasoning():
    with open('./automatic_prompt_engineer/data/bigbench-ii/epistemic_reasoning/task.json', 'r' ) as f:
        json_data = json.load(f)
    text_input = []
    labels = []
    for data in json_data['examples']:
        text_input.append(data['input'])
        label = data['target_scores']
        if label['non-entailment'] == 1 :
            labels.append(1)
        else:
            labels.append(0)
    train_dataset = TextDataset(text_input[:30], labels[:30])
    train_labels = labels[:30]
    test_dataset = TextDataset(text_input[30:], labels[30:])
    test_labels = labels[30:]
    return train_dataset,train_labels,test_dataset,test_labels

def load_sst2():
    from datasets import load_dataset
    train_sentences = load_dataset( 'sst2', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset( 'sst2', split='validation')
    test_labels = test_sentences['label']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    for item in train_sentences:
        item['text'] = 'Input :' + item['sentence'] + '\n'
    for item in test_sentences:
        item['text'] = 'Input :' + item['sentence'] + '\n'
    return train_sentences, train_labels, test_sentences, test_labels

def load_qnli():
    from datasets import load_dataset
    train_sentences = load_dataset('glue', 'qnli', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('glue', 'qnli', split='validation')
    test_labels = test_sentences['label']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    for item in train_sentences:
        item['text'] = 'question : ' + item['question'] + '\n sentence : ' + item['sentence']
    for item in test_sentences:
        item['text'] = 'question : ' + item['question'] + '\n sentence : ' + item['sentence']
    return train_sentences, train_labels, test_sentences, test_labels

def load_mnli():
    from datasets import load_dataset
    train_sentences = load_dataset('glue', 'mnli', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('glue', 'mnli', split='validation_matched')
    test_labels = test_sentences['label']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    for item in train_sentences:
        item['text'] = 'premise : ' + item['premise'] + '\n hypothesis : ' + item['hypothesis']
    for item in test_sentences:
        item['text'] = 'premise : ' + item['premise'] + '\n hypothesis : ' + item['hypothesis']
    return train_sentences, train_labels, test_sentences, test_labels

def load_agnews():
    from datasets import load_dataset
    train_sentences = load_dataset('ag_news', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('ag_news', split='test')
    test_labels = test_sentences['label']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences] 
    for item in train_sentences:
        item['text'] = 'Article : ' + item['text']
    for item in test_sentences:
        item['text'] = 'Article : ' + item['text']
    return train_sentences, train_labels, test_sentences, test_labels

def load_yelp_polarity():
    from datasets import load_dataset
    train_sentences = load_dataset('yelp_polarity', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('yelp_polarity', split='test')
    test_labels = test_sentences['label']
    str2int = train_sentences.features['label']._str2int
    int2str = inv_map = {v: k for k, v in str2int.items()}
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    return train_sentences, train_labels, test_sentences, test_labels, str2int, int2str


def load_snli():
    train_dataset = load_dataset('snli', split='train')
    test_dataset = load_dataset('snli', split='validation')

    # Filter out entries with label -1
    train_dataset = train_dataset.filter(lambda example: example['label'] != -1)
    test_dataset = test_dataset.filter(lambda example: example['label'] != -1)

    # Extract sentences and labels
    train_sentences = [sentence for sentence in train_dataset]
    test_sentences = [sentence for sentence in test_dataset]
    train_labels = train_dataset['label']
    test_labels = test_dataset['label']

    # Map for label conversion
    str2int = train_dataset.features['label']._str2int
    int2str = {v: k for k, v in str2int.items()}

    # Add formatted text field
    for item in train_sentences:
        item['text'] = 'premise : ' + item['premise'] + '\n hypothesis : ' + item['hypothesis']
    for item in test_sentences:
        item['text'] = 'premise : ' + item['premise'] + '\n hypothesis : ' + item['hypothesis']

    return train_sentences, train_labels, test_sentences, test_labels, str2int, int2str

def load_rte():
    from datasets import load_dataset
    # file_dict = {'train': 'data/k-shot/RTE/16-13/train.tsv'}
    # train_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
    train_sentences = load_dataset('super_glue', 'rte', split='train')
    train_labels = train_sentences['label']
    unique = {label: idx for idx, label in enumerate(set(train_labels))}
    train_labels = [unique[label] for label in train_sentences['label']]
    # file_dict = {'train': 'data/k-shot/RTE/16-13/test.tsv'}
    # test_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
    test_sentences = load_dataset('super_glue', 'rte', split='validation')
    test_labels = test_sentences['label']
    unique = {label: idx for idx, label in enumerate(set(test_labels))}
    test_labels = [unique[label] for label in test_sentences['label']]
    # str2int = train_sentences.features['label']._str2int
    # int2str = inv_map = {v: k for k, v in str2int.items()}
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    for item in train_sentences:
        item['text'] = 'premise : ' + item['premise'] + '\n hypothesis : ' + item['hypothesis']
    for item in test_sentences:
        item['text'] = 'premise : ' + item['premise'] + '\n hypothesis : ' + item['hypothesis']
    return train_sentences, train_labels, test_sentences, test_labels


def load_mrpc():
    from datasets import load_dataset
    train_sentences = load_dataset('glue', 'mrpc', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('glue', 'mrpc', split='validation')
    test_labels = test_sentences['label']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    for item in train_sentences:
        item['text'] = 'sentence1 : ' + item['sentence1'] + '\n sentence2 : ' + item['sentence2']
    for item in test_sentences:
        item['text'] = 'sentence1 : ' + item['sentence1'] + '\n sentence2 : ' + item['sentence2']
    return train_sentences, train_labels, test_sentences, test_labels

def load_customer_review():
    from datasets import load_dataset
    file_dict = {'train': 'cr/16-42/train.tsv'}
    train_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
    train_labels = train_sentences['label']
    file_dict = {'train': 'cr/16-42/test.tsv'}
    test_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
    test_labels = test_sentences['label']
    # str2int = train_sentences.features['label']._str2int
    # int2str = inv_map = {v: k for k, v in str2int.items()}
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    return train_sentences, train_labels, test_sentences, test_labels

def load_mr():
    from datasets import load_dataset
    file_dict = {'train': 'mr/train.tsv'}
    train_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
    train_labels = train_sentences['label']
    file_dict = {'train': 'mr/test.tsv'}
    test_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
    test_labels = test_sentences['label']
    # str2int = train_sentences.features['label']._str2int
    # int2str = inv_map = {v: k for k, v in str2int.items()}
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    return train_sentences, train_labels, test_sentences, test_labels


def instruct_dataset(dataset_name):
    INDUCTION_TASKS = ['active_to_passive', 'antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
                   'informal_to_formal', 'larger_animal', 'letters_list', 'negation', 'num_to_verbal',
                   'orthography_starts_with', 'rhymes', 'second_word_letter', 'sentence_similarity', 'sentiment',
                   'singular_to_plural', 'sum', 'synonyms', 'taxonomy_animal', 'translation_en-de', 'translation_en-es',
                   'translation_en-fr', 'word_in_context']
    train_data_path = 'instruction-induction/data/raw/induce'
    test_data_path = 'instruction-induction/data/raw/execute'
    task_name = dataset_name
    if dataset_name not in INDUCTION_TASKS:
        raise NotImplementedError
        return None
    with open(f'{train_data_path}/{task_name}.json', encoding='utf-8') as f_examples:
        data = json.load(f_examples)
    examples = data['examples']
    train_sentences = [examples[key]['input'] for key in examples.keys()]
    train_labels = [examples[key]['output'] for key in examples.keys()]
    train_dataset = dict()
    train_dataset['text'] = train_sentences
    train_dataset['label'] = train_labels
    with open(f'{test_data_path}/{task_name}.json', encoding='utf-8') as f_examples:
        data = json.load(f_examples)
    test_sentences = [examples[key]['input'] for key in examples.keys()]
    test_labels = [examples[key]['output'] for key in examples.keys()]
    test_dataset = dict()
    test_dataset['text'] = train_sentences
    test_dataset['label'] = train_labels
    return train_dataset, None,test_dataset,None


def load_all_dataset(dataset_name):
    print(dataset_name)
    if 'mmlu' in dataset_name:
        return load_mmlu(dataset_name)
    elif dataset_name == 'sst2':
        return load_sst2()
    elif dataset_name == 'qnli':
        return load_qnli()
    elif dataset_name == 'mnli':
        return load_mnli()
    elif dataset_name == 'agnews':
        return load_agnews()
    elif dataset_name == 'yelp_polarity':
        return load_yelp_polarity()
    elif dataset_name == 'rte':
        return load_rte()
    elif dataset_name == 'mrpc':
        return load_mrpc()
    elif dataset_name == 'mr':
        return load_mr()
    elif dataset_name == 'customer_review':
        return load_customer_review()
    elif dataset_name == 'snli':
        return load_snli()
    elif dataset_name == 'bigbench_causal_judgement':
        return load_bigbench_causal_judgment()
    elif dataset_name == 'bigbench_epistemic_reasoning':
        return load_bigbench_epistemic_reasoning()
    elif dataset_name == 'bigbench_implicatures':
        return load_bigbench_implicatures()
    elif dataset_name == 'bigbench_presuppositions_as_nli':
        return load_bigbench_presuppositions_as_nli()
    elif dataset_name == 'bigbench_snarks':
        return load_bigbench_snarks()
    elif dataset_name == 'bigbench_sports_understanding':
        return load_bigbench_sports_understanding()
    elif dataset_name == 'bigbench_navigate':
        return load_bigbench_navigate()
    elif dataset_name == 'mmlu_electrical_engineering':
        return load_qa_electrical()
    elif dataset_name == 'mmlu_econometrics':
        return load_qa_econometrics()
    else:
        raise instruct_dataset(dataset_name)
    
def dataset_names():
    return ['sst2','qnli','mnli','agnews','yelp_polarity','rte','mrpc','snli']

def load_qa_dataset(dataset_name):
    if dataset_name == 'openbookqa':
        return load_openbookqa()
    else:
        return load_mmlu(dataset_name)

def load_generation_dataset(dataset_name):
    if dataset_name == 'squad':
        return load_squad()
    elif dataset_name == 'bigbench_causal_judgement':
        return load_bigbench_causal_judgment()
    elif dataset_name == 'bigbench_epistemic_reasoning':
        return load_bigbench_epistemic_reasoning()
    elif dataset_name == 'bigbench_implicatures':
        return load_bigbench_implicatures()
    elif dataset_name == 'bigbench_presuppositions_as_nli':
        return load_bigbench_presuppositions_as_nli()
    elif dataset_name == 'bigbench_snarks':
        return load_bigbench_snarks()
    elif dataset_name == 'bigbench_sports_understanding':
        return load_bigbench_sports_understanding()
    elif dataset_name == 'bigbench_navigate':
        return load_bigbench_navigate()

def qa_dicts():
    return {0 : 'A',1 : 'B',2 : 'C',3 : 'D'}

def dataset_dicts(dataset_name):
    if 'mmlu' in dataset_name:
        return {0 : 'A',1 : 'B',2 : 'C',3 : 'D'}

    elif dataset_name == 'sst2':
        return {0 : 'no',1 : 'yes'}
    elif dataset_name == 'qnli':
        return {0 : 'yes',1 : 'no'}
    elif dataset_name == 'mnli':
        return {0 : 'Yes',1 : 'Maybe',2 : 'No'}
    elif dataset_name == 'agnews':
        return {0 : 'World',1 : 'Sports',2 : 'Business',3 : 'Technology'}
    elif dataset_name == 'yelp_polarity':
        return {0 : 'No',1 : 'Yes'}
    elif dataset_name == 'rte':
        return {0 : 'yes',1 : 'no'}
    elif dataset_name == 'mrpc':
        return {0 : 'No',1 : 'Yes'}
    elif dataset_name == 'customer_review':
        return {0 : 'negative',1 : 'positive'}
    elif dataset_name == 'mr':
        return {0 : 'No',1 : 'Yes'}
    elif dataset_name == 'snli':
        return {0 : 'Yes',1 : 'Maybe',2 : 'No'}    
    elif dataset_name == 'bigbench_causal_judgement':
        return {0 : 'No',1 : 'Yes'}
    elif dataset_name == 'bigbench_epistemic_reasoning':
        return {0 : 'Yes',1 : 'No'}
    elif dataset_name == 'bigbench_implicatures':
        return {0 : 'No',1 : 'Yes'}
    elif dataset_name == 'bigbench_presuppositions_as_nli':
        return {0: 'Yes',1 :'Maybe',2 : 'No'}
    elif dataset_name == 'bigbench_snarks':
        return {0:'A',1:'B'}
    elif dataset_name == 'bigbench_sports_understanding':
        return {0:'Yes',1 : 'No'}
    elif dataset_name == 'bigbench_navigate':
        return {0:'Yes',1 : 'No'}
    
    else:
        raise NotImplementedError
    
    
    
def load_annotation(dataset):
    annotation = 'Read carefully'
    if dataset == 'sst2':
        annotation = '''
        In this task, you are given sentences from movie reviews. The task is to classify a sentence as "yes" if the sentiment of the sentence is positive or as "no" if the sentiment of the sentence is negative.',
        '''
    elif dataset == 'mnli':
        annotation = '''
        In this task, you’re given a pair of sentences, sentence 1 and sentence 2. Your job is to choose whether the two sentences clearly agree (entailment)/disagree (contradiction) with each other, or if this cannot be determined (neutral). Your answer must be in the form of the letters Yes, Maybe, and No respectively.',
        '''
    elif dataset == 'qnli':
        annotation = '''
        You are given two sentences(Sentence1 and Sentence2). Answer “yes” if these sentences are a paraphrase of one another, otherwise answer “no”.',
        '''
    elif dataset == 'snli':
        annotation = '''
        In this task, you’re given a pair of sentences, sentence 1 and sentence 2. Your job is to choose whether the two sentences clearly agree (entailment)/disagree (contradiction) with each other, or if this cannot be determined (neutral). Your answer must be in the form of the letters Yes, Maybe, and No respectively',
        '''
    elif dataset == 'rte':
        annotation = '''
        Dose the premise follow from the fact that the hypothesis? Please answer either "yes" or "no".',
        '''
    elif dataset == 'mrpc':
        annotation = '''
        You are given two sentences(Sentence1 and Sentence2). Answer \"Yes\" if these sentences are a paraphrase of one another, otherwise answer \"No\".
        '''
        
        
    return annotation