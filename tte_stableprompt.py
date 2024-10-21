import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig,AutoModelForCausalLMWithValueHead
import argparse
import numpy as np
import wandb
import copy
import random
import heapq
import utils
from dataset_utils import load_all_dataset,dataset_dicts,load_qa_dataset,qa_dicts,load_generation_dataset,load_annotation
from peft import LoraConfig
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model',type=str,default='google/gemma-1.1-7b-it')
    parser.add_argument('--agent_model',type=str,default='google/gemma-1.1-7b-it')
    parser.add_argument('--task',type=str,default='classification')
    parser.add_argument('--dataset',type=str,default='sst2')
    parser.add_argument(
        '--verbalizer',
        type = str,
        nargs = '+',
        default = None
    )
    parser.add_argument('--cache_dir',type=str,default='/mnt/sdb/llm/')
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--max_prompt_length',type=int,default=100)
    parser.add_argument('--train_data_per_labels',type=int,default=10)
    parser.add_argument('--num_example',type=int,default=3)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--meta_prompt',type=str,
                        default = '''I want to give the appropriate hint to help
                        a friend who needs to look at the input and guess the output.
                        Plase write instruction to help my friends.
                        This is an example input-output pair that you can reference when writing your instructions.
                        ''',)
    parser.add_argument('--prompt_per_example',type=int,default=2)
    parser.add_argument('--learning_rate',type=float,default=1e-5)
    parser.add_argument('--update_term',type=int,default=10)
    parser.add_argument('--threshold',type=float,default=0.05)
    parser.add_argument('--test_batch_size',type=int,default=16)
    parser.add_argument('--add_manual',action='store_true')
    parser.add_argument('--test_mode',action='store_true')
    parser.add_argument('--no_ref',action='store_true')
    parser.add_argument('--no_rollback',action='store_true')
    args = parser.parse_args()
    return args

def main():
    #torch.backends.cuda.enable_mem_efficient_sdp(False)
    #torch.backends.cuda.enable_flash_sdp(False)
    args = parser_args()
    device=  'cuda:0'
    agent_name = args.agent_model.split('/')[-1]
    target_name = args.target_model.split('/')[-1]
    if args.max_prompt_length < 100 :
        name = 'tta_TC_shorts'
    else:
        name = 'tta_TC'
    small_name = args.dataset+'_'
    if args.test_mode:
        small_name += 'test_'
    if args.no_ref:
        small_name += 'no_ref_'
    if args.no_rollback:
        small_name += 'no_rollback_'
    wandb.init(project=name, 
               config=args,
               name = small_name)
    
    

    
    #load dataset
    if args.task == 'classification':
        dataset = load_all_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        if args.test_mode:
            test_dataset = utils.create_balanced_subset(test_dataset,100)
        if args.verbalizer is None:
            verbalizer = dataset_dicts(args.dataset)
        num_labels = len(verbalizer)
        train_dataset,validation_dataset = utils.create_balanced_subset_and_validation(train_dataset,
                                                                                       args.train_data_per_labels * num_labels,
                                                                                       )
        
    elif args.task == 'qa':
        dataset = load_qa_dataset(args.dataset)
        train_dataset = dataset[4]
        validation_dataset = dataset[0]
        test_dataset = dataset[2]
        #test_dataset = utils.create_balanced_subset(test_dataset,100)
        if args.verbalizer is None:
            verbalizer = qa_dicts()
        num_labels = len(verbalizer)
        #validation_dataset = train_dataset
    
    elif args.task == 'generation':
        dataset = load_generation_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        test_dataset = utils.create_balanced_subset(test_dataset,100)
        verbalizer = None
        validation_dataset = train_dataset
    
    else:
        #TODO
        pass
    print('train_data_size' , len(train_dataset))
    print('test_data_size' , len(test_dataset))
    print('Verbalizer : ', verbalizer)        
    #make dataloader
    test_dataloader = DataLoader(test_dataset,batch_size = 1,shuffle = True)
    train_dataloader = DataLoader(train_dataset,batch_size = 1,shuffle = True)
    
    

    annotation = load_annotation(args.dataset)
    
    
    
    #load agent model
    config = PPOConfig(
        model_name = args.agent_model,
        learning_rate = args.learning_rate,
        batch_size = args.prompt_per_example,
        mini_batch_size= args.prompt_per_example,
        log_with='wandb',
    )
    lora_config = LoraConfig(
        r= 16,
        lora_alpha = 32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    agent_tokenizer = AutoTokenizer.from_pretrained(args.agent_model,cache_dir = args.cache_dir)
    agent_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.agent_model,
        torch_dtype=torch.bfloat16,
        device_map = 'auto',
        peft_config = lora_config,
        cache_dir = args.cache_dir
    )
    if args.no_ref:
        ref_model = None
    else:
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            args.agent_model,
            torch_dtype=torch.bfloat16,
            device_map = 'auto',
            peft_config = lora_config,
            cache_dir = args.cache_dir
        )
    ppo_trainer = PPOTrainer(config= config,
                            model = agent_model,
                            ref_model = ref_model,
                            tokenizer = agent_tokenizer)

    agent_tokenizer.pad_token = agent_tokenizer.eos_token
    
    
    #load target model
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model,cache_dir = args.cache_dir)
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model,
                                                        torch_dtype=torch.bfloat16,
                                                        cache_dir = args.cache_dir,
                                                        device_map='auto')
    target_model.config.pad_token_id = target_tokenizer.eos_token_id
    target_tokenizer.pad_token = target_tokenizer.eos_token
    
    #generation kwargs setting
    generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": agent_tokenizer.eos_token_id,
    "max_new_tokens":args.max_prompt_length,
    "min_length": -1,
    }
    
    
    #setting verbalizer ids
    verbalizer_ids=  []
    if verbalizer is not None:
        for i in range(len(verbalizer)):
            verbalizer_ids.append(agent_tokenizer.convert_tokens_to_ids(verbalizer[i]))
    
    
    examples = utils.got_example(validation_dataset,verbalizer,shot=args.num_example)
    if args.add_manual:
        context = 'Look at the instruction and current input, rewrite instruction for current input. \n instruction : '
        context += annotation 
    else:
        context = args.meta_prompt + '\n' + examples
    test_acc = utils.tta_evaluation(
        test_dataset,
        agent_model,
        agent_tokenizer,
        target_model,
        target_tokenizer,
        device,
        context,
        generation_kwargs,
        verbalizer_ids,
        verbalizer,
        batch_size=args.test_batch_size
    )

    print('Test Accuracy : ', test_acc)
    wandb.log({
        'test_acc' : test_acc
    })
    
    test_accs = [test_acc]
    change_num = 0
    step = 0
    #start training
    for ep in tqdm(range(args.epochs)):
        
        
        
        for batch in train_dataloader:
            step +=1
            inputs = batch['text']
            labels = batch['label']
            examples = utils.got_example(validation_dataset,verbalizer,shot=args.num_example)
            with torch.no_grad():
                if args.add_manual:
                    context = 'Look at the instruction and current input, rewrite instruction for current input. \n instruction : '
                    context += annotation +'\n Current Input : ' + inputs[0] + '\n Rewritten Instruction : '        
                else:
                    context = args.meta_prompt + '\n' + examples
                query_text = [
                    {"role" : "user", "content" : context},
                ]
                
                query_encoded = agent_tokenizer.apply_chat_template(
                    query_text,
                    return_tensors='pt'
                ).view(-1)
                
                response_tensors =ppo_trainer.generate(
                    query_encoded.to(device),
                    **generation_kwargs,
                    return_prompt=False,
                    num_return_sequences = args.prompt_per_example
                )
                
                used_prompt = [agent_tokenizer.decode(r.squeeze(),skip_special_tokens=True) for r in response_tensors]
            
            
            rewards = []
            losses = []
            accs = []
            with torch.no_grad(): 
                for prompt in used_prompt:
                    template = prompt +  inputs[0] + "\nOutput : "
                    prompt_encoded = target_tokenizer(template,return_tensors='pt').to(device)
                    #print(prompt_encoded)
                    outputs = target_model(**prompt_encoded)
                    logits = outputs.logits
                    verbalizer_logits = logits[:, -1, verbalizer_ids]
                    label = torch.tensor(labels).to(device)
                    pred = torch.argmax(verbalizer_logits,dim=1)
                    acc = torch.sum(pred == label).item() / len(label)
                    loss = -torch.nn.functional.cross_entropy(verbalizer_logits,label).item()
                    rewards.append(loss + acc * 10)
                    losses.append(loss)
                    accs.append(acc)
            
        
                    
            
            np_rewards = np.array(rewards)
            pt_rewards = [torch.tensor(reward_) for reward_ in rewards]
            bs = len(pt_rewards)
            stats = ppo_trainer.step(
                [query_encoded] * bs,
                [response for response in response_tensors],
                pt_rewards,
            )

            for i in range(len(rewards)):
                print('Input : ', inputs[0])
                print('Prompt : ', used_prompt[i])
                print('Reward : ', rewards[i])
                print('Acc : ', accs[i])
                wandb.log({
                    'mean_reward': np.mean(np_rewards),
                    'max_reward' : np.max(np_rewards),
                    'min_reward' : np.min(np_rewards),
                })


            #start evaluation
            if step % args.update_term == 0:
                test_acc = utils.tta_evaluation(
                    test_dataset,
                    agent_model,
                    agent_tokenizer,
                    target_model,
                    target_tokenizer,
                    device,
                    context,
                    generation_kwargs,
                    verbalizer_ids,
                    verbalizer,
                    batch_size=args.test_batch_size
                )
                if args.no_ref:
                    change_num=0
                else:
                    diff = test_acc - test_accs[-1]
                    if diff < - args.threshold:
                        ppo_trainer.model = ppo_trainer.ref_model
                        change_num -= 1
                    elif diff > args.threshold and args.no_rollback==False:
                        ppo_trainer.ref_model = ppo_trainer.model
                        change_num +=1
                    else:
                        change_num = change_num
                    if change_num < 0 :
                        change_num = 0
                test_accs.append(test_acc)
                print(test_acc)
                wandb.log({
                    'test_acc' : test_acc,
                    'change_num' : change_num
                })
        
            
if __name__ == '__main__':
    main()
                
                    
                    
    
    
    