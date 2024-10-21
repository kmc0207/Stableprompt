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
from dataset_utils import load_all_dataset,dataset_dicts,load_qa_dataset,qa_dicts,load_generation_dataset
from peft import LoraConfig
from datasets import Dataset
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
    parser.add_argument('--batch_size',type=int,default=16)
    parser.add_argument('--max_prompt_length',type=int,default=100)
    parser.add_argument('--train_data_per_labels',type=int,default=16)
    parser.add_argument('--num_example',type=int,default=5)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--meta_prompt',type=str,
                        default = '''I gave a friend an instruction and five inputs. 
                        The friend read the instruction and wrote an output for every one of the inputs.
                        Here are the input-output pairs: \n
                        ''',)
    parser.add_argument('--prompt_per_example',type=int,default=4)
    parser.add_argument('--update_term',type=int,default=15)
    parser.add_argument('--update_threshold',type=float,default=0.05)   
    parser.add_argument('--num_test_example',type=int,default=20)

    args = parser.parse_args()
    return args

def main():
    
    args = parser_args()
    device= 'cuda:0'
    wandb.init(project='algprompt_' +args.task + '_' + args.dataset, 
               config=args,
               name = args.task + '_' + args.dataset + '_' + args.agent_model + '_' + args.target_model)
    
    
    if args.task == 'classification':
        dataset = load_all_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        #test_dataset = utils.create_balanced_subset(test_dataset,100)
        if args.verbalizer is None:
            verbalizer = dataset_dicts(args.dataset)
        num_labels = len(verbalizer)
        train_dataset,validation_dataset = utils.create_balanced_subset_and_validation(train_dataset,
                                                                                       args.train_data_per_labels * num_labels,
                                                                                       )
    elif args.task == 'qa':
        dataset = load_qa_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        test_dataset = utils.create_balanced_subset(test_dataset,100)
        if args.verbalizer is None:
            verbalizer = qa_dicts()
        num_labels = len(verbalizer)
        validation_dataset = train_dataset
    
    elif args.task == 'generation':
        dataset = load_generation_dataset(args.dataset)
        train_dataset = dataset[0]
        test_dataset = dataset[2]
        test_dataset = utils.create_balanced_subset(test_dataset,100)
        verbalizer = None
        validation_dataset = train_dataset
        
    #make dataloader
    test_dataloader = DataLoader(test_dataset,batch_size = args.batch_size,shuffle = True)
    train_dataloader = DataLoader(train_dataset,batch_size = args.batch_size,shuffle = True)
    
    print('train_data_size' , len(train_dataset))
    print('test_data_size' , len(test_dataset))
        #load agent model
    config = PPOConfig(
        model_name = args.agent_model,
        learning_rate = 1e-5,
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
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.agent_model,
        torch_dtype=torch.bfloat16,
        device_map = 'auto',
        peft_config = lora_config,
        cache_dir = args.cache_dir
    )
    agent_tokenizer.pad_token = agent_tokenizer.eos_token
    ppo_trainer = PPOTrainer(config,agent_model,ref_model,agent_tokenizer)
    
    #load target model
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model,cache_dir = args.cache_dir)
    target_model = AutoModelForCausalLM.from_pretrained(args.target_model,
                                                        cache_dir = args.cache_dir,
                                                        torch_dtype=torch.bfloat16,
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
    for i in range(len(verbalizer)):
        verbalizer_ids.append(agent_tokenizer.convert_tokens_to_ids(verbalizer[i]))
    
    queue = utils.TopAccuracyTextsNoDuplicates(max_size=5)
    change_num = 0
    #start training
    for ep in tqdm(range(args.epochs)):
        max_total_loss = 0
        min_total_loss = 0
        mean_total_loss = 0
        sum_total_loss = 0
        
        
        
        
        for batch in train_dataloader:
            inputs = batch['text']
            labels = batch['label']
            examples = utils.got_example(validation_dataset,verbalizer,shot=args.num_example)
            with torch.no_grad():
                
                query_text = [
                    {"role" : "user", "content" : args.meta_prompt + '\n' + examples},
                    {"role": "assistant","content" : "The Instruction is : "}
                ]
                
                query_encoded = agent_tokenizer.apply_chat_template(
                    query_text,
                    return_tensors='pt'
                ).view(-1).to(device)
                
                response_tensors =ppo_trainer.generate(
                    query_encoded,
                    **generation_kwargs,
                    return_prompt=False,
                    num_return_sequences = args.prompt_per_example
                )
                
                used_prompt = [agent_tokenizer.decode(r.squeeze(),skip_special_tokens=True) for r in response_tensors]
                
            #나온 프롬프트 중 너무 길이가 짧은게 많으면 종료
            if sum([len(p) for p in used_prompt]) < args.prompt_per_example * 10:
                break
            
            rewards = []
            losses = []
            new_dict ={
                'text' : inputs,
                'label' : labels
            }
            new_ds = Dataset.from_dict(new_dict)
            with torch.no_grad(): 
                accuracys,softmax_diff = utils.evaluation_sd(
                    used_prompt,
                    new_ds,
                    target_model,
                    target_tokenizer,
                    'cuda:0',
                    verbalizer.values(),
                )
            rewards = [  0.01 * softmax_diff[i] + 30 * accuracys[i] for i in range(len(used_prompt))]
            np_rewards = np.array(rewards)
            np_acc = np.array(accuracys)
            rewards = [ torch.tensor(reward) for reward in rewards]
            for i in range(len(rewards)):
                print('reward : ', rewards[i].item(),'acc :', accuracys[i],' prompt : ', used_prompt[i], '\n')
                queue.add(rewards[i].item(),used_prompt[i],ep)
            bs = len(np_rewards)
            #print([query_encoded.view(-1) for i in range(bs)],response_tensors,[torch.tensor(reward) for reward in rewards])
            stats = ppo_trainer.step([query_encoded.view(-1) for i in range(bs)],
                         [response for response in response_tensors],
                         rewards)
            rewards = torch.stack(rewards)
            mean_reward = torch.mean(rewards)
            max_reward = torch.max(rewards)
            wandb.log({
                'rewards' : rewards,
                'mean_reward' : mean_reward,
                'max_reward' : max_reward,
            })
            
            
        #reference model update
        if ep % args.update_term == 0 and ep!=0:
            response_tensors,ref_response_tensors = ppo_trainer.generate(query_encoded.view(-1),**generation_kwargs,return_prompt=False, num_return_sequences=2,generate_ref_response=True)
            used_prompt = [agent_tokenizer.decode(r.squeeze(),skip_special_tokens=True) for r in response_tensors]
            ref_used_prompt = [agent_tokenizer.decode(r.squeeze(),skip_special_tokens=True) for r in ref_response_tensors]
            acc = utils.evaluation(
                used_prompt,
                validation_dataset,
                target_model,
                target_tokenizer,
                device,
                verbalizer.values(),
            )
            ref_acc = utils.evaluation(
                ref_used_prompt,
                validation_dataset,
                target_model,
                target_tokenizer,
                device,
                verbalizer.values(),
            )
            print('acc : ', acc)
            print('ref_acc : ', ref_acc)
            mean_acc = np.mean(np.array(acc))
            mean_ref_acc = np.mean(np.array(ref_acc))
            diff = mean_acc - mean_ref_acc
            if diff > args.update_threshold:
                ppo_trainer.ref_model =  ppo_trainer.model
                print('update ref model')
                change_num +=1
            elif diff < -args.update_threshold:
                ppo_trainer.model = ppo_trainer.ref_model
                print('rollback model')
                change_num -=1
            else:
                change_num=change_num
            if change_num < 0 :
                change_num = 0
            wandb.log({
                'change_num' : change_num,
                'valid_acc' : mean_acc,
                'ref_valid_acc' : mean_ref_acc,
            })
                            
            
    print('Final test Start')
    prompt_queue = queue.get_top_texts()
    new_acc = utils.evaluation(
        [prompt[1] for prompt in prompt_queue],
        test_dataset,
        target_model,
        target_tokenizer,
        device,
        verbalizer.values(),
    )
    for i in range(len(prompt_queue)):
        print('prompt : ',prompt_queue[i][1],'acc : ',new_acc[i])
    max_new_acc = np.max(np.array(new_acc))
    wandb.log({
        'final_acc' : max_new_acc,
        'final_mean_acc' : np.mean(np.array(new_acc))
    })
    
            
if __name__ == '__main__':
    main()
                
                    