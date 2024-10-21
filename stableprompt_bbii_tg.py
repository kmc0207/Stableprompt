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
from ii_utils import load_ii_data,evaluation_ii,got_example_ii, TASK_TO_METRIC, load_annotation,evaluation_ii_batch
from dataset_utils import load_all_dataset,dataset_dicts,load_qa_dataset,qa_dicts,load_generation_dataset,load_bigbench
from peft import LoraConfig
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model',type=str,default='google/gemma-1.1-7b-it')
    parser.add_argument('--agent_model',type=str,default='google/gemma-1.1-7b-it')
    parser.add_argument('--task',type=str,default='classification')
    parser.add_argument('--dataset',type=str,default='gender_inclusive_sentences_german')
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
    parser.add_argument('--no_rollback',action='store_true')
    args = parser.parse_args()
    return args

def main():
    
    args = parser_args()
    device= 'cuda:0'
    name = ''
    if args.no_rollback:
        name += 'no_rollback_'
    wandb.init(project='bbh_tg_algprompt', 
               config=args,
               name = args.dataset+ '_'+name )
    
    
    #load dataset
    #train_dataset, test_dataset, validation_dataset = load_ii_data(args.dataset)
    metrics,train_dataset,test_dataset,verbalizer,task_prefix = load_bigbench(args.dataset)
    validation_dataset = train_dataset
    print('task :' , args.dataset)
    print('metric :', TASK_TO_METRIC.get(args.dataset, 'em'))
    print('train_data_size' , len(train_dataset))
    print('test_data_size' , len(test_dataset))
    #make dataloader
    test_dataloader = DataLoader(test_dataset,batch_size = args.batch_size,shuffle = True)
    train_dataloader = DataLoader(train_dataset,batch_size = args.batch_size,shuffle = True)
    examples = utils.got_example_bbh(train_dataset,verbalizer,shot=args.num_example,metrics=metrics)
    
    print('Example : ', examples)

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
            examples = utils.got_example_bbh(train_dataset,verbalizer,shot=args.num_example,metrics=metrics)
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
            for prompt in used_prompt:
                reward = evaluation_ii_batch(
                    prompt,
                    new_ds,
                    target_model,
                    target_tokenizer,
                    device,
                    prompt,
                    generation_kwargs,
                    args.dataset,
                    batch_size=16,
                )
                rewards.append(reward)
            accuracys = rewards
            #rewards = [  0.01 * softmax_diff[i] + 30 * accuracys[i] for i in range(len(used_prompt))]
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
            acc = []
            for prompt in used_prompt:
                ac = evaluation_ii_batch(
                    prompt,
                    new_ds,
                    target_model,
                    target_tokenizer,
                    device,
                    prompt,
                    generation_kwargs,
                    args.dataset,
                    batch_size=16,
                )
                acc.append(ac)
            ref_acc = []
            for prompt in ref_used_prompt:
                ac = evaluation_ii_batch(
                    prompt,
                    new_ds,
                    target_model,
                    target_tokenizer,
                    device,
                    prompt,
                    generation_kwargs,
                    args.dataset,
                    batch_size=16,
                )
                ref_acc.append(ac)
            print('acc : ', acc)
            print('ref_acc : ', ref_acc)
            mean_acc = np.mean(np.array(acc))
            mean_ref_acc = np.mean(np.array(ref_acc))
            diff = mean_acc - mean_ref_acc
            if diff > args.update_threshold:
                ppo_trainer.ref_model =  ppo_trainer.model
                print('update ref model')
                change_num +=1
            elif diff < -args.update_threshold and not args.no_rollback:
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
    used_prompt = [prompt[1] for prompt in prompt_queue]
    new_acc= []
    for prompt in used_prompt : 
        reward = evaluation_ii_batch(
            prompt,
            new_ds,
            target_model,
            target_tokenizer,
            device,
            prompt,
            generation_kwargs,
            args.dataset,
            batch_size=16,
        )
        new_acc.append(reward)
    print(len(prompt_queue))
    print(len(new_acc))
    
    for i in range(len(prompt_queue)):
        
        print('prompt : ',prompt_queue[i][1],'acc : ',new_acc[i])
    max_new_acc = np.max(np.array(new_acc))
    wandb.log({
        'final_acc' : max_new_acc,
        'final_mean_acc' : np.mean(np.array(new_acc))
    })
    
            
if __name__ == '__main__':
    main()
                
                    