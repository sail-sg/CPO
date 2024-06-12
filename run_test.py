import os
import json
import argparse
import torch 
import sys
import random
from tot.tasks import get_task
from tot.methods.bfs_test import solve, naive_solve
# from tot.models import gpt_usage
import warnings
import csv
import transformers
from peft import PeftModel, LoraConfig
from transformers import GenerationConfig, AutoModel, AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig, LlamaTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
# import tensor_parallel as tp
import pandas as pd
from load_data import *
from datasets import load_dataset
# from vllm import LLM
import torch.distributed as dist
import time


warnings.filterwarnings("ignore")

def run(args, load_8bit: bool = False,
    base_model: str = "",   

    instruct_dir: str = "",
    use_lora: bool = False,
    lora_weights: str = "",
    # The prompt template to use, will default to med_template.
    prompt_template: str = "med_template"):

    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )
    import time

    print('===base_model===')
    base_model = args.base_model
    print(base_model)
    start_time = time.time()
    if args.fast_test:
        print('=====fast test using distilgpt2=====')
        checkpoint = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
    else:
        tokenizer = LlamaTokenizer.from_pretrained(base_model,padding_side = "left")
        end_time = time.time()
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
        )
        end_time = time.time()

    if use_lora:
        print(f"using lora {lora_weights}")
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float32,
        )
        # LoRA Config

    
    model.config.pad_token_id = tokenizer.pad_token_id = 2  # unk
    tokenizer.pad_token = tokenizer.eos_token
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.config.pad_token_id = model.config.eos_token_id
    # model.resize_token_embeddings(len(tokenizer))
    # if not load_8bit:
    #     model.half()  # seems to fix bugs for some users.
    end_time = time.time()


    accelerator = Accelerator()
    device = accelerator.device
    # model = model.to(device)
    model = accelerator.prepare_model(model)
    model.eval()

    end_time = time.time()

    logs, cnt_avg, cnt_any = [], 0, 0

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # model = torch.nn.DataParallel(model)

    end_time = time.time()

    if args.add_more != '':
        with open(args.add_more,'r') as f:
            lines = f.readlines()
        q_list = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            if (' sum(accs)' in line) & (lines[i-1].startswith('====GR====')):
                q = line.split(' sum(accs)')[0]
                if q not in q_list:
                    q_list.append(q)
    start_time = time.time()
    if args.task in ['game24', 'bamboogle', '2wiki', 'qasc', 'hotpotqa','fever','feverous','tabfacts','vitaminc']:
        if args.task == 'game24':
            data = list(pd.read_csv('tot/data/24/24.csv')['Puzzles'])[args.task_start_index:args.task_end_index]
        elif args.task == 'bamboogle':
            if args.train == False:
                data = list(pd.read_csv('tot/data/bamboogle/Bamboogle Prerelease - Sheet1.csv')['Question'])
                ground_truths = list(pd.read_csv('tot/data/bamboogle/Bamboogle Prerelease - Sheet1.csv')['Answer'])
                data_dic = {}
                for i in range(len(data)):
                    data_dic[data[i]] = ground_truths[i]
            else:
                file = json.load(open('tot/data/bamboogle/train.json'))
                data = []
                data_dic = {}
                # for i in range(len(file['data'])):
                for i in range(200):
                    data.append(file['data'][i]['Question'])
                    data_dic[file['data'][i]['Question']] = file['data'][i]['Answer'][0]

        elif args.task == '2wiki':
            if args.train == False:
                file = json.load(open('tot/data/2wiki/dev.json'))
            else:   
                file = json.load(open('tot/data/2wiki/train.json'))
            data = []
            data_dic = {}
            for i in range(len(file)):
                data.append(file[i]['question'])
                data_dic[file[i]['question']] = file[i]['answer']
            args.task = 'bamboogle'
        elif args.task == 'hotpotqa':
            if args.train == True:
                dataset = load_dataset('hotpot_qa', 'fullwiki', split='train')
            else:
                dataset = load_dataset('hotpot_qa', 'fullwiki', split='validation')

            data = []
            data_dic = {}
            for d in dataset:
                n += 1
                data.append(d['question'])
                data_dic[d['question']] = d['answer'] 
            args.task = 'bamboogle'
        elif args.task in ['fever','feverous','tabfacts','vitaminc']:
            if args.train == False:
                if args.task == 'fever':
                    path_fever = 'tot/data/fever/dev.jsonl'
                if args.task == 'tabfact':
                    path_fever = 'tot/data/tabfact/test_data.jsonl'
                if args.task == 'vitaminc':
                    path_fever = 'tot/data/vitaminc/test.jsonl'
                if args.task == 'feverous':
                    path_fever = 'tot/data/feverous/feverous_dev_challenges.jsonl'
                with open(path_fever) as jsonl_file:
                    lines = jsonl_file.readlines()
                file = {}
                for i, concept in enumerate(lines):
                    concept_item = json.loads(concept)
                    file[i] = {}
                    file[i]['question'] = concept_item['claim']
                    file[i]['answer'] = concept_item['label']
            else: 
                if args.task == 'fever':
                    path_fever = 'tot/data/fever/train.jsonl'
                if args.task == 'tabfact':
                    path_fever = 'tot/data/tabfact/train_data.jsonl'
                if args.task == 'vitaminc':
                    path_fever = 'tot/data/vitaminc/train.jsonl'    
                if args.task == 'feverous':
                    path_fever = 'tot/data/feverous/feverous_train_challenges.jsonl'
                with open(path_fever) as jsonl_file:
                    lines = jsonl_file.readlines()
                file = {}
                for i, concept in enumerate(lines):
                    concept_item = json.loads(concept)
                    file[i] = {}

                    file[i]['question'] = concept_item['claim']
                    file[i]['answer'] = concept_item['label']    
                dev_ids_ori = []
            data = []
            data_dic = {}
            for i in range(len(file)):
                    data.append(file[i]['question'])
                    data_dic[file[i]['question']] = file[i]['answer']

            args.task = 'fever'
        task = get_task(args.task)
        d_l = data
        
        i = args.task_start_index
        if args.data_json_file == 'output.json':
            instances = []
            with open('output.json','r') as f:
                ins = json.load(f)
            for in_ in ins:
                instances.append(list(in_.keys())[0])
        file_out = open(args.data_json_file, 'a')

        dic = []
        if len(args.add_more)>1:
            with open(args.add_more,'r') as f:
                lines = f.readlines()
            for line in lines:
                if 'sum(accs)' in line:
                    if line.split(' sum(accs)')[0] not in dic:
                        dic.append(line.split(' sum(accs)')[0])
        # print(dic[0])
        data_filtered = []
        for d in data:
            if len(args.add_more)>1:
            ##filter out those have been tested
                if d in dic:
                    continue

            if args.data_json_file == 'output.json':
                if d in instances:
                    continue
            data_filtered.append(d)
        if len(data_filtered) == 0:
            exit()

        data = torch.utils.data.DataLoader(data_filtered,batch_size=1)
        data = accelerator.prepare_data_loader(data)

        for d in data:
            if args.naive_run:
                # d is batch_size*instances
                ys, info = naive_solve(args, task, d, model, tokenizer, device) 
                out = {}
            else:
                ys, info, out = solve(args, task, d, model, tokenizer, device)
            # log
            for d_idx, d_i in enumerate(d):
                infos = []
                y = ys[d_idx]
                if args.naive_run:
                    out[d_i] = {}
                if args.task in ['2wiki', 'bamboogle','bbh', 'qasc','fever']:
                    if args.task in ['2wiki', 'bamboogle','bbh','qasc','fever']:
                        if 'the final answer is' not in y.lower():
                            continue
                    info, out = task.test_output(data_dic[d_i], y, out) 
                else:
                    info, out = task.test_output(d_i, y, out)
                infos.append(info)
                # infos, out = [task.test_output(d, y, out) for y in ys]
                # info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': gpt_usage(args.backend)})
                info.update({'idx': d_i, 'ys': ys, 'infos': infos})
                logs.append(info)
                # with open(file, 'w') as f:
                #     json.dump(logs, f, indent=4)
                # log main metric
                accs = [info['r'] for info in infos]
                if len(accs) == 0:
                    print('====wrong case===='+d_i)
                    continue
                cnt_avg += sum(accs) / len(accs)
                cnt_any += any(accs)
                if len(args.output_file) == 0:
                    print(d_i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, '\n')
                else:
                    process_id = dist.get_rank()  # 获取当前进程的ID
                    output_file = args.output_file + str(process_id) + '.out'
                    with open(output_file, "a") as f:
                        print(d_i, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, '\n', file=f)
                # ppo_trainer.save_model()
                file_out.write(json.dumps(out,ensure_ascii=False,indent=1))
                file_out.write(',\n')
                file_out.flush() 
        # n = args.task_end_index - args.task_start_index
        n = len(data)
        print(cnt_avg / n, cnt_any / n)
            # print('usage_so_far', gpt_usage(args.backend))
        file_out.write('\n]')
    elif args.task in ['math','gsm8k','svamp','asdiv']:
        if args.train ==True:
            if args.task == 'svamp':
                data = load_svamp_test('tot/data/SVAMP/train.json')
            elif args.task == 'asdiv':
                data = load_svamp_test('tot/data/asdiv/train.json')
            else:
                data = load_gsm8k_test(split="train")
        else:
            if args.task == 'svamp':
                data = load_svamp_test('tot/data/SVAMP/test.json')
            elif args.task == 'asdiv':
                data = load_svamp_test('tot/data/asdiv/test.json')
            else:
                data = load_gsm8k_test(split="test")
        d_l = data
        args.task = 'math'
        data = torch.utils.data.DataLoader(data)
        data = accelerator.prepare_data_loader(data)
        i = args.task_start_index
        if args.data_json_file != 'test.json':
            instances = []
            with open(args.data_json_file,'r') as f:
                ins = json.load(f)
            for in_ in ins:
                instances.append(list(in_.keys())[0])
        file_out = open(args.data_json_file, 'a')
        # file_out.write('[\n')
        task_prompt = create_demo_text()
        for d in data:
            if args.data_json_file != 'test.json':
                if d['question'][0] in instances:
                    continue
            if args.add_more != '':
                if str(d) in q_list:
                    continue
            if args.naive_run:
                ys, info = naive_solve(args, task_prompt, d['question'][0], model, tokenizer, device) 
                out = {}
            else:
                ys, info, out = solve(args, task_prompt, d['question'][0], model, tokenizer, device)
            infos = []
            for y in ys:
                # print(y)
                info, out = math_test_output(d, y, out)
                infos.append(info)
            # infos, out = [task.test_output(d, y, out) for y in ys]
            # info.update({'idx': i, 'ys': ys, 'infos': infos, 'usage_so_far': gpt_usage(args.backend)})
            info.update({'idx': d['question'][0], 'ys': ys, 'infos': infos})
            logs.append(info)
            # with open(file, 'w') as f:
            #     json.dump(logs, f, indent=4)
            # log main metric
            accs = [info['r'] for info in infos]
            if len(accs) == 0:
                print(d, 'sum(accs)', sum(accs))
            else:
                cnt_avg += sum(accs) / len(accs)
                cnt_any += any(accs)
                print(d, 'sum(accs)', sum(accs), 'cnt_avg', cnt_avg, 'cnt_any', cnt_any, '\n')
            # ppo_trainer.save_model()
            file_out.write(json.dumps(out,ensure_ascii=False,indent=1))
            file_out.write(',\n')
            file_out.flush() 
        # n = args.task_end_index - args.task_start_index
        print(cnt_avg / len(data), cnt_any / len(data))
            # print('usage_so_far', gpt_usage(args.backend))
        file_out.write('\n]')
    end_time = time.time()


    
    

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'llama2-7b'], default='llama2-7b')
    args.add_argument('--temperature', type=float, default=0.9)

    args.add_argument('--task', type=str, required=True, choices=['game24', 'text', 'crosswords', 'math','gsm8k','svamp','asdiv', 'bamboogle', '2wiki', 'bbh', 'qasc', 'hotpotqa','fever','feverous','tabfacts','vitaminc'])
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)
    args.add_argument('--base_model', type=str, default='')
    args.add_argument('--naive_run', action='store_true')
    args.add_argument('--sr_run', action='store_true')
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'], default='cot')  # only used when method_generate = sample, or naive_run

    args.add_argument('--method_generate', type=str, choices=['sample', 'propose'])
    args.add_argument('--method_evaluate', type=str, choices=['value', 'vote'])
    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--n_generate_sample', type=int, default=1)  # only thing needed if naive_run
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--n_select_sample', type=int, default=1)
    args.add_argument('--data_json_file', type=str, default='test.json')
    args.add_argument('--fast_test', type=bool, default=False)
    args.add_argument('--train', type=bool, default=False)
    args.add_argument('--add_more', type=str, default='')    
    args.add_argument('--percentage', type=float, default=1.0)   
    args.add_argument('--epoch', type=float, default=-1)   
    args.add_argument('--output_file', type=str, default='')   
    args = args.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)
