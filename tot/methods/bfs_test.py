import itertools
import numpy as np
from functools import partial
import re
import sys
import json
import fire
import gradio as gr
import torch
import bisect
import transformers
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from load_data import *

import re
import string

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def math_test_output(y):
    expression = y.strip()
    if 'arabic numerals' in expression:
        expression = expression.split('(arabic numerals) is ')[1]
        expression = expression.replace(',','')
    numbers = re.findall(r'\d+', expression)
    if len(numbers)>0:
        numbers = numbers[-1]
    else:
        # print(y)
        numbers = '-1'
    return numbers
  
def final_evaluate_fever(new_ys, values, final_sentence):
    choices_set = {}
    for i in range(len(new_ys)):
        new_ys[i] = new_ys[i].replace('\"','\'').replace('-',' ').replace('  ',' ').split('\n\n')[0].lower().replace('so the final answer is ', 'so the final answer is: ')
        choices_i_list = new_ys[i].split('\n')
        if final_sentence not in new_ys[i]:
            # print(new_ys[i])
            continue
        if len(choices_i_list)>1:
            while final_sentence not in choices_i_list[-1].lower():
                choices_i_list = choices_i_list[:-1]
                if len(choices_i_list)<=1:
                    break
            if len(choices_i_list)<=1:
                continue
            new_ys[i] = ' '.join(choices_i_list)
        if new_ys[i] not in choices_set:
            choices_set[new_ys[i]] = [float(values[i])]
        else:
            choices_set[new_ys[i]].append(float(values[i]))
    for i in range(len(choices_set)):
        choices = list(choices_set.keys())
        choices_set[choices[i]] = sum(choices_set[choices[i]])/len(choices_set[choices[i]])
    sorted_dict = sorted(choices_set.items(), key=lambda item: item[1], reverse=True)
    sorted_dict = {item[0]: item[1] for item in sorted_dict}
    pre_set = {}
    for choice_item in list(sorted_dict.keys()):
        pre = choice_item.lower().split(final_sentence)[1].replace('.','')
        if '-' in pre:
            continue
        if 'refute' in pre:
            pre = 'refutes'
        if 'reje' in pre:
            pre = 'refutes'
        if ('support' in pre) or ('suport' in pre) or ('correct' in pre):
            pre = 'supports'
        if ('not enough info' in pre )or ('no enough info' in pre) or ('no evidence'in pre):
            pre = 'not enough info'
        if pre not in ['refutes','supports','not enough info']:
            print(pre)
            continue
        if pre not in pre_set:
            pre_set[pre] = {}
            pre_set[pre]['value'] = [choices_set[choice_item]]
            pre_set[pre]['item'] = [choice_item]
        else:
            pre_set[pre]['value'].append(choices_set[choice_item])
            pre_set[pre]['item'].append(choice_item)
            
    if len(pre_set)>0:
        pre_ = ''
        len_ = 0
        max_val = 0
        # for pre_item in list(pre_set.keys()):
        #     if sum(pre_set[pre_item]['value'])>max_val:
        #         pre_ = pre_item
        #         len_ = len(pre_set[pre_item]['value'])
        #         max_val = sum(pre_set[pre_item]['value'])
        for pre_item in list(pre_set.keys()):
            if len(pre_set[pre_item]['value'])>len_:
                pre_ = pre_item
                len_ = len(pre_set[pre_item]['value'])
                max_val = sum(pre_set[pre_item]['value'])
        if pre_ == '':
            return [], []
        else:
            return pre_set[pre_]['item'], pre_set[pre_]['value']
    else:
        return [],[]

def final_evaluate(new_ys, values, final_sentence):
    choices_set = {}
    for i in range(len(new_ys)):
        new_ys[i] = new_ys[i].replace('\"','\'').replace('  ',' ').split('\n\n')[0].lower().replace('so the final answer is ', 'so the final answer is: ')
        choices_i_list = new_ys[i].split('\n')
        
        if final_sentence not in new_ys[i]:
            print('line 57')
            print(new_ys[i])
            continue

        if len(choices_i_list)>1:
            while final_sentence not in choices_i_list[-1].lower():
                choices_i_list = choices_i_list[:-1]
                if len(choices_i_list)<=1:
                    break
            if len(choices_i_list)==1:
                if final_sentence not in choices_i_list[0]:
                    print('line 68')
                    print(new_ys[i])
                    continue
            if len(choices_i_list)==0:
                print('line 72')
                print(new_ys[i])
                continue

            new_ys[i] = ' '.join(choices_i_list)
        if new_ys[i] not in choices_set:
            choices_set[new_ys[i]] = [float(format(values[i],'.3f'))]
        else:
            choices_set[new_ys[i]].append(float(format(values[i],'.3f')))
    for i in range(len(choices_set)):
        choices = list(choices_set.keys())
        choices_set[choices[i]] = sum(choices_set[choices[i]])/len(choices_set[choices[i]])
    sorted_dict = sorted(choices_set.items(), key=lambda item: item[1], reverse=True)
    sorted_dict = {item[0]: item[1] for item in sorted_dict}
    pre_set = {}
    for choice_item in list(sorted_dict.keys()):

        pre = choice_item.lower().split(final_sentence)[1].replace('.','')

        pre = normalize_answer(pre)
        if pre not in pre_set:
            pre_set[pre] = {}
            pre_set[pre]['value'] = [choices_set[choice_item]]
            pre_set[pre]['item'] = [choice_item]
        else:
            pre_set[pre]['value'].append(choices_set[choice_item])
            pre_set[pre]['item'].append(choice_item)
            
    if len(pre_set)>0:
        pre_ = ''
        len_ = 0
        max_val = 0
        tem_list = []
        for pre_item in list(pre_set.keys()):
            if sum(pre_set[pre_item]['value'])>max_val:
                if 'arabic numerals' in final_sentence:
                    if math_test_output(pre_item) =='-1':
                        continue
                pre_ = pre_item
                len_ = len(pre_set[pre_item]['value'])
                max_val = sum(pre_set[pre_item]['value'])
                tem_list = pre_set[pre_item]['value']
                
            elif sum(pre_set[pre_item]['value'])==max_val:
                if sorted(pre_set[pre_item]['value'])[0]>sorted(tem_list)[0]:
                    pre_ = pre_item
                    len_ = len(pre_set[pre_item]['value'])
                    max_val = sum(pre_set[pre_item]['value'])
                    tem_list = pre_set[pre_item]['value']
        # for pre_item in list(pre_set.keys()):
        #     if len(pre_set[pre_item]['value'])>len_:
        #         pre_ = pre_item
        #         len_ = len(pre_set[pre_item]['value'])
        #         max_val = sum(pre_set[pre_item]['value'])
        if pre_ == '':
            print(pre_set)
            return [], []
        else:
            return pre_set[pre_]['item'], pre_set[pre_]['value']
    else:
        return [],[]

def get_value(task, x, y, n_evaluate_sample, tokenizer, GenerationConfig, model, device, step,cache_value=True):
    # if x.lower().strip() in y.lower().strip():
    #     return 0.001
    if isinstance(task,str) == False:
        if (task.steps == 3):
            if (step == 2):
                if 'the final answer is' not in y.lower():
                    return 0
        if task.stops == 'qasc':
            if (step == 3):
                if 'so the final answer is' not in y.lower():
                    return 0
                if y.lower().split('so the final answer is ')[1].replace(': ','').replace('.','') not in x.lower():
                    return 0
                if (y.lower().split('so the final answer is')[1].strip().startswith('(')) ==False:
                    return 0
        if (task.steps == 4):
            value_prompts = task.value_prompt_wrap(x, y, step)
            if value_prompts == 0:
                return 0
        else:
            value_prompts = task.value_prompt_wrap(x, y)
    else:
        value_prompts = math_evaluate + 'Question: ' + x + '\nThought: ' + y +'\nEvaluation Process: '
        # if 'answer (arabic numerals) is ' in y:
        #     value_prompts = math_final_evaluate + 'Question: ' + x + '\nThought: ' + y +'\nEvaluation Process: '
        # else:
        #     value_prompts = math_evaluate + 'Question: ' + x + '\nThought: ' + y +'\nEvaluation Process: '
    if isinstance(value_prompts,list):
        value_prompts_list = value_prompts
        value = 1
        for value_prompts in value_prompts_list:
            value_outputs, _ = gpt(value_prompts, tokenizer, GenerationConfig, model, device, n=n_evaluate_sample, temperature=0.4,stop=None)
            
            value_item = 0
            for i in range(len(value_outputs)):
                value_output = value_outputs[i]
                # if isinstance(value_prompts,list):
                #     value_prompt = value_prompts[i]
                # else:
                value_prompt = value_prompts
                # print(f'========\n {value_output}')
                value_output = value_output.replace(value_prompt, "").replace('</s>', '').replace('<s>', '')
                # print(f'========\nQuestion: {x}\nThought: {y}\nvalue_output: {value_output}\n========')
                value_output = value_output.split('\n')
                value_item += value_outputs_unwrap(x, y, value_output)
            value = value*value_item
    else:
        value_outputs, _ = gpt(value_prompts, tokenizer, GenerationConfig, model, device, n=n_evaluate_sample, max_tokens=256, temperature=0.4,stop=None)
        value = 0
        
        for i in range(len(value_outputs)):
            value_output = value_outputs[i]
            if isinstance(value_prompts,list):
                value_prompt = value_prompts[i]
            else:
                value_prompt = value_prompts
            # print(f'========\n {value_output}')
            value_output = value_output.replace(value_prompt, "").replace('</s>', '').replace('<s>', '')
            # print(f'========\nQuestion: {x}\nThought: {y}\nvalue_output: {value_output}\n_generate_sample========')
            value_output = value_output.split('\n')
            value += value_outputs_unwrap(x, y, value_output)
    return value

def get_values(task, x, ys, n_evaluate_sample, tokenizer, GenerationConfig, model, device, step,cache_value=True):
    values = []
    for y in ys:  # each partial output
        value = get_value(task, x, y, n_evaluate_sample, tokenizer, GenerationConfig, model, device,step, cache_value=cache_value)
        values.append(round(value, 3))
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y, tokenizer, GenerationConfig, model): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, tokenizer, GenerationConfig, model, n=1, stop='None').split('\n')
    indices = [index for index, value in enumerate(proposals) if 'Input' in value]
    if len(indices)>2:
        proposals = proposals[indices[1]+2:indices[2]]
    else:
        proposals = proposals[indices[1]+2:]
    return [y + _ + '\n' for _ in proposals]

def get_samples_wiki(task, x, y, n_generate_sample, tokenizer, GenerationConfig, model, device, prompt_sample, step, bbh_flag):
    if bbh_flag != None:
        prompt = ''
        with open('/home/ducunxiao/Distill-ToT/tot/BIG-Bench-Hard/cot-prompts/' + bbh_flag.split('.')[0]+'.json') as f:
            lines = f.readlines()
        for line in lines:
            prompt += line
        prompt = prompt  + '\nQ: '+ x + '\nA: '
    else:
        if prompt_sample == 'standard':
            prompt = task.standard_prompt_wrap(x, y)
        elif prompt_sample == 'cot':
            prompt = task.cot_prompt_wrap(x, y)
        else:
            raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    out_sample = []
    # if 'so the final answer is' in y.lower():
    #     return [y], []
    # else:
    if step == 0:
        prompt_ = prompt + 'Answer: Step 1, '
    else:
        prompt_ = prompt + ' Step ' + str(step+1) + ', '
    if bbh_flag != None:
        if step == 4:
            prompt_ = prompt_ + 'so the final answer is: '
    else:
        if 'Options:' in prompt:
            if step == 3:
                prompt_ = prompt_ + 'So the final answer is ('
        else:
            if step == 2:
                prompt_ = prompt_ + 'So the final answer is: '
    # while len(out_sample)<n_generate_sample:
    samples, input_ids = gpt(prompt_, tokenizer, GenerationConfig, model, device, n=n_generate_sample-len(out_sample), temperature=0.4)
    out_sample = clean_generated_thought(prompt, samples, out_sample, y)
    # print(out_sample)
    return out_sample, input_ids

def clean_generated_thought(prompt, samples, out_sample, y):
    for i in range(len(samples)):
        sample = samples[i]
        sample = sample.replace(prompt, "").replace('</s>', '').replace('<s>', '').lower()
        if 'question:' in sample:
            sample = sample.split('question:')[0]
        if 'claim:' in sample:
            sample = sample.split('claim:')[0]
        if 'task:' in sample:
            sample = sample.split('task:')[0]
        if len(sample.strip()) < 2:
            continue 
        if 'step' in sample:
            splited_sample = sample.split('step')
            if len(splited_sample) >2:
                sample = splited_sample[0]+'step' + splited_sample[1]
        else:
            print('====clean_generated_thought====')
            print(sample)
            print('====prompt====')
            print(prompt)
            exit()
            sample = sample.split('\n')[0]
        if sample.count('so the final answer is') > 1:
            first_idx = sample.lower().find('so the final answer is')
            sample = sample[:sample.lower().find('so the final answer is', first_idx+1)]
        if len(sample.strip()) < 2:
            continue
        sample = sample.strip()
        if (y + sample) not in out_sample:
            out_sample.append(y + sample)
    return out_sample
    
def get_samples(task, x, y, n_generate_sample, tokenizer, GenerationConfig, model, device, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples, input_ids = gpt(prompt, tokenizer, GenerationConfig, model, device, n=n_generate_sample, temperature=0.3, stop=stop)
    out_sample = []
    for i in range(len(samples)):
        sample = samples[i]
        sample = sample.replace(prompt, "").replace('</s>', '').replace('<s>', '')
        splited_sample = sample.split('\n')
        if 'Steps' in splited_sample[0]:
            samples[i] = splited_sample[0] + '\n' + splited_sample[1] + '\n'
        elif 'Answer' in splited_sample[0]:
            samples[i] = splited_sample[0]
        else:
            samples[i] = splited_sample[0] + '\n'
        if len(samples[i].strip()) < 2:
            continue
        out_sample.append(y + samples[i])
    return out_sample, input_ids

def get_samples_cot(task, x, y, n_generate_sample, tokenizer, GenerationConfig, model, device, prompt_sample, stop=None,bbh_flag=None):  
    if bbh_flag != None:
        prompt = ''
        with open('/home/ducunxiao/Distill-ToT/tot/BIG-Bench-Hard/cot-prompts/' + bbh_flag.split('.')[0]+'.txt') as f:
            lines = f.readlines()
        for line in lines:
            prompt += line
        prompt_ = prompt + '\nQ: '+ x + '\nA: '
    else:
        if isinstance(x,str):
            if prompt_sample == 'standard':
                prompt = task.standard_prompt_wrap(x, y)
            elif prompt_sample == 'cot':
                prompt = task.cot_prompt_wrap(x, y)
            else:
                raise ValueError(f'prompt_sample {prompt_sample} not recognized')
            prompt_ = prompt+' Answer: Step 1, '
        else:
            prompt_ = []
            prompts = []
            for x_i in x:
                if prompt_sample == 'standard':
                    prompt = task.standard_prompt_wrap(x_i, y)
                elif prompt_sample == 'cot':
                    prompt = task.cot_prompt_wrap(x_i, y)
                else:
                    raise ValueError(f'prompt_sample {prompt_sample} not recognized')
                prompt_.append(prompt+' Answer: Step 1, ')
                prompts.append(prompt)

    while True:
        samples, input_ids = gpt(prompt_, tokenizer, GenerationConfig, model, device, n=n_generate_sample, stop=stop, temperature=0.4)
        for i in range(len(samples)):
            sample = samples[i]
            sample = sample.replace(prompts[i], "").replace('</s>', '').replace('<s>', '')
            splited_sample = sample.split('\n')
            tem_s = []
            for splited_s in splited_sample:
                if 'nput' in splited_s:
                    break
                if 'question:' in splited_s.lower():
                    break
                if 'nswer' in splited_s:
                    tem_s.append(splited_s)
                    break
                if 'claim: 'in splited_s.lower():
                    break
                tem_s.append(splited_s)
            samples[i] = '\n'.join(tem_s)
        if bbh_flag != None:
            break

        break_flag= True
        for i in range(len(samples)): 
            if 'step 2' not in samples[i].lower():
                prompt_[i] = prompts[i]+samples[i]+' Step 2, '
                break_flag = False
            elif 'step 3, so the final answer is:' not in samples[i].lower():
                prompt_[i] = prompts[i]+samples[i]+' Step 3, so the final answer is: '
                break_flag = False
            else:
                prompt_[i] = prompts[i]+samples[i]
        if break_flag:
            break

    return [y + _ for _ in samples], input_ids


def get_samples_math(task_prompt, x, ys, n_generate_sample, tokenizer, GenerationConfig, model, device, prompt_sample, stop=None):
    if isinstance(ys,list):
        prompt_ = []
        task_prompt_ = []
        for y in ys:
            task_prompt = task_prompt+'Q: '+x+'\nA: '+y + 'The answer (arabic numerals) is ' 
            task_prompt_.append(task_prompt)
            prompt_.append(task_prompt_)
    else:
        y = ys
        task_prompt = task_prompt+'Q: '+x+'\n'
        task_prompt_ = (task_prompt + 'A: Step 1, '+y)
        prompt_ = [task_prompt_]
        task_prompt = [task_prompt]
    splited_samples = []
    while True:
        samples, input_ids = gpt(prompt_, tokenizer, GenerationConfig, model, device, n=n_generate_sample, stop=stop)
        splited_samples = []
        for i in range(len(samples)):
            sample = samples[i]
            sample = sample.replace(task_prompt[i], "").replace('</s>', '').replace('<s>', '').lower()
            if 'q:' in sample:
                sample = sample.split('q:')[0]
            if '\n\n' in sample:
                sample = sample.split('\n\n')[0]
            samples[i] = sample
        break_flag= True
        for i in range(len(samples)): 
            if 'step 2' not in samples[i].lower():
                prompt_[i] = prompt_[i]+samples[i]+' Step 2, '
                break_flag = False
            elif 'step 3' not in samples[i].lower():
                prompt_[i] = prompt_[i]+samples[i]+' Step 3, '
                break_flag = False
            elif 'step 4' not in samples[i].lower():
                prompt_[i] = prompt_[i]+samples[i]+' Step 4, The answer (arabic numerals) is '
                break_flag = False
            elif ('step 4' in samples[i].lower()) & ('answer (arabic numerals' not in samples[i].lower()):
                prompt_[i] = prompt_[i]+samples[i].lower().split('step 4')[0]+' Step 4, The answer (arabic numerals) is '
            else:
                prompt_[i] = prompt_[i]+samples[i]
        if break_flag:
            break
            # if '(arabic numerals) is ' in sample:   
            #     splited_samples.append(sample)
    return samples, samples, input_ids

def get_samples_math_tot(task_prompt, x, y, n_generate_sample, tokenizer, GenerationConfig, model, device, prompt_sample, step):
    task_prompt = task_prompt+'Q: '+x+'\nA: '+y
    sample_last = y
    if step == 3:
        prompt_ = task_prompt + 'Step 4, The answer (arabic numerals) is '
    else:
        prompt_ = task_prompt + 'Step ' + str(step+1) + ', '
    samples, input_ids = gpt(prompt_, tokenizer, GenerationConfig, model, device, n=n_generate_sample, temperature=0.7,  max_tokens=128, stop=None,top_p=0.9)
    for i in range(len(samples)):
        sample = samples[i]          
        sample = sample.replace(task_prompt, "").replace('</s>', '').replace('<s>', '').replace('\n',' ').lower()
        if '\n\nq:' in sample:
            sample = sample.split('\n\nq:')[0]
        if 'q:' in sample:
            sample = sample.split('q:')[0]
        if 'step' in sample:
            splited_sample = sample.split('step')
            if len(splited_sample) >2:
                sample = splited_sample[0]+'step' + splited_sample[1]
        
        samples[i] = y + sample
    # print(samples)
    # print(len(samples))
    return samples, input_ids

def solve(args, task, x, model, tokenizer, device, to_print=True,
    # The prompt template to use, will default to med_template.
    prompt_template: str = "med_template", bbh_flag = None):
    if args.backend == 'llama2-7b':
       from tot.llama_models import gpt
    else:
        from tot.models import gpt
    global gpt
    gpt = partial(gpt)
    # print(gpt)
    out = {}
    ys = ['']  # current output candidates
    infos = []
    if 'math' not in args.task:
        x = x[0]
    out[x] = {}
    if isinstance(task, str):
        steps = 4
        stops = ['.']*steps
    else:
        steps = task.steps
        stops = task.stops
    if bbh_flag != None:
        steps = 5
    for step in range(steps):
        # generation
        select_new_ys = []
        n_select_sample = args.n_select_sample-len(select_new_ys)
        if args.method_generate == 'sample':
            if 'math' in args.task:
                new_ys = [get_samples_math_tot(task, x, y, args.n_generate_sample, tokenizer, GenerationConfig, model, device, prompt_sample=args.prompt_sample, step=step)[0] for y in ys]          
            elif '24' in args.task:
                new_ys = [get_samples(task, x, y, args.n_generate_sample, tokenizer, GenerationConfig, model, device, prompt_sample=args.prompt_sample, stop=stops[step])[0] for y in ys]
            else:
                new_ys = [get_samples_wiki(task, x, y, args.n_generate_sample, tokenizer, GenerationConfig, model, device, prompt_sample=args.prompt_sample, step=step, bbh_flag = bbh_flag)[0] for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y, tokenizer, GenerationConfig, model) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        new_ys = list(set(new_ys))
        out[x][str(step)] = {}
        out[x][str(step)]['candiate'] = new_ys
        ids = list(range(len(new_ys)))
        
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample, tokenizer, GenerationConfig, model, device = device, step=step)
        out[x][str(step)]['values'] = values
        
        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            if ((step == 2 )&(args.task not in ['math','game24','qasc'])):
                if args.task == 'fever':
                    new_select_new_ys, sorted_values = final_evaluate_fever(new_ys, values,'so the final answer is: ')
                else:
                    new_select_new_ys, sorted_values = final_evaluate(new_ys, values,'so the final answer is: ')
                select_new_ys.extend(new_select_new_ys)
                print(f'-- new_ys --: {new_ys}\n-- sol values --: {values}\n-- choices --: {select_new_ys}\n')
            elif (step == 3 )&(args.task in ['math']):
                new_select_new_ys, sorted_values = final_evaluate(new_ys, values,'answer (arabic numerals) is ')
                select_new_ys.extend(new_select_new_ys)
                print(f'-- new_ys --: {new_ys}\n-- sol values --: {values}\n-- choices --: {select_new_ys}\n')
            elif ((step == 3 )&(args.task in ['qasc'])):
                new_select_new_ys, sorted_values = final_evaluate(new_ys, values,'so the final answer is: ')
                select_new_ys.extend(new_select_new_ys)
                print(f'-- new_ys --: {new_ys}\n-- sol values --: {values}\n-- choices --: {select_new_ys}\n')
            else:
                sorted_ids = sorted(ids, key=lambda x: values[x], reverse=True)
                select_ids = sorted_ids[:n_select_sample] 

                for i in range(n_select_sample,len(sorted_ids)):
                    id = sorted_ids[i]
                    p_id = sorted_ids[i-1]
                    if (values[id]==values[p_id]):
                        select_ids.append(id)
                    else:
                        break
                select_new_ys.extend([new_ys[select_id] for select_id in select_ids])
                # log
                if to_print:
                    try:
                        sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
                    except:
                        print('====error====')
                        print('new_ys')
                        print(new_ys)
                        print('values')
                        print(values)
                        exit()
                    print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    if to_print: 
        print(ys)
    out[x]['correct'] = ys
    return ys, {'steps': infos}, out

def naive_solve(args, task, x, model, tokenizer, device, to_print=True, bbh_flag = None):
    if args.backend == 'llama2-7b':
       from tot.llama_models import gpt
    else:
        from tot.models import gpt
    global gpt
    out = {}
    gpt = partial(gpt, temperature=args.temperature)
    # x = task.get_input(idx)  # input
    if isinstance(x,list):
        for x_i in x:
            out[x_i] = {}
    else:
        out[x] = {}
    if 'math' in args.task:
        ys,ys_ori,_ = get_samples_math(task, x, '', args.n_generate_sample, tokenizer, GenerationConfig, model, device, args.prompt_sample, stop=None)
        if len(ys) == 0:
            ys,ys_ori,_ = get_samples_math(task, x, ys_ori, args.n_generate_sample, tokenizer, GenerationConfig, model, device, args.prompt_sample, stop=None)
    else:
        ys,_ = get_samples_cot(task, x, '', args.n_generate_sample, tokenizer, GenerationConfig, model, device, args.prompt_sample, stop=None, bbh_flag = bbh_flag)
    print(ys)
    return ys, out

