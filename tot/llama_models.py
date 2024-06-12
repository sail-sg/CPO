import sys
import json
import fire
import gradio as gr
import torch
import transformers
from typing import List
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, \
    STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings


def gpt(prompt, tokenizer, GenerationConfig, model, device, temperature=0.5, max_tokens=1000, n=1, stop=None,top_p=0.8,
):
    stopping_criteria = None
    def evaluate(
        prompt,
        input=None,
        temperature=0.4,
        top_p=top_p,
        top_k=40,
        num_beams=n,
        max_new_tokens=32,
        **kwargs,
    ):  
        if isinstance(prompt,list):
            try:
                inputs = tokenizer(prompt, padding=True, return_tensors="pt", truncation=True, max_length=4000)
            except:
                print('tokenizer')
                print(prompt)
                exit()
        else:
            inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        if isinstance(prompt,list):
            # generation_output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, attention_mask=attention_mask, repetition_penalty=1.1, do_sample=True, num_beams=1, temperature=temperature, num_return_sequences=1)
            try:
                generation_output = model.generate(input_ids=input_ids, pad_token_id = 2, max_new_tokens=max_new_tokens, attention_mask=attention_mask, do_sample=False, num_return_sequences=1)
                s = tokenizer.batch_decode(generation_output, skip_special_tokens = True)
            except Exception as error:
                print(error)
                print(prompt)
                s = []
        else:
            if n == 10:
                generation_output = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, attention_mask=attention_mask, do_sample=False, num_return_sequences=1)
                s = tokenizer.batch_decode(generation_output, skip_special_tokens = True)
            else:
                try:
                  s = []
                  for i in range(n):
                      generation_output = model.generate(input_ids=input_ids, pad_token_id = 2, max_new_tokens=max_new_tokens, attention_mask=attention_mask, repetition_penalty=1.1, do_sample=True, temperature=temperature, num_return_sequences=1)
                      s.append(tokenizer.batch_decode(generation_output, skip_special_tokens = True)[0])
                  except Exception as error:
                      print(error)
                      # print('===oom===')
                      print(prompt)
                      s = []
                            
        return s, input_ids

    return evaluate(prompt, temperature=temperature, max_new_tokens=max_tokens,stopping_criteria=stopping_criteria)

