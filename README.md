# CPO: Chain of Preference Optimization

The official implementation of paper: Chain of Preference Optimization: Improving Chain-of-Thought Reasoning in LLMs.

## Overview

The recent development of chain-of-thought (CoT) decoding has enabled large language models (LLMs) to generate explicit logical reasoning paths for complex problem-solving. However, research indicates that these paths are not always deliberate and optimal. The tree-of-thought (ToT) method employs tree-searching to extensively explore the reasoning space and find better reasoning paths that CoT decoding might overlook. This deliberation, however, comes at the cost of significantly increased inference complexity. In this work, we demonstrate that fine-tuning LLMs leveraging the search tree constructed by ToT allows CoT to achieve similar or better performance, thereby avoiding the substantial inference burden. This is achieved through \emph{Chain of Preference Optimization} (CPO), where LLMs are fine-tuned to align each step of the CoT reasoning paths with those of ToT using the inherent preference information in the tree-search process. Extensive experimental results show that CPO significantly improves LLM performance in solving a variety of complex problems, including question answering, fact verification, and arithmetic reasoning, demonstrating its effectiveness. 

![](https://github.com/sail-sg/CPO/blob/main/Figures/intro_figure.png)

## Setup

```
pip install -r requirement.txt
```

## Quick Start

We show examples of one task. By simply changing the task's name, the approach can be applied to other tasks.

### Data Collection via ToT

1. Selecting reasoning path via ToT.

```
accelerate launch run_test.py --task bamboogle --method_generate sample --method_evaluate value --method_select greedy --n_evaluate_sample 5 --n_generate_sample 15 --n_select_sample 3 --base_model ./model/Llama-2-7b-hf --data_json_file bamboogle_7b.json --train True  >>logs/bamboogle_7b_tot_test.out
```

2. Collect paired preference thoughts for optimization.

```
python clean_dataset.py
```

### Training via CPO

```
accelerate launch dpo_training.py --dataset bam_7b_data.json --wandb_name dpo_7b_bam --base_model ./model/Llama-2-7b-hf --output_dir ./results/results_bam_7b_dpo 
```

### Testing over CoT

```
accelerate launch run_test.py --task bamboogle --naive_run --method_generate greedy --base_model ./results/results_bam_7b_dpo >>logs/bam_7b_dpo.out

```

## Reference Repositories

- Tree-of-thought(ToT) [https://github.com/princeton-nlp/tree-of-thought-llm/](https://github.com/princeton-nlp/tree-of-thought-llm/)
- Direct Preference Optimization (DPO) [https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py)

## Citation

If you find CPO helpful or intriguing and decide to use it, kindly acknowledge the paper by citing it and consider starring this repo, thanks!

```bibtex

