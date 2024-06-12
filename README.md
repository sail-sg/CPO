# CPO: Chain of Preference Optimization

The official implementation of paper: Chain of Preference Optimization: Improving Chain-of-Thought Reasoning in LLMs.

## Overview

The recent development of chain-of-thought (CoT) decoding has enabled large language models (LLMs) to generate explicit logical reasoning paths for complex problem-solving. However, research indicates that these paths are not always deliberate and optimal. The tree-of-thought (ToT) method employs tree-searching to extensively explore the reasoning space and find better reasoning paths that CoT decoding might overlook. This deliberation, however, comes at the cost of significantly increased inference complexity. In this work, we demonstrate that fine-tuning LLMs leveraging the search tree constructed by ToT allows CoT to achieve similar or better performance, thereby avoiding the substantial inference burden. This is achieved through \emph{Chain of Preference Optimization} (CPO), where LLMs are fine-tuned to align each step of the CoT reasoning paths with those of ToT using the inherent preference information in the tree-search process. Extensive experimental results show that CPO significantly improves LLM performance in solving a variety of complex problems, including question answering, fact verification, and arithmetic reasoning, demonstrating its effectiveness. 

![](https://github.com/sail-sg/CPO/blob/main/Figures/intro_figure.pdf)

## Setup

```
pip install -r requirement.txt
```

## Quick Start

## Reference Repositories

- Tree-of-thought(ToT) [https://github.com/princeton-nlp/tree-of-thought-llm/](https://github.com/princeton-nlp/tree-of-thought-llm/)
- Direct Preference Optimization (DPO) [https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py)

## Citation

If you find CPO helpful or intriguing and decide to use it, kindly acknowledge the paper by citing it and consider starring this repo, thanks!

```bibtex

