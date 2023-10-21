# Federated Learning of Large Language Models with Parameter-Efficient Prompt Tuning and Adaptive Optimization

## Introduction

FedPepTAO is a Parameter-efficient prompt Tuning approach with Adaptive Optimization to enable efficient and effective Federated Learning of Large Language Models. First, an efficient partial prompt tuning approach is proposed to improve performance and efficiency simultaneously. Second, a novel adaptive optimization method is developed to address the client drift problems on both the device and server sides to enhance performance further. We implement our method on LLMs with up to 7 billion parameters, i.e., LLaMA 7B. More details are provided in our paper [Federated Learning of Large Language Models with Parameter-Efficient Prompt Tuning and Adaptive Optimization](pending url), which has been accepted by the EMNLP 2023 main conference.


## Prepare your environment

All the necessary packages are in environment.yml, and you can install the Conda environment as follows:
For experiments on RoBERTa Large and GPT2 Large
```bash
conda env create -f environment.yml
```
For experiments on LLaMA 3B and LLaMA 7B
```bash
conda env create -f environment_llama.yml
```


## Run FedPepTAO

A detailed walk through can be found in `example_roberta.sh`.


## Citation

If you find this work helpful to your research, please consider cite our paper: (pending bibtex)

```bibtex
```
