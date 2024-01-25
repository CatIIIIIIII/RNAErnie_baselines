# RNAErnie_baselines
Official implement of BERT-like baselines (RNABERT, RNA-MSM, RNA-FM) for paper "Multi-purpose RNA Language Modeling with Motif-aware Pre-training and Type-guided Fine-tuning" with pytorch.

- [RNAErnie\_baselines](#rnaernie_baselines)
  - [Installation](#installation)
  - [Pre-training](#pre-training)
  - [Downstream Tasks](#downstream-tasks)
    - [RNA sequence classification](#rna-sequence-classification)
      - [1. Data Preparation](#1-data-preparation)
      - [2. Fine-tuning](#2-fine-tuning)
    - [RNA RNA interaction prediction](#rna-rna-interaction-prediction)
      - [1. Data Preparation](#1-data-preparation-1)
      - [2. Fine-tuning](#2-fine-tuning-1)
    - [RNA secondary structure prediction](#rna-secondary-structure-prediction)
      - [1. Data Preparation](#1-data-preparation-2)
      - [2. Adaptation](#2-adaptation)

## Installation
First, download the repository and create the environment.

```bash
git clone https://github.com/CatIIIIIIII/RNAErnie_baselines.git
cd ./RNAErnie_baselines
conda env create -f environment.yml
```

Then, activate the "RNAErnie" environment.

```bash
conda activate ErnieFold
```

## Pre-training
You need to download the pre-training model weight from [RNABERT](https://drive.google.com/file/d/10gJBl24OGQ_aZfxtj09dik0rrM4dOk_R/view?usp=sharing), [RNA-MSM](https://drive.google.com/file/d/1-Gl9LGjR_dmDfOARrIVvuOmb7w_tJvGA/view?usp=sharing) and place them in the `./checkpoints` folder. The pre-training model weight of RNA-FM would be downloaded automatically when you run the pre-training script.

## Downstream Tasks

### RNA sequence classification

#### 1. Data Preparation

You can download training data from [Google Drive](https://drive.google.com/drive/folders/1flh2rXiMKIreHE2l4sbjMmwAqfURj4vv?usp=sharing) and place them in the `./data/seq_cls` folder. For baselines, only dataset nRC is available for this task.

#### 2. Fine-tuning

Fine-tune BERT-style large-scale pre-trained language model on RNA sequence classification task with the following command:

```bash
python run_seq_cls.py \
    --device 'cuda:0' \
    --model_name RNAFM
```
You could configure backbone model by changing `--model_name` to `RNAMSM` or `RNABERT`.

### RNA RNA interaction prediction

#### 1. Data Preparation

You can download training data from [Google Drive](https://drive.google.com/drive/folders/1iZK3-rw0QCyustOEUaEII8t2wXS2SwFc?usp=sharing) and place them in the `./data/rr_inter` folder.

#### 2. Fine-tuning

Fine-tune RNAErnie on RNA-RNA interaction task with the following command:

```bash
python run_rr_inter.py \
    --device 'cuda:0' \
    --model_name RNAFM
```
You could configure backbone model by changing `--model_name` to `RNAMSM` or `RNABERT`.

### RNA secondary structure prediction

#### 1. Data Preparation

You can download training data from [Google Drive](https://drive.google.com/drive/folders/1XUBVXAUyIB6NqWmwEdLLlnWFaoU_l3XN?usp=sharing) and unzip and place them in the `./data/ssp` folder. Two tasks (RNAStrAlign-ArchiveII, bpRNA1m) are available for this task.

#### 2. Adaptation

Adapt RNAErnie on RNA secondary structure prediction task with the following command:

```bash
python run_ssp.py \
    --device 'cuda:0' \
    --model_name RNAFM
```
You could configure backbone model by changing `--model_name` to `RNAMSM` or `RNABERT`. Or test on different tasks by changing `--task_name` to `RNAStrAlign` or `bpRNA1m`.
