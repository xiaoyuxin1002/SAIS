#!/bin/bash

stage="Prepare"
dataset="DocRED"
transformer="bert-base-cased" # "roberta-large"
max_seq_length=1024
    
    
python3 prepare.py --stage=${stage} --dataset=${dataset} --transformer=${transformer} --max_seq_length=${max_seq_length}