#!/bin/bash

stage="Main"
dataset="DocRED"
seed=2021

transformer="bert-base-cased" # "roberta-large"
hidden_size=768
bilinear_block_size=64

RE_max=4
CR_focal_gamma=2
PER_focal_gamma=2
FER_threshold=0.5

loss_weight_CR=0.1
loss_weight_ET=0.1
loss_weight_PER=0.1
loss_weight_FER=0.1

num_epoch=20
batch_size=4 
update_freq=1 

new_lr=1e-4 
pretrained_lr=5e-5
warmup_ratio=0.06 
max_grad_norm=1.0


python3 main.py --stage=${stage} --dataset=${dataset} --seed=${seed} --transformer=${transformer} --hidden_size=${hidden_size} --bilinear_block_size=${bilinear_block_size} --RE_max=${RE_max} --CR_focal_gamma=${CR_focal_gamma} --PER_focal_gamma=${PER_focal_gamma} --FER_threshold=${FER_threshold} --loss_weight_CR=${loss_weight_CR} --loss_weight_ET=${loss_weight_ET} --loss_weight_PER=${loss_weight_PER} --loss_weight_FER=${loss_weight_FER} --num_epoch=${num_epoch} --batch_size=${batch_size} --update_freq=${update_freq} --new_lr=${new_lr} --pretrained_lr=${pretrained_lr} --warmup_ratio=${warmup_ratio} --max_grad_norm=${max_grad_norm}