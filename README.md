# SAIS

Code for <a href="https://arxiv.org/abs/2109.12093">SAIS: Supervising and Augmenting Intermediate Steps for Document-Level Relation Extraction</a> (NAACL 2022).

## Requirements

```
PyTorch = 1.9.0
HuggingFace Transformers = 4.8.1
```

## Run

Given an input dataset (e.g., _DocRED_):

1. The ```Data/{dataset}/Original``` folder contains the original files provided by the corresponding dataset that are necessary for our experiments.
1. The command ```bash Code/prepare.sh``` transforms the original data structure into the structure acceptable to our model and stores the output files in the ```Data/{dataset}/Processed``` folder.
1. The command ```bash Code/main.sh``` trains the model, writes the standard output in the ```Data/{dataset}/Stdout``` folder, and delivers the set of predicted relations and corresponding evidence for the develop and test sets in the ```Data/{dataset}/Processed``` folder.

The set of hyperparameters for Step 2 and 3 are specified in ```prepare.sh``` and ```main.sh```, respectively. 

Our model trained on _DocRED_ can be downloaded <a href="https://drive.google.com/drive/folders/1_xM8GdK0G5geYn0t4_L4ONOobLHSpznO?usp=sharing">here</a>.

## Citation

```
@inproceedings{xiao2021sais,
  title={SAIS: Supervising and Augmenting Intermediate Steps for Document-Level Relation Extraction},
  author={Xiao, Yuxin and Zhang, Zecheng and Mao, Yuning and Yang, Carl and Han, Jiawei},
  booktitle={NAACL},
  year={2022}
}
```