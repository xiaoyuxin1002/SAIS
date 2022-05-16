import json
import math
import time
import random
import dill as pk
import numpy as np
from itertools import permutations
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
        
        
def myprint(text, file):
    
    file = open(file, 'a')
    print(time.strftime("%Y %b %d %a, %H:%M:%S: ", time.localtime()) + text, file=file, flush=True)
    file.close()
    
    
def prepare_autos(args, info):
    
    config = AutoConfig.from_pretrained(args.transformer, num_labels=info.NUM_REL)
    tokenizer = AutoTokenizer.from_pretrained(args.transformer)
    transformer = AutoModel.from_pretrained(args.transformer, config=config)
    
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    
    return config, tokenizer, transformer


def prepare_optimizer_scheduler(args, num_train_docs, model):
    
    grouped_parameters = defaultdict(list)
    for name, param in model.named_parameters():
        if 'transformer_module' in name: grouped_parameters['pretrained_lr'].append(param)
        else: grouped_parameters['new_lr'].append(param)
    grouped_lrs = [{'params':grouped_parameters[group], 'lr':lr} for group, lr in zip(['pretrained_lr', 'new_lr'], [args.pretrained_lr, args.new_lr])]
    optimizer = AdamW(grouped_lrs)

    num_updates = math.ceil(math.ceil(num_train_docs / args.batch_size) / args.update_freq) * args.num_epoch
    num_warmups = int(num_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmups, num_training_steps=num_updates)

    return optimizer, scheduler


def cal_f1(relations, predictions):
        
    TP = ((relations == predictions) & predictions).sum().item()
    P = predictions.sum().item()
    T = relations.sum().item()

    precision = TP / P
    recall = TP / T
    f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1


def evaluate(args, info, mode, triplets, sids):
    
    myprint(f'Start Evaluating {mode} Result', info.FILE_STDOUT)
        
    outputs = []
    for triplet, sid in zip(triplets, sids):
        outputs.append({'title':triplet[0], 'h_idx':triplet[1], 't_idx':triplet[2], 'r':info.DATA_ID2REL[triplet[3]], 'evidence':sid.gt(args.FER_threshold).nonzero().flatten().tolist()})

    json.dump(outputs, open(f'{info.FILE_RESULTS[mode]}_{info.index}.json', 'w'))

    if mode == info.MODE_DEV:

        truths = pk.load(open(info.FILE_TRUTHS[mode], 'rb'))
        dev_in_train = pk.load(open(info.FILE_DEV_IN_TRAIN, 'rb'))

        Pred, Pred_sent = 0, 0
        Correct, Correct_sent = 0, 0
        Correct_ign_train, Incorrect_ign_train = 0, 0

        for output in outputs:

            Pred += 1
            Pred_sent += len(output['evidence'])

            output_key = (output['title'], output['h_idx'], output['t_idx'], output['r'])
            if output_key in truths:
                Correct += 1
                Correct_sent += len(set(output['evidence']) & truths[output_key])

            if output_key not in dev_in_train:
                if output_key in truths: Correct_ign_train += 1
                else: Incorrect_ign_train += 1

        Precision = Correct / Pred if Pred!=0 else 0
        Recall = Correct / len(truths)
        F1 = 2 * Precision * Recall / (Precision + Recall) if Precision+Recall!=0 else 0
        myprint(f'{mode} Set - Base Precision: {Precision:.4f} | Recall: {Recall:.4f} | F1: {F1:.4f}', info.FILE_STDOUT)

        Precision_ign_train = Correct_ign_train / (Correct_ign_train + Incorrect_ign_train) if Correct_ign_train+Incorrect_ign_train!=0 else 0
        Recall_ign_train = Correct_ign_train / (len(truths) - len(dev_in_train))
        F1_ign_train = 2 * Precision_ign_train * Recall_ign_train / (Precision_ign_train + Recall_ign_train) if Precision_ign_train+Recall_ign_train!=0 else 0
        myprint(f'{mode} Set - Ignore Train Precision: {Precision_ign_train:.4f} | Recall: {Recall_ign_train:.4f} | F1: {F1_ign_train:.4f}', info.FILE_STDOUT)

        Precision_sent = Correct_sent / Pred_sent if Pred_sent!=0 else 0
        Recall_sent = Correct_sent / sum([len(evidence) for evidence in truths.values()])
        F1_sent = 2 * Precision_sent * Recall_sent / (Precision_sent + Recall_sent) if Precision_sent+Recall_sent!=0 else 0
        myprint(f'{mode} Set - Evidence Precision: {Precision_sent:.4f} | Recall: {Recall_sent:.4f} | F1: {F1_sent:.4f}', info.FILE_STDOUT)

    myprint('-'*20, info.FILE_STDOUT)
    
    
def register_hyperparameters_prepare(args, info):
    
    myprint(f'Transformer = {args.transformer}', info.FILE_STDOUT)
    myprint(f'Max Sequence Length = {args.max_seq_length}', info.FILE_STDOUT)
    
    
def register_hyperparameters_main(args, info):
    
    myprint(f'Seed = {args.seed}', info.FILE_STDOUT)
    myprint(f'Transformer = {args.transformer}', info.FILE_STDOUT)
    myprint(f'Hidden Dim = {args.hidden_size}', info.FILE_STDOUT)
    myprint(f'Bilinear Block Size = {args.bilinear_block_size}', info.FILE_STDOUT)
    
    myprint(f'RE Max Relations = {args.RE_max}', info.FILE_STDOUT)
    myprint(f'CR Focal Gamma = {args.CR_focal_gamma}', info.FILE_STDOUT)
    myprint(f'PER Focal Gamma = {args.PER_focal_gamma}', info.FILE_STDOUT)
    myprint(f'FER Threshold = {args.FER_threshold}', info.FILE_STDOUT)
    
    myprint(f'CR Loss Weight = {args.loss_weight_CR}', info.FILE_STDOUT)
    myprint(f'ET Loss Weight = {args.loss_weight_ET}', info.FILE_STDOUT)
    myprint(f'PER Loss Weight = {args.loss_weight_PER}', info.FILE_STDOUT)
    myprint(f'FER Loss Weight = {args.loss_weight_FER}', info.FILE_STDOUT)
    
    myprint(f'Num of Epoch = {args.num_epoch}', info.FILE_STDOUT)
    myprint(f'Batch Size = {args.batch_size}', info.FILE_STDOUT)
    myprint(f'Updating Frequency = {args.update_freq}', info.FILE_STDOUT)
    
    myprint(f'New Learning Rate = {args.new_lr}', info.FILE_STDOUT)
    myprint(f'Pretrained Learning Rate = {args.pretrained_lr}', info.FILE_STDOUT)
    myprint(f'Warmup Ratio = {args.warmup_ratio}', info.FILE_STDOUT)    
    myprint(f'Max Gradient Norm = {args.max_grad_norm}', info.FILE_STDOUT)
    
    
def prepare_batch_train(info, inputs, batch_size):
    
    doc_titles = list(inputs.keys())
    np.random.shuffle(doc_titles)
    num_batch = math.ceil(len(doc_titles) / batch_size)
    
    for idx_batch in range(num_batch):
        
        batch_titles = doc_titles[idx_batch*batch_size:(idx_batch+1)*batch_size]
        batch_inputs = [inputs[doc_title] for doc_title in batch_titles]
    
        batch_token_seqs, batch_token_masks, batch_token_types = [], [], []
        batch_sent_tids, batch_num_sents_per_doc = [], []
        batch_mention_tids, batch_mention_coreferences, batch_num_mentions_per_doc = [], [], []
        batch_epair_tids, batch_epair_types, batch_epair_masks, batch_epair_relations, batch_epair_pooled_evidences, batch_epair_finegrained_evidences, batch_num_epairs_per_doc = [], [], [], [], [], [], []
                
        for doc_input in batch_inputs:

            batch_token_seqs.append(doc_input.doc_tokens)
            batch_token_masks.append(torch.ones(doc_input.doc_tokens.shape[0]))

            doc_token_types = torch.zeros(doc_input.doc_tokens.shape[0])
            for sid in range(len(doc_input.sid2tids)):
                batch_sent_tids.append(doc_input.sid2tids[sid][0])
                doc_token_types[doc_input.sid2tids[sid][0] : doc_input.sid2tids[sid][1]] = sid % 2
            batch_token_types.append(doc_token_types)
            batch_num_sents_per_doc.append(len(doc_input.sid2tids))

            doc_mention_tids, doc_num_mentions_per_doc = [], []
            for tids in doc_input.eid2tids.values():
                doc_mention_tids += list(tids)
                doc_num_mentions_per_doc.append(len(tids))
            batch_mention_tids += doc_mention_tids
            batch_num_mentions_per_doc.append(sum(doc_num_mentions_per_doc))

            doc_mention_tids = np.cumsum([0] + doc_mention_tids)
            doc_mention_coreferences = torch.zeros((batch_num_mentions_per_doc[-1], batch_num_mentions_per_doc[-1]))
            for mid in range(len(doc_mention_tids)-1):
                doc_mention_coreferences[doc_mention_tids[mid]:doc_mention_tids[mid+1], doc_mention_tids[mid]:doc_mention_tids[mid+1]] = 1
            batch_mention_coreferences.append(doc_mention_coreferences.flatten())

            max_epair_tids = max([len(tids) for tids in doc_input.eid2tids.values()])
            doc_epair_tids, doc_epair_types, doc_epair_masks, doc_epair_relations, doc_epair_pooled_evidences, doc_epair_finegrained_evidences, doc_num_epairs_per_doc = [], [], [], [], [], [], 0            
            for eid_i, eid_j in permutations(doc_input.eid2etype, 2):
                
                doc_num_epairs_per_doc += 1
                doc_epair_types.append((doc_input.eid2etype[eid_i], doc_input.eid2etype[eid_j]))

                epair_tids = torch.full((2, max_epair_tids), -1)
                epair_tids[0, :len(doc_input.eid2tids[eid_i])] = torch.Tensor(list(doc_input.eid2tids[eid_i])).long()
                epair_tids[1, :len(doc_input.eid2tids[eid_j])] = torch.Tensor(list(doc_input.eid2tids[eid_j])).long()
                doc_epair_tids.append(epair_tids)

                epair_masks = torch.ones(doc_input.doc_tokens.shape[0])
                doc_epair_masks.append(epair_masks)

                epair_relations = torch.zeros(info.NUM_REL)
                epair_pooled_evidences = torch.zeros(len(doc_input.sid2tids))                
                for rid, sids in sorted(doc_input.eids2rid2sids[(eid_i, eid_j)].items(), key=lambda x:x[0]):
                    epair_relations[rid] = 1
                    epair_finegrained_evidences = torch.zeros(len(doc_input.sid2tids))
                    for sid in sids:
                        epair_pooled_evidences[sid] += 1
                        epair_finegrained_evidences[sid] = 1
                    doc_epair_finegrained_evidences.append(epair_finegrained_evidences)            
                doc_epair_relations.append(epair_relations)
                doc_epair_pooled_evidences.append(epair_pooled_evidences)
            
            batch_epair_tids.append(torch.stack(doc_epair_tids))
            batch_epair_types.append(torch.Tensor(doc_epair_types))
            batch_epair_masks.append(torch.stack(doc_epair_masks))
            batch_epair_relations.append(torch.stack(doc_epair_relations))
            batch_epair_pooled_evidences.append(torch.cat(doc_epair_pooled_evidences))
            if len(doc_epair_finegrained_evidences) != 0: batch_epair_finegrained_evidences.append(torch.cat(doc_epair_finegrained_evidences))
            batch_num_epairs_per_doc.append(doc_num_epairs_per_doc)

        batch_token_seqs = rnn.pad_sequence(batch_token_seqs, batch_first=True, padding_value=0).long().to(info.DEVICE_GPU)
        batch_token_masks = rnn.pad_sequence(batch_token_masks, batch_first=True, padding_value=0).float().to(info.DEVICE_GPU)
        batch_token_types = rnn.pad_sequence(batch_token_types, batch_first=True, padding_value=0).long().to(info.DEVICE_GPU)

        batch_sent_tids = torch.Tensor(batch_sent_tids).long().to(info.DEVICE_GPU)
        batch_num_sents_per_doc = torch.Tensor(batch_num_sents_per_doc).long()

        batch_mention_tids = torch.Tensor(batch_mention_tids).long().to(info.DEVICE_GPU)
        batch_mention_coreferences = torch.cat(batch_mention_coreferences).float().to(info.DEVICE_GPU)

        max_epair_tids = max([doc_epair_tids.shape[-1] for doc_epair_tids in batch_epair_tids])
        batch_epair_tids = torch.cat([F.pad(doc_epair_tids, (0, max_epair_tids-doc_epair_tids.shape[-1]), value=-1) for doc_epair_tids in batch_epair_tids]).long().to(info.DEVICE_GPU)
        batch_epair_types = torch.cat(batch_epair_types).long().to(info.DEVICE_GPU)
        batch_epair_masks = torch.cat([F.pad(doc_epair_masks, (0, batch_token_seqs.shape[-1]-doc_epair_masks.shape[-1]), value=0) for doc_epair_masks in batch_epair_masks]).float().to(info.DEVICE_GPU)
        batch_epair_relations = torch.cat(batch_epair_relations).float().to(info.DEVICE_GPU)
        batch_epair_pooled_evidences = torch.cat(batch_epair_pooled_evidences).float().to(info.DEVICE_GPU)
        batch_epair_finegrained_evidences = torch.cat(batch_epair_finegrained_evidences).float().to(info.DEVICE_GPU) if len(batch_epair_finegrained_evidences) != 0 else None
        batch_num_epairs_per_doc = torch.Tensor(batch_num_epairs_per_doc).long()

        batch_inputs = {'batch_token_seqs': batch_token_seqs,
                   'batch_token_masks': batch_token_masks,
                   'batch_token_types': batch_token_types,
                   'batch_sent_tids': batch_sent_tids,
                   'batch_num_sents_per_doc': batch_num_sents_per_doc,
                   'batch_mention_tids': batch_mention_tids,
                   'batch_mention_coreferences': batch_mention_coreferences,
                   'batch_num_mentions_per_doc': batch_num_mentions_per_doc,
                   'batch_epair_tids': batch_epair_tids,
                   'batch_epair_types': batch_epair_types,
                   'batch_epair_masks': batch_epair_masks,
                   'batch_epair_relations': batch_epair_relations,
                   'batch_epair_pooled_evidences': batch_epair_pooled_evidences,
                   'batch_epair_finegrained_evidences': batch_epair_finegrained_evidences,
                   'batch_num_epairs_per_doc': batch_num_epairs_per_doc}

        yield batch_inputs


def prepare_batch_test(info, inputs, batch_size, infer_round, preds=None):
    
    doc_titles = list(inputs.keys()) if preds is None else list(preds.keys())
    num_batch = math.ceil(len(doc_titles) / batch_size)
    
    for idx_batch in range(num_batch):
        
        batch_titles = doc_titles[idx_batch*batch_size:(idx_batch+1)*batch_size]
        batch_inputs = [inputs[doc_title] for doc_title in batch_titles]

        batch_token_seqs, batch_token_masks, batch_token_types = [], [], []
        batch_sent_tids, batch_num_sents_per_doc = [], []
        batch_epair_ids, batch_epair_tids, batch_epair_masks, batch_epair_relations, batch_num_epairs_per_doc = [], [], [], [], []
        
        for doc_title, doc_input in zip(batch_titles, batch_inputs):
            
            batch_token_seqs.append(doc_input.doc_tokens)
            batch_token_masks.append(torch.ones(doc_input.doc_tokens.shape[0]))
        
            doc_token_types = torch.zeros(doc_input.doc_tokens.shape[0])
            for sid in range(len(doc_input.sid2tids)):
                if infer_round == info.INFER_ROUND_FER: batch_sent_tids.append(doc_input.sid2tids[sid][0])
                doc_token_types[doc_input.sid2tids[sid][0] : doc_input.sid2tids[sid][1]] = sid % 2
            batch_token_types.append(doc_token_types)
            if infer_round == info.INFER_ROUND_FER: batch_num_sents_per_doc.append(len(doc_input.sid2tids))

            if infer_round == info.INFER_ROUND_FIRST:
                epair2rids = [[epair, list(doc_input.eids2rid2sids[epair].keys())] for epair in permutations(doc_input.eid2etype, 2)]
            elif infer_round == info.INFER_ROUND_FER:
                epair2rids = [[epair, list(rids.keys())] for epair, rids in preds[doc_title].items()]
            elif infer_round == info.INFER_ROUND_MASK:
                epair2rids = [[epair, [rid]] for epair, rids in preds[doc_title].items() for rid in rids]
            elif infer_round == info.INFER_ROUND_DOC:
                epair2rids = [[epair, list(rids.keys())] for epair, rids in doc_input.eids2rid2sids.items()]
                
            max_epair_tids = max([len(tids) for tids in doc_input.eid2tids.values()])
            doc_epair_ids, doc_epair_tids, doc_epair_masks, doc_epair_relations = [], [], [], []
            for epair, rids in epair2rids:
                
                doc_epair_ids.append(epair)
                
                epair_tids = torch.full((2, max_epair_tids), -1)
                epair_tids[0, :len(doc_input.eid2tids[epair[0]])] = torch.Tensor(list(doc_input.eid2tids[epair[0]])).long()
                epair_tids[1, :len(doc_input.eid2tids[epair[1]])] = torch.Tensor(list(doc_input.eid2tids[epair[1]])).long()
                doc_epair_tids.append(epair_tids)

                if infer_round == info.INFER_ROUND_MASK:
                    epair_masks = torch.zeros(doc_input.doc_tokens.shape[0])
                    for sid, tids in doc_input.sid2tids.items():
                        epair_masks[tids[0]:tids[1]] = preds[doc_title][epair][rids[0]][sid]
                else:
                    epair_masks = torch.ones(doc_input.doc_tokens.shape[0])
                doc_epair_masks.append(epair_masks)

                epair_relations = torch.zeros(info.NUM_REL)
                for rid in rids:
                    epair_relations[rid] = 1
                doc_epair_relations.append(epair_relations)

            batch_epair_ids.append(doc_epair_ids)
            batch_epair_tids.append(torch.stack(doc_epair_tids))
            batch_epair_masks.append(torch.stack(doc_epair_masks))
            batch_epair_relations.append(torch.stack(doc_epair_relations))
            batch_num_epairs_per_doc.append(len(epair2rids))
            
        if infer_round == info.INFER_ROUND_DOC:
            batch_titles = [doc_title[0] for doc_title in batch_titles]

        batch_token_seqs = rnn.pad_sequence(batch_token_seqs, batch_first=True, padding_value=0).long().to(info.DEVICE_GPU)
        batch_token_masks = rnn.pad_sequence(batch_token_masks, batch_first=True, padding_value=0).float().to(info.DEVICE_GPU)
        batch_token_types = rnn.pad_sequence(batch_token_types, batch_first=True, padding_value=0).long().to(info.DEVICE_GPU)
        
        if infer_round == info.INFER_ROUND_FER:
            batch_sent_tids = torch.Tensor(batch_sent_tids).long().to(info.DEVICE_GPU)
            batch_num_sents_per_doc = torch.Tensor(batch_num_sents_per_doc).long()
            
        max_epair_tids = max([doc_epair_tids.shape[-1] for doc_epair_tids in batch_epair_tids])
        batch_epair_tids = torch.cat([F.pad(doc_epair_tids, (0, max_epair_tids-doc_epair_tids.shape[-1]), value=-1) for doc_epair_tids in batch_epair_tids]).long().to(info.DEVICE_GPU)
        batch_epair_masks = torch.cat([F.pad(doc_epair_masks, (0, batch_token_seqs.shape[-1]-doc_epair_masks.shape[-1]), value=0) for doc_epair_masks in batch_epair_masks]).float().to(info.DEVICE_GPU)
        batch_epair_relations = torch.cat(batch_epair_relations).float().to(info.DEVICE_GPU)
        batch_num_epairs_per_doc = torch.Tensor(batch_num_epairs_per_doc).long()
            
        batch_inputs = {'batch_titles': batch_titles,
                   'batch_token_seqs': batch_token_seqs, 
                   'batch_token_masks': batch_token_masks,
                   'batch_token_types': batch_token_types,
                   'batch_sent_tids': batch_sent_tids,
                   'batch_num_sents_per_doc': batch_num_sents_per_doc,
                   'batch_epair_ids': batch_epair_ids,
                   'batch_epair_tids': batch_epair_tids,               
                   'batch_epair_masks': batch_epair_masks,
                   'batch_epair_relations': batch_epair_relations, 
                   'batch_epair_finegrained_evidences': None,
                   'batch_num_epairs_per_doc': batch_num_epairs_per_doc}

        yield batch_inputs
        
        
def feed_batch(info, batch_inputs, batch_preds, infer_round):
    
    batch_titles, batch_epair_ids = batch_inputs['batch_titles'], batch_inputs['batch_epair_ids']
    batch_num_epairs_per_doc, batch_num_sents_per_doc = batch_inputs['batch_num_epairs_per_doc'].tolist(), batch_inputs['batch_num_sents_per_doc']
    batch_epair_relations, batch_preds = batch_inputs['batch_epair_relations'].to(info.DEVICE_CPU), batch_preds.to(info.DEVICE_CPU)
    
    if infer_round != info.INFER_ROUND_FIRST:
        batch_epair_relations = torch.split(batch_epair_relations, batch_num_epairs_per_doc, dim=0)
        if infer_round == info.INFER_ROUND_FER: cum_num_sents = 0
        else: batch_preds = torch.split(batch_preds, batch_num_epairs_per_doc, dim=0)

    batch_triplets, batch_predictions = [], []
    for doc_idx, (doc_title, doc_epair_ids) in enumerate(zip(batch_titles, batch_epair_ids)):
        for epair_idx, epair_ids in enumerate(doc_epair_ids):
            if infer_round == info.INFER_ROUND_FIRST:
                batch_triplets.append((doc_title, *epair_ids))
            else:
                for rid in batch_epair_relations[doc_idx][epair_idx].nonzero().flatten().tolist():
                    batch_triplets.append((doc_title, *epair_ids, rid))
                    if infer_round == info.INFER_ROUND_FER:
                        doc_num_sents = batch_num_sents_per_doc[doc_idx]
                        batch_predictions.append(batch_preds[cum_num_sents:cum_num_sents+doc_num_sents])
                        cum_num_sents += doc_num_sents
                    else:
                        epair_preds = batch_preds[doc_idx][epair_idx]
                        batch_predictions.append(epair_preds[rid].item())
                
    if infer_round == info.INFER_ROUND_FIRST:
        return batch_triplets, batch_epair_relations, batch_preds
    
    return batch_triplets, batch_predictions