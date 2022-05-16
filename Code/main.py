import json
import math
import argparse
import dill as pk
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from info import Info
from model import Model
from prepare import DocInput
from util import set_seed, myprint, prepare_autos, prepare_optimizer_scheduler, register_hyperparameters_main, cal_f1, evaluate, prepare_batch_train, prepare_batch_test, feed_batch


def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--stage', type=str, default='Main')
    parser.add_argument('--dataset', type=str, default='DocRED')
    parser.add_argument('--seed', type=int, default=66)
    
    parser.add_argument('--transformer', type=str, default='bert-base-cased')
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--bilinear_block_size', type=int, default=64)
        
    parser.add_argument('--RE_max', type=int, default=4)
    parser.add_argument('--CR_focal_gamma', type=int, default=2)
    parser.add_argument('--PER_focal_gamma', type=int, default=2)
    parser.add_argument('--FER_threshold', type=float, default=0.5)
    
    parser.add_argument('--loss_weight_CR', type=float, default=0.1)
    parser.add_argument('--loss_weight_ET', type=float, default=0.1)
    parser.add_argument('--loss_weight_PER', type=float, default=0.1)
    parser.add_argument('--loss_weight_FER', type=float, default=0.1)
    
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--update_freq', type=int, default=1)
    
    parser.add_argument('--new_lr', type=float, default=1e-4)
    parser.add_argument('--pretrained_lr', type=float, default=5e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)    
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    args = parser.parse_args()
    
    return args


def train(args, info, idx_epoch, inputs_train, model, optimizer, scheduler):
    
    myprint(f'Start Training Epoch {idx_epoch}', info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)
    
    model.train()
    optimizer.zero_grad()
    batch_tasks = set([info.TASK_RE, info.TASK_CR, info.TASK_ET, info.TASK_PER, info.TASK_FER])

    num_batch = math.ceil(len(inputs_train) / args.batch_size)
    report_batch = num_batch // 5
    for idx_batch, batch_inputs in enumerate(prepare_batch_train(info, inputs_train, args.batch_size)):

        batch_loss, _, = model(batch_tasks, batch_inputs, to_evaluate=True, to_predict=False)
        (batch_loss / args.update_freq).backward()

        if (idx_batch+1) % args.update_freq == 0 or idx_batch+1 == num_batch:
            clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if idx_batch % report_batch == 0:
            myprint(f'Finish Training Epoch {idx_epoch} | Report Batch {idx_batch//report_batch} | Loss {batch_loss.item():.4f}', info.FILE_STDOUT)

    myprint('-'*20, info.FILE_STDOUT)
    
    
def test(args, info, mode, inputs, tokenizer, model, if_final=False, rej_rate=0, all_shifts=None):
    
    myprint(f'Start Testing {mode}', info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)

    batch_tasks, batch_size, infer_round = set([info.TASK_RE]), args.batch_size, info.INFER_ROUND_FIRST
    myprint(f'Start {infer_round} Round of Inference', info.FILE_STDOUT)
    model.eval()
    with torch.no_grad():
        
        all_triplets, all_relations, all_first_predictions = [], [], []
        for idx_batch, batch_inputs in enumerate(prepare_batch_test(info, inputs, batch_size, infer_round)):
            
            _, batch_preds = model(batch_tasks, batch_inputs, to_evaluate=False, to_predict=True)
            batch_triplets, batch_relations, batch_first_predictions = feed_batch(info, batch_inputs, batch_preds, infer_round)
            all_triplets += batch_triplets; all_relations.append(batch_relations); all_first_predictions.append(batch_first_predictions)
            
    all_relations, all_first_predictions = torch.cat(all_relations).bool(), torch.cat(all_first_predictions)
    
    if not if_final:
        all_first_P, all_first_R, all_first_F1 = cal_f1(all_relations, model.loss_module.pred_RE_results(all_first_predictions))
        
        all_first_F1_threshold = 0.62
        if all_first_F1 <= all_first_F1_threshold:
            myprint(f'All {info.INFER_ROUND_FIRST} Precision: {all_first_P:.4f} | Recall: {all_first_R:.4f} | F1: {all_first_F1:.4f}', info.FILE_STDOUT)
            return all_first_F1

    myprint(f'Identify Rejection Triplets', info.FILE_STDOUT)
    all_logits = all_first_predictions[:, info.ID_REL_THRE+1:].flatten()
    all_sorted_indices = all_logits.abs().argsort()
    all_size = all_sorted_indices.shape[0]
    
    if mode == info.MODE_DEV:
        all_results = all_relations[:, info.ID_REL_THRE+1:].flatten() != all_logits.gt(0)
        all_risks = (all_results.sum() - all_results[all_sorted_indices].cumsum(0)) / all_size
        all_rejects = torch.arange(all_size) / all_size        
        minmax_normalize = lambda x: (x-x.min())/x.max()
        rej_index = (minmax_normalize(all_risks).square() + minmax_normalize(all_rejects).square()).argmin()
        rej_rate = rej_index / all_size
        
    rej_threshold = all_logits[all_sorted_indices[int(rej_rate*all_size)]].abs().item()
    rej_triplet_masks = all_first_predictions.abs().lt(rej_threshold)
    rej_triplet_masks[:, info.ID_REL_THRE] = False
    
    rej_max = 10
    if rej_max > 0:
        rej_lowest_confs, _ = torch.topk(-all_first_predictions.abs(), rej_max+1, dim=1)
        rej_lowest_confs = -rej_lowest_confs[:, -1].unsqueeze(1)
        rej_triplet_masks = rej_triplet_masks & all_first_predictions.abs().le(rej_lowest_confs)
        
    rej_triplets_eids2rid2sids = defaultdict(lambda: defaultdict(dict))
    for (doc_title, eid_i, eid_j), mask in zip(all_triplets, rej_triplet_masks):
        for rid in mask.nonzero().flatten().tolist():
            rej_triplets_eids2rid2sids[doc_title][(eid_i, eid_j)][rid] = []

    batch_tasks, batch_size, infer_round = set([info.TASK_FER]), args.batch_size, info.INFER_ROUND_FER
    myprint(f'Start {infer_round} Round of Inference', info.FILE_STDOUT)
    model.eval()
    with torch.no_grad():
        
        rej_FER_triplets, rej_FER_predictions = [], []
        for idx_batch, batch_inputs in enumerate(prepare_batch_test(info, inputs, batch_size, infer_round, preds=rej_triplets_eids2rid2sids)):
            
            _, batch_preds = model(batch_tasks, batch_inputs, to_evaluate=False, to_predict=True)
            batch_FER_triplets, batch_FER_predictions = feed_batch(info, batch_inputs, batch_preds, infer_round)
            rej_FER_triplets += batch_FER_triplets; rej_FER_predictions += batch_FER_predictions

    rej_triplets_FER2eids2rids = defaultdict(lambda: defaultdict(list))
    for (doc_title, eid_i, eid_j, rid), FER_prediction in zip(rej_FER_triplets, rej_FER_predictions):
        rej_triplets_eids2rid2sids[doc_title][(eid_i, eid_j)][rid] = FER_prediction
        FER_doc = FER_prediction.gt(args.FER_threshold).nonzero().flatten().tolist()
        if len(FER_doc) > 0:
            rej_triplets_FER2eids2rids[(doc_title, *FER_doc)][(eid_i, eid_j)].append(rid)
        
    batch_tasks, batch_size, infer_round = set([info.TASK_RE]), 2, info.INFER_ROUND_MASK
    myprint(f'Start {infer_round} Round of Inference', info.FILE_STDOUT)
    model.eval()
    with torch.no_grad():
        
        rej_mask_triplets, rej_mask_predictions = [], []
        for idx_batch, batch_inputs in enumerate(prepare_batch_test(info, inputs, batch_size, infer_round, preds=rej_triplets_eids2rid2sids)):
            
            _, batch_preds = model(batch_tasks, batch_inputs, to_evaluate=False, to_predict=True)
            batch_mask_triplets, batch_mask_predictions = feed_batch(info, batch_inputs, batch_preds, infer_round)
            rej_mask_triplets += batch_mask_triplets; rej_mask_predictions += batch_mask_predictions
            
    corpus = json.load(open(info.FILE_CORPUSES[mode], 'r'))
    corpus_title2idx = {doc['title']:did for did, doc in enumerate(corpus)}
    myprint(f'Construct FER Docs for Rejection Triplets', info.FILE_STDOUT)
    rej_triplets_FER_docs = {}    
    for FER_title, eids2rids in rej_triplets_FER2eids2rids.items():
        
        doc_title, FER = FER_title[0], set(FER_title[1:])
        eids = set([eid for epair in eids2rids for eid in epair])
        doc = corpus[corpus_title2idx[doc_title]]

        wid2eids, start_wids, end_wids = defaultdict(set), set(), set()
        for eid, entity in enumerate(doc['vertexSet']):
            for mid, mention in enumerate(entity):
                wid2eids[(mention['sent_id'], mention['pos'][0])].add(eid)
                start_wids.add((mention['sent_id'], mention['pos'][0]))
                end_wids.add((mention['sent_id'], mention['pos'][1]-1))
        
        sid2tids, eid2tids, doc_tokens, sid_ = {}, defaultdict(set), [], 0
        for sid, sent in enumerate(doc['sents']):
            if sid not in FER: continue
            sent_tokens = []
            for wid, word in enumerate(sent):
                tokens = tokenizer.tokenize(word)
                if (sid, wid) in start_wids:
                    tokens = [info.MARKER_ENTITY] + tokens
                    for eid in wid2eids[(sid, wid)]:
                        if eid in eids: eid2tids[eid].add(len(doc_tokens) + len(sent_tokens) + 1)
                if (sid, wid) in end_wids:
                    tokens = tokens + [info.MARKER_ENTITY]
                sent_tokens.extend(tokens)
            sent_tokens = tokenizer.convert_tokens_to_ids(sent_tokens)
            sent_tokens = tokenizer.build_inputs_with_special_tokens(sent_tokens)
            sid2tids[sid_] = (len(doc_tokens), len(doc_tokens)+len(sent_tokens)); sid_ += 1
            doc_tokens.extend(sent_tokens)
        doc_tokens = torch.Tensor(doc_tokens).int()
        
        eids2rid2sids = defaultdict(dict)
        for epair, rids in eids2rids.items():
            if epair[0] in eid2tids and epair[1] in eid2tids:
                for rid in rids:
                    eids2rid2sids[epair][rid] = []
       
        if len(eids2rid2sids) != 0: 
            rej_triplets_FER_docs[FER_title] = DocInput(FER_title, doc_tokens, sid2tids, eid2tids, None, eids2rid2sids)
            
    batch_tasks, batch_size, infer_round = set([info.TASK_RE]), 16, info.INFER_ROUND_DOC
    myprint(f'Start {infer_round} Round of Inference', info.FILE_STDOUT)
    model.eval()
    with torch.no_grad():
        
        rej_doc_triplets, rej_doc_predictions = [], []
        for idx_batch, batch_inputs in enumerate(prepare_batch_test(info, rej_triplets_FER_docs, batch_size, infer_round)):
            
            _, batch_preds = model(batch_tasks, batch_inputs, to_evaluate=False, to_predict=True)
            batch_doc_triplets, batch_doc_predictions = feed_batch(info, batch_inputs, batch_preds, infer_round)
            rej_doc_triplets += batch_doc_triplets; rej_doc_predictions += batch_doc_predictions
    
    myprint(f'Blend All Rounds of Inference', info.FILE_STDOUT)
    rej_mask_triplet2idx = {triplet:idx for idx, triplet in enumerate(rej_mask_triplets)}
    rej_doc_triplet2idx = {triplet:idx for idx, triplet in enumerate(rej_doc_triplets)}
    all_predictions = all_first_predictions.expand(4,-1,-1).clone()
    for idx, ((doc_title, eid_i, eid_j), mask) in enumerate(zip(all_triplets, rej_triplet_masks)):
        for rid in mask.nonzero().flatten().tolist():
            triplet = (doc_title, eid_i, eid_j, rid)
            all_predictions[1, idx, rid] = rej_mask_predictions[rej_mask_triplet2idx[triplet]]
            all_predictions[2, idx, rid] = rej_doc_predictions[rej_doc_triplet2idx[triplet]] if triplet in rej_doc_triplet2idx else 0
    
    blend = lambda shifts, preds, masks: ((preds*masks).sum(0) - shifts.expand(preds.shape[1],-1)*masks)[masks]
    def eval():
        loss = F.binary_cross_entropy_with_logits(blend(all_shifts, all_predictions, rej_triplet_masks), all_relations[rej_triplet_masks])
        optimizer.zero_grad()
        loss.backward()
        return loss
    
    all_relations, all_predictions, rej_triplet_masks = all_relations.float().to(info.DEVICE_GPU), all_predictions.to(info.DEVICE_GPU), rej_triplet_masks.to(info.DEVICE_GPU)
    if all_shifts is None:
        all_shifts = nn.Parameter(torch.zeros(info.NUM_REL).to(info.DEVICE_GPU))
        optimizer = optim.LBFGS([all_shifts], lr=0.001, max_iter=100)
        optimizer.step(eval)
        all_shifts = all_shifts.detach()
    all_predictions[3][rej_triplet_masks] = blend(all_shifts, all_predictions, rej_triplet_masks)
    
    if not if_final:
        
        all_relations = all_relations.bool()
        myprint('-'*20, info.FILE_STDOUT)
        
        rej_first_P, rej_first_R, rej_first_F1 = cal_f1(all_relations[rej_triplet_masks], all_predictions[0][rej_triplet_masks].gt(0))
        rej_mask_P, rej_mask_R, rej_mask_F1 = cal_f1(all_relations[rej_triplet_masks], all_predictions[1][rej_triplet_masks].gt(0))
        rej_doc_P, rej_doc_R, rej_doc_F1 = cal_f1(all_relations[rej_triplet_masks], all_predictions[2][rej_triplet_masks].gt(0))
        rej_blend_P, rej_blend_R, rej_blend_F1 = cal_f1(all_relations[rej_triplet_masks], all_predictions[3][rej_triplet_masks].gt(0))
        myprint(f'Rej {info.INFER_ROUND_FIRST} Precision: {rej_first_P:.4f} | Recall: {rej_first_R:.4f} | F1: {rej_first_F1:.4f}', info.FILE_STDOUT)
        myprint(f'Rej {info.INFER_ROUND_MASK} Precision: {rej_mask_P:.4f} | Recall: {rej_mask_R:.4f} | F1: {rej_mask_F1:.4f}', info.FILE_STDOUT)
        myprint(f'Rej {info.INFER_ROUND_DOC} Precision: {rej_doc_P:.4f} | Recall: {rej_doc_R:.4f} | F1: {rej_doc_F1:.4f}', info.FILE_STDOUT)
        myprint(f'Rej {info.INFER_ROUND_BLEND} Precision: {rej_blend_P:.4f} | Recall: {rej_blend_R:.4f} | F1: {rej_blend_F1:.4f}', info.FILE_STDOUT)
        
        all_mask_P, all_mask_R, all_mask_F1 = cal_f1(all_relations, model.loss_module.pred_RE_results(all_predictions[1]))
        all_doc_P, all_doc_R, all_doc_F1 = cal_f1(all_relations, model.loss_module.pred_RE_results(all_predictions[2]))
        all_blend_P, all_blend_R, all_blend_F1 = cal_f1(all_relations, model.loss_module.pred_RE_results(all_predictions[3]))
        myprint(f'All {info.INFER_ROUND_FIRST} Precision: {all_first_P:.4f} | Recall: {all_first_R:.4f} | F1: {all_first_F1:.4f}', info.FILE_STDOUT)
        myprint(f'All {info.INFER_ROUND_MASK} Precision: {all_mask_P:.4f} | Recall: {all_mask_R:.4f} | F1: {all_mask_F1:.4f}', info.FILE_STDOUT)
        myprint(f'All {info.INFER_ROUND_DOC} Precision: {all_doc_P:.4f} | Recall: {all_doc_R:.4f} | F1: {all_doc_F1:.4f}', info.FILE_STDOUT)
        myprint(f'All {info.INFER_ROUND_BLEND} Precision: {all_blend_P:.4f} | Recall: {all_blend_R:.4f} | F1: {all_blend_F1:.4f}', info.FILE_STDOUT)
        
        return all_blend_F1
        
    else:
        
        all_blend_predictions = model.loss_module.pred_RE_results(all_predictions[3])
        all_triplets_eids2rid2sids = defaultdict(lambda: defaultdict(dict))
        for (doc_title, eid_i, eid_j), pred in zip(all_triplets, all_blend_predictions):
            for rid in pred.nonzero().flatten().tolist():
                all_triplets_eids2rid2sids[doc_title][(eid_i, eid_j)][rid] = []

        batch_tasks, batch_size, infer_round = set([info.TASK_FER]), args.batch_size, info.INFER_ROUND_FER
        myprint(f'Start {infer_round} Round of Inference', info.FILE_STDOUT)
        model.eval()
        with torch.no_grad():

            all_FER_triplets, all_FER_predictions = [], []
            for idx_batch, batch_inputs in enumerate(prepare_batch_test(info, inputs, batch_size, infer_round, preds=all_triplets_eids2rid2sids)):

                _, batch_preds = model(batch_tasks, batch_inputs, to_evaluate=False, to_predict=True)
                batch_FER_triplets, batch_FER_predictions = feed_batch(info, batch_inputs, batch_preds, infer_round)
                all_FER_triplets += batch_FER_triplets; all_FER_predictions += batch_FER_predictions

        myprint('-'*20, info.FILE_STDOUT)
        evaluate(args, info, mode, all_FER_triplets, all_FER_predictions)
        if mode == info.MODE_DEV: return rej_rate, all_shifts


def main():
    
    args = parse_args()
    info = Info(args)
    
    myprint('='*20, info.FILE_STDOUT)
    myprint(f'Start {args.stage} {args.dataset}', info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)
    
    register_hyperparameters_main(args, info)
    
    myprint('-'*20, info.FILE_STDOUT)
    myprint('Initialize Relevant Objects', info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)
    
    set_seed(args.seed)
    rtype_embeddings = torch.from_numpy(np.load(info.DATA_REL2VEC)).float()
    inputs_train, inputs_dev, inputs_test = map(lambda x: pk.load(open(info.FILE_INPUTS[x], 'rb')), [info.MODE_TRAIN, info.MODE_DEV, info.MODE_TEST])

    _, tokenizer, transformer = prepare_autos(args, info)
    model = Model(args, info, transformer, rtype_embeddings).to(info.DEVICE_GPU)
    optimizer, scheduler = prepare_optimizer_scheduler(args, len(inputs_train), model)
    
    best_f1, best_epoch = 0, 0
    for idx_epoch in range(args.num_epoch):
        
        train(args, info, idx_epoch, inputs_train, model, optimizer, scheduler)
        epoch_f1 = test(args, info, info.MODE_DEV, inputs_dev, tokenizer, model, if_final=False, rej_rate=0, all_shifts=None)
        
        if epoch_f1 >= best_f1:
            best_f1, best_epoch = epoch_f1, idx_epoch
            torch.save(model.state_dict(), f'{info.FILE_MODEL}')
            myprint(f'This is the Best Performing Epoch by far - Epoch {idx_epoch} F1 {epoch_f1:.4f}', info.FILE_STDOUT)
        else:
            myprint(f'Not the Best Performing Epoch by far - Epoch {idx_epoch} F1 {epoch_f1:.4f} vs Best F1 {best_f1:.4f}', info.FILE_STDOUT)
        myprint('-'*20, info.FILE_STDOUT)
        
    model.load_state_dict(torch.load(f'{info.FILE_MODEL}', map_location=info.DEVICE_GPU))
    myprint(f'Load the Model from Epoch {best_epoch} with {info.MODE_DEV} F1 {best_f1:.4f}', info.FILE_STDOUT)
    myprint('-'*20, info.FILE_STDOUT)
    
    rej_rate, all_shifts = test(args, info, info.MODE_DEV, inputs_dev, tokenizer, model, if_final=True, rej_rate=0, all_shifts=None)
    test(args, info, info.MODE_TEST, inputs_test, tokenizer, model, if_final=True, rej_rate=rej_rate, all_shifts=all_shifts)
    
    myprint(f'Finish {args.stage} {args.dataset}', info.FILE_STDOUT)
    myprint('='*20, info.FILE_STDOUT)
    
    
if __name__=='__main__':
    main()