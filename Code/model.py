import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn

from opt_einsum import contract


class Transformer(nn.Module):
    
    def __init__(self, info, transformer):
        super(Transformer, self).__init__()
        
        self.info = info
        self.transformer = transformer
        self.max_num_tokens = 512
        
        self.start_token_len, self.end_token_len = 1, 1
        self.start_token_ids = torch.Tensor([transformer.config.cls_token_id]).to(info.DEVICE_GPU)
        self.end_token_ids = torch.Tensor([transformer.config.sep_token_id]).to(info.DEVICE_GPU)
        
        
    def forward(self, batch_token_seqs, batch_token_masks, batch_token_types):
        
        if 'roberta' in self.transformer.config._name_or_path: batch_token_types = torch.zeros_like(batch_token_types)

        batch_size, batch_num_tokens = batch_token_seqs.size()
        
        if batch_num_tokens <= self.max_num_tokens:
            batch_output = self.transformer(input_ids=batch_token_seqs, attention_mask=batch_token_masks, token_type_ids=batch_token_types, output_attentions=True)
            batch_token_embs, batch_token_atts = batch_output[0], batch_output[-1][-1]
            
        else:
            new_token_seqs, new_token_masks, new_token_types, new_token_segs = [], [], [], []
            real_num_tokens = batch_token_masks.sum(1).int().tolist()
            for doc_id, real_num_token in enumerate(real_num_tokens):
                if real_num_token <= self.max_num_tokens:
                    new_token_seqs.append(batch_token_seqs[doc_id, :self.max_num_tokens])
                    new_token_masks.append(batch_token_masks[doc_id, :self.max_num_tokens])
                    new_token_types.append(batch_token_types[doc_id, :self.max_num_tokens])
                    new_token_segs.append(1)
                else:
                    new_token_seq1 = torch.cat([batch_token_seqs[doc_id, :self.max_num_tokens - self.end_token_len], self.end_token_ids], dim=-1)
                    new_token_mask1 = batch_token_masks[doc_id, :self.max_num_tokens]
                    new_token_type1 = torch.cat([batch_token_types[doc_id, :self.max_num_tokens - self.end_token_len], batch_token_types[doc_id, self.max_num_tokens - self.end_token_len - 1].repeat(self.end_token_len)], dim=-1)
                    
                    new_token_seq2 = torch.cat([self.start_token_ids, batch_token_seqs[doc_id, real_num_token - self.max_num_tokens + self.start_token_len : real_num_token]], dim=-1)                    
                    new_token_mask2 = batch_token_masks[doc_id, real_num_token - self.max_num_tokens : real_num_token]
                    new_token_type2 = torch.cat([batch_token_types[doc_id, real_num_token - self.max_num_tokens + self.start_token_len].repeat(self.start_token_len), batch_token_types[doc_id, real_num_token - self.max_num_tokens + self.start_token_len : real_num_token]], dim=-1)
                    
                    new_token_seqs.extend([new_token_seq1, new_token_seq2])
                    new_token_masks.extend([new_token_mask1, new_token_mask2])
                    new_token_types.extend([new_token_type1, new_token_type2])
                    new_token_segs.append(2)
                    
            batch_token_seqs, batch_token_masks, batch_token_types = torch.stack(new_token_seqs, dim=0).long(), torch.stack(new_token_masks, dim=0).float(), torch.stack(new_token_types, dim=0).long()
            batch_output = self.transformer(input_ids=batch_token_seqs, attention_mask=batch_token_masks, token_type_ids=batch_token_types, output_attentions=True)
            batch_token_embs, batch_token_atts = batch_output[0], batch_output[-1][-1]
            
            seg_id, new_token_embs, new_token_atts = 0, [], []
            for (new_token_seq, real_num_token) in zip(new_token_segs, real_num_tokens):
                if new_token_seq == 1:
                    new_token_emb = F.pad(batch_token_embs[seg_id], (0, 0, 0, batch_num_tokens - self.max_num_tokens))
                    new_token_att = F.pad(batch_token_atts[seg_id], (0, batch_num_tokens - self.max_num_tokens, 0, batch_num_tokens - self.max_num_tokens))
                    new_token_embs.append(new_token_emb)
                    new_token_atts.append(new_token_att)
                    
                elif new_token_seq == 2:
                    valid_num_token1 = self.max_num_tokens - self.end_token_len
                    new_token_emb1 = F.pad(batch_token_embs[seg_id][:valid_num_token1], (0, 0, 0, batch_num_tokens - valid_num_token1))
                    new_token_mask1 = F.pad(batch_token_masks[seg_id][:valid_num_token1], (0, batch_num_tokens - valid_num_token1))
                    new_token_att1 = F.pad(batch_token_atts[seg_id][:, :valid_num_token1, :valid_num_token1], (0, batch_num_tokens - valid_num_token1, 0, batch_num_tokens - valid_num_token1))
                    
                    valid_num_token2 = real_num_token - self.max_num_tokens
                    new_token_emb2 = F.pad(batch_token_embs[seg_id+1][self.start_token_len:], (0, 0, valid_num_token2 + self.start_token_len, batch_num_tokens - real_num_token))
                    new_token_mask2 = F.pad(batch_token_masks[seg_id+1][self.start_token_len:], (valid_num_token2 + self.start_token_len, batch_num_tokens - real_num_token))
                    new_token_att2 = F.pad(batch_token_atts[seg_id+1][:, self.start_token_len:, self.start_token_len:], (valid_num_token2 + self.start_token_len, batch_num_tokens - real_num_token, valid_num_token2 + self.start_token_len, batch_num_tokens - real_num_token))
                    
                    new_token_mask = new_token_mask1 + new_token_mask2 + self.info.EXTREME_SMALL_POSI
                    new_token_emb = (new_token_emb1 + new_token_emb2) / new_token_mask.unsqueeze(-1)
                    new_token_att = (new_token_att1 + new_token_att2)
                    new_token_att /= (new_token_att.sum(-1, keepdim=True) + self.info.EXTREME_SMALL_POSI)
                    new_token_embs.append(new_token_emb)
                    new_token_atts.append(new_token_att)
                    
                seg_id += new_token_seq
            batch_token_embs, batch_token_atts = torch.stack(new_token_embs, dim=0), torch.stack(new_token_atts, dim=0)
            
        return batch_token_embs, batch_token_atts
    
    
class Loss:
    
    def __init__(self, args, info):
        
        self.args = args
        self.info = info
        
        
    def cal_RE_loss(self, batch_RE_reps, batch_epair_relations):
        
        batch_epair_thresholds = torch.zeros_like(batch_epair_relations).float().to(self.info.DEVICE_GPU)
        batch_epair_thresholds[:, self.info.ID_REL_THRE] = 1
        batch_epair_relations[:, self.info.ID_REL_THRE] = 0
        
        batch_posi_masks = batch_epair_relations + batch_epair_thresholds        
        batch_posi_reps = batch_RE_reps + (1 - batch_posi_masks) * self.info.EXTREME_SMALL_NEGA
        batch_posi_loss = - (F.log_softmax(batch_posi_reps, dim=-1) * batch_epair_relations).sum(1)
        
        batch_nega_masks = 1 - batch_epair_relations
        batch_nega_reps = batch_RE_reps + (1 - batch_nega_masks) * self.info.EXTREME_SMALL_NEGA
        batch_nega_loss = - (F.log_softmax(batch_nega_reps, dim=-1) * batch_epair_thresholds).sum(1)
        
        batch_RE_loss = (batch_posi_loss + batch_nega_loss).mean()

        return batch_RE_loss
    
    
    def cal_CR_loss(self, batch_CR_reps, batch_mention_coreferences):
        
        batch_weights = batch_mention_coreferences.long().bincount()
        batch_weights = batch_weights.sum() / (2 * batch_weights)
        batch_weights = batch_weights.gather(0, batch_mention_coreferences.long())
        
        batch_CR_loss = F.binary_cross_entropy_with_logits(batch_CR_reps, batch_mention_coreferences, reduction='none')
        batch_CR_focal = (1 - torch.exp(-batch_CR_loss)) ** self.args.CR_focal_gamma
        batch_CR_loss = (batch_weights * batch_CR_focal * batch_CR_loss).mean()

        return batch_CR_loss
        
    
    def cal_ET_loss(self, batch_ET_reps, batch_epair_types):
        
        batch_epair_types = batch_epair_types.T.flatten()
        batch_ET_loss = F.cross_entropy(batch_ET_reps, batch_epair_types)
        
        return batch_ET_loss
    
    
    def cal_PER_loss(self, batch_PER_reps, batch_epair_pooled_evidences):
        
        batch_weights = torch.zeros(2).float().to(self.info.DEVICE_GPU)
        batch_weights[0] = (batch_epair_pooled_evidences==0).sum()
        batch_weights[1] = batch_epair_pooled_evidences.sum()
        batch_weights = batch_weights.sum() / (2 * batch_weights)
        batch_weights = batch_weights.gather(0, batch_epair_pooled_evidences.clamp(max=1).long())
        batch_weights *= batch_epair_pooled_evidences.clamp(min=1)
        
        batch_PER_loss = F.binary_cross_entropy_with_logits(batch_PER_reps, batch_epair_pooled_evidences.clamp(max=1), reduction='none')
        batch_PER_focal = (1 - torch.exp(-batch_PER_loss)) ** self.args.PER_focal_gamma
        batch_PER_loss = (batch_weights * batch_PER_focal * batch_PER_loss).mean()
                
        return batch_PER_loss
    
    
    def cal_FER_loss(self, batch_FER_reps, batch_epair_finegrained_evidences):
        
        if batch_epair_finegrained_evidences is None: batch_FER_loss = 0
        else: batch_FER_loss = F.binary_cross_entropy_with_logits(batch_FER_reps, batch_epair_finegrained_evidences)

        return batch_FER_loss
    
    
    def cal_RE_results(self, batch_RE_reps):
        
        batch_pred_relations = batch_RE_reps - batch_RE_reps[:, self.info.ID_REL_THRE].unsqueeze(1)
        
        return batch_pred_relations
    
    
    def cal_FER_results(self, batch_FER_reps):
        
        batch_pred_evidences = torch.sigmoid(batch_FER_reps)
        
        return batch_pred_evidences
    
    
    def pred_RE_results(self, batch_RE_reps):
        
        batch_pred_relations = batch_RE_reps.gt(0)
        
        if self.args.RE_max > 0:
            batch_top_labels, _ = torch.topk(batch_RE_reps, self.args.RE_max, dim=1)
            batch_top_labels = batch_top_labels[:, -1].unsqueeze(1)
            batch_pred_relations = batch_pred_relations & batch_RE_reps.ge(batch_top_labels)
    
        return batch_pred_relations
    
    
    def pred_FER_results(self, batch_FER_reps):
        
        batch_pred_evidences = batch_pred_evidences.gt(self.args.FER_threshold)
        
        return batch_pred_evidences
    
    
    
class Model(nn.Module):
    
    def __init__(self, args, info, transformer, rtype_embeddings):
        super(Model, self).__init__()
        
        self.args = args
        self.info = info

        self.rtype_emb_module = nn.Embedding.from_pretrained(rtype_embeddings, freeze=False)
        self.transformer_module = Transformer(info, transformer)
        self.loss_module = Loss(args, info)
        
        self.head_extractor_module = nn.Linear(transformer.config.hidden_size, args.hidden_size)
        self.tail_extractor_module = nn.Linear(transformer.config.hidden_size, args.hidden_size)
        self.head_context_extractor_module = nn.Linear(transformer.config.hidden_size, args.hidden_size)
        self.tail_context_extractor_module = nn.Linear(transformer.config.hidden_size, args.hidden_size)
        self.triplet_extractor_module = nn.Linear(4*transformer.config.hidden_size, transformer.config.hidden_size)
        
        self.RE_predictor_module = nn.Linear(args.hidden_size*args.bilinear_block_size, info.NUM_REL)
        self.CR_predictor_module = nn.Linear(transformer.config.hidden_size*args.bilinear_block_size, 1)
        self.ET_predictor_module = nn.Linear(args.hidden_size, info.NUM_NER)
        self.PER_predictor_module = nn.Linear(transformer.config.hidden_size*args.bilinear_block_size, 1)
        self.FER_predictor_module = nn.Linear(transformer.config.hidden_size*args.bilinear_block_size, 1)
        
        
    def get_epair_infos(self, batch_token_embs, batch_token_atts, batch_epair_tids, batch_epair_masks, batch_num_epairs_per_doc):
        
        batch_epair_dids = torch.arange(batch_num_epairs_per_doc.shape[0]).repeat_interleave(batch_num_epairs_per_doc).unsqueeze(-1).unsqueeze(-1)
        batch_token_embs = F.pad(batch_token_embs, (0,0,0,1), value=self.info.EXTREME_SMALL_NEGA)
        batch_epair_embs = batch_token_embs[batch_epair_dids, batch_epair_tids].logsumexp(-2)
        batch_epair_reps = [self.head_extractor_module(batch_epair_embs[:,0,:]), self.tail_extractor_module(batch_epair_embs[:,1,:])]
        
        batch_token_atts = F.pad(batch_token_atts.mean(1), (0,0,0,1))
        batch_epair_atts = batch_token_atts[batch_epair_dids, batch_epair_tids].sum(-2)
        batch_epair_atts /= (batch_epair_tids!=-1).sum(-1, keepdim=True).clamp(min=1).to(self.info.DEVICE_GPU)
        batch_epair_atts = batch_epair_atts.prod(-2) * batch_epair_masks
        batch_epair_atts /= batch_epair_atts.sum(-1, keepdim=True).clamp(min=self.info.EXTREME_SMALL_POSI)
        batch_epair_atts = torch.split(batch_epair_atts, batch_num_epairs_per_doc.tolist(), dim=0)
        batch_epair_contexts = torch.cat([doc_epair_atts @ batch_token_embs[did,:-1,:] for did, doc_epair_atts in enumerate(batch_epair_atts)])
                
        return batch_epair_embs, batch_epair_reps, batch_epair_contexts
    
    
    def get_sent_infos(self, batch_token_embs, batch_sent_tids, batch_num_sents_per_doc):
        
        batch_sent_dids = torch.arange(batch_num_sents_per_doc.shape[0]).repeat_interleave(batch_num_sents_per_doc)
        batch_sent_embs = batch_token_embs[batch_sent_dids, batch_sent_tids]        
        batch_sent_embs = batch_sent_embs.view(-1, batch_sent_embs.shape[-1] // self.args.bilinear_block_size, self.args.bilinear_block_size)
        batch_sent_embs = torch.split(batch_sent_embs, batch_num_sents_per_doc.tolist(), dim=0) # doc: num_sent * num_block * size_block
        
        return batch_sent_embs
    
    
    def get_RE_reps(self, batch_epair_reps, batch_epair_contexts):
        
        batch_head_reps = torch.tanh(batch_epair_reps[0] + self.head_context_extractor_module(batch_epair_contexts))
        batch_tail_reps = torch.tanh(batch_epair_reps[1] + self.tail_context_extractor_module(batch_epair_contexts))
        batch_head_reps = batch_head_reps.view(-1, self.args.hidden_size // self.args.bilinear_block_size, self.args.bilinear_block_size)
        batch_tail_reps = batch_tail_reps.view(-1, self.args.hidden_size // self.args.bilinear_block_size, self.args.bilinear_block_size)
        batch_RE_reps = (batch_head_reps.unsqueeze(3) * batch_tail_reps.unsqueeze(2)).view(-1, self.args.hidden_size * self.args.bilinear_block_size)
        batch_RE_reps = self.RE_predictor_module(batch_RE_reps)
        
        return batch_RE_reps
    
    
    def get_CR_reps(self, batch_token_embs, batch_mention_tids, batch_num_mentions_per_doc):
        
        batch_mention_tids = torch.split(batch_mention_tids, batch_num_mentions_per_doc, dim=0)
        
        batch_CR_reps = []
        for doc_token_embs, doc_mention_tids in zip(batch_token_embs, batch_mention_tids):
            
            doc_mention_embs = doc_token_embs[doc_mention_tids]
            doc_mention_embs = doc_mention_embs.view(-1, doc_mention_embs.shape[-1] // self.args.bilinear_block_size, self.args.bilinear_block_size) # num_mention * num_block * size_block
            doc_mention_reps = contract('mbi,nbj->mnbij', doc_mention_embs, doc_mention_embs).flatten(2,4).flatten(0,1) # num_mention * num_mention * num_block * size_block * size_block
            batch_CR_reps.append(doc_mention_reps)
            
        batch_CR_reps = torch.cat(batch_CR_reps)
        batch_CR_reps = self.CR_predictor_module(batch_CR_reps).flatten()
        
        return batch_CR_reps
    
    
    def get_ET_reps(self, batch_epair_reps):
        
        batch_ET_reps = torch.tanh(torch.cat(batch_epair_reps, dim=0))
        batch_ET_reps = self.ET_predictor_module(batch_ET_reps)
        
        return batch_ET_reps
    
    
    def get_PER_reps(self, batch_epair_contexts, batch_num_epairs_per_doc, batch_sent_embs):

        batch_epair_contexts = batch_epair_contexts.view(-1, batch_epair_contexts.shape[-1] // self.args.bilinear_block_size, self.args.bilinear_block_size)
        batch_epair_contexts = torch.split(batch_epair_contexts, batch_num_epairs_per_doc.tolist(), dim=0) # doc: num_epair * num_block * size_block
        batch_PER_reps = [contract('ebi,sbj->esbij', doc_epair_contexts, doc_sent_embs) for doc_epair_contexts, doc_sent_embs in zip(batch_epair_contexts, batch_sent_embs)] # doc: num_epair * num_sent * num_block * size_block * size_block
        batch_PER_reps = torch.cat([doc_PER_reps.flatten(2,4).flatten(0,1) for doc_PER_reps in batch_PER_reps])
        batch_PER_reps = self.PER_predictor_module(batch_PER_reps).flatten()
        
        return batch_PER_reps
    
    
    def get_FER_reps(self, batch_epair_embs, batch_epair_contexts, batch_epair_relations, batch_num_epairs_per_doc, batch_sent_embs):
        
        batch_epair_embs, batch_epair_contexts, batch_epair_relations = map(lambda x: torch.split(x, batch_num_epairs_per_doc.tolist(), dim=0), [batch_epair_embs, batch_epair_contexts, batch_epair_relations])
        
        batch_triplet_reps, batch_num_triplet_per_doc = [], []
        for doc_epair_embs, doc_epair_contexts, doc_epair_relations in zip(batch_epair_embs, batch_epair_contexts, batch_epair_relations):
            
            doc_epair_num_relations = doc_epair_relations.sum(1)
            doc_epair_with_relations = torch.nonzero(doc_epair_num_relations).flatten()
            doc_epair_real_relations = torch.nonzero(doc_epair_relations[doc_epair_with_relations])[:,1]
            doc_epair_num_relations = doc_epair_num_relations[doc_epair_with_relations].long()
            
            doc_epair_embs = doc_epair_embs[doc_epair_with_relations]
            doc_epair_reps = torch.cat([doc_epair_embs[:,0,:], doc_epair_embs[:,1,:], doc_epair_contexts[doc_epair_with_relations]], dim=1)
            doc_epair_reps = torch.cat([epair_reps.expand(epair_num_relations, -1) for epair_reps, epair_num_relations in zip(doc_epair_reps, doc_epair_num_relations)]) if doc_epair_num_relations.sum()>0 else doc_epair_reps.repeat(0,1)
            
            doc_triplet_reps = torch.cat([doc_epair_reps, self.rtype_emb_module(doc_epair_real_relations)], dim=1)
            batch_triplet_reps.append(doc_triplet_reps)
            batch_num_triplet_per_doc.append(doc_triplet_reps.shape[0])
            
        batch_triplet_reps = torch.cat(batch_triplet_reps)
        batch_triplet_reps = torch.tanh(self.triplet_extractor_module(batch_triplet_reps))
        batch_triplet_reps = torch.split(batch_triplet_reps, batch_num_triplet_per_doc, dim=0)
            
        batch_FER_reps = []
        for doc_triplet_reps, doc_sent_embs in zip(batch_triplet_reps, batch_sent_embs):
        
            doc_triplet_reps = doc_triplet_reps.view(-1, doc_triplet_reps.shape[-1] // self.args.bilinear_block_size, self.args.bilinear_block_size)
            doc_FER_reps = contract('tbi,sbj->tsbij', doc_triplet_reps, doc_sent_embs).flatten(2,4).flatten(0,1)
            batch_FER_reps.append(doc_FER_reps)
            
        batch_FER_reps = torch.cat(batch_FER_reps)
        batch_FER_reps = self.FER_predictor_module(batch_FER_reps).flatten()
        
        return batch_FER_reps
    
    
    def forward(self, batch_tasks, batch_inputs, to_evaluate=False, to_predict=False):
        
        batch_token_seqs, batch_token_masks, batch_token_types = batch_inputs['batch_token_seqs'], batch_inputs['batch_token_masks'], batch_inputs['batch_token_types']
        batch_token_embs, batch_token_atts = self.transformer_module(batch_token_seqs, batch_token_masks, batch_token_types)
        
        batch_epair_tids, batch_epair_masks, batch_num_epairs_per_doc = batch_inputs['batch_epair_tids'], batch_inputs['batch_epair_masks'], batch_inputs['batch_num_epairs_per_doc']
        batch_epair_embs, batch_epair_reps, batch_epair_contexts = self.get_epair_infos(batch_token_embs, batch_token_atts, batch_epair_tids, batch_epair_masks, batch_num_epairs_per_doc)
        
        if self.info.TASK_PER in batch_tasks or self.info.TASK_FER in batch_tasks:
            batch_sent_tids, batch_num_sents_per_doc = batch_inputs['batch_sent_tids'], batch_inputs['batch_num_sents_per_doc']
            batch_sent_embs = self.get_sent_infos(batch_token_embs, batch_sent_tids, batch_num_sents_per_doc)
        
        batch_loss, batch_preds = 0, None
        
        if self.info.TASK_RE in batch_tasks:
            batch_epair_relations = batch_inputs['batch_epair_relations']
            batch_RE_reps = self.get_RE_reps(batch_epair_reps, batch_epair_contexts)
            if to_evaluate: batch_loss += self.loss_module.cal_RE_loss(batch_RE_reps, batch_epair_relations)
            if to_predict: batch_preds = self.loss_module.cal_RE_results(batch_RE_reps)
                
        if self.info.TASK_CR in batch_tasks:
            batch_mention_tids, batch_mention_coreferences, batch_num_mentions_per_doc = batch_inputs['batch_mention_tids'], batch_inputs['batch_mention_coreferences'], batch_inputs['batch_num_mentions_per_doc']
            batch_CR_reps = self.get_CR_reps(batch_token_embs, batch_mention_tids, batch_num_mentions_per_doc)
            if to_evaluate: batch_loss += self.args.loss_weight_CR * self.loss_module.cal_CR_loss(batch_CR_reps, batch_mention_coreferences)
                
        if self.info.TASK_ET in batch_tasks:
            batch_epair_types = batch_inputs['batch_epair_types']
            batch_ET_reps = self.get_ET_reps(batch_epair_reps)
            if to_evaluate: batch_loss += self.args.loss_weight_ET * self.loss_module.cal_ET_loss(batch_ET_reps, batch_epair_types)
        
        if self.info.TASK_PER in batch_tasks:
            batch_epair_pooled_evidences = batch_inputs['batch_epair_pooled_evidences']
            batch_PER_reps = self.get_PER_reps(batch_epair_contexts, batch_num_epairs_per_doc, batch_sent_embs)
            if to_evaluate: batch_loss += self.args.loss_weight_PER * self.loss_module.cal_PER_loss(batch_PER_reps, batch_epair_pooled_evidences)
                
        if self.info.TASK_FER in batch_tasks:
            batch_epair_relations, batch_epair_finegrained_evidences = batch_inputs['batch_epair_relations'], batch_inputs['batch_epair_finegrained_evidences']
            batch_FER_reps = self.get_FER_reps(batch_epair_embs, batch_epair_contexts, batch_epair_relations, batch_num_epairs_per_doc, batch_sent_embs)
            if to_evaluate: batch_loss += self.args.loss_weight_FER * self.loss_module.cal_FER_loss(batch_FER_reps, batch_epair_finegrained_evidences)
            if to_predict: batch_preds = self.loss_module.cal_FER_results(batch_FER_reps)

        return batch_loss, batch_preds