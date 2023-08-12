import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contiguous_params import ContiguousParams

from .knowformer import Knowformer
from .utils import get_ranks, get_norms, get_scores
from torch.nn.modules.loss import _WeightedLoss

class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                                                              self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss

class HyperFormer(nn.Module):
    def __init__(self, args: dict, bert_encoder: Knowformer):
        super(HyperFormer, self).__init__()

        self.device = torch.device(args['device'])
        self.add_ent_neighbors = True if args['add_ent_neighbors'] == 'True' else False
        self.add_rel_neighbors = True if args['add_rel_neighbors'] == 'True' else False
        self.use_rel_loss = True if args['use_interacte'] == 'False' else False
        self.ent_neighbor_num = args['ent_neighbor_num']
        self.rel_neighbor_num = args['rel_neighbor_num']
        self.lr = args['kge_lr']

        self.entity_begin_idx = args['entity_begin_idx']
        self.relation_begin_idx = args['relation_begin_idx']
        self.entity_end_idx = args['entity_end_idx']
        self.use_extra_encoder = args['extra_encoder']

        self.dataset_mode = args['dataset_mode']

        self.bert_encoder = bert_encoder
        self.loss_fc = LabelSmoothCrossEntropyLoss(smoothing=args['kge_label_smoothing'])

    def forward(self, batch_data):
        output = self.link_prediction(batch_data)
        return output['loss'], output['rank']

    def training_step(self, batch, batch_idx):
        loss, _ = self.forward(batch)
        return loss

    def training_epoch_end(self, outputs):
        return np.round(np.mean([loss.item() for loss in outputs]), 4)

    def validation_step(self, batch, batch_idx):
        output = self.link_prediction_validation(batch)
        loss, rank = output['loss'], output['rank']
        return loss.item(), rank

    def validation_epoch_end(self, outputs):
        loss, rank = list(), list()
        for batch_loss, batch_rank in outputs:
            loss.append(batch_loss)
            rank += batch_rank
        loss = np.mean(loss)
        scores = get_scores(rank, loss)
        return scores

    def link_prediction_validation(self, batch):
        # 1. prepare data
        if self.dataset_mode == 'statement':
            input_ids = batch['qual_triple_seq']['input_ids'].to(self.device)
        elif self.dataset_mode == 'triple':
            input_ids = batch['triple_seq']['input_ids'].to(self.device)

        context_input_ids = None
        rel_local_context_ids = None
        labels = batch['labels'].to(self.device)
        filters = batch['filters'].to(self.device)

        # 2. get output from knowformer
        output = self.bert_encoder(input_ids, context_input_ids, rel_local_context_ids, self.use_extra_encoder)
        origin_logits = output['triple_neighbors']

        # 3. compute loss and rank
        origin_loss = self.loss_fc(origin_logits, labels + self.entity_begin_idx)
        origin_logits = origin_logits[:, self.entity_begin_idx: self.entity_end_idx]
        rank = get_ranks(F.softmax(origin_logits, dim=-1), labels, filters)

        return {'loss': origin_loss, 'rank': rank, 'logits': origin_logits}

    def link_prediction(self, batch):
        # 1. prepare data
        if self.dataset_mode == 'statement':
            input_ids = batch['qual_triple_seq']['input_ids'].to(self.device)
        elif self.dataset_mode == 'triple':
            input_ids = batch['triple_seq']['input_ids'].to(self.device)

        if self.add_ent_neighbors:
            if self.dataset_mode == 'statement':
                context_input_ids = [t['input_ids'].to(self.device) for t in batch['ent_qual_neighbors']]
            elif self.dataset_mode == 'triple':
                context_input_ids = [t['input_ids'].to(self.device) for t in batch['ent_neighbors']]
            neighbors_labels = batch['neighbors_labels'].to(self.device)
        else:
            context_input_ids = None
            neighbors_labels = None

        if self.add_rel_neighbors:
            rel_loc_context_input_ids = [t['input_ids'].to(self.device) for t in batch['rel_qual_neighbors']]
            relations_labels = batch['relations_labels'].to(self.device)
        else:
            rel_loc_context_input_ids = None

        labels = batch['labels'].to(self.device)
        filters = batch['filters'].to(self.device)

        # 2. encode
        output = self.bert_encoder(input_ids, context_input_ids, rel_loc_context_input_ids, self.use_extra_encoder)
        triple_logits = output['triple_neighbors']
        context_ent_logits = output['ent_neighbors']
        context_rel_logits = output['rel_neighbors']
        mixed_logits = output['neighbors']

        # 3. compute lossed
        # 3.1 loss from the current triplet
        triple_loss = self.loss_fc(triple_logits, labels + self.entity_begin_idx)
        if not self.add_ent_neighbors and not self.add_rel_neighbors:
            triple_logits = triple_logits[:, self.entity_begin_idx: self.entity_end_idx]
            rank = get_ranks(F.softmax(triple_logits, dim=-1), labels, filters)
            return {'loss': triple_loss, 'rank': rank, 'logits': triple_logits}

        # 3.2 losses from neighboring triplets
        if self.add_ent_neighbors:
            loss_for_ent_neighbors = None
            for i in range(self.ent_neighbor_num):
                logits = context_ent_logits[i]
                loss = self.loss_fc(logits, neighbors_labels + self.entity_begin_idx)
                if loss_for_ent_neighbors is None:
                    loss_for_ent_neighbors = loss
                else:
                    loss_for_ent_neighbors += loss
            loss_for_ent_neighbors = loss_for_ent_neighbors / self.ent_neighbor_num

        if self.add_rel_neighbors and self.use_rel_loss:
            loss_for_rel_neighbors = None
            for i in range(self.rel_neighbor_num):
                logits = context_rel_logits[i]
                loss = self.loss_fc(logits, relations_labels + self.relation_begin_idx)
                if loss_for_rel_neighbors is None:
                    loss_for_rel_neighbors = loss
                else:
                    loss_for_rel_neighbors += loss
            loss_for_rel_neighbors = loss_for_rel_neighbors / self.rel_neighbor_num

        # 3.3 loss from mixed embeddings
        if self.add_ent_neighbors == True and self.add_rel_neighbors == False:
            mixed_loss = self.loss_fc(mixed_logits, labels + self.entity_begin_idx)
            # 3.4 merge all losses
            loss = triple_loss + mixed_loss + 0.5 * loss_for_ent_neighbors
            logits = mixed_logits[:, self.entity_begin_idx: self.entity_end_idx] \
                     + triple_logits[:, self.entity_begin_idx: self.entity_end_idx]
            rank = get_ranks(F.softmax(logits, dim=-1), labels, filters)
        if self.add_ent_neighbors == False and self.add_rel_neighbors == True:
            mixed_loss = self.loss_fc(mixed_logits, labels + self.entity_begin_idx)
            # 3.4 merge all losses
            if self.use_rel_loss:
                loss = triple_loss + mixed_loss + 0.5 * loss_for_rel_neighbors
            else:
                loss = triple_loss + mixed_loss
            logits = mixed_logits[:, self.entity_begin_idx: self.entity_end_idx] \
                     + triple_logits[:, self.entity_begin_idx: self.entity_end_idx]
            rank = get_ranks(F.softmax(logits, dim=-1), labels, filters)
        if self.add_ent_neighbors == True and self.add_rel_neighbors == True:
            mixed_loss = self.loss_fc(mixed_logits, labels + self.entity_begin_idx)
            # 3.4 merge all losses
            if self.use_rel_loss:
                loss = triple_loss + mixed_loss + 0.5 * loss_for_ent_neighbors + 0.25 * loss_for_rel_neighbors
            else:
                loss = triple_loss + mixed_loss + 0.5 * loss_for_ent_neighbors
            logits = mixed_logits[:, self.entity_begin_idx: self.entity_end_idx] \
                     + triple_logits[:, self.entity_begin_idx: self.entity_end_idx]
            rank = get_ranks(F.softmax(logits, dim=-1), labels, filters)

        return {'loss': loss, 'rank': rank, 'logits': logits}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(ContiguousParams(self.bert_encoder.parameters()).contiguous(), lr=self.lr)
        scheduler = None
        return {'optimizer': opt, 'scheduler': scheduler}

    def get_parameters(self):
        decay_param = []
        no_decay_param = []
        for n, p in self.bert_encoder.named_parameters():
            if not p.requires_grad:
                continue
            if ('bias' in n) or ('LayerNorm.weight' in n):
                no_decay_param.append(p)
            else:
                decay_param.append(p)
        return [
            {'params': decay_param, 'weight_decay': 1e-2, 'lr': self.lr},
            {'params': no_decay_param, 'weight_decay': 0, 'lr': self.lr}
        ]

    def freeze(self):
        for n, p in self.bert_encoder.named_parameters():
            p.requires_grad = False

    def clip_grad_norm(self):
        norms = get_norms(self.bert_encoder.parameters()).item()
        info = f'grads for N-Former: {round(norms, 4)}'
        return info

    def grad_norm(self):
        norms = get_norms(self.bert_encoder.parameters()).item()
        return round(norms, 4)

