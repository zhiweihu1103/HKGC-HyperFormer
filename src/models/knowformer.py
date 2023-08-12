import torch
import torch.nn as nn
import numpy as np
from .knowformer_encoder import Encoder, get_param

from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_

class FuseNet(nn.Module):
    def __init__(self, hidden_size):
        super(FuseNet, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, p, q):
        lq = self.linear(q)
        lp = self.linear(p)
        mid = nn.Sigmoid()(lq+lp)
        output = p * mid + q * (1-mid)
        return output

class Knowformer(nn.Module):
    def __init__(self, config):
        super(Knowformer, self).__init__()
        self._emb_size = config['hidden_size']
        self._n_layer = config['num_hidden_layers']
        self._n_head = config['num_attention_heads']
        self._input_dropout_prob = config['input_dropout_prob']
        self._context_dropout_prob = config['context_dropout_prob']
        self._qual_dropout_prob = config['qual_dropout_prob']
        self._attention_dropout_prob = config['attention_dropout_prob']
        self._hidden_dropout_prob = config['hidden_dropout_prob']
        self._entity_dropout_prob = config['entity_dropout_prob']
        self._residual_dropout_prob = config['residual_dropout_prob']
        self._intermediate_size = config['intermediate_size']
        self._initializer_range = config['initializer_range']
        self.device = config['device']

        self.add_ent_neighbors = True if config['add_ent_neighbors'] == 'True' else False
        self.add_rel_neighbors = True if config['add_rel_neighbors'] == 'True' else False

        self._voc_size = config['vocab_size']
        self._n_entity = config['num_entities']
        self._n_relation = config['num_relations']
        self._n_special = config['num_specials']

        self._entity_begin_idx = config['entity_begin_idx']
        self._entity_end_idx = config['entity_end_idx']
        self._relation_begin_idx = config['relation_begin_idx']
        self._relation_end_idx = config['relation_end_idx']
        self._special_begin_idx = config['special_begin_idx']
        self._special_end_idx = config['special_end_idx']

        self._ablation_mode = config['ablation_mode']
        self.use_interacte = True if config['use_interacte'] == 'True' else False

        self.ele_embedding = get_param((self._voc_size, self._emb_size))
        self.triple_encoder = Encoder(config, moe_mode=config['moe_mode'])
        self.context_encoder = Encoder(config, moe_mode=config['moe_mode'])
        self.input_dropout_layer = nn.Dropout(p=self._input_dropout_prob)
        self.context_dropout_layer = nn.Dropout(p=self._context_dropout_prob)
        self.qual_dropout_layer = nn.Dropout(p=self._qual_dropout_prob)
        self.entity_or_relation_layernorm_layer = nn.LayerNorm(config['hidden_size'])
        self.entity_dropout_layer = torch.nn.Dropout(self._entity_dropout_prob)

        self.conv_input_dropout_prob = config['conv_input_dropout_prob']
        self.conv_hidden_dropout_prob = config['conv_hidden_dropout_prob']
        self.conv_feature_dropout_prob = config['conv_feature_dropout_prob']
        self.conv_padding = config['conv_padding']
        self.conv_number_channel = config['conv_number_channel']
        self.conv_kernel_size = config['conv_kernel_size']
        self.conv_kernel_width = config['conv_kernel_width']
        self.conv_kernel_height = config['conv_kernel_height']
        self.conv_permution_size = config['conv_permution_size']

        self.flatten_size = self.conv_kernel_width * 2 * self.conv_kernel_height * self.conv_number_channel * self.conv_permution_size
        self.bn0 = torch.nn.BatchNorm2d(self.conv_permution_size)
        self.bn1 = torch.nn.BatchNorm2d(self.conv_number_channel * self.conv_permution_size)
        self.bn2 = torch.nn.BatchNorm1d(self._emb_size)
        self.bn3 = torch.nn.BatchNorm1d(self._emb_size*2)
        self.conv_input_dropout = torch.nn.Dropout(self.conv_input_dropout_prob)
        self.conv_hidden_dropout = torch.nn.Dropout(self.conv_hidden_dropout_prob)
        self.conv_feature_dropout = torch.nn.Dropout2d(self.conv_feature_dropout_prob)
        self.fc = torch.nn.Linear(self.flatten_size, self._emb_size)
        self.fc2 = torch.nn.Linear(self.flatten_size, self._emb_size * 2)
        self.register_parameter('conv_filt', Parameter(torch.zeros(self.conv_number_channel, 1, self.conv_kernel_size, self.conv_kernel_size)))
        xavier_normal_(self.conv_filt)
        if self.use_interacte:
            self.chequer_perm = self.get_chequer_perm()
        self.fuse_net = FuseNet(self._emb_size)

    def get_chequer_perm(self):
        ent_perm = np.int32([np.random.permutation(self._emb_size) for _ in range(self.conv_permution_size)])
        rel_perm = np.int32([np.random.permutation(self._emb_size) for _ in range(self.conv_permution_size)])

        comb_idx = []
        for k in range(self.conv_permution_size):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(self.conv_kernel_height):
                for j in range(self.conv_kernel_width):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx]);
                            ent_idx += 1;
                            temp.append(rel_perm[k, rel_idx] + self._emb_size);
                            rel_idx += 1;
                        else:
                            temp.append(rel_perm[k, rel_idx] + self._emb_size);
                            rel_idx += 1;
                            temp.append(ent_perm[k, ent_idx]);
                            ent_idx += 1;
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self._emb_size);
                            rel_idx += 1;
                            temp.append(ent_perm[k, ent_idx]);
                            ent_idx += 1;
                        else:
                            temp.append(ent_perm[k, ent_idx]);
                            ent_idx += 1;
                            temp.append(rel_perm[k, rel_idx] + self._emb_size);
                            rel_idx += 1;

            comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
        return chequer_perm

    def circular_padding_chw(self, batch, padding):
        upper_pad = batch[..., -padding:, :]
        lower_pad = batch[..., :padding, :]
        temp = torch.cat([upper_pad, batch, lower_pad], dim=2)

        left_pad = temp[..., -padding:]
        right_pad = temp[..., :padding]
        padded = torch.cat([left_pad, temp, right_pad], dim=3)
        return padded

    def __combine_vector(self, first_embedding, second_embedding):
        comb_emb = torch.cat([first_embedding, second_embedding], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, self.conv_permution_size, 2 * self.conv_kernel_width, self.conv_kernel_height))
        stack_inp = self.bn0(stack_inp)
        x = self.conv_input_dropout(stack_inp)
        x = self.circular_padding_chw(x, self.conv_kernel_size // 2)
        x = F.conv2d(x, self.conv_filt.repeat(self.conv_permution_size, 1, 1, 1), padding=self.conv_padding, groups=self.conv_permution_size)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv_feature_dropout(x)
        x = x.view(-1, self.flatten_size)
        x = self.fc2(x)
        x = self.conv_hidden_dropout(x)
        emb_out = self.bn3(x)
        first_embedding, second_embedding = torch.chunk(emb_out, 2, dim=1)

        return first_embedding, second_embedding

    def __forward_bidirection_fusion(self, triple_entity_embedding, triple_relation_embedding, triple_qual_embedding):
        entity_relation_embedding, relation_entity_embedding = self.__combine_vector(triple_entity_embedding, triple_relation_embedding)
        entity_qual_embedding, qual_entity_embedding = self.__combine_vector(triple_entity_embedding, triple_qual_embedding)
        relation_qual_embedding, qual_relation_embedding = self.__combine_vector(triple_relation_embedding, triple_qual_embedding)
        entity_embedding = self.fuse_net(entity_relation_embedding, entity_qual_embedding)
        relation_embedding = self.fuse_net(relation_entity_embedding, relation_qual_embedding)
        qual_embedding = self.fuse_net(qual_entity_embedding, qual_relation_embedding)

        return entity_embedding, relation_embedding, qual_embedding

    def __forward_intersect(self, triple_ids):
        entity_embeddings = torch.index_select(self.ele_embedding, 0, torch.tensor(range(self._entity_begin_idx, self._entity_end_idx)).to(self.device))
        entity_embeddings = self.entity_or_relation_layernorm_layer(entity_embeddings)
        entity_embeddings = self.entity_dropout_layer(entity_embeddings)
        relation_embeddings = torch.index_select(self.ele_embedding, 0, torch.tensor(range(self._relation_begin_idx, self._relation_end_idx)).to(self.device))
        relation_embeddings = self.entity_or_relation_layernorm_layer(relation_embeddings)
        special_embedding = torch.index_select(self.ele_embedding, 0, torch.tensor(range(self._special_begin_idx, self._special_end_idx)).to(self.device))
        ele_embedding = torch.cat((special_embedding, entity_embeddings, relation_embeddings), 0)

        if 'transe' in self._ablation_mode:
            batch_size, triple_len = triple_ids.shape[0], triple_ids.shape[1]
            triple_embedding = torch.index_select(ele_embedding, 0, triple_ids.view(-1)).view(batch_size, triple_len, -1)
            triple_ent_embedding = triple_embedding[:, 0, :]
            triple_rel_embedding = triple_embedding[:, 1, :]
            emb_out = triple_ent_embedding + triple_rel_embedding
        elif 'dismult' in self._ablation_mode:
            batch_size, triple_len = triple_ids.shape[0], triple_ids.shape[1]
            triple_embedding = torch.index_select(ele_embedding, 0, triple_ids.view(-1)).view(batch_size, triple_len, -1)
            triple_ent_embedding = triple_embedding[:, 0, :]
            triple_rel_embedding = triple_embedding[:, 1, :]
            emb_out = triple_ent_embedding * triple_rel_embedding
        elif 'complex' in self._ablation_mode:
            batch_size, triple_len = triple_ids.shape[0], triple_ids.shape[1]
            triple_embedding = torch.index_select(ele_embedding, 0, triple_ids.view(-1)).view(batch_size, triple_len, -1)
            triple_ent_embedding = triple_embedding[:, 0, :]
            triple_rel_embedding = triple_embedding[:, 1, :]
            re_head, im_head = torch.chunk(triple_ent_embedding, 2, dim=1)
            re_relation, im_relation = torch.chunk(triple_rel_embedding, 2, dim=1)
            re_part = re_head * re_relation - im_head * im_relation
            im_part = re_head * im_relation + im_head * re_relation
            emb_out = torch.cat((re_part, im_part), 1)
        elif 'rotate' in self._ablation_mode:
            pi = 3.14159265358979323846
            batch_size, triple_len = triple_ids.shape[0], triple_ids.shape[1]
            triple_embedding = torch.index_select(ele_embedding, 0, triple_ids.view(-1)).view(batch_size, triple_len, -1)
            triple_ent_embedding = triple_embedding[:, 0, :]
            triple_rel_embedding = triple_embedding[:, 1, :]
            re_head, im_head = torch.chunk(triple_ent_embedding, 2, dim=1)
            phase_relation = triple_rel_embedding / (self._initializer_range / pi)
            re_relation, im_relation = torch.chunk(phase_relation, 2, dim=1)
            re_relation = torch.cos(re_relation)
            im_relation = torch.sin(im_relation)
            re_part = re_head * re_relation - im_head * im_relation
            im_part = re_head * im_relation + im_head * re_relation
            emb_out = torch.cat((re_part, im_part), 1)

        return emb_out

    def __forward_triples(self, triple_ids, context_emb=None, qual_emb=None, encoder_mask=None, encoder_type="triple"):
        entity_embeddings = torch.index_select(self.ele_embedding, 0, torch.tensor(range(self._entity_begin_idx, self._entity_end_idx)).to(self.device))
        entity_embeddings = self.entity_or_relation_layernorm_layer(entity_embeddings)
        entity_embeddings = self.entity_dropout_layer(entity_embeddings)
        relation_embeddings = torch.index_select(self.ele_embedding, 0, torch.tensor(range(self._relation_begin_idx, self._relation_end_idx)).to(self.device))
        relation_embeddings = self.entity_or_relation_layernorm_layer(relation_embeddings)
        special_embedding = torch.index_select(self.ele_embedding, 0, torch.tensor(range(self._special_begin_idx, self._special_end_idx)).to(self.device))
        ele_embedding = torch.cat((special_embedding, entity_embeddings, relation_embeddings), 0)
        batch_size, triple_len = triple_ids.shape[0], triple_ids.shape[1]
        emb_out = torch.index_select(ele_embedding, 0, triple_ids.view(-1)).view(batch_size, triple_len, -1)

        if context_emb is not None:
            context_emb = self.context_dropout_layer(context_emb)
            emb_out[:, 0, :] = (emb_out[:, 0, :] + context_emb) / 2
        if qual_emb is not None:
            qual_emb = self.qual_dropout_layer(qual_emb)
            emb_out[:, 1, :] = (emb_out[:, 1, :] + qual_emb) / 2

        emb_out = self.input_dropout_layer(emb_out)
        encoder = self.triple_encoder if encoder_type == "triple" else self.context_encoder
        emb_out = encoder(emb_out, mask=encoder_mask)
        return emb_out

    def __process_mask_feat(self, mask_feat):
        return torch.matmul(mask_feat, self.ele_embedding.transpose(0, 1))

    def forward(self, src_ids, context_ids=None, rel_local_context_ids=None, double_encoder=False):
        encoder_mask = (src_ids != 0).unsqueeze(1).unsqueeze(2)
        seq_emb_out = self.__forward_triples(src_ids, context_emb=None, encoder_mask=encoder_mask)
        triple_relation_emb = seq_emb_out[:, 1, :]
        mask_emb = seq_emb_out[:, 2, :]

        logits_from_triplets = self.__process_mask_feat(mask_emb)

        if context_ids is not None:
            logits_from_ent_neighbors = []
            embeds_from_ent_neighbors = []
            for i in range(len(context_ids)):
                if double_encoder:
                    seq_emb_out = self.__forward_triples(context_ids[i], context_emb=None, qual_emb=None, encoder_type='context')
                else:
                    seq_emb_out = self.__forward_triples(context_ids[i], context_emb=None, qual_emb=None, encoder_type='triple')
                mask_emb = seq_emb_out[:, 2, :]
                logits = self.__process_mask_feat(mask_emb)
                embeds_from_ent_neighbors.append(mask_emb)
                logits_from_ent_neighbors.append(logits)
            context_ent_embeds = torch.stack(embeds_from_ent_neighbors, dim=0)
            context_ent_embeds = torch.mean(context_ent_embeds, dim=0)

        if rel_local_context_ids is not None:
            logits_from_rel_neighbors = []
            embeds_from_rel_neighbors = []
            for i in range(len(rel_local_context_ids)):
                if double_encoder:
                    seq_emb_out = self.__forward_triples(rel_local_context_ids[i], context_emb=None, qual_emb=None, encoder_type='context')
                else:
                    if self.use_interacte:
                        mask_emb = self.__forward_intersect(rel_local_context_ids[i])
                    else:
                        seq_emb_out = self.__forward_triples(rel_local_context_ids[i], context_emb=None, qual_emb=None, encoder_type='triple')
                if self.use_interacte:
                    logits = None
                else:
                    mask_emb = seq_emb_out[:, 2, :]
                    logits = self.__process_mask_feat(mask_emb)
                embeds_from_rel_neighbors.append(mask_emb)
                logits_from_rel_neighbors.append(logits)
            context_rel_embeds = torch.stack(embeds_from_rel_neighbors, dim=0)
            context_rel_embeds = torch.mean(context_rel_embeds, dim=0)

        if context_ids is not None and rel_local_context_ids is not None:
            context_ent_embeds, triple_relation_emb, context_rel_embeds = self.__forward_bidirection_fusion(context_ent_embeds, triple_relation_emb, context_rel_embeds)
            seq_emb_out = self.__forward_triples(src_ids, context_emb=context_ent_embeds, qual_emb=context_rel_embeds, encoder_type='triple')
            mask_embed = seq_emb_out[:, 2, :]
            logits_from_ent_rel_and_triplets = self.__process_mask_feat(mask_embed)
            return {
                'triple_neighbors': logits_from_triplets,
                'ent_neighbors': logits_from_ent_neighbors,
                'rel_neighbors': logits_from_rel_neighbors,
                'neighbors': logits_from_ent_rel_and_triplets
            }

        if context_ids is None and rel_local_context_ids is None:
            return {
                'triple_neighbors': logits_from_triplets,
                'ent_neighbors': None,
                'rel_neighbors': None,
                'neighbors': None
            }
