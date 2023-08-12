import os
import copy
import torch
import random
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

class KGCDataset(Dataset):
    def __init__(self, data: list):
        super(KGCDataset, self).__init__()

        self.data = data
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]

class KGCDataModule:
    def __init__(self, args: dict):
        # 0. some variables used in this class
        self.task = args['task']
        self.data_path = args['data_path']
        self.batch_size = args['batch_size']
        self.num_workers = args['num_workers']
        self.pin_memory = args['pin_memory']

        self.add_ent_neighbors = True if args['add_ent_neighbors'] == 'True' else False
        self.add_rel_neighbors = True if args['add_rel_neighbors'] == 'True' else False
        self.ent_neighbor_num = args['ent_neighbor_num']
        self.rel_neighbor_num = args['rel_neighbor_num']
        self.ent_qual_neighbor_num = args['ent_qual_neighbor_num']
        self.no_relation_token = args['no_relation_token']
        self.no_qual_relation_token = args['no_qual_relation_token']
        self.no_qual_entity_token = args['no_qual_entity_token']
        self.dataset_mode = args['dataset_mode']
        self.train_mode = args['train_mode']
        self.device = args['device']

        # 1. read all data lines
        self.lines = self.read_lines()
        # 2。 get the entities and relations mapping information
        self.entities, self.relations = self.read_entities_and_relations()
        print(f'Number of entities: {len(self.entities)}; Number of relations: {len(self.relations)}')
        # 3. construct the vocab dictionary
        self.vocab, self.vocab_offset = self.get_vocab()
        args.update(self.vocab_offset)
        # 4. get the number of relations and entities
        args['vocab_size'] = len(self.vocab)
        args['num_relations'] = self.vocab_offset['relation_end_idx'] - self.vocab_offset['relation_begin_idx']
        args['num_entities'] = self.vocab_offset['entity_end_idx'] - self.vocab_offset['entity_begin_idx']
        args['num_specials'] = self.vocab_offset['entity_begin_idx']
        # 5. get the entities neighbor
        self.neighbors = self.get_neighbors()
        # 6 entities to be filtered when predict some triplet
        self.entity_filter = self.get_entity_filter()
        # 7. create examples
        examples = self.create_examples()
        # 8. the above inputs are used to construct pytorch Dataset objects
        self.train_ds = KGCDataset(examples['train'])
        self.dev_ds = KGCDataset(examples['dev'])
        self.test_ds = KGCDataset(examples['test'])

    def read_lines(self):
        """
        read triplets from files
        :return: a Python Dict, {train: [], dev: [], test: []}
        """
        data_paths = {
            'train': os.path.join(self.data_path, 'train.txt'),
            'dev': os.path.join(self.data_path, 'dev.txt'),
            'test': os.path.join(self.data_path, 'test.txt')
        }

        lines = dict()
        for mode in data_paths:
            data_path = data_paths[mode]
            raw_data = list()

            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if self.dataset_mode == 'triple':
                        split_line = str(line).strip().split('\t')
                        h, r, t = split_line[0:3]
                        raw_data.append((h, r, t))
                    elif self.dataset_mode == 'statement':
                        split_line = str(line).strip().split('\t')
                        raw_data.append(tuple(split_line))

            lines[mode] = raw_data

        return lines

    def read_entities_and_relations(self):
        """
        read entities and realtions information
        :return:
        """
        entities_list = []
        relations_list = []
        for split_name in self.lines:
            split_data = self.lines[split_name]
            for data in split_data:
                q1, p, q2 = data[0], data[1], data[2]
                if q1 not in entities_list:
                    entities_list.append(q1)
                if p not in relations_list:
                    relations_list.append(p)
                if q2 not in entities_list:
                    entities_list.append(q2)
                if len(data) > 3:
                    data = data[3:]
                    for i in range(len(data)):
                        if i % 2 == 0:
                            if data[i] not in relations_list:
                                relations_list.append(data[i])
                        else:
                            if data[i] not in entities_list:
                                entities_list.append(data[i])
        entities_dict = {}
        relations_dict = {}
        count = 0
        for ent in entities_list:
            entities_dict[ent] = {'token_id': count}
            count = count + 1
        count = 0
        for rel in relations_list:
            relations_dict[rel] = {'token_id': count}
            count = count + 1

        return entities_dict, relations_dict

    def get_vocab(self):
        """
        construct the vocab
        :return: two Python Dict
        """
        tokens = ['[PAD]', '[MASK]', '[SEP]', '[CLS]', self.no_relation_token, self.no_qual_relation_token, self.no_qual_entity_token]
        entity_names = [e for e in self.entities]
        relation_names = []
        for r in self.relations:
            relation_names += [r, f'{r}_reverse']

        entity_begin_idx = len(tokens)
        entity_end_idx = len(tokens) + len(entity_names)
        relation_begin_idx = len(tokens) + len(entity_names)
        relation_end_idx = len(tokens) + len(entity_names) + len(relation_names)

        tokens = tokens + entity_names + relation_names
        vocab = dict()
        for idx, token in enumerate(tokens):
            vocab[token] = idx

        return vocab, {
            'entity_begin_idx': entity_begin_idx,
            'entity_end_idx': entity_end_idx,
            'relation_begin_idx': relation_begin_idx,
            'relation_end_idx': relation_end_idx,
        }

    def get_neighbors(self):
        """
        construct neighbor prompts from training dataset
        :return:
        """
        if self.train_mode == 'without_valid':
            lines = self.lines['train']
        elif self.train_mode == 'with_valid':
            lines = self.lines['train'] + self.lines['dev']
        if self.dataset_mode == 'statement':
            ent_data = {e: {'triple_neighbors': [], 'triple_qual_neighbors': []} for e in self.entities}
        else:
            ent_data = {e: {'triple_neighbors': []} for e in self.entities}

        if self.dataset_mode == 'triple':
            triple_lines = lines
        elif self.dataset_mode == 'statement':
            triple_lines = []
            for line in lines:
                triple_lines.append(tuple(list(line)[0:3]))

        for h, r, t in triple_lines:
            head_tri_neighbors = [self.vocab[t], self.vocab[f'{r}_reverse'], self.vocab['[MASK]']]
            ent_data[h]['triple_neighbors'].append(head_tri_neighbors)
            tail_tri_neighbors = [self.vocab[h], self.vocab[r], self.vocab['[MASK]']]
            ent_data[t]['triple_neighbors'].append(tail_tri_neighbors)

        if self.dataset_mode == 'statement':
            for line in lines:
                head_tri_qual_neighbors = [self.vocab[line[2]], self.vocab[f'{line[1]}_reverse'], self.vocab['[MASK]']]
                qual_line = line[3:]
                pad_len = 2 * self.ent_qual_neighbor_num - len(qual_line)
                for i in range(len(qual_line)):
                    if i % 2 == 0:
                        head_tri_qual_neighbors.append(self.vocab[qual_line[i]])
                    else:
                        head_tri_qual_neighbors.append(self.vocab[qual_line[i]])
                for i in range(pad_len):
                    head_tri_qual_neighbors.append(self.vocab['[PAD]'])
                if pad_len < 0:
                    head_tri_qual_neighbors = head_tri_qual_neighbors[0: 3+2*self.ent_qual_neighbor_num]
                ent_data[line[0]]['triple_qual_neighbors'].append(head_tri_qual_neighbors)
                tail_tri_qual_neighbors = [self.vocab[line[0]], self.vocab[line[1]], self.vocab['[MASK]']]
                for i in range(len(qual_line)):
                    if i % 2 == 0:
                        tail_tri_qual_neighbors.append(self.vocab[qual_line[i]])
                    else:
                        tail_tri_qual_neighbors.append(self.vocab[qual_line[i]])
                for i in range(pad_len):
                    tail_tri_qual_neighbors.append(self.vocab['[PAD]'])
                if pad_len < 0:
                    tail_tri_qual_neighbors = tail_tri_qual_neighbors[0: 3+2*self.ent_qual_neighbor_num]
                ent_data[line[2]]['triple_qual_neighbors'].append(tail_tri_qual_neighbors)

        # add a fake neighbor if there is no neighbor for the entity
        for ent in ent_data:
            if len(ent_data[ent]['triple_neighbors']) == 0:
                tri_neighbors = [self.vocab[ent], self.vocab[self.no_relation_token], self.vocab['[MASK]']]
                ent_data[ent]['triple_neighbors'].append(tri_neighbors)
                if self.dataset_mode == 'statement':
                    tri_qual_neighbors = [self.vocab[ent], self.vocab[self.no_relation_token], self.vocab['[MASK]']]
                    pad_len = 2 * self.ent_qual_neighbor_num
                    for i in range(pad_len):
                        tri_qual_neighbors.append(self.vocab['[PAD]'])
                    ent_data[ent]['triple_qual_neighbors'].append(tri_qual_neighbors)

        return ent_data

    def get_entity_filter(self):
        """
        for given h, r, collect all t
        :return:
        """
        train_lines = self.lines['train']
        dev_lines = self.lines['dev']
        test_lines = self.lines['test']
        lines = train_lines + dev_lines + test_lines

        entity_filter = defaultdict(set)

        if self.dataset_mode == 'triple':
            triple_lines = lines
        elif self.dataset_mode == 'statement':
            triple_lines = []
            for line in lines:
                triple_lines.append(tuple(list(line)[0:3]))
        for h, r, t in triple_lines:
            entity_filter[h, r].add(self.entities[t]['token_id'])
            entity_filter[t, r].add(self.entities[h]['token_id'])
        return entity_filter

    def create_examples(self):
        """
        :return: {train: [], dev: [], test: []}
        """
        examples = dict()
        for mode in self.lines:
            data = list()
            lines = self.lines[mode]
            if self.dataset_mode == 'triple':
                triple_lines = lines
            elif self.dataset_mode == 'statement':
                triple_lines = []
                for line in lines:
                    triple_lines.append(tuple(list(line)[0:3]))
            count = 0
            for h, r, t in tqdm(triple_lines, desc=f'[{mode}]create examples'):
                head_example, tail_example = self.create_one_example(lines[count], h, r, t)
                data.append(head_example)
                data.append(tail_example)
                count = count + 1
            examples[mode] = data
        return examples

    def create_one_example(self, tri_qual, h, r, t):
        """
        create one example
        """
        mask_token = '[MASK]'
        pad_token = '[PAD]'

        head, rel, tail = self.entities[h], self.relations[r], self.entities[t]

        # 1. prepare inputs
        struc_head_prompt = [self.vocab[t], self.vocab[f'{r}_reverse'], self.vocab[mask_token]]
        struc_tail_prompt = [self.vocab[h], self.vocab[r], self.vocab[mask_token]]
        # 2. get filters
        head_filters = list(self.entity_filter[t, r] - {head['token_id']})
        tail_filters = list(self.entity_filter[h, r] - {tail['token_id']})
        # 3. get qual pairs and qual sequence
        tri_qual_list = tri_qual[3:]
        tri_qual_count = len(tri_qual_list) // 2
        # 3.1 if the qual count larger than qual length, then sample
        if tri_qual_count > self.rel_neighbor_num:
            filter_tri_qual_list = []
            random_list = []
            for i in range(tri_qual_count):
                random_list.append([tri_qual_list[2 * i], tri_qual_list[2 * i + 1]])
            random.shuffle(random_list)
            for i in range(self.rel_neighbor_num):
                filter_tri_qual_list.append(random_list[i][0])
                filter_tri_qual_list.append(random_list[i][1])
            tri_qual_list = filter_tri_qual_list
        # 3.2 get the triple and reverse triple qual information
        tri_qual_prompt = []
        reverse_tri_qual_prompt = []
        for i in range(len(tri_qual_list) // 2):
            tri_qual_prompt.append(
                [self.vocab[tri_qual_list[2 * i]], self.vocab[tri_qual_list[2 * i + 1]], self.vocab[mask_token]])
            reverse_tri_qual_prompt.append(
                [self.vocab[f'{tri_qual_list[2 * i]}_reverse'], self.vocab[tri_qual_list[2 * i + 1]],
                 self.vocab[mask_token]])
        # 3.3 if the qual number is smaller than rel_neighbor_num, then adding the padding information
        pad_len = self.rel_neighbor_num - len(tri_qual_prompt)
        if len(tri_qual_prompt) == 0:
            tri_qual_prompt.append([self.vocab[self.no_qual_relation_token], self.vocab[self.no_qual_entity_token],
                                    self.vocab[mask_token]])
            reverse_tri_qual_prompt.append(
                [self.vocab[self.no_qual_relation_token], self.vocab[self.no_qual_entity_token],
                 self.vocab[mask_token]])
        # 4. get qual seguence
        tri_qual_sequence = []
        tri_qual_sequence.append(self.vocab[h])
        tri_qual_sequence.append(self.vocab[r])
        tri_qual_sequence.append(self.vocab[mask_token])
        reverse_tri_qual_sequence = []
        reverse_tri_qual_sequence.append(self.vocab[t])
        reverse_tri_qual_sequence.append(self.vocab[f'{r}_reverse'])
        reverse_tri_qual_sequence.append(self.vocab[mask_token])
        for i in range(len(tri_qual_list) // 2):
            tri_qual_sequence.append(self.vocab[tri_qual_list[2 * i]])
            tri_qual_sequence.append(self.vocab[tri_qual_list[2 * i + 1]])
            reverse_tri_qual_sequence.append(self.vocab[f'{tri_qual_list[2 * i]}_reverse'])
            reverse_tri_qual_sequence.append(self.vocab[tri_qual_list[2 * i + 1]])
        for i in range(pad_len):
            tri_qual_sequence.append(self.vocab[pad_token])
            tri_qual_sequence.append(self.vocab[pad_token])
            reverse_tri_qual_sequence.append(self.vocab[pad_token])
            reverse_tri_qual_sequence.append(self.vocab[pad_token])
        # 5. prepare examples
        head_example = {
            'triple': (t, r, h),
            'triple_seq': struc_head_prompt,
            'triple_qual_seq': reverse_tri_qual_sequence,
            'triple_rel_qual_neighbors': reverse_tri_qual_prompt,
            'neighbors_label': tail['token_id'],
            'label': head["token_id"],
            'rel_labels': reverse_tri_qual_sequence[1] - self.vocab_offset['relation_begin_idx'],
            'filters': head_filters,
        }
        tail_example = {
            'triple': (h, r, t),
            'triple_seq': struc_tail_prompt,
            'triple_qual_seq': tri_qual_sequence,
            'triple_rel_qual_neighbors': tri_qual_prompt,
            'neighbors_label': head['token_id'],
            'label': tail["token_id"],
            'rel_labels': tri_qual_sequence[1] - self.vocab_offset['relation_begin_idx'],
            'filters': tail_filters,
        }

        return head_example, tail_example

    def struc_batch_encoding(self, inputs):
        input_ids = torch.tensor(inputs)
        return {'input_ids': input_ids}

    def collate_fn(self, batch_data):
        data_triple = [data_dit['triple'] for data_dit in batch_data]
        triple_seqs = [copy.deepcopy(data_dit['triple_seq']) for data_dit in batch_data]
        triple_seq_data = self.struc_batch_encoding(triple_seqs)

        # 1. get the entity neighbors
        if self.add_ent_neighbors:
            batch_ent_neighbors = [[] for _ in range(self.ent_neighbor_num)]
            batch_ent_neighbors_with_qual = [[] for _ in range(self.ent_neighbor_num)]
            for ent, _, _ in data_triple:
                ent_neighbors = self.neighbors[ent]['triple_neighbors']
                idxs = list(range(len(ent_neighbors)))
                if len(idxs) >= self.ent_neighbor_num:
                    idxs = random.sample(idxs, self.ent_neighbor_num)
                else:
                    tmp_idxs = []
                    for _ in range(self.ent_neighbor_num - len(idxs)):
                        tmp_idxs.append(random.sample(idxs, 1)[0])
                    idxs = tmp_idxs + idxs
                assert len(idxs) == self.ent_neighbor_num
                for i, idx in enumerate(idxs):
                    batch_ent_neighbors[i].append(ent_neighbors[idx])
                if self.dataset_mode == 'statement':
                    ent_qual_neighbors = self.neighbors[ent]['triple_qual_neighbors']
                    for i, idx in enumerate(idxs):
                        batch_ent_neighbors_with_qual[i].append(ent_qual_neighbors[idx])
            # ent_neighbor_num * batch_size
            ent_neighbors = [self.struc_batch_encoding(batch_ent_neighbors[i]) for i in range(self.ent_neighbor_num)]
            if self.dataset_mode == 'statement':
                ent_qual_neighbors = [self.struc_batch_encoding(batch_ent_neighbors_with_qual[i]) for i in range(self.ent_neighbor_num)]
            else:
                ent_qual_neighbors = None
        else:
            ent_neighbors = None
            ent_qual_neighbors = None

        # 2. get the relation qual neighbors
        if self.dataset_mode == 'statement':
            rel_qual_neighbors = [data_dit['triple_rel_qual_neighbors']for data_dit in batch_data]
            batch_rel_qual_neighbors = [[] for _ in range(self.rel_neighbor_num)]
            for single_rel_qual_neighbors in rel_qual_neighbors:
                idxs = list(range(len(single_rel_qual_neighbors)))
                if len(idxs) >= self.rel_neighbor_num:
                    idxs = random.sample(idxs, self.rel_neighbor_num)
                else:
                    tmp_idxs = []
                    for _ in range(self.rel_neighbor_num - len(idxs)):
                        tmp_idxs.append(random.sample(idxs, 1)[0])
                    idxs = tmp_idxs + idxs
                assert len(idxs) == self.rel_neighbor_num
                for i, idx in enumerate(idxs):
                    batch_rel_qual_neighbors[i].append(single_rel_qual_neighbors[idx])
            rel_qual_neighbors = [self.struc_batch_encoding(batch_rel_qual_neighbors[i]) for i in range(self.rel_neighbor_num)]
            triple_qual_seq = [copy.deepcopy(data_dit['triple_qual_seq']) for data_dit in batch_data]
            triple_qual_seq_data = self.struc_batch_encoding(triple_qual_seq)
        else:
            rel_qual_neighbors = None
            triple_qual_seq_data = None

        neighbors_labels = torch.tensor([data_dit['neighbors_label']for data_dit in batch_data]) if self.add_ent_neighbors else None
        relations_labels = torch.tensor([data_dit['rel_labels'] for data_dit in batch_data]) if self.add_rel_neighbors else None
        labels = torch.tensor([data_dit['label'] for data_dit in batch_data])
        filters = torch.tensor([[i, j] for i, data_dit in enumerate(batch_data) for j in data_dit['filters']])

        return {
            'data': data_triple,
            'triple_seq': triple_seq_data, 'qual_triple_seq': triple_qual_seq_data,
            'ent_neighbors': ent_neighbors, 'ent_qual_neighbors': ent_qual_neighbors, 'rel_qual_neighbors': rel_qual_neighbors,
            'labels': labels, 'neighbors_labels': neighbors_labels, 'relations_labels': relations_labels, 'filters': filters
        }

    def get_train_dataloader(self):
        dataloader = DataLoader(self.train_ds, collate_fn=self.collate_fn,
                                batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=True)
        return dataloader

    def get_train_dev_dataloader(self):
        dataloader = DataLoader(self.train_ds + self.dev_ds, collate_fn=self.collate_fn,
                                batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=True)
        return dataloader

    def get_dev_dataloader(self):
        dataloader = DataLoader(self.dev_ds, collate_fn=self.collate_fn,
                                batch_size=2 * self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=False)
        return dataloader

    def get_test_dataloader(self):
        dataloader = DataLoader(self.test_ds, collate_fn=self.collate_fn,
                                batch_size=2 * self.batch_size, num_workers=self.num_workers,
                                pin_memory=self.pin_memory, shuffle=False)
        return dataloader
