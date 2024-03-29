import numpy as np
import os
import random
import torch
import logging

from .__init__ import max_seq_lengths, BERT_Loader, benchmark_labels


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class DataManager:

    def __init__(self, args, logger_name='Detection'):

        self.logger = logging.getLogger(logger_name)

        set_seed(args.seed)
        args.max_seq_length = max_seq_lengths[args.dataset]  # 55
        self.data_dir = os.path.join(args.data_dir, args.dataset)

        self.all_label_list = self.get_labels(args.dataset)
        # ["Refund_not_showing_up", "activate_my_card", "age_limit"..]

        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)  # 0.75 The number of known classes
        self.known_label_list = np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False)
        self.known_label_list = list(self.known_label_list)

        self.data_dir_neg = os.path.join(args.data_dir, args.dataset_neg)

        self.logger.info('The number of known intents is %s', self.n_known_cls)
        self.logger.info('Lists of known labels are: %s', str(self.known_label_list))

        args.num_labels = self.num_labels = len(self.known_label_list)

        
        self.unseen_label = '<UNK>'

        args.unseen_label_id = self.unseen_label_id = self.num_labels
        self.label_list = self.known_label_list + [self.unseen_label]

        self.dataloader = self.get_loader(args, self.get_attrs())

    def get_labels(self, dataset):

        labels = benchmark_labels[dataset]

        return labels

    def get_loader(self, args, attrs):

        dataloader = BERT_Loader(args, attrs, args.logger_name)

        return dataloader

    def get_attrs(self):

        attrs = {}
        for name, value in vars(self).items():
            attrs[name] = value

        return attrs
