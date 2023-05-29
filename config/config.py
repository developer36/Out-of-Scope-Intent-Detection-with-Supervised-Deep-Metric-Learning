import argparse
import os
import importlib
from easydict import EasyDict


class ParamManager:

    def __init__(self, args):

        output_path_param = self.add_output_path_param(args)
        model_hyper_params = ModelHyperParam(args).hyper_param

        self.args = EasyDict(
            dict(
                vars(args),
                **output_path_param,
                **model_hyper_params
            )
        )

    def add_output_path_param(self, args):

        task_output_dir = os.path.join(args.output_dir, args.type)
        if not os.path.exists(task_output_dir):
            os.makedirs(task_output_dir)

        concat_names = [args.method, args.dataset, args.known_cls_ratio, args.labeled_ratio, args.backbone, args.seed]
        method_output_name = "_".join([str(x) for x in concat_names])

        method_output_dir = os.path.join(task_output_dir, method_output_name)
        if not os.path.exists(method_output_dir):
            os.makedirs(method_output_dir)

        model_output_dir = os.path.join(method_output_dir, args.model_dir)
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)

        output_path_param = {
            'method_output_dir': method_output_dir,
            'model_output_dir': model_output_dir
        }

        return output_path_param


class ModelHyperParam(object):
    def __init__(self, args):
        self.hyper_param = self.get_hyper_parameters(args)

    def get_hyper_parameters(self, args):
        """
        Args:
            bert_model (directory): The path for the pre-trained bert model.
            num_train_epochs (int): The number of training epochs.
            num_labels (autofill): The output dimension.
            max_seq_length (autofill): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            freeze_backbone_parameters (binary): Whether to freeze all parameters but the last layer.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            lr_boundary (float): The learning rate of the decision boundary.
            lr (float): The learning rate of backbone.
            activation (str): The activation function of the hidden layer (support 'relu' and 'tanh').
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation.
            test_batch_size (int): The batch size for testing.
            wait_patient (int): Patient steps for Early Stop.
        """
        hyper_parameters = {

            'bert_model': "./pretrained_embedding/bert/uncased_L-12_H-768_A-12/",
            'num_labels': None,
            'max_seq_length': None,
            'freeze_backbone_parameters': True,
            'feat_dim': 768,
            'warmup_proportion': 0.1,
            'lr_boundary': 0.05,
            'activation': 'relu',
            'train_batch_size': 128,
            'eval_batch_size': 64,
            'test_batch_size': 64,
            'wait_patient': 10,
            'lr': 2e-5,
            'num_train_epochs': 100,

        }
        return hyper_parameters