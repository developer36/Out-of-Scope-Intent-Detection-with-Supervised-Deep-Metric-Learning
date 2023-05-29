import torch
import logging
from transformers import AdamW, get_linear_schedule_with_warmup
from .utils import freeze_bert_parameters
from .__init__ import backbones_map

class ModelManager:

    def __init__(self, args, data, logger_name='Detection'):

        self.logger = logging.getLogger(logger_name)
        if args.backbone.startswith('bert'):
            self.model = self.set_model(args, 'bert')
            self.optimizer, self.scheduler = self.set_optimizer(self.model, data.dataloader.num_train_examples,
                                                                args.train_batch_size, args.num_train_epochs, args.lr, args.warmup_proportion)

    def set_optimizer(self, model, num_train_examples, train_batch_size, num_train_epochs, lr, warmup_proportion):
        num_train_optimization_steps = int(num_train_examples / train_batch_size) * num_train_epochs

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False, no_deprecation_warning=True)
        num_warmup_steps = int(num_train_examples * num_train_epochs * warmup_proportion / train_batch_size)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        return optimizer, scheduler

    def set_model(self, args, pattern):
        print('='*30,args.backbone)
        #print(backbones_map)
        backbone = backbones_map[args.backbone]  # ** BERT
        args.device = self.device = torch.device('cuda:%d' % int(args.gpu_id) if torch.cuda.is_available() else 'cpu')

        if pattern == 'bert':
            model = backbone.from_pretrained(args.pretrain_model_dir, cache_dir="cache", args=args) ##'/home/jovyan/oos可视化/my_pretrain'
            if args.freeze_backbone_parameters:  # True
                self.logger.info('Freeze all parameters but the last layer for efficiency')
                model = freeze_bert_parameters(model)
        model.to(self.device)

        '''
        def freeze_bert_parameters(model):
            for name, param in model.bert.named_parameters():
                param.requires_grad = False
                if "encoder.layer.11" in name or "pooler" in name:
                    param.requires_grad = True
            return model
        '''
        return model

