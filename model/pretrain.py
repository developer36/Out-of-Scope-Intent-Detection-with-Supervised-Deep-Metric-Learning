import torch
import torch.nn.functional as F
import os
import copy
import logging
from torch import nn
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tqdm import trange, tqdm
from utils.functions import save_model, restore_model, centroids_cal
import numpy as np

loss_map = {
                'CrossEntropyLoss': nn.CrossEntropyLoss(),
                'Binary_CrossEntropyLoss': nn.BCELoss(),
            }

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.0, emb_name='word_embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                print(name, param.requires_grad, param.grad) 
                norm = torch.norm(param.grad) 
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

        
class PretrainManager:

    def __init__(self, args, data, model, logger_name='Detection'):

        self.logger = logging.getLogger(logger_name)

        self.model = model.model
        self.optimizer = model.optimizer
        self.scheduler = model.scheduler
        self.device = model.device

        self.train_dataloader = data.dataloader.train_labeled_loader
        self.eval_dataloader = data.dataloader.eval_loader
        self.test_dataloader = data.dataloader.test_loader

        self.loss_fct = nn.CrossEntropyLoss()
        self.centroids = None
        self.best_eval_score = None

        if args.pretrain or (not os.path.exists(args.model_output_dir)):
            print('Pre-training Begin...')

            self.train(args, data)

            self.logger.info('Pre-training finished...')

        else:
            self.model = restore_model(self.model, args.model_output_dir)

    def train(self, args, data):

        wait = 0
        best_model = None
        best_eval_score = 0

        EPOCH = int(args.num_train_epochs)
        avg_cost = np.zeros([EPOCH, 2], dtype=np.float32)
        lambda_weight = np.ones([2, EPOCH])
        T = 1

        for epoch in trange(EPOCH, desc="Epoch"):  # 100
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            cost = np.zeros(2, dtype=np.float32)
            if epoch == 0 or epoch == 1:
                avg_cost[:, epoch] = 1.0
            else:
                w_1 = (avg_cost[epoch - 2, 0] - avg_cost[epoch - 1, 0])*10
                w_2 = (avg_cost[epoch - 2, 1] - avg_cost[epoch - 1, 1])*10
                lambda_weight[0, epoch] = 2 * np.exp(w_1 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))
                lambda_weight[1, epoch] = 2 * np.exp(w_2 / T) / (np.exp(w_1 / T) + np.exp(w_2 / T))


            BATCH = 0
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                BATCH += 1
            
             # 初始化
            fgm = FGM(self.model)
            #pgd = PGD(self.model)
            #K = 3
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
               
                with torch.set_grad_enabled(True):
                    loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode="train", loss_fct=self.loss_fct)
                    loss.backward()
                    
                    fgm.attack() # 在embedding上添加对抗扰动
                    loss_adv = self.model(input_ids, segment_ids, input_mask, label_ids, mode="train", loss_fct=self.loss_fct)
                    loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore() # 恢复embedding参数
                    
#                     pgd.backup_grad()
#                     for t in range(K):
#                         pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
#                         if t != K - 1:
#                             self.model.zero_grad()
#                         else:
#                             pgd.restore_grad()
#                         loss_adv1, loss_adv2 = self.model(input_ids, segment_ids, input_mask, label_ids, mode="train", loss_fct=self.loss_fct)
#                         loss_adv = loss_adv1 + loss_adv2
#                         loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
#                     pgd.restore()  # 恢复embedding参数
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            loss = tr_loss / nb_tr_steps

            y_true, y_pred = self.get_outputs(args, data, mode='eval')
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_eval_score': best_eval_score,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))

            if eval_score > best_eval_score:

                best_model = copy.deepcopy(self.model)
                wait = 0
                best_eval_score = eval_score

            elif eval_score > 0:

                wait += 1
                if wait >= args.wait_patient:
                    break

        self.model = best_model

        if args.save_model:
            self.logger.info('Trained models are saved in %s', args.model_output_dir)
            save_model(self.model, args.model_output_dir)

    def get_outputs(self, args, data, mode='eval', get_feats=False):

        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader

        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)

        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)

            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):

                pooled_output, logits = self.model(input_ids, segment_ids, input_mask, mode=mode)

                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, pooled_output))
                total_logits = torch.cat((total_logits, logits))

        if get_feats:
            feats = total_features.cpu().numpy()
            return feats

        else:

            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim=1)

            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()

            return y_true, y_pred
