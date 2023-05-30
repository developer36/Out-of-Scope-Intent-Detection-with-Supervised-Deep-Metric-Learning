import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from tqdm import trange, tqdm
from .boundary import BoundaryLoss
from utils.functions import save_model, euclidean_metric
from utils.metrics import F_measure
from utils.functions import restore_model, centroids_cal
from .pretrain import PretrainManager
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd

loss_map = {
                'CrossEntropyLoss': nn.CrossEntropyLoss(),  # 这个
                'Binary_CrossEntropyLoss': nn.BCELoss()
            }

class DRTManager:

    def __init__(self, args, data, model, logger_name='Detection'):

        self.logger = logging.getLogger(logger_name)

        pretrain_model = PretrainManager(args, data, model)
        self.model = pretrain_model.model
        self.centroids = pretrain_model.centroids
        self.pretrain_best_eval_score = pretrain_model.best_eval_score

        if args.pretrain:  # True
            from dataloader.base import DataManager
            data = DataManager(args, logger_name=None)

            from model.base import TextRepresentation
            model = TextRepresentation(args, data, logger_name='')

        self.device = model.device

        self.train_dataloader = data.dataloader.train_labeled_loader
        self.eval_dataloader = data.dataloader.eval_loader
        self.test_dataloader = data.dataloader.test_loader
        self.label_map = data.dataloader.label_map

        self.loss_fct = loss_map[args.loss_fct]  # nn.CrossEntropyLoss()
        self.best_eval_score = None

        if args.train:
            self.delta = None
            self.delta_points = []

        else:
            self.model = restore_model(self.model, args.model_output_dir)
            self.delta = np.load(os.path.join(args.method_output_dir, 'deltas.npy'))
            self.delta = torch.from_numpy(self.delta).to(self.device)
            self.centroids = np.load(os.path.join(args.method_output_dir, 'centroids.npy'))
            self.centroids = torch.from_numpy(self.centroids).to(self.device)

    def train(self, args, data):
        criterion_boundary = BoundaryLoss(num_labels=data.num_labels, feat_dim=args.feat_dim, device=self.device)

        self.delta = F.softplus(criterion_boundary.delta)
        self.delta_points.append(self.delta)
        optimizer = torch.optim.Adam(criterion_boundary.parameters(), lr=args.lr_boundary)

        if self.centroids is None:
            self.centroids = centroids_cal(self.model, args, data, self.train_dataloader, self.device)
            

        best_eval_score, best_delta, best_centroids = 0, None, None
        wait = 0
        
        print("********** num_train_epochs", int(args.num_train_epochs), '******')

        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                    
                    
                with torch.set_grad_enabled(True):
                    features = self.model(input_ids, segment_ids, input_mask, feature_ext=True)
                    loss, self.delta = criterion_boundary(features, self.centroids, label_ids)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    tr_loss += loss.item()

                    nb_tr_examples += features.shape[0]
                    nb_tr_steps += 1

            self.delta_points.append(self.delta)

            loss = tr_loss / nb_tr_steps

            y_true, y_pred = self.get_outputs(args, data, mode='eval')
            eval_score = round(f1_score(y_true, y_pred, average='macro') * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_eval_score': best_eval_score,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))

            if eval_score > best_eval_score:
                wait = 0
                best_delta = self.delta
                best_eval_score = eval_score
            else:
                if best_eval_score > 0:
                    wait += 1
                    if wait >= args.wait_patient:
                        break

        if best_eval_score > 0:
            self.delta = best_delta
            self.best_eval_score = best_eval_score

        if args.save_model:
            np.save(os.path.join(args.method_output_dir, 'centroids.npy'), self.centroids.detach().cpu().numpy())
            np.save(os.path.join(args.method_output_dir, 'deltas.npy'), self.delta.detach().cpu().numpy())
            np.save(os.path.join(args.method_output_dir, 'all_deltas.npy'), self.delta_points)

    def get_outputs(self, args, data, mode='eval', get_feats=False, pre_train=False, delta=None):

        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)

        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, data.num_labels)).to(self.device)

        for batch in tqdm(dataloader, desc="Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                pooled_output = self.model(input_ids, segment_ids, input_mask, feature_ext=True)

                preds = self.open_classify(data, pooled_output)
                total_preds = torch.cat((total_preds, preds))
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, pooled_output))

        if mode == 'test':
            y_true = total_labels.cpu().numpy()
            y_pred = total_preds.cpu().numpy()
            feats = total_features.cpu().numpy()
            return feats, y_true, y_pred

        if not get_feats:
            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            return y_true, y_pred

    def open_classify(self, data, features):
        #print("open_classify:self.centroids[0, :5]", self.centroids[0, :5])
        logits = euclidean_metric(features, self.centroids)
        probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
        euc_dis = torch.norm(features - self.centroids[preds], 2, 1).view(-1)
        preds[euc_dis >= self.delta[preds]] = data.unseen_label_id

        return preds

    def test(self, args, data, show=True):
        feats, y_true, y_pred = self.get_outputs(args, data, mode='test')

        '''tsne '''
        if args.dataset!='oos':
            colors = ['brown', 'salmon', 'orange', 'darkkhaki', 'olivedrab', 'seagreen', 'teal',
                  'steelblue', 'palevioletred', 'thistle', 'cyan', 'navy', 'pink', 'maroon', 'violet', 'peru', 'indigo', 'plum', 
                  'gold', 'skyblue', 'yellow', 'olive', 'aqua', 'hotpink', 'purple', 'sienna', 'lightblue', 'darkseagreen', 'wheat', 'slategrey',
                'orangered', 'tan', 'bisque', 'lightcoral', 'orchid', 'royalblue', 'forestgreen', 'burlywood', 'darkslateblue']
            oos_id = self.label_map['<UNK>']
            #oos_id = self.label_map['oos']  # oos dataset
#             print("---start tsne----")
            
#             tsne = TSNE(n_components=2)
#             X = tsne.fit_transform(feats)
#             l = feats.shape[0]
#             plt.figure(figsize=(8, 8))
#             plt.grid(c='white')
#             for i in set(y_true):
#                 indexs = np.where(y_true == i)
#                 X_tsne = X[indexs[0]]
#                 if i != oos_id:
#                     plt.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], c=colors[i], s=3)
#                 else:
#                     plt.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], c='grey', s=3)
#             ax = plt.gca()
#             ax.set_facecolor('gainsboro')
#             ax.spines['right'].set_visible(False)
#             ax.spines['top'].set_visible(False)
#             ax.spines['left'].set_visible(False)
#             ax.spines['bottom'].set_visible(False)

#             plt.savefig(f'figs/tsne可视化{args.dataset}_triple_ADB_{args.seed}.png')
#             print("--tsne可视化完成--")
            
        

        cm = confusion_matrix(y_true, y_pred)
        test_results = F_measure(cm)

        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        test_results['Acc'] = acc

        if show:
            self.logger.info("***** Test: Confusion Matrix *****")
            self.logger.info("%s", str(cm))
            self.logger.info("***** Test results *****")

            for key in sorted(test_results.keys()):
                self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred

        return test_results

