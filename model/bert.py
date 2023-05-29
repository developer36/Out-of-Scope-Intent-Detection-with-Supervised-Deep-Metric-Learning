from torch import nn
from transformers import BertPreTrainedModel, BertModel
from torch.nn.parameter import Parameter
from .utils import ConvexSampler
import torch

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}

class BERT(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BERT, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.init_weights()

        self.trip_loss = TripletLoss(args.margin)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None, centroids=None):
        '''
        :param input_ids:
        :param token_type_ids:
        :param attention_mask:
        :param labels:
        :param feature_ext:
        :param mode:
        :param loss_fct:   loss_fct =nn.CrossEntropyLoss()
        :param centroids:
        :return:
        '''
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)
        encoded_layer_12 = outputs.hidden_states
        pooled_output = outputs.pooler_output
        #pooled_output = encoded_layer_12[-1][:, 1:, :].mean(dim=1)  # mean pooling t1..tN
        pooled_output = encoded_layer_12[-1].mean(dim=1)  # mean pooling cls t1..tN
        #pooled_output = encoded_layer_12[-1][:, 0, :]  # [cls]

        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                loss_c = loss_fct(logits, labels)
                loss_t = self.trip_loss(pooled_output, labels)
                #return loss_c
                return loss_c+loss_t
                #return loss_t
            else:
                return pooled_output, logits

            
class AdvBERT(BertPreTrainedModel):
    def __init__(self, config, args):
        super(AdvBERT, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.init_weights()

        self.trip_loss = TripletLoss()

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                feature_ext=False, mode=None, loss_fct=None, centroids=None, inputs_embeds=None, gan=False):
        '''
        :param input_ids:
        :param token_type_ids:
        :param attention_mask:
        :param labels:
        :param feature_ext:
        :param mode:
        :param loss_fct:   loss_fct =nn.CrossEntropyLoss()
        :param centroids:
        :return:
        '''
        if not gan:
            outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)
            encoded_layer_12 = outputs.hidden_states
            pooled_output = outputs.pooler_output
            #pooled_output = encoded_layer_12[-1][:, 1:, :].mean(dim=1)  # mean pooling t1..tN
            pooled_output = encoded_layer_12[-1].mean(dim=1)  # mean pooling cls t1..tN
            #pooled_output = encoded_layer_12[-1][:, 0, :]  # [cls]

            pooled_output = self.dense(pooled_output)
            pooled_output = self.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)
        else:
            pooled_output = inputs_embeds

        logits = self.classifier(pooled_output)

        if feature_ext:
            return pooled_output
        else:
            if mode == 'train':
                loss_c = loss_fct(logits, labels)
                loss_t = self.trip_loss(pooled_output, labels)
                return (loss_c, loss_t), pooled_output
            else:
                return pooled_output, logits


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=1.2, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        '''t_loss = tripletloss(outputs[1], labels)'''
        n = inputs.size(0)
        #inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()  # t() 转置
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        # If y =1 then it assumed the first input should be ranked higher (have a larger value) than the second input, and vice-versa for y = -1
        if self.mutual:
            return loss, dist
        return loss

