import math
import torch
import logging
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
logger = logging.getLogger(__name__)


class Observer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-2-v2')
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(768, 1)
        #self.init_weights()
        print(self.bert)

    def forward(self, input_ids, token_type_ids, attention_mask, weight=None, is_training=True):
        batch_size = input_ids.size()[0]
        device = input_ids.device

        pool_output = self.bert(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)#[1]
        #score = self.classifier(self.dropout(pool_output))  # [B, 1]
        output = pool_output.logits
        score = pool_output.logits

        if is_training:
            similarities = torch.cat((score[0::2], score[1::2]), dim=1)  # [bs//2, 2]
            labels = torch.zeros((batch_size//2,), dtype=torch.int64).to(device)  # [bs//2,]

            loss_fct = nn.CrossEntropyLoss(reduce=False)
            # print(similarities, labels)
            loss = loss_fct(similarities, labels)  # [bs//2]
            if weight is None:
                weight = torch.ones((batch_size//2,), dtype=torch.float).to(device)
            loss_origin = loss.sum() / (batch_size//2)
            loss_weight = torch.dot(weight, loss) / (batch_size//2)

            output = loss_origin, loss_weight, output
        return output


class Reranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-2-v2')
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(768, 1)
        #self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, weight=None, is_training=True):
        batch_size = input_ids.size()[0]
        device = input_ids.device

        pool_output = self.bert(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)#[1]
        #score = self.classifier(self.dropout(pool_output))  # [B, 1]
        output = pool_output.logits
        score = pool_output.logits

        if is_training:
            similarities = torch.cat((score[0::2], score[1::2]), dim=1)  # [bs//2, 2]
            labels = torch.zeros((batch_size//2,), dtype=torch.int64).to(device)  # [bs//2,]

            loss_fct = nn.CrossEntropyLoss(reduce=False)
            loss = loss_fct(similarities, labels)  # [bs//2]
            if weight is None:
                weight = torch.ones((batch_size//2,), dtype=torch.float).to(device)
            loss_origin = loss.sum() / (batch_size//2)
            loss_weight = torch.dot(weight, loss) / (batch_size//2)

            output = loss_origin, loss_weight, output
        return output