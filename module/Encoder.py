from transformers import BertModel, BertConfig, BertTokenizer
import torch.nn as nn
from torch import Tensor as T

class Encoder(nn.Module):

    def __init__(self, use_pretrained_model=False, bert_path='bert-base-uncased'):
        super(Encoder, self).__init__()
        if (use_pretrained_model):
            self.model = BertModel.from_pretrained(bert_path)
        else:
            config = BertConfig(hidden_dropout_prob=0.1,attention_probs_dropout_prob=0.1)
            self.model = BertModel(BertConfig)
            self.model.init_weights()

    def forward(self, input_ids: T, token_type_ids: T, attention_mask: T): # output of the whole batch
        last_hidden_states = self.model(input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)
        cls_embedding = last_hidden_states.pooler_output
        return cls_embedding
