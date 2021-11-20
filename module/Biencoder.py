import collections
import torch
import torch.nn as nn
from torch import Tensor as T
from typing import List
from transformers import BertModel, BertConfig, BertTokenizer


BiencoderBatch = collections.namedtuple(
    "BiencoderBatch", # B questions, each has one passage (B passages)
    [
        'q_input_ids',      # tensor refer to B questions representation
        'q_token_type_ids',
        'q_attention_mask',
        'p_input_ids',      # tensor refer to B passage representation
        'p_token_type_ids',
        'p_attention_mask',
    ]
)


class Biencoder(nn.Module):

    def __init__(
            self,
            question_model: nn.Module,
            passage_model: nn.Module
    ):
        super(Biencoder, self).__init__()
        self.question_model = question_model
        self.passage_model = passage_model

    def forward(self,
        q_input_ids: T,
        q_token_type_ids: T,
        q_attention_mask: T,
        p_input_ids: T,
        p_token_type_ids: T,
        p_attention_mask: T
    ):
        q_vectors = self.question_model(q_input_ids, q_token_type_ids, q_attention_mask)
        p_vectors = self.passage_model(p_input_ids, p_token_type_ids, p_attention_mask)
        return q_vectors, p_vectors


class BiencoderDatasetTraining:     # this class is for dataloader to generate batch
    def __init__(self, questions: List[str], passages: List[str], tokenizer, max_length):
        self.questions = questions
        self.passages = passages
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        question = self.questions[item]
        passage = self.passages[item]
        with torch.no_grad():
            q_input = self.tokenizer(question, padding=True, return_tensors='pt')
            p_input = self.tokenizer(passage, max_length=self.max_length, truncation=True,
                                     padding='max_length', return_tensors='pt')

        return {
            'q_input_ids': q_input['input_ids'],
            'q_token_type_ids': q_input['token_type_ids'],
            'q_attention_mask': q_input['attention_mask'],
            'p_input_ids': p_input['input_ids'],
            'p_token_type_ids': p_input['token_type_ids'],
            'p_attention_mask': p_input['attention_mask']
        }

    def get_batch(self, batch_size):
        with torch.no_grad():
            batches = []
            for idx, batch_start in enumerate(range(0,len(self.questions),batch_size)):
                batch_questions = self.questions[batch_start : batch_start + batch_size]
                batch_token_tensors = self.tokenizer.batch_encode_plus(batch_questions, padding=True, return_tensors='pt')


                q_ids_batch = batch_token_tensors['input_ids']
                q_seg_batch = batch_token_tensors['token_type_ids']
                q_attn_batch = batch_token_tensors['attention_mask']

                batch_passages = self.passages[batch_start : batch_start + batch_size]
                batch_token_tensors = self.tokenizer.batch_encode_plus(batch_passages, padding='max_length', truncation=True,
                                                                       max_length=self.max_length, return_tensors='pt')
                p_ids_batch = batch_token_tensors['input_ids']
                p_seg_batch = batch_token_tensors['token_type_ids']
                p_attn_batch = batch_token_tensors['attention_mask']

                batches.append({
                    'q_input_ids': q_ids_batch,
                    'q_token_type_ids': q_seg_batch,
                    'q_attention_mask': q_attn_batch,
                    'p_input_ids': p_ids_batch,
                    'p_token_type_ids': p_seg_batch,
                    'p_attention_mask': p_attn_batch,
                    # 'BM25_negative': ???
                })

        return batches

# BatchInstance = collections.namedtuple(
#     "BiencoderBatchInstance",   # B instances in a batch, each instance has one question, one positive passage
#     [                           # B-1 negative passages
#         'q_input_ids',
#         'q_token_type_ids',
#         'q_attention_mask',
#         'p_input_ids',
#         'p_token_type_ids',
#         'p_attention_mask',
#         'is_positive'           # List[bool]
#     ]
# )
#
#
# class BiencoderBatchInstance:   # from batch generate batch instances
#     def __init__(self, batch_data):
#         self.q_input_ids = batch_data['q_input_ids']
#         self.q_token_type_ids = batch_data['q_token_type_ids']
#         self.q_attention_mask = batch_data['q_attention_mask']
#         self.p_input_ids = batch_data['p_input_ids']
#         self.p_token_type_ids = batch_data['p_token_type_ids']
#         self.p_attention_mask = batch_data['p_attention_mask']
#
#     def get_train_instance(self):
#         instances = []
#         for i in range(len(self.q_input_ids)):
#             is_positive = []
#             for j in range(len(self.p_input_ids)):
#                 if i == j:
#                     is_positive.append(True)
#                 else:
#                     is_positive.append(False)
#             instances.append({
#                 'q_input_ids': self.q_input_ids[i],
#                 'q_token_type_ids': self.q_token_type_ids[i],
#                 'q_attention_mask': self.q_attention_mask[i],
#                 'p_input_ids': self.p_input_ids,
#                 'p_token_type_ids': self.p_token_type_ids,
#                 'p_attention_mask': self.p_attention_mask,
#                 'is_positive': is_positive
#             })
#         return instances
#
