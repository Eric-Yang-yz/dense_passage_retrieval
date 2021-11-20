import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from module.Biencoder import BiencoderDatasetTraining, Biencoder
from module.Encoder import Encoder
from train_encoder import train_loop, eval_loop
import pandas as pd
from tqdm import tqdm
import time

def test():
    print("start main")
    reader = pd.read_csv('D:\D\MSMARCO_dataset\collection\collection.tsv', sep='\t', error_bad_lines=False, iterator=True, header=None, names=['index','passage'])
    chunk: pd.DataFrame = reader.get_chunk(5).reset_index(drop=True)
    chunk.reset_index(drop=True)
    corpus = chunk.passage.values
    corpus = np.array(chunk['passage'])
    print(chunk)
    print(corpus)
    print(type(corpus))



    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print("model gazed!")
    text = ["did animal planet cancel the zoo?","how long is the screening process at uic dental school","I'm focusing on NLP tasks"]
    passage = "The average Walgreens salary ranges from approximately $15,000 per year for Customer Service Associate / Cashier to $179,900 per year for District Manager. Average Walgreens hourly pay ranges from approximately $7.35 per hour for Laboratory Technician to $68.90 per hour for Pharmacy Manager. Salary information comes from 7,810 data points collected directly from employees, users, and jobs on Indeed. The average Walgreens salary ranges from approximately $15,000 per year for Customer Service Associate / Cashier to $179,900 per year for District Manager. Average Walgreens hourly pay ranges from approximately $7.35 per hour for Laboratory Technician to $68.90 per hour for Pharmacy Manager. Salary information comes from 7,810 data points collected directly from employees, users, and jobs on Indeed.The average Walgreens salary ranges from approximately $15,000 per year for Customer Service Associate / Cashier to $179,900 per year for District Manager. Average Walgreens hourly pay ranges from approximately $7.35 per hour for Laboratory Technician to $68.90 per hour for Pharmacy Manager. Salary information comes from 7,810 data points collected directly from employees, users, and jobs on Indeed. The average Walgreens salary ranges from approximately $15,000 per year for Customer Service Associate / Cashier to $179,900 per year for District Manager. Average Walgreens hourly pay ranges from approximately $7.35 per hour for Laboratory Technician to $68.90 per hour for Pharmacy Manager. Salary information comes from 7,810 data points collected directly from employees, users, and jobs on Indeed. The average Walgreens salary ranges from approximately $15,000 per year for Customer Service Associate / Cashier to $179,900 per year for District Manager. Average Walgreens hourly pay ranges from approximately $7.35 per hour for Laboratory Technician to $68.90 per hour for Pharmacy Manager. Salary information comes from 7,810 data points collected directly from employees, users, and jobs on Indeed. The average Walgreens salary ranges from approximately $15,000 per year for Customer Service Associate / Cashier to $179,900 per year for District Manager. Average Walgreens hourly pay ranges from approximately $7.35 per hour for Laboratory Technician to $68.90 per hour for Pharmacy Manager. Salary information comes from 7,810 data points collected directly from employees, users, and jobs on Indeed. yes a a a a a a"
    list_passage = []
    list_passage.append(passage)
    list_passage.append(passage)
    encoded_input = tokenizer.batch_encode_plus(text, padding=True, return_tensors='pt')
    passage_encode = tokenizer.batch_encode_plus(list_passage, add_special_tokens=True, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    last_hidden_states = model(**encoded_input)
    # cls = last_hidden_states[0][:,0,:]
    # cls.squeeze()
    # print('text:%s' % text)
    print('{0}:{1}'.format('encoded_input', encoded_input))
    print('{0}:{1}'.format('passage_encode', passage_encode['input_ids']))
    print(passage_encode['input_ids'].size())
    # print(f'size:{passage_encode["input_ids"].size()}')
    print('output:{}'.format(last_hidden_states.pooler_output))
    # print(f'{type(last_hidden_states)}')
    # print(f'{cls}')
    # print(f'len_cls:{cls.size()[1]}')

    from module.BM25 import BM25Retriever
    split_text = []
    for p in text:
        tokens = tokenizer.tokenize(p)
        print(tokens)
        split_text.append(tokens)
    bm25 = BM25Retriever(split_text)
    query = 'You have asked the question that did animal planet cancel the zoo'
    query = tokenizer.tokenize(query)
    for i, _ in enumerate(text):
        print(f"{i} score: {bm25.get_score(i,query)}")


import faiss


def faiss_test():
    print('start faiss test')
    dim = 3                 # dimension of the search space (768)
    m = 512                 # the connection that each node has (neighbor to store per node)
    ef_search = 128         # the depth of search
    ef_construction = 200   # how much of networks will be searched during the construction

    a = np.array([[1,2,3],[4,5,6],[7,8,9]]).astype('float32')
    a_tensor = torch.Tensor(a)
    print(a_tensor)
    a = np.array(a_tensor).astype('float32')

    index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_L2)
    index.hnsw.efSearch = ef_search
    index.hnsw.efConstruction = ef_construction
    print(index.is_trained)
    index.add(a)
    print(index.ntotal)


    k = 3
    q_tensor = torch.Tensor(np.array([1,2,3]))
    query = np.array(q_tensor.unsqueeze(dim=0)).astype('float32')
    print(f"{type(query)}, {query.dtype}")
    dis, idx = index.search(query, k)
    print(idx)
    print(dis)
    print(idx.shape)

def fake_train(epoch):
    batch_num = int(8e3 / 128)
    with tqdm(total=batch_num, desc='EPOCH '+ str(epoch + 1), leave=True, unit_scale=True) as pbar:
        for i in range(batch_num):
            time.sleep(0.01)
            pbar.update(1)

def tqdm_test():
    for i in range(10):
        fake_train(i)
        print()
        print("loss")

from pandas.io.json._json import JsonReader
def pandas_test():
    print("strat reading dataset")
    with pd.read_json("D:\D\MSMARCO_dataset\\NaturalQuestions\\simplified-nq-train.jsonl", lines=True, chunksize=5, orient='index', numpy=True) as reader:
        print(reader)

    return
    df = pd.read_json("D:\\D\\MSMARCO_dataset\\triviaqa-unfiltered\\unfiltered-web-dev.json")
    dataset = df.head().Data.values
    for data in dataset:
        print(data['Answer']['Value'])
        for p in data['SearchResults']:
            print(f'rank: {p["Rank"]}  Description: {p["Description"]}')


if __name__ == "__main__":
    test()