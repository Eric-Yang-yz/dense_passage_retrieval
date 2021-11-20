import numpy as np
from collections import Counter


class BM25Retriever:

    def __init__(self, passages, k1=2, k2=1, b=0.75): # input split passages
        self.passages = passages
        self.passage_num = len(passages)
        self.avg_passage_len = sum(len(p) for p in passages) / self.passage_num
        self.f = []
        self.idf = {}
        self.k1 = k1
        self.k2 = k2
        self.b = b
        self.preprocess()

    def preprocess(self): # figure out f & idf in passages
        df = {}
        for p in self.passages:
            tmp = {}
            for word in p:
                tmp[word] = tmp.get(word, 0) + 1
            self.f.append(tmp)
            for key in tmp.keys():
                df[key] = df.get(key, 0) + 1
        for key, value in df.items():
            self.idf[key] = np.log((self.passage_num - value + 0.5) / value + 0.5)

    def get_score(self, idx, query):
        score = 0.0
        cur_passage_len = len(self.passages[idx])
        qf = Counter(query)
        for q in query:
            if q not in self.f[idx]:
                continue
            score += self.idf[q] * (self.f[idx][q] * (self.k1 + 1) / (
                    self.f[idx][q] + self.k1 * (1 - self.b + self.b * cur_passage_len / self.avg_passage_len))) * (
                             qf[q] * (self.k2 + 1) / (qf[q] + self.k2))
        return score
