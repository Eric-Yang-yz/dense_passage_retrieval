import pandas as pd
import numpy as np


def load_corpus(q_path, p_path):
    """ load train/eval queries and passages in MSMARCO """
    df = pd.read_csv(q_path, sep='\t', header=None)
    qid_list = df[0].values
    q_text = df[1].values

    qid_text = {}
    for qid, text in zip(qid_list, q_text):
        qid_text[qid] = text
    print(f"load questions: {len(qid_text)}")

    df = pd.read_csv(p_path, sep='\t', header=None)
    pid_list = df[0].values
    p_text = df[1].values

    pid_text = {}
    for pid, text in zip(pid_list, p_text):
        pid_text[pid] = text
    print(f"load passages: {len(pid_text)}")

    return qid_text, pid_text


def load_positives(qrels_path):
    df = pd.read_csv(qrels_path, sep='\t', header=None)
    qid_list = df[0].values
    pid_list = df[1].values

    pos_qp = {}
    for qid, pid in zip(qid_list, pid_list):
        if qid not in pos_qp:
            pos_qp[qid] = []
        pos_qp[qid].append(pid)
    print("load positive qids: %s" % len(pos_qp))
    return pos_qp


def get_qp_pair(qid_text, pid_text, pos_qp):
    questions = []
    passages = []
    for key, value in qid_text.items():
        questions.append(value)
        pos_pid = pos_qp[key][0]    # if a question has multiple positive passages, select the first
        passages.append(pid_text[pos_pid])
    print("process question-passage pairs:{0} {1}".format(len(questions),len(passages)))
    return questions, passages


def test():
    # reader = pd.read_csv("D:\\D\\MSMARCO_dataset\\Research-master\\NLP\\ACL2021-PAIR\\corpus\\marco\\para.txt",
    #                  sep='\t', iterator=True, header=None)
    # chunk = reader.get_chunk(4252)
    # chunk.to_csv(".\\para.txt", sep='\t', header=None, index=False)
    # pos_qp = load_positives("D:\\D\\MSMARCO_dataset\\Research-master\\NLP\\ACL2021-PAIR\\corpus\\marco\\qrels.train.tsv")
    # output = pd.DataFrame(columns=['a','b'])
    # for i in range(502939):
    #     print(output.shape[0])
    #     if output.shape[0] == 128:
    #         break
    #     for key, value in pos_qp.items():
    #         if i in value:
    #             output = output.append({'a':key, 'b':i}, ignore_index=True)
    # output.to_csv(".\\qrels.txt", sep='\t', header=None, index=False)
    df = pd.read_csv(".\\qrels.txt", sep='\t', header=None)
    qids = df[0].values
    cnt = {}
    for qid in qids:
        if qid not in cnt:
            cnt[qid] = 0
        else:
            print(qid)



if __name__ == '__main__':
    test()