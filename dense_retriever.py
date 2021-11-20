import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from module.Biencoder import BiencoderDatasetTraining, Biencoder
from module.Encoder import Encoder
from train_encoder import train_loop, eval_loop
from util.dpr_data_process import load_corpus, load_positives, get_qp_pair
import pandas as pd

MAX_LENGTH = 10
BATCH_SIZE = 16
EPOCH = 5
LR = 1e-5
BERT_PATH = 'D:\\D\\MSMARCO_dataset\\bert-base-uncased'


def main():
    qid_text, pid_text = load_corpus(".\\util\\query.txt", ".\\util\\para.txt")
    pos_qp = load_positives(".\\util\\qrels.txt")
    questions, passages = get_qp_pair(qid_text, pid_text, pos_qp)

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    train_dataset = BiencoderDatasetTraining(questions, passages, tokenizer, max_length=MAX_LENGTH)
    batches = train_dataset.get_batch(BATCH_SIZE)

    # use data loader #
    # train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    q_encoder = Encoder(use_pretrained_model=True, bert_path=BERT_PATH)
    p_encoder = Encoder(use_pretrained_model=True, bert_path=BERT_PATH)
    model = Biencoder(question_model=q_encoder, passage_model=p_encoder).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)

    num_train_steps = int(len(train_dataset) / BATCH_SIZE * EPOCH)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=50, num_training_steps=num_train_steps)

    print("start training...")
    for epoch in range(EPOCH):
        train_loop(batches, model, optimizer, device, scheduler)


if __name__ == '__main__':
    main()
