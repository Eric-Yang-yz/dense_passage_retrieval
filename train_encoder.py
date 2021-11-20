import torch
import torch.nn as nn
from torch import Tensor as T
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List
from module.Biencoder import Biencoder
from tqdm import tqdm


class BiencoderLoss(nn.Module):
    def __init__(self):
        super(BiencoderLoss, self).__init__()

    def forward(self, q_vectors: T, p_vectors: T):
        score_matrix = torch.mm(q_vectors, torch.transpose(p_vectors,0,1))
        score_softmax = F.softmax(score_matrix, dim=1)
        scores = torch.diag(score_softmax)
        loss = torch.mean(torch.neg(torch.log(scores)))
        return loss


def train_loop(
    batches,
    model: Biencoder,
    optimizer,
    device,
    scheduler=None
):
    model.train()
    avgloss = []
    loss_function = BiencoderLoss()
    with tqdm(total=len(batches), desc='EPOCH ', leave=True, unit_scale=True) as pbar:
        for idx, batch_data in enumerate(batches):
            q_input_ids = batch_data['q_input_ids']
            q_token_type_ids = batch_data['q_token_type_ids']
            q_attention_mask = batch_data['q_attention_mask']

            p_input_ids = batch_data['p_input_ids']
            p_token_type_ids = batch_data['p_token_type_ids']
            p_attention_mask = batch_data['p_attention_mask']

            q_input_ids = q_input_ids.to(device, dtype=torch.long)
            q_token_type_ids = q_token_type_ids.to(device, dtype=torch.long)
            q_attention_mask = q_attention_mask.to(device, dtype=torch.long)

            p_input_ids = p_input_ids.to(device, dtype=torch.long)
            p_token_type_ids = p_token_type_ids.to(device, dtype=torch.long)
            p_attention_mask = p_attention_mask.to(device, dtype=torch.long)

            optimizer.zero_grad()
            q_vectors, p_vectors = model(q_input_ids,q_token_type_ids,q_attention_mask,
                                         p_input_ids,p_token_type_ids,p_attention_mask)
            loss = loss_function(q_vectors, p_vectors)
            loss.backward()
            avgloss.append(loss)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            pbar.update(1)
        print(f"loss:{avgloss}")


def eval_loop(data_loader: DataLoader, model: Biencoder, device):
    model.eval()
    with torch.no_grad():
        for idx, batch_data in enumerate(data_loader):
            q_input_ids = batch_data['q_input_ids']
            q_token_type_ids = batch_data['q_token_type_ids']
            q_attention_mask = batch_data['q_attention_mask']

            p_input_ids = batch_data['p_input_ids']
            p_token_type_ids = batch_data['p_token_type_ids']
            p_attention_mask = batch_data['p_attention_mask']

            q_input_ids = q_input_ids.to(device, dtype=torch.long)
            q_token_type_ids = q_token_type_ids.to(device, dtype=torch.long)
            q_attention_mask = q_attention_mask.to(device, dtype=torch.long)

            p_input_ids = p_input_ids.to(device, dtype=torch.long)
            p_token_type_ids = p_token_type_ids.to(device, dtype=torch.long)
            p_attention_mask = p_attention_mask.to(device, dtype=torch.long)

            q_vectors, p_vectors = model(q_input_ids,q_token_type_ids,q_attention_mask,
                                         p_input_ids,p_token_type_ids,p_attention_mask)
            loss_function = BiencoderLoss()
            loss = loss_function(q_vectors, p_vectors)


