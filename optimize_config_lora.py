import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from setuptools.dist import sequence
from torch.utils.data import Dataset, DataLoader, random_split


num_bits_list = [2, 3, 4, 8]
num_bits_1_list = ["bf16", "fp16", "fp32"]
block_size_0_list = [16, 32, 64]
block_size_1_list = [16, 64, 256]

def map_value_to_index(value, value_list):
    return value_list.index(value)

def sequence_to_config_dict(sequence, layer_keys):
    assert len(sequence) == 5 * len(layer_keys)
    cfg_dict = {}
    for i, key in enumerate(layer_keys):
        idx = i * 5
        cfg_dict[key] = {
            "num_bits":      num_bits_list[sequence[idx]],
            "num_bits_0":    num_bits_list[sequence[idx + 1]],
            "num_bits_1":    num_bits_1_list[sequence[idx + 2]],
            "block_size_0":  block_size_0_list[sequence[idx + 3]],
            "block_size_1":  block_size_1_list[sequence[idx + 4]],
        }
    return cfg_dict

def load_sample_from_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    keys = data.keys()


    sample_indices = []
    for k in keys:
        cfg = data[k]
        idx_num_bits = map_value_to_index(cfg["num_bits"], num_bits_list)
        idx_num_bits_0 = map_value_to_index(cfg["num_bits_0"], num_bits_list)
        idx_num_bits_1 = map_value_to_index(cfg["num_bits_1"], num_bits_1_list)
        idx_block_size_0 = map_value_to_index(cfg["block_size_0"], block_size_0_list)
        idx_block_size_1 = map_value_to_index(cfg["block_size_1"], block_size_1_list)

        sample_indices.extend([idx_num_bits, idx_num_bits_0, idx_num_bits_1, idx_block_size_0, idx_block_size_1])

    return torch.tensor(sample_indices)  # (720,)

class QuantConfigDataset(Dataset):
    def __init__(self, data_dir, sample_list):
        self.samples = []
        for i in sample_list:
            path = os.path.join(data_dir, f"qconfig_dict_{i}.json")
            sample = load_sample_from_json(path)
            self.samples.append(sample)
        self.samples = torch.stack(self.samples, dim=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class TaskDecoderModel(nn.Module):
    def __init__(self, num_categories=4, d_model=32, max_seq_len=720):
        super().__init__()
        self.layer_num = 6
        self.block_num = 24
        self.param_per_layer = 5
        # 每个位置的类别数列表
        base_list = [4,4,3,3,3]
        self.num_classes_list = base_list * self.block_num * self.layer_num  # 长度3600
        self.max_vocab_size = max(self.num_classes_list)
        self.d_model = d_model

        self.category_embedding = nn.Embedding(num_categories, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.layer_emb = nn.Embedding(self.layer_num, d_model)
        self.block_emb = nn.Embedding(self.block_num, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=1, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.embedding_ln = nn.LayerNorm(d_model)
        self.pooling_fc = nn.Linear(d_model, d_model)
        self.classification_head = nn.Linear(d_model, self.max_vocab_size)

    def forward(self, input_categories, emb=False):
        device = input_categories.device
        B, S = input_categories.shape
        T = S
        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B, T)

        block_ids = (position_ids // (self.layer_num * self.param_per_layer))
        layer_ids = ((position_ids // self.param_per_layer) % self.layer_num)

        raw_input = self.category_embedding(input_categories)

        enc_input = self.embedding_ln(raw_input)+ self.block_emb(block_ids) \
                                                + self.layer_emb(layer_ids) \
                                                + self.position_embedding(position_ids)
        memory = self.encoder(enc_input)

        pooled = memory.mean(dim=1)
        pooled_transformed = self.pooling_fc(pooled)
        if emb:
            return pooled_transformed

        memory = pooled_transformed.unsqueeze(1).repeat(1, T, 1)
        tgt_position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        tgt_embed = self.position_embedding(tgt_position_ids)

        decoder_output = self.decoder(tgt=tgt_embed, memory=memory)
        logits = self.classification_head(decoder_output)  # (B, T, max_vocab_size)

        invalid_position_mask = torch.ones((T, self.max_vocab_size), dtype=torch.bool, device=device)
        for pos in range(T):
            valid_num = self.num_classes_list[pos]
            invalid_position_mask[pos, :valid_num] = False

        invalid_mask_expanded = invalid_position_mask.unsqueeze(0).expand(B, T, self.max_vocab_size)
        logits = logits.masked_fill(invalid_mask_expanded, float('-inf'))
        return logits

def train_one_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    for input_seq in dataloader:
        input_seq = input_seq.to(device)
        tgt_seq = input_seq.to(device)
        optimizer.zero_grad()
        logits = model(input_seq)  # (B, T, V)
        B, T, V = logits.shape
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_seq.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * input_seq.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for input_seq in dataloader:
            input_seq = input_seq.to(device)
            tgt_seq = input_seq.to(device)
            logits = model(input_seq)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), tgt_seq.view(-1), ignore_index=-100)
            total_loss += loss.item() * input_seq.size(0)
            preds = logits.argmax(dim=2)  # (B,T)
            correct += (preds == tgt_seq).sum().item()
            total += tgt_seq.numel()
    return total_loss / total, correct / total

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def decode_candidate(model, candidate_emb, max_seq_len=720, device='cuda'):
    model.eval()
    with torch.no_grad():
        B = candidate_emb.shape[0]
        S = max_seq_len
        d_model = candidate_emb.size(-1)

        memory = candidate_emb.unsqueeze(1).expand(B, S, d_model).to(device)

        tgt_seq = torch.zeros(B, 1, dtype=torch.long, device=device)

        generated_seq = []
        for i in range(S):
            tgt_position_ids = torch.arange(tgt_seq.size(1), device=device).unsqueeze(0).expand(B, -1)
            tgt_embed = model.position_embedding(tgt_position_ids)

            # decoder调用时要保证 tgt_embed 的 seq_len 对应 tgt_seq 的长度
            decoder_output = model.decoder(tgt=tgt_embed, memory=memory)
            logits = model.classification_head(decoder_output)  # (B, T, V)

            # 添加mask，禁止非法类别
            invalid_position_mask = torch.ones((model.max_vocab_size,), dtype=torch.bool, device=device)
            valid_num = model.num_classes_list[i]
            invalid_position_mask[:valid_num] = False
            next_token_logits = logits[:, -1, :]
            next_token_logits = next_token_logits.masked_fill(invalid_position_mask, float('-inf'))

            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            tgt_seq = torch.cat([tgt_seq, next_token], dim=1)
            generated_seq.append(next_token)

        generated_seq = torch.cat(generated_seq, dim=1)  # (B, S)
        return generated_seq[:, :S].cpu()  # 返回固定长度序列


from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
import numpy as np
from botorch.utils.transforms import normalize, unnormalize
from tqdm import tqdm
from gpytorch.kernels import Kernel

def estimate_storage_from_config_numel(
    n_param,
    qconfig,
) -> float:
    b = float(qconfig.num_bits)
    b0 = float(qconfig.num_bits_0)
    B0 = float(qconfig.block_size_0)
    B1 = float(qconfig.block_size_1)
    if qconfig.num_bits_1 == "fp32":
        b1 = 32.
    elif qconfig.num_bits_1 in ["bf16", "fp16"]:
        b1 = 16.
    else:
        raise ValueError

    overhead_0 = b0 / B0
    overhead_1 = b1 / (B0 * B1)
    total_bits = b + overhead_0 + overhead_1
    return total_bits * n_param

def finite_diff_guided_search(
    model,
    current_x: torch.Tensor,
    all_y: torch.Tensor,
    value_ranges: list[int],
    ref_point: list[float],
    steps: int = 10,
    bounds: torch.Tensor = None,
):
    device = current_x.device
    partitioning = NondominatedPartitioning(
        ref_point=torch.tensor(ref_point, device=device),
        Y=all_y
    )

    x = current_x.clone()
    for step in range(steps):
        grad = finite_diff_gradient(
            x, model, bounds, ref_point, partitioning, value_ranges
        )


        top_dim = torch.argmax(grad.abs())
        direction = 1 if grad[top_dim] > 0 else -1

        new_x = x.clone()
        new_x[top_dim] += direction
        if 0 <= new_x[top_dim] < value_ranges[top_dim]:
            x = new_x
        else:
            break

    return x.long()


def finite_diff_gradient(
    x: torch.Tensor,
    model,
    bounds: torch.Tensor,
    ref_point: list[float],
    partitioning,
    value_ranges: list[int],
):
    d = x.shape[0]
    grad = torch.zeros_like(x, dtype=torch.float)
    acq_func = qLogExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning
    )

    for i in range(d):
        diffs = []
        for delta in [-1, 1]:
            new_x = x.clone()
            new_x[i] += delta

            if 0 <= new_x[i] < value_ranges[i]:
                new_x_norm = normalize(new_x.unsqueeze(0).float(), bounds=bounds)
                score = acq_func(new_x_norm.unsqueeze(1))  # shape: [1, 1]
                diffs.append((delta, score.item()))

        if len(diffs) == 2:
            grad[i] = (diffs[1][1] - diffs[0][1]) / (diffs[1][0] - diffs[0][0])
        elif len(diffs) == 1:
            grad[i] = diffs[0][1] * diffs[0][0]

    return grad

from botorch.acquisition import qLogExpectedImprovement
from gpytorch.kernels import ScaleKernel
from math import sqrt
from gpytorch.kernels import MaternKernel, RBFKernel
from botorch.optim import optimize_acqf_discrete

def train_and_optimize_candidate(
    all_x,
    all_y,
    device=None,
    d =720,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Preparing Data for surrogate modeling...')
    choices_block = torch.tensor([4, 4, 3, 3, 3], device=device)

    repeat_num = d // 5


    value_ranges = choices_block.repeat(repeat_num).tolist()
    kernel = RBFKernel(ard_num_dims=d, lengthscale=torch.tensor([d ** 0.5], device=device))


    all_x = all_x.to(device)
    x_min = all_x.min(dim=0).values.float()
    x_max = all_x.max(dim=0).values.float()
    bounds = torch.stack([x_min, x_max])
    all_x_norm = normalize(all_x, bounds=bounds)


    N = all_x.shape[0]
    y = all_y.double().to(device)  # Replace with real y if available

    gps = []
    for i in range(2):
        gp = SingleTaskGP(all_x_norm.double(), y[:, i:i + 1], covar_module=kernel)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception as e:
            print(f"Fit failed for output {i}:", e)

        gps.append(gp)

    multi_gp = ModelListGP(*gps)
    idx_sample = torch.randperm(N)[:5]
    candidates = []
    for i in idx_sample:
        data = all_x[i]
        try:
            next_candidate = finite_diff_guided_search(
                model=multi_gp,
                current_x=data.long(),
                all_y=y,
                value_ranges=value_ranges,
                ref_point=[0.0, 0.0],
                steps=d,
                bounds=bounds.to(device),
            )
            candidates.append(next_candidate)
        except Exception as e:
            print(f"Error at index {i}: {e}")
            pass

    return candidates

