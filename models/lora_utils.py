import os
import math
import torch
import click
import time
from transformers import (
    GPTQConfig,
    LlamaForCausalLM,
    BitsAndBytesConfig,
    RobertaForSequenceClassification)
import csv
from torch.utils.data import Dataset, DataLoader
from transformers.trainer import Trainer
from torch.utils import _pytree as pytree
from peft.tuners import lora
from peft import (
    LoraConfig,
    PeftModelForCausalLM,
    PeftModelForSequenceClassification,
    get_peft_model,
    prepare_model_for_kbit_training)
from typing import List, Optional, Union, Dict, Any, cast
from torch.utils.data import Subset, random_split
from models import misc_utils
from models import allocation_utils as allocation_utils_LLaMA
from models import allocation_utils_2 as allocation_utils_RoBERTa
from models import quantization_utils_2
from models import tensor_container_utils
from models.lq_utils import (
    QuantConfig,
    maybe_sparsify_or_quantize,
    lowrank_quantized_sparse_decomposition_maybe_cast)
import copy
from optimize_config_lora import train_and_optimize_candidate, estimate_storage_from_config_numel, sequence_to_config_dict, QuantConfigDataset



def get_model_best_lora(
        loss_table,
        num_models):
    if num_models is None:
        num_models = max(model_idx for model_idx, _ in loss_table.keys()) + 1

    model_best_lora = [-1] * num_models
    model_best_loss = [float('inf')] * num_models

    for (model_idx, lora_idx), loss in loss_table.items():
        if loss < model_best_loss[model_idx]:
            model_best_loss[model_idx] = loss
            model_best_lora[model_idx] = lora_idx

    return model_best_lora
def get_hf_quantization_config(method: Optional[str], sequence_length: int) -> Dict[str, Any]:

    if method is None:
        return {}

    if method == "bnb-4bit":
        click.secho("Loading in 4-bit", fg="red")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
        return {"device_map": "auto",
                "quantization_config": bnb_config}

    if method == "gptq-3bit":
        click.secho("Loading in GPTQ 3-bit", fg="red")
        gptq_config = GPTQConfig(
            bits=3,
            dataset="c4",
            model_seqlen=sequence_length,
            disable_exllama=True)
        return {"device_map": "auto",
                "quantization_config": gptq_config}

    if method == "gptq-4bit":
        click.secho("Loading in GPTQ 4-bit", fg="red")
        gptq_config = GPTQConfig(
            bits=4,
            dataset="c4",
            model_seqlen=sequence_length,
            disable_exllama=True)
        return {"device_map": "auto",
                "quantization_config": gptq_config}

    raise ValueError(f"Unknown quantization method: {method}")


def save_full_model(trainer: Trainer) -> None:
    if not isinstance(trainer.model, (PeftModelForCausalLM, PeftModelForSequenceClassification)):
        raise TypeError(
            f"Expected `PeftModelForCausalLM`, or "
            f"`PeftModelForSequenceClassification`, "
            f"but got {type(trainer.model)}")
    if not trainer.args.should_save:
        return

    state_dict = trainer.model.state_dict()
    file_name = os.path.join(
        trainer.args.output_dir,
        "full_model.pth")
    torch.save(state_dict, file_name)
    click.secho(f"Saved model state dict to {file_name}", fg="green")


def load_peft_model(
    model: LlamaForCausalLM,
    checkpoint_dir: str,
    checkpoint_preprocess_embedding: bool = False,
) -> PeftModelForCausalLM:
    if not isinstance(model, LlamaForCausalLM):
        raise TypeError(f"Expected LlamaForCausalLM, but got {type(model)}")

    peft_model = PeftModelForCausalLM.from_pretrained(
        model=model,
        model_id=checkpoint_dir,
        is_trainable=True)
    checkpoint_path = os.path.join(
        checkpoint_dir,
        "full_model.pth")
    # We need to load the state dict to CUDA instead of CPU.
    # This is because for `QuantizedTensor`, `tensor.device`
    # is not reflective of the actual underlying device. Rather,
    # we set this attribute to the device of the tensor during
    # creation/quantization (which is usually GPU). Unfortunately,
    # loading the tensor to CPU will not update this attribute.
    state_dict = torch.load(
        checkpoint_path,
        map_location=torch.device("cuda"))
    # The above command makes sure that the underlying tensors are on GPU, and
    # the folllowing command asserts that the tensor "behaves" as if it's on GPU
    if pytree.tree_all_only(
        tensor_container_utils.QuantizedTensor,
        lambda qtensor: qtensor.device.type == "cuda",
        state_dict) is False:
        raise ValueError

    if checkpoint_preprocess_embedding is True:
        # LLaMA-specific ad-hoc fix.
        state_dict = _checkpoint_handle_mismatched_embedding_shape_for_LLaMA(
            model=peft_model,
            state_dict=state_dict)

    # Unfortunately, `load_state_dict` does in-place copy behind the scene, but we
    # cannot in-place copy a `QuantizedTensor` into a `torch.Tensor`. Instead, we will
    # first do out-of-place assignment (so they are compatible), before in-place copy.
    transform_lora_layers_for_loading(
        model=peft_model,
        state_dict=state_dict,
        device="cuda")

    # Note: PyTorch 2.1 has a new feature that allows us to load a state dict via assignment,
    # `new_model.load_state_dict(state_dict, assign=True)`. Unfortunately, this is still
    # bit fragile, so we will do the old-school way instead.
    peft_model.load_state_dict(state_dict)

    click.secho(
        f"Loaded \n"
        f"- PEFT model from {checkpoint_dir}\n"
        f"- state dict from {checkpoint_path}",
        fg="green")

    # Mostly to get the type-checker to stop complaining
    if not isinstance(peft_model, PeftModelForCausalLM):
        raise TypeError(f"Expected PeftModelForCausalLM, but got {type(peft_model)}")
    return peft_model


def transform_lora_layers_for_loading(
    model: PeftModelForCausalLM,
    state_dict: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None,
) -> None:
    if not isinstance(model, PeftModelForCausalLM):
        raise TypeError

    for name, submodule in model.named_modules():

        # This implicitly assumes that `LoraLayer`
        # do not include `LoraLayer` within the module.
        if isinstance(submodule, lora.LoraLayer):

            # We will move the submodule to the GPU first
            # before we do the transformation. This is
            # because after transformation, moving data
            # to GPU is not supported yet.
            if device is not None:
                # This is in-place
                submodule.to(device=device)

            transform_lora_layer_for_loading(
                name=name,
                patch=True,
                module=submodule,
                qweight=state_dict[f"{name}.weight"])


def _checkpoint_handle_mismatched_embedding_shape_for_LLaMA(
    model: PeftModelForCausalLM,
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if not isinstance(model, PeftModelForCausalLM):
        raise TypeError
    if not isinstance(model.base_model.model, LlamaForCausalLM):
        raise TypeError

    lm_head_weight_ = state_dict["base_model.model.lm_head.weight"]
    lm_head_weight = model.base_model.model.lm_head.weight
    lm_head_weight = lm_head_weight.to(device=lm_head_weight_.device)
    embed_tokens_weight_ = state_dict["base_model.model.model.embed_tokens.weight"]
    embed_tokens_weight = model.base_model.model.model.embed_tokens.weight
    embed_tokens_weight = embed_tokens_weight.to(device=embed_tokens_weight_.device)

    if lm_head_weight.shape != (
        lm_head_weight_.shape[0] + 1,
        lm_head_weight_.shape[1]):
        raise ValueError
    if embed_tokens_weight.shape != (
        embed_tokens_weight_.shape[0] + 1,
        embed_tokens_weight_.shape[1]):
        raise ValueError
    if not (lm_head_weight[:-1, :] == lm_head_weight_).all():
        raise ValueError
    if not (embed_tokens_weight[:-1, :] == embed_tokens_weight_).all():
        raise ValueError

    state_dict["base_model.model.lm_head.weight"] = lm_head_weight
    state_dict["base_model.model.model.embed_tokens.weight"] = embed_tokens_weight
    return state_dict


def _enable_gradient_checkpointing(model: LlamaForCausalLM) -> None:
    # For backward compatibility
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()


def prepare_model_for_lora(
    model: LlamaForCausalLM,
    num_ranks: int,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    use_gradient_checkpointing: bool = False,
    checkpoint_dir: Optional[str] = None,
    checkpoint_preprocess_embedding: bool = False,
) -> PeftModelForCausalLM:

    # if not isinstance(model, LlamaForCausalLM):
    #     raise TypeError(f"Expected LlamaForCausalLM, but got {type(model)}")
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ]

    click.secho(
        f"Applying LoRA with the following configurations:\n"
        f"\t -num_ranks: {num_ranks}\n"
        f"\t -lora_alpha: {lora_alpha}\n"
        f"\t -lora_dropout: {lora_dropout}\n"
        f"\t -target_modules: {target_modules}",
        fg="blue")

    peft_config = LoraConfig(
        r=num_ranks,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM")

    # This function is useful even if we are not doing int8 training.
    # (1) Freezing all the parameters before we add LoRA.
    # (2) Casting all fp16/bf16 parameters to fp32.
    # (3) Including gradient checkpointing when the model is loaded in 4/8-bit and some details.
    new_model = prepare_model_for_kbit_training(
        model=model,
        use_gradient_checkpointing=use_gradient_checkpointing)
    # The above only enables gradient checkpointing for BNB-quantized models
    # https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py#L81C1-L117C17
    if use_gradient_checkpointing is True:
        _enable_gradient_checkpointing(new_model)
    if checkpoint_dir is not None:
        click.secho(
            f"Loading PEFT model from {checkpoint_dir}. "
            f"Aforementioned arguments will be ignored",
            fg="blue")
        new_model = load_peft_model(
            model=new_model,
            checkpoint_dir=checkpoint_dir,
            checkpoint_preprocess_embedding=checkpoint_preprocess_embedding)
    else:
        new_model = get_peft_model(new_model, peft_config)
    new_model.print_trainable_parameters()
    if not isinstance(new_model, PeftModelForCausalLM):
        raise TypeError(f"Expected PeftModelForCausalLM, but got {type(new_model)}")
    return new_model


def prepare_model_for_lora_classification(
    model: RobertaForSequenceClassification,
    num_ranks: int,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    use_gradient_checkpointing: bool = False,
) -> PeftModelForSequenceClassification:

    if not isinstance(model, RobertaForSequenceClassification):
        raise TypeError(f"Expected RobertaForSequenceClassification, but got {type(model)}")
    if target_modules is None:
        target_modules = [
            "query",
            "key",
            "value",
            "output.dense",
            "intermediate.dense",
        ]

    click.secho(
        f"Applying LoRA with the following configurations:\n"
        f"\t -num_ranks: {num_ranks}\n"
        f"\t -lora_alpha: {lora_alpha}\n"
        f"\t -lora_dropout: {lora_dropout}\n"
        f"\t -target_modules: {target_modules}",
        fg="blue")

    peft_config = LoraConfig(
        r=num_ranks,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_CLS")

    # This function is useful even if we are not doing int8 training.
    # (1) Freezing all the parameters before we add LoRA.
    # (2) Casting all fp16/bf16 parameters to fp32.
    # (3) Including gradient checkpointing when the model is loaded in 4/8-bit and some details.
    new_model = prepare_model_for_kbit_training(
        model=model,
        use_gradient_checkpointing=use_gradient_checkpointing)
    # The above only enables gradient checkpointing for BNB-quantized models
    # https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py#L81C1-L117C17
    if use_gradient_checkpointing is True:
        _enable_gradient_checkpointing(new_model)
    new_model = get_peft_model(new_model, peft_config)
    # new_model = PeftMixedModel(new_model, peft_config=peft_config)
    new_model.print_trainable_parameters()
    if not isinstance(new_model, PeftModelForSequenceClassification):
        raise TypeError(f"Expected PeftModelForSequenceClassification or PeftMixedModel, but got {type(new_model)}")
    return new_model

import os
import json
import random
from collections import defaultdict
from tqdm import tqdm
from torch import nn

from dataclasses import dataclass, asdict
import re

def parse_layer_pos_and_module(full_name,
                               target_modules = ["query", "key", "value", "output.dense","intermediate.dense"]):
    import re

    m = re.search(r"layer\.(\d+)", full_name)
    layer_pos = int(m.group(1)) if m else 0
    parts = full_name.split('.')

    last_two = '.'.join(parts[-2:])
    if last_two in target_modules:
        module = last_two
    else:
        last_one = parts[-1]
        if last_one in target_modules:
            module = last_one
        else:
            raise ValueError(f"Module name '{last_one}' or '{last_two}' not in target_modules")

    return layer_pos, module

def patch_lora_forward_with_quant_embedding(model, qconfig_dict, bit, target_modules = ["query",  "key", "value", "output.dense", "intermediate.dense"]):
    import types
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            module.quant_embedding = model.quant_embedding

            def new_forward(self, x, full_module_name=name, bit=bit):

                match = re.search(r"layers\.(\d+)", full_module_name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx > 31:
                        config = {
                            "num_bits": 8,
                            "num_bits_0": 8,
                            "num_bits_1": "bf16",
                            "block_size_0": 64,
                            "block_size_1": 256
                        }
                    else:
                        config = qconfig_dict[full_module_name]
                else:
                    config = qconfig_dict[full_module_name]
                layer_pos, module = parse_layer_pos_and_module(full_module_name, target_modules=target_modules)

                Q = self.quant_embedding(config, layer_pos, module, bit/8)
                A = self.lora_A["default"].weight  # shape: [r, in_features]
                B = self.lora_B["default"].weight  # shape: [out_features, r]
                scaling = self.scaling["default"] if hasattr(self, "scaling") else 1.0
                dropout = self.lora_dropout["default"] if hasattr(self, "lora_dropout") else nn.Identity()

                x_proj = dropout(x) @ A.t()  # shape: [bsz, r]
                x_proj = x_proj @ Q.t()
                delta = x_proj @ B.t()  # shape: [bsz, out_features]
                delta = delta * scaling

                output = x @ self.weight.t() + delta
                if hasattr(self, "bias") and self.bias is not None:
                    output += self.bias

                return output

            module.forward = types.MethodType(new_forward, module)

class QuantConfigEmbedding(nn.Module):
    def __init__(self, device, r=8, target_modules = ["query",  "key", "value", "output.dense", "intermediate.dense"]):
        super().__init__()
        emb_dim = 8
        self.target_modules = target_modules
        self.num_bits_list = [2, 3, 4, 8]
        self.num_bits_1_list = ["bf16", "fp16", "fp32"]
        self.block_size_0_list = [16, 32, 64]
        self.block_size_1_list = [16, 64, 256]
        self.module_embed = nn.Embedding(len(self.target_modules), emb_dim)
        self.num_bits_embed = nn.Embedding(len(self.num_bits_list), emb_dim)
        self.num_bits_0_embed = nn.Embedding(len(self.num_bits_list), emb_dim)
        self.num_bits_1_embed = nn.Embedding(len(self.num_bits_1_list), emb_dim)
        self.block_size_0_embed = nn.Embedding(len(self.block_size_0_list), emb_dim)
        self.block_size_1_embed = nn.Embedding(len(self.block_size_1_list), emb_dim)
        self.scale = nn.Parameter(torch.tensor(1.0))
        # self.position_embed = nn.Embedding(24, emb_dim)  #  Bloom
        self.position_embed = nn.Embedding(32, emb_dim)  # Llama 7B
        # self.position_embed = nn.Embedding(28, emb_dim)  # Qwen 1.5B
        # self.position_embed = nn.Embedding(36, emb_dim)  # Qwen 3B
        self.device = device
        # self.global_bit = nn.Sequential(
        #     nn.Linear(1, emb_dim),
        #     nn.ReLU(),
        #     nn.Linear(emb_dim, emb_dim),
        #     nn.ReLU(),
        #     nn.Linear(emb_dim, emb_dim)
        # )
        # MLP
        input_dim = emb_dim * 7
        hidden_dim = 128
        self.r = r
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, r * r),
        )
        nn.init.zeros_(self.mlp[-1].bias)
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)

    def _index_of(self, x, lst):
        # 将具体值转成索引，没有找到抛异常
        if x not in lst:
            raise ValueError(f"Value {x} not in list {lst}")
        return lst.index(x)

    def forward(self, config, layer_pos, module, bit):

        num_bits_idx = torch.tensor([self._index_of(config['num_bits'], self.num_bits_list)], device=self.device)
        num_bits_0_idx = torch.tensor([self._index_of(config['num_bits_0'], self.num_bits_list)], device=self.device)
        num_bits_1_idx = torch.tensor([self._index_of(config['num_bits_1'], self.num_bits_1_list)], device=self.device)
        block_size_0_idx = torch.tensor([self._index_of(config['block_size_0'], self.block_size_0_list)], device=self.device)
        block_size_1_idx = torch.tensor([self._index_of(config['block_size_1'], self.block_size_1_list)], device=self.device)
        modules_idx = torch.tensor([self._index_of(module, self.target_modules)], device=self.device)
        layer_pos_idx = torch.tensor([layer_pos], device=self.device)

        mod = self.module_embed(modules_idx)
        nb = self.num_bits_embed(num_bits_idx)
        nb0 = self.num_bits_0_embed(num_bits_0_idx)
        nb1 = self.num_bits_1_embed(num_bits_1_idx)
        bs0 = self.block_size_0_embed(block_size_0_idx)
        bs1 = self.block_size_1_embed(block_size_1_idx)
        pos_emb = self.position_embed(layer_pos_idx)
        # global_bit_emb = self.global_bit(bit).unsqueeze(0)


        x = torch.cat([nb, nb0, nb1, bs0, bs1, pos_emb, mod], dim=-1).squeeze(0)  # shape (input_dim,)
        out = self.mlp(x)  # shape (r*r)
        out = out.view(self.r, self.r)  # reshape成矩阵

        return  self.scale * out + torch.eye(self.r, device=out.device)


@dataclass
class QuantConfig:
    num_bits: int
    num_bits_0: int
    num_bits_1: str
    block_size_0: int
    block_size_1: int

def transform_lora_layers(
    lpq: bool,
    model: Union[PeftModelForCausalLM, PeftModelForSequenceClassification],
    model_name: str,
    num_iterations: int = 100,
    num_oversampling: int = 10,
    randomized: bool = True,
    given_budget: float = 0.0,
    device: Optional[torch.device] = None,
    idx: int = -1,
    serialized_qconfig = {}
) -> None:

    click.secho(
        f"Transforming LoRA layers with the following configurations:\n"
        f"\t -lpq: {lpq}\n"
        f"\t -model_name: {model_name}\n"
        f"\t -num_iterations: {num_iterations}\n"
        f"\t -num_oversampling: {num_oversampling}\n"
        f"\t -randomized: {randomized}",
        fg="blue")

    if isinstance(model, PeftModelForCausalLM):
        qconfig_dict, sensitivity_dict = (
            allocation_utils_LLaMA
            .create_qconfig_and_sensitivity_dict_LLaMA(
                identifier=model_name, given_budget=given_budget))
    elif isinstance(model, PeftModelForSequenceClassification):
        qconfig_dict, sensitivity_dict = (
            allocation_utils_RoBERTa
            .create_qconfig_and_sensitivity_dict_RoBERTa(
                identifier=model_name, given_budget=given_budget))
    else:
        raise NotImplementedError(f"Unknown model type: {type(model)}")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.bias = None
    #################################################
    serialized_qconfig.update({
        name: asdict(qconfig) for name, qconfig in qconfig_dict.items()
    })
     ###########################################################

    for name, submodule in model.named_modules():

        # This implicitly assumes that `LoraLayer`
        # do not include `LoraLayer` within the module.
        if isinstance(submodule, lora.LoraLayer):

            match = re.search(r"layers\.(\d+)", name)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx > 31:
                    continue
            # These operations will be too slow on CPU
            if device is not None:
                # This is in-place
                submodule.to(device=device)

            assert_lora_Linear_layer(submodule)
            # qconfig = qconfig_dict[name]
            qconfig = qconfig_dict[name]
            if lpq is False:
                # print(f"{name:<50}\tqconfig={qconfig}")
                transform_lora_layer(
                    submodule,
                    qconfig=qconfig)
            else:
                num_ranks = cast(
                    int,
                    submodule.r[submodule.active_adapter])
                # Possibly empty
                sensitivity = None

                # print(f"{name:<50}\tqconfig={qconfig}")
                transform_lora_layer_lpq(
                    submodule,
                    num_ranks=num_ranks,
                    num_iterations=1,
                    num_oversampling=num_oversampling,
                    randomized=False,
                    qconfig=qconfig,
                    W=sensitivity,
                    heuristic="two-sided")

    # ===== Save qconfig_dict and model =====
    save_path = "/mnt/data1/big_file/yerg/quant_conf_llama2-7"
    os.makedirs(save_path, exist_ok=True)

    # Save qconfig_dict
    with open(os.path.join(save_path, f"qconfig_dict_{idx}.json"), "w") as f:
        json.dump(serialized_qconfig, f, indent=2)

    # Save model weights
    torch.save(model, os.path.join(save_path, f"model_state_dict_{idx}.pth"))
    click.secho(
        f"\nSaved qconfig_dict and model to {save_path}",
        fg="green"
    )



@torch.no_grad()
def transform_lora_layer_lpq(
    module: lora.LoraLayer,
    num_ranks: int,
    num_iterations: int,
    num_oversampling: int,
    randomized: bool,
    qconfig: Optional[QuantConfig],
    W: Optional[torch.Tensor] = None,
    heuristic: Optional[str] = None,
) -> None:
    if type(module) is lora.Linear:
        L1, L2, Q, _ = lowrank_quantized_sparse_decomposition_maybe_cast(
            module.weight,
            num_ranks=num_ranks,
            num_iterations=num_iterations,
            num_oversampling=num_oversampling,
            randomized=randomized,
            qconfig=qconfig,
            W=W,
            heuristic=heuristic)
        replace_weight_(
            module=module,
            new_weight=Q)
    else:
        raise TypeError

    # The LoRA layer essentially does the following computation:
    # ```
    # 1. x_ = dropout(x)
    # 2. y  = x @ W.T + s * x_ @ A.T @ B.T
    # When dropout is turned off,
    #    y = x @  W.T +  s * x @ A.T @          B.T
    #      = x @ (W.T +  s *     A.T @          B.T)
    #      = x @ (W   +  s *     B   @          A  ).T
    #      = x @ (W   + [sqrt(s) B]  @ [sqrt(s) A] ).T
    # ```
    # Since LPQ applies the following computation: `W + L1 @ L2`, we want
    # ```
    # 1. L1 = sqrt(s) B
    # 2. L2 = sqrt(s) A
    # ```
    # Hence we assign
    # ```
    # 1. A = L2 / sqrt(s)
    # 2. B = L1 / sqrt(s)
    # ```
    scale_sqrt = math.sqrt(module.scaling[module.active_adapter])
    module.lora_A[module.active_adapter].weight.copy_(L2 / scale_sqrt)
    module.lora_B[module.active_adapter].weight.copy_(L1 / scale_sqrt)


@torch.no_grad()
def transform_lora_layer(
    module: lora.LoraLayer,
    qconfig: Optional[QuantConfig],
) -> None:

    if type(module) is lora.Linear:
        Q = maybe_sparsify_or_quantize(
            module.weight,
            qconfig=qconfig)
        replace_weight_(
            module=module,
            new_weight=Q)
    else:
        raise TypeError


@torch.no_grad()
def transform_lora_layer_for_loading(
    name: str,
    patch: bool,
    module: lora.LoraLayer,
    qweight: tensor_container_utils.QuantizedTensor,
) -> None:

    if not isinstance(qweight, tensor_container_utils.QuantizedTensor):
        raise TypeError

    if patch is True:
        click.secho(f"[Fast Dequantization]: {name}", fg="blue")
        quantization_utils_2.patch_qtensor_for_fast_dequantization_(
            qtensor=qweight)

    if type(module) is lora.Linear:
        replace_weight_(
            module=module,
            new_weight=qweight)
    else:
        raise TypeError


def assert_lora_Linear_layer(
    module: lora.Linear,
) -> None:
    if type(module) is not lora.Linear:
        raise TypeError
    if module.fan_in_fan_out:
        raise ValueError
    if (len(module.lora_embedding_A) != 0 or
        len(module.lora_embedding_B) != 0):
        raise ValueError
    if (module.bias is not None or
        module.lora_A[module.active_adapter].bias is not None or
        module.lora_B[module.active_adapter].bias is not None):

        raise ValueError

    lora_B_weight = module.lora_B[module.active_adapter].weight
    lora_B_weight_all_zeros = (lora_B_weight == 0.).all().item()
    if not lora_B_weight_all_zeros:
        raise ValueError("Expected `module.lora_B.weight` to be zero.")

def find_nth_linear(model, n_from_last=1):
    linear_layers = [(name, m) for name, m in model.base_model.named_modules() if isinstance(m, torch.nn.Linear)]
    if n_from_last > len(linear_layers):
        raise RuntimeError("Not enough linear layers")

    name, module = linear_layers[-n_from_last]
    return module


def collect_last_layer_activation(model, dataloader, device="cuda"):
    last_linear = find_nth_linear(model)
    last_activation = {}

    def hook_fn(module, input, output):
        last_activation['out'] = output.detach().to(torch.float16).cpu()  # 可改 float32 如需更高精度

    hook = last_linear.register_forward_hook(hook_fn)

    model.eval()
    batch = next(iter(dataloader))
    inputs = batch['input_ids'].to(device)
    attention_mask = batch.get('attention_mask', None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        model(input_ids=inputs, attention_mask=attention_mask)

    hook.remove()
    return last_activation['out']

def backup_lora_weights(model):
    backup = {}
    for name, module in model.base_model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            A_weights = {}
            B_weights = {}
            for key in module.lora_A.keys():
                A_weights[key] = module.lora_A[key].weight.data.clone()
            for key in module.lora_B.keys():
                B_weights[key] = module.lora_B[key].weight.data.clone()
            backup[name] = (A_weights, B_weights)
    return backup

from torch import nn
def get_lora_svd_from_diff(model1, model2, r, target_modules=None):

    if target_modules is None:
        target_modules = [
            "query",
            "key",
            "value",
            "output.dense",
            "intermediate.dense",
        ]

    A_svd_layers = {}
    B_svd_layers = {}

    for name, param in model1.named_parameters():
        if not any(t in name for t in target_modules):
            continue
        if ("lora_A" in name) or ("lora_B" in name):
            continue
        if "weight" in name and name in model2.state_dict():
            W1 = param.data
            W2 = model2.state_dict()[name].data
            if W1.ndim < 2:
                continue
            delta_W = W2 - W1
            U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)

            U_r = U[:, :r]
            S_r = S[:r]
            Vh_r = Vh[:r, :]

            sqrt_S = torch.sqrt(S_r)
            B_svd = U_r @ torch.diag(sqrt_S)   # shape: [m, r]
            A_svd = torch.diag(sqrt_S) @ Vh_r  # shape: [r, n]

            lora_A_key = name.replace(".weight", ".lora_A.default.weight")
            lora_B_key = name.replace(".weight", ".lora_B.default.weight")

            A_svd_layers[lora_A_key] = A_svd
            B_svd_layers[lora_B_key] = B_svd

    return A_svd_layers, B_svd_layers



def select_first_point_by_minimax(dist_matrix: torch.Tensor):
    """
    选初始点：计算每个点到所有点的最大距离，选最大距离最小的点
    也就是找“中心”点，保证初始点尽可能覆盖得好
    """
    max_dists = dist_matrix.max(dim=1).values
    return torch.argmin(max_dists).item()

def strict_k_center_greedy_with_init(dist_matrix: torch.Tensor, k: int):
    """
    严格贪心k-center算法，选初始点为最大最小距离最小点，
    每步选能最大程度降低最大最小距离的点
    """
    n = dist_matrix.size(0)
    selected = []
    not_selected = set(range(n))
    first = select_first_point_by_minimax(dist_matrix)
    # first = torch.randint(0, n, (1,)).item()
    selected.append(first)
    not_selected.remove(first)

    min_dist_to_S = dist_matrix[first, :].clone()

    while len(selected) < k:
        best_candidate = None
        best_max_min = float('inf')

        for candidate in not_selected:
            candidate_row = dist_matrix[candidate, :]
            new_min_dist = torch.minimum(min_dist_to_S, candidate_row)
            max_min_dist = new_min_dist.max().item()

            if max_min_dist < best_max_min:
                best_max_min = max_min_dist
                best_candidate = candidate

        selected.append(best_candidate)
        not_selected.remove(best_candidate)
        min_dist_to_S = torch.minimum(min_dist_to_S, dist_matrix[best_candidate, :])

    return selected

from peft import get_peft_model_state_dict
from tqdm import tqdm
from peft import get_peft_model
def extract_lora_weights(model):
    # lora_state_dict = get_peft_model_state_dict(model)
    return {k: v.clone().detach() for k, v in model.named_parameters() if v.requires_grad}

def train_one_epoch(
    model,
    optimizer,
    train_dataloader,
    eval_dataloader,
    accelerator,
    training_args,
    data_args,
    metric,
    run_eval,
    logger,
    task_to_metrics,
    epoch,
    starting_epoch,
    resume_step=None,
    checkpointing_steps=500,
    performace_dict=None,
    total_loss=0.0,
    is_regression=False,
    metric_name = None
):
    completed_steps = 0
    model.train()
    progress_bar = tqdm(range(training_args.max_train_steps), desc="Training")

    dataset_size = len(train_dataloader)
    for step, batch in enumerate(train_dataloader):
        if training_args.resume_from_checkpoint and epoch == starting_epoch:
            if resume_step is not None and step < resume_step:
                completed_steps += 1
                continue
        start_time = time.time()
        with accelerator.accumulate(model):
            # 1️⃣ Forward
            t1 = time.time()
            outputs = model(**batch)
            forward_time = time.time() - t1

            # 2️⃣ Loss
            t2 = time.time()
            loss = outputs.loss
            loss_time = time.time() - t2

            total_loss += loss.detach().float()

            # 4️⃣ Backward
            t3 = time.time()
            accelerator.backward(loss)
            backward_time = time.time() - t3

            # 5️⃣ Clip Gradients
            t4 = time.time()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            clip_time = time.time() - t4

            # 6️⃣ Optimizer Step
            t5 = time.time()
            optimizer.step()
            optimizer.zero_grad()
            opt_time = time.time() - t5
        total_step_time = time.time() - start_time

        if step % 10 == 0:
            accelerator.print(
                f"Epoch: {epoch} | Step: {completed_steps} | Loss: {loss.item():.4f}\n"
                f"Time (s) - Forward: {forward_time:.3f}, Backward: {backward_time:.3f}, "
                f"Clip: {clip_time:.3f}, Optimizer: {opt_time:.3f}, Total: {total_step_time:.3f}"
            )

        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1

        # Step-level checkpointing
        if isinstance(checkpointing_steps, int):
            if completed_steps % checkpointing_steps == 0:
                output_dir = f"step_{completed_steps}"
                if training_args.output_dir is not None:
                    output_dir = os.path.join(training_args.output_dir, output_dir)
                accelerator.save_state(output_dir)

        if completed_steps >= training_args.max_train_steps:
            break
        # Logging
        if completed_steps % 500 == 0 and step % training_args.gradient_accumulation_steps == 0:
            logger.info(f"The current loss is {loss.item():.4f}")


        if completed_steps == dataset_size and step % training_args.gradient_accumulation_steps == 0:
            model.eval()
            if metric_name is None:
                eval_metric = run_eval(eval_dataloader, model, accelerator, is_regression, metric=metric)
                metric_field = task_to_metrics[data_args.task_name] if data_args.task_name else "accuracy"
            else:
                eval_metric = run_eval(eval_dataloader, model, accelerator)
                metric_field = "perplexity"

            logger.info(
                f"seed {training_args.seed} lr {training_args.learning_rate} epoch {epoch}: {eval_metric}"
            )

            if performace_dict is not None:
                performace_dict[completed_steps] = eval_metric[metric_field]

            csv_path = os.path.join(training_args.output_dir, "metrics.csv")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

            write_header = not os.path.exists(csv_path)

            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", metric_field])
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "epoch": epoch,
                    metric_field: eval_metric[metric_field]
                })

            # accelerator.log(
            #     {
            #         metric_field: eval_metric[metric_field],
            #         "train_loss": total_loss.item() / len(train_dataloader),
            #         "epoch": epoch,
            #         "step": completed_steps,
            #     },
            #     step=completed_steps,
            # )

from peft import set_peft_model_state_dict
from peft.tuners.lora import LoraModel
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
import eco2ai
import warnings
import pandas as pd
import numpy as np
from collections import defaultdict
from collections import Counter
warnings.filterwarnings("ignore", category=FutureWarning)



def check_lora_distribution(model_best_lora, m):

    all_lora_indices = list(range(m))
    lora_counts = Counter(model_best_lora)
    for lora_index in all_lora_indices:
        if lora_index not in lora_counts:
            lora_counts[lora_index] = 0

    n = len(model_best_lora)
    min_count = math.floor(n / m)

    for lora, count in lora_counts.items():
        if count < min_count:
            return False

    return True

def load_acc_history(csv_path):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return {}

    acc_history_dict = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        model_indices = [int(h.split('_')[0][1:]) for h in reader.fieldnames if h.startswith('M') and h.endswith('_Accuracy')]
        for idx in model_indices:
            acc_history_dict[idx] = []

        for row in reader:
            for idx in model_indices:
                fieldname = f"M{idx}_Accuracy"
                acc_val = float(row[fieldname])
                acc_history_dict[idx].append(acc_val)

    return acc_history_dict

from peft import get_peft_model, PeftModel
import torch.nn.functional as F
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import random


def compute_similarity_matrix(lora_model_grad, lora_idx, lora_model_grad_count, target_model_indices):
    gradients = []
    valid_mask = []

    for i in target_model_indices:
        count = lora_model_grad_count[lora_idx][i]
        if count == 0:
            gradients.append(None)
            valid_mask.append(False)
        else:
            grad = lora_model_grad[lora_idx][i] / count
            gradients.append(grad)
            valid_mask.append(True)

    num_models = len(gradients)
    sim_matrix = torch.zeros((num_models, num_models))

    for i in range(num_models):
        for j in range(i, num_models):
            if valid_mask[i] and valid_mask[j]:
                sim = F.cosine_similarity(
                    gradients[i].unsqueeze(0),
                    gradients[j].unsqueeze(0),
                    dim=1
                ).item()
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

    mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
    valid_values = sim_matrix[mask][sim_matrix[mask] != 0]
    avg_sim = valid_values.mean().item() if len(valid_values) > 0 else 0.1

    for i in range(num_models):
        if not valid_mask[i]:
            sim_matrix[i, :] = avg_sim
            sim_matrix[:, i] = avg_sim
    sim_matrix.fill_diagonal_(1.0)

    return sim_matrix


def load_quant_model(model_path, qconfig_path):
    model = torch.load(model_path)
    with open(qconfig_path, 'r') as f:
        qconfig_dict = json.load(f)
    return model, qconfig_dict

def strip_prefix(name, prefix="base_model.model."):
    if name.startswith(prefix):
        return name[len(prefix):]
    return name

def fast_non_dominated_sort(Y):
    """Return Pareto rank (0 = best, 1 = second front, ...) for each point in Y."""
    n = Y.shape[0]
    ranks = torch.full((n,), -1, dtype=torch.long)
    domination_counts = torch.zeros(n, dtype=torch.long)
    dominated_sets = [[] for _ in range(n)]
    fronts = [[]]
    # Compare every pair of points
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(Y[p], Y[q]):
                dominated_sets[p].append(q)
            elif dominates(Y[q], Y[p]):
                domination_counts[p] += 1
        if domination_counts[p] == 0:
            ranks[p] = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_sets[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    ranks[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return ranks

def dominates(y1, y2):
    """Check if y1 dominates y2 (for minimization)."""
    return torch.all(y1 <= y2) and torch.any(y1 < y2)
num_bits_list = [2, 3, 4, 8]
num_bits_1_list = ["bf16", "fp16", "fp32"]
block_size_0_list = [16, 32, 64]
block_size_1_list = [16, 64, 256]


def segmented_pareto_selection(all_y, storage_dim=0, bin_start=2.25, bin_end=6.5, bin_step=0.25):

    storage_costs = -all_y[:, storage_dim]

    bins = torch.arange(bin_start, bin_end + bin_step, bin_step)

    selected_indices = []

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]

        idx_in_bin = ((storage_costs >= low) & (storage_costs < high)).nonzero(as_tuple=True)[0]
        if len(idx_in_bin) == 0:
            continue

        y_in_bin = all_y[idx_in_bin]
        ranks = fast_non_dominated_sort(-y_in_bin)
        pareto_front_local = (ranks == 0).nonzero(as_tuple=True)[0]

        selected_indices.extend(idx_in_bin[pareto_front_local].tolist())

    return selected_indices

def fake_cuda(self, *args, **kwargs):
    return self.to("cpu")

def train_one_epoch_lora_generate(
    train_dataloader,
    model,
    param_shapes,
    dummy_model_plus_lora,
    num_meta_model,
    eval_dataloader,
    accelerator,
    training_args,
    data_args,
    model_args,
    metric,
    trainable_modules,
    optimizer,
    run_eval,
    logger,
    task_to_metrics,
    epoch,
    starting_epoch,
    target_modules,
    resume_step=None,
    checkpointing_steps=500,
    performace_dict=None,
    total_loss = 0.0,
    is_regression= False,
    base_path = "/mnt/data1/big_file/yerg/quant_conf_mn",
    value_dict=None,
    meta_data_numbers=None,
    meta_data_numbers_select=None,
    all_x_ref=[],
    all_y_ref=[],
    metric_name=None,
    model_total_params= 301989888,
    d=720
):
    completed_steps = 0
    progress_bar = tqdm(range(training_args.max_train_steps), desc=f"Epoch {epoch}")
    dataset_size = len(train_dataloader)
    BATCH_ACCUM_SIZ = 2
    for step, batch in enumerate(train_dataloader):
        if training_args.resume_from_checkpoint and epoch == starting_epoch:
            if resume_step is not None and step < resume_step:
                completed_steps += 1
                continue
        batch = {k: v.to(training_args.device) for k, v in batch.items()}

        idx_list = random.sample(meta_data_numbers_select, BATCH_ACCUM_SIZ)
        for idx in idx_list:
            state_path = os.path.join(base_path, f"model_state_dict_{idx}.pth")
            qconfig_path = os.path.join(base_path, f"qconfig_dict_{idx}.json")
            q_model, qconfig_dict = load_quant_model(state_path, qconfig_path)
            q_model.add_module("quant_embedding",
                             QuantConfigEmbedding(device=training_args.device,
                                                  r=model_args.lora_num_ranks,
                                                  target_modules=target_modules))
            bit_tensor = torch.tensor([value_dict[idx]]).to(training_args.device)
            patch_lora_forward_with_quant_embedding(q_model, qconfig_dict, bit_tensor, target_modules)
            q_model = q_model.to(training_args.device)

            q_model.load_state_dict(trainable_modules, strict=False)
            # 2️⃣ ⏱️ forward
            q_model.train()
            outputs = q_model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            # 3️⃣ ⏱️ backward
            accelerator.backward(loss)
            q_named = dict(q_model.named_parameters())

            m_named = dict(model.named_parameters())
            with torch.no_grad():
                for name in q_named:
                    if name in m_named and q_named[name].grad is not None:
                        grad = q_named[name].grad.detach().cpu()
                        if m_named[name].grad is None:
                            m_named[name].grad = grad.clone().to(training_args.device)
                        else:
                            m_named[name].grad += grad.to(training_args.device)
            del q_model

        # 4️⃣ ⏱️ step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        with torch.no_grad():
            trainable_modules.update(extract_lora_weights(model))

        if step % 10 == 0:
            # model_best_lora_true = get_model_best_lora(loss_table, len(model_best_lora))
            accelerator.print(
                f"Epoch {epoch} | Step {completed_steps} | "
                f"Train on model {idx} | "
                f"Loss: {loss.item():.4f}\n"
            )
        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1
        if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
            output_dir = os.path.join(training_args.output_dir, f"step_{completed_steps}")
            accelerator.save_state(output_dir)
        if completed_steps >= training_args.max_train_steps:
            break
        # Eval and save model using best LoRA per Q model
        if completed_steps == (dataset_size):
            # Checkpoint by epoch
            if data_args.use_bay:
                # 1. Load dataset and dataloaders
                sequence = train_and_optimize_candidate(all_x_ref[0], all_y_ref[0],
                                                               training_args.device, d)
                for seq in sequence:
                    mid_du_model = copy.deepcopy(dummy_model_plus_lora)

                    cfg_dict = sequence_to_config_dict(seq, qconfig_dict.keys())
                    quant_config = {k: QuantConfig(**v) for k, v in cfg_dict.items()}
                    total_storage = 0.0

                    for name, numel in param_shapes.items():
                        qconfig = quant_config[name]
                        cost = estimate_storage_from_config_numel(numel, qconfig)
                        total_storage += cost / model_total_params
                    for name, submodule in mid_du_model.named_modules():
                        if isinstance(submodule, lora.LoraLayer):
                            transform_lora_layer(submodule, qconfig=quant_config[name])

                    first_batch = next(iter(train_dataloader))
                    first_batch = {k: v.cpu() for k, v in first_batch.items()}

                    with torch.no_grad():
                        outputs = mid_du_model(**first_batch)
                        loss = outputs.loss.item()

                    new_y = [-total_storage, -loss]
                    new_y = torch.tensor(new_y, dtype=torch.float32, device=all_y_ref[0].device)
                    new_y = new_y.unsqueeze(0)  # [1, 2]
                    new_x = seq.unsqueeze(0).float()  # 或 .double() 取决于 all_x 的 dtype
                    all_x_ref[0] = torch.cat([all_x_ref[0], new_x], dim=0)
                    all_y_ref[0] = torch.cat([all_y_ref[0], new_y], dim=0)
                    assert mid_du_model is not dummy_model_plus_lora,"⚠ deepcopy fail"
                    state_path = os.path.join(base_path, f"model_state_dict_{num_meta_model[0]}.pth")
                    torch.save(mid_du_model, state_path)
                    del mid_du_model
                    # Save qconfig_dict
                    with open(os.path.join(base_path, f"qconfig_dict_{num_meta_model[0]}.json"), "w") as f:
                        json.dump(cfg_dict, f, indent=2)
                    value_dict[num_meta_model[0]] = total_storage
                    meta_data_numbers.append(num_meta_model[0])
                    num_meta_model[0] += 1


                selected_indices = segmented_pareto_selection(all_y_ref[0], storage_dim=0)
                # selected_indices = list(range(len(all_y_ref[0])))

                target_values = torch.tensor([-2.5, -3.0, -3.5, -4.0], device=all_y_ref[0].device)  # shape: [4]


                selected_y = all_y_ref[0][selected_indices, 0]  # shape: [top_k]

                # diff[i][j] = |target_values[i] - selected_y[j]|
                diff = torch.abs(target_values.unsqueeze(1) - selected_y.unsqueeze(0))  # [4, top_k]


                min_indices_in_selected = diff.argmin(dim=1)  # shape: [4]

                nearest_global_indices = [selected_indices[i.item()] for i in min_indices_in_selected]
                nearest_model_indices = [meta_data_numbers[i] for i in nearest_global_indices]
                save_idx = [meta_data_numbers[i] for i in selected_indices]
                # inplace update
                meta_data_numbers_select[:] = [meta_data_numbers[i] for i in selected_indices]
            if training_args.checkpointing_steps == "epoch":
                output_dir = os.path.join(training_args.output_dir, f"epoch_{epoch}")
                accelerator.save_state(output_dir)
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    save_dict = {
                            k: v.cpu() for k, v in trainable_modules.items()
                    }
                    torch.save(save_dict, os.path.join(output_dir, f"lora_embedding.pth"))
                    if data_args.use_bay:
                        select_idx_path = os.path.join(output_dir, "select_idx.txt")
                        with open(select_idx_path, "w") as f:
                            for idx in save_idx:
                                f.write(f"{idx}\n")
            all_metrics = {}
            if data_args.use_bay:
                eval_idx = nearest_model_indices
            else:
                eval_idx = [5, 15, 25, 35]
            for idx, model_idx in enumerate(eval_idx):
                state_path = os.path.join(base_path, f"model_state_dict_{model_idx}.pth")
                qconfig_path = os.path.join(base_path, f"qconfig_dict_{model_idx}.json")
                q_model, qconfig_dict = load_quant_model(state_path, qconfig_path)
                q_model.add_module("quant_embedding",
                                 QuantConfigEmbedding(device=training_args.device,
                                                      r=model_args.lora_num_ranks,
                                                      target_modules=target_modules))
                bit_tensor = torch.tensor([value_dict[model_idx]]).to(training_args.device)
                patch_lora_forward_with_quant_embedding(q_model, qconfig_dict, bit_tensor, target_modules)
                q_model.to(training_args.device)
                q_model.load_state_dict(trainable_modules, strict=False)
                q_model.eval()
                if metric_name is None:
                    eval_metric = run_eval(eval_dataloader, q_model, accelerator, is_regression, metric=metric)
                    metric_field = task_to_metrics[data_args.task_name] if data_args.task_name else "accuracy"
                else:
                    eval_metric = run_eval(eval_dataloader, q_model, accelerator)
                    metric_field = "perplexity"


                logger.info(f"Model {model_idx} eval metric: {eval_metric[metric_field]}")
                all_metrics[model_idx] = eval_metric[metric_field]
                output_model_path = os.path.join(training_args.output_dir, f"step_{completed_steps}_model_{model_idx}.pth")
                state_dict = accelerator.unwrap_model(q_model).state_dict()
                torch.save(state_dict, output_model_path)
                del q_model
                if performace_dict is not None:
                    for model_idx, metrics in all_metrics.items():
                        performace_dict[f"{completed_steps}_model_{model_idx}"] = metrics

            if performace_dict is not None:
                csv_path = os.path.join(training_args.output_dir, "metrics.csv")

                # Always define fieldnames regardless of file existence
                fieldnames = ["epoch"]
                for model_idx in all_metrics:
                    fieldnames.append(f"M{model_idx}_{metric_field}")
                    fieldnames.append(f"M{model_idx}_bit")

                write_header = not os.path.exists(csv_path)

                row = {"epoch": epoch}
                for idx, (model_idx, metrics) in enumerate(all_metrics.items()):
                    acc = metrics
                    fieldname = f"M{model_idx}_{metric_field}"
                    fieldname_bit = f"M{model_idx}_bit"
                    row[fieldname] = acc
                    if data_args.use_bay:
                        row[fieldname_bit] = selected_y[min_indices_in_selected[idx]].item()
                try:
                    with open(csv_path, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
                        if write_header:
                            writer.writeheader()
                        writer.writerow(row)
                except Exception as e:
                    print(f"[Warning] Failed to write metrics to CSV at epoch {epoch}: {e}")
                # acc_history_dict = load_acc_history(csv_path)


def train_one_epoch_lora_shared_with_tracking(model_set, share_lora, train_dataloader, eval_dataloader, accelerator, training_args, data_args, metric,
                                              run_eval, logger, task_to_metrics, epoch, starting_epoch, optimizer_dict,
                                              resume_step, checkpointing_steps, performace_dict, total_loss, is_regression, metric_name=None):
    completed_steps = 0
    progress_bar = tqdm(range(training_args.max_train_steps), desc=f"Epoch {epoch}")
    dataset_size = len(train_dataloader)
    for step, batch in enumerate(train_dataloader):
        if training_args.resume_from_checkpoint and epoch == starting_epoch:
            if resume_step is not None and step < resume_step:
                completed_steps += 1
                continue
        select_idx = random.randint(0, len(model_set) - 1)
        model = model_set[select_idx]
        model.load_state_dict(share_lora, strict=False)
        model.train()
        optimizer = optimizer_dict[0]
        optimizer.param_groups[0]["params"] = [
            p for n, p in model.named_parameters() if p.requires_grad
        ]
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()
        share_lora.update(extract_lora_weights(model))
        if step % 10 == 0:
            # model_best_lora_true = get_model_best_lora(loss_table, len(model_best_lora))
            accelerator.print(
                f"Epoch {epoch} | Step {completed_steps} | "
                f"Train on model {select_idx} | "
                f"Loss: {loss.item():.4f}\n"
            )
        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1

        if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
            output_dir = os.path.join(training_args.output_dir, f"step_{completed_steps}")
            accelerator.save_state(output_dir)

        if completed_steps >= training_args.max_train_steps:
            break
        if completed_steps == len(train_dataloader)-1:
            all_metrics = {}
            for model_idx, model in enumerate(model_set):
                model.load_state_dict(share_lora, strict=False)
                model.eval()
                if metric_name is None:
                    eval_metric = run_eval(eval_dataloader, model, accelerator, is_regression, metric=metric)
                    metric_field = task_to_metrics[data_args.task_name] if data_args.task_name else "accuracy"
                else:
                    eval_metric = run_eval(eval_dataloader, model, accelerator)
                    metric_field = "perplexity"

                logger.info(f"Model {model_idx} eval metric: {eval_metric[metric_field]}")
                all_metrics[model_idx] = eval_metric[metric_field]

                if performace_dict is not None:
                    for model_idx, metrics in all_metrics.items():
                        performace_dict[f"{completed_steps}_model_{model_idx}"] = metrics

            # Checkpoint by epoch
            if training_args.checkpointing_steps == "epoch":
                output_dir = os.path.join(training_args.output_dir, f"epoch_{epoch}")

                save_dict = {
                    k: v.cpu() for k, v in share_lora.items()
                }
                os.makedirs(output_dir, exist_ok=True)
                torch.save(save_dict, os.path.join(output_dir, f"share_lora.pth"))


            row = {"epoch": epoch}
            fieldnames = ["epoch"]
            for model_idx, metrics in all_metrics.items():
                acc = metrics
                performace_dict[f"epoch{epoch}_model_{model_idx}"] = acc
                fieldname = f"M{model_idx}_Accuracy"
                row[fieldname] = acc
                fieldnames.append(fieldname)

            csv_path = os.path.join(training_args.output_dir, "metrics.csv")
            write_header = not os.path.exists(csv_path)

            try:
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)
            except Exception as e:
                print(f"[Warning] Failed to write metrics to CSV at epoch {epoch}: {e}")

def replace_weight_(
    module: Union[lora.Linear, torch.nn.Linear],
    new_weight: Union[torch.Tensor, tensor_container_utils.QuantizedTensor],
) -> None:
    if isinstance(new_weight, tensor_container_utils.QuantizedTensor):
        if not isinstance(module.weight, torch.nn.Parameter):
            raise TypeError
        if module.weight.requires_grad is not False:
            raise ValueError
        module.weight = torch.nn.Parameter(
            new_weight,
            requires_grad=module.weight.requires_grad)
    else:
        module.weight.copy_(new_weight)


def transform_lora_adapters_nf8(model: PeftModelForCausalLM) -> None:
    if not isinstance(model, PeftModelForCausalLM):
        raise TypeError

    nf8_qconfig = QuantConfig(
        num_bits=8,
        num_bits_0=8,
        num_bits_1="fp32",
        block_size_0=64,
        block_size_1=256)

    click.secho(f"Transforming LoRA adapters with NF8 quantization", fg="blue")
    for name, submodule in model.named_modules():
        # This implicitly assumes that `LoraLayer`
        # do not include `LoraLayer` within the module.
        if isinstance(submodule, lora.LoraLayer):
            print(f"{name:<50}")
            with torch.no_grad():
                if type(submodule) is lora.Linear:
                    submodule_lora_A = submodule.lora_A[submodule.active_adapter]
                    submodule_lora_B = submodule.lora_B[submodule.active_adapter]
                    submodule_lora_A.weight.requires_grad_(False)
                    submodule_lora_B.weight.requires_grad_(False)
                    qLA = maybe_sparsify_or_quantize(
                        submodule_lora_A.weight,
                        qconfig=nf8_qconfig)
                    qLB = maybe_sparsify_or_quantize(
                        submodule_lora_B.weight,
                        qconfig=nf8_qconfig)
                    replace_weight_(
                        module=submodule_lora_A,
                        new_weight=qLA)
                    replace_weight_(
                        module=submodule_lora_B,
                        new_weight=qLB)
                else:
                    raise TypeError
