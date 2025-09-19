#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
import copy
import time
from email.policy import default
from peft import set_peft_model_state_dict
import eco2ai
from collections import defaultdict
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
import torch
import logging
import os
import random
from peft import get_peft_model
from tqdm import tqdm
import sys
import json
import click
import warnings
from dataclasses import dataclass, field
from typing import Optional
from torch.utils.data import DataLoader
import datasets
import evaluate
import numpy as np
from datasets import load_dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    get_scheduler,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from optimize_config_lora import QuantConfigDataset
from models import misc_utils
from models import tensor_container_utils
from models import lora_utils
import math

os.environ["WANDB_MODE"] = "disabled"
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.34.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
task_to_metrics = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "f1",
    "qnli": "accuracy",
    "qqp": "f1",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",
}
logger = logging.getLogger(__name__)

def init_base_model(model_args, config, tokenizer, training_args, label2id, id2label):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    model.config.pad_token_id = 0
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    model = model.to(training_args.device)
    model.label2id = label2id
    model.id2label = id2label
    return model


def copy_model_weights(model):
    new_model = type(model)(model.config)  # 用相同config初始化
    new_model.load_state_dict(copy.deepcopy(model.state_dict()))
    return new_model

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    use_bay: Optional[bool] = field(default=True)
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    config_data_dir: Optional[str] = field(
        default= "/mnt/data1/big_file/yerg/quant_conf_mip")

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    with_tracking: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to enable experiment trackers for logging. "
                "If set to True, will log to the default tracker (e.g., TensorBoard, WandB, etc.)"
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


def run_eval(eval_dataloader, model, accelerator, is_regression, metric):
    samples_seen = 0

    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            predictions = model(**batch)

        references = batch["labels"]
        predictions, references = accelerator.gather_for_metrics((predictions, references))

        if predictions.logits.isnan().any():
            print("Warning: some of the output logits for evaluation were NaN!")

        predictions = predictions.logits.argmax(dim=-1) if not is_regression else predictions.logits.squeeze()
        # If we are in a multiprocess environment, the last batch has duplicates
        if step == len(eval_dataloader) - 1:
            predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
            references = references[: len(eval_dataloader.dataset) - samples_seen]
        else:
            samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    return metric.compute()

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    lora_num_ranks: int = field(default=8)
    lora_dropout: float = field(default=0.05)
    lora_config: Optional[str] = field(default=None)
    lora_model_name: Optional[str] = field(default=None)
    hf_quantization_method: Optional[str] = field(default=None)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    accelerator_log_kwargs = {}
    if data_args.with_tracking:
        accelerator_log_kwargs["project_dir"] = training_args.output_dir
        accelerator_log_kwargs['log_with'] = training_args.report_to

    accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps, **accelerator_log_kwargs)
    if model_args.use_auth_token is not None:
        warnings.warn("The `use_auth_token` argument is deprecated and will be removed in v4.34.", FutureWarning)
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.info(accelerator.state, main_process_only=False)
    # only main process output logging when using distributed training
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    # setting logging level
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    if accelerator.is_main_process:
        if training_args.seed is not None:
            set_seed(training_args.seed)
        if training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    accelerator.wait_for_everyone()
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
    model.config.pad_token_id = 0
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    tokenizer.truncation_side = "left"
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(training_args.device)
    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if sorted(label_name_to_id.keys()) == sorted(label_list):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {sorted(label_name_to_id.keys())}, dataset labels: {sorted(label_list)}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    remove_columns = [col for col in raw_datasets["train"].column_names if col not in ["label"]]
    with accelerator.main_process_first():
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            remove_columns=remove_columns,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name)
    elif is_regression:
        metric = evaluate.load("mse")
    else:
        metric = evaluate.load("accuracy")
    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    # def compute_metrics(p: EvalPrediction):
    #     preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    #     preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    #     result = metric.compute(predictions=preds, references=p.label_ids)
    #     if len(result) > 1:
    #         result["combined_score"] = np.mean(list(result.values())).item()
    #     return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    if training_args.do_train:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=training_args.per_device_train_batch_size
        )

    if training_args.do_eval:
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=training_args.per_device_eval_batch_size)
    if model_args.lora_config in ["lora", "lora-lpq", 'lorasync', 'lorashare']:
        click.secho(f"LoRA Finetuning with `{model_args.lora_config}`", bg="yellow")

        if not all([
            model_args.lora_config is not None,
            model_args.lora_model_name is not None,
            model_args.hf_quantization_method is None]):
            raise ValueError
        if model_args.lora_config in ["lora", "lora-lpq"]:
            model = lora_utils.prepare_model_for_lora_classification(
                model=model,
                num_ranks=model_args.lora_num_ranks,
                lora_dropout=model_args.lora_dropout)
            lora_utils.transform_lora_layers(
                lpq=(model_args.lora_config == "lora-lpq"),
                model=model,
                model_name=model_args.lora_model_name,
                device="cuda")
        elif model_args.lora_config == 'lorashare':
            model_idx = [5, 15, 25, 35]
            Model_set = []
            base_path = data_args.config_data_dir
            for idx in model_idx:
                model = torch.load(os.path.join(base_path, f"model_state_dict_{idx}.pth"))
                Model_set.append(model)
            share_lora = lora_utils.extract_lora_weights(Model_set[0])
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=training_args.learning_rate
            )
            model_set = [model.to(training_args.device) for model in Model_set]
            model_set = [accelerator.prepare(model) for model in model_set]
            optimizer_dict = {}
            optimizer_dict[0] = optimizer
        elif model_args.lora_config == 'lorasync':
            # pass
            # label2id = model.config.label2id
            # id2label = model.config.id2label
            # # for idx, budget in enumerate(np.arange(2.28, 6.53, 0.25)):
            # for idx, budget in enumerate(np.arange(2.25, 7.3, 0.05)):
            # # for idx, budget in enumerate(np.arange(6.55, 7.3, 0.05)):
            #     model = init_base_model(model_args, config, tokenizer, training_args, label2id, id2label)
            #     model_plus_lora = lora_utils.prepare_model_for_lora_classification(
            #         model=model,
            #         num_ranks=model_args.lora_num_ranks,
            #         lora_dropout=model_args.lora_dropout)
            #     lora_utils.transform_lora_layers(
            #         lpq=(model_args.lora_config == "lora-lpq"),
            #         model=model_plus_lora,
            #         model_name=model_args.lora_model_name,
            #         device="cuda",
            #         given_budget=budget,
            #         idx=idx)
            # exit()


            param_names = []
            param_shapes = {}
            target_modules = [
                "query",
                "key",
                "value",
                "output.dense",
                "intermediate.dense",
            ]
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):  # 这里筛选你想的模块
                    module.bias = None  # 你可以这么设置
                if any(t in name for t in target_modules):
                    full_name = "base_model.model." + name
                    param_names.append(full_name)
                    # 统计模块内所有参数数量（weight 和 bias）
                    num_params = sum(p.numel() for p in module.parameters())
                    param_shapes[full_name] = num_params

            label2id = model.config.label2id
            id2label = model.config.id2label
            dummy_model = init_base_model(model_args, config, tokenizer, training_args, label2id, id2label)
            for name, module in dummy_model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    module.bias = None
            dummy_model_plus_lora = lora_utils.prepare_model_for_lora_classification(
                model=dummy_model,
                num_ranks=model_args.lora_num_ranks,
                lora_dropout=model_args.lora_dropout)
    elif model_args.lora_config in ["lora-gptq"]:
        raise NotImplementedError
    else:
        click.secho(f"Full Finetuning", bg="yellow")

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True
    training_args.warmup_steps = training_args.warmup_ratio * training_args.max_train_steps
    if model_args.lora_config != 'lorasync':
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=training_args.learning_rate
        )
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )
    elif model_args.lora_config == 'FullFinetuning':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_args.learning_rate
        )
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )
    else:
        values = [round(2.25 + 0.05 * i, 2) for i in range(101)]
        values = [v for v in values if v != 6.5]

        value_dict = {i: v for i, v in enumerate(values)}

        base_path = data_args.config_data_dir
        model = torch.load(os.path.join(base_path, f"model_state_dict_0.pth"))
        qconfig_dict = json.load(open(os.path.join(base_path, f"qconfig_dict_0.json")))
        model.add_module("quant_embedding",
                           lora_utils.QuantConfigEmbedding(device=training_args.device, r=model_args.lora_num_ranks))
        lora_utils.patch_lora_forward_with_quant_embedding(model, qconfig_dict, torch.tensor([2.25]).to(training_args.device))
        model.to(training_args.device)
        trainable_modules = lora_utils.extract_lora_weights(model)

        optimizer = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if p.requires_grad],
            lr=training_args.learning_rate
        )
        train_dataloader, eval_dataloader = accelerator.prepare(train_dataloader, eval_dataloader)

        meta_data_numbers = list(range(0, 100, 2))
        meta_data_numbers_select = list(range(0, 100, 2))
        all_y = []
        dataset = QuantConfigDataset(data_args.config_data_dir, sample_list=meta_data_numbers)
        all_x = dataset.samples
        for _, model_idx in tqdm(enumerate(meta_data_numbers), desc="Evaluating models"):

            state_path = os.path.join(base_path, f"model_state_dict_{model_idx}.pth")
            qconfig_path = os.path.join(base_path, f"qconfig_dict_{model_idx}.json")

            q_model, qconfig_dict = lora_utils.load_quant_model(state_path, qconfig_path)

            q_model.add_module("quant_embedding", lora_utils.QuantConfigEmbedding(
                device=training_args.device,
                r=model_args.lora_num_ranks,
            ))
            bit_tensor = torch.tensor([value_dict[model_idx]]).to(training_args.device)
            lora_utils.patch_lora_forward_with_quant_embedding(q_model, qconfig_dict, bit_tensor)
            q_model.to(training_args.device)

            q_model.load_state_dict(trainable_modules, strict=False)
            q_model.eval()

            qconfig_obj_dict = {k: lora_utils.QuantConfig(**v) for k, v in qconfig_dict.items()}
            total_storage = 0.0
            model_total_params = 301989888
            for name, numel in param_shapes.items():
                qconfig = qconfig_obj_dict[name]
                cost = lora_utils.estimate_storage_from_config_numel(numel, qconfig)
                total_storage += cost / model_total_params


            if model_idx == 0:
                first_batch = next(iter(train_dataloader))
                first_batch = {k: v.to(training_args.device) for k, v in first_batch.items()}

            with torch.no_grad():
                outputs = q_model(**first_batch)
                loss = outputs.loss.item()

            all_y.append([-total_storage, -loss])
        all_y = torch.tensor(all_y, dtype=torch.float32).to(training_args.device)
        all_x = all_x.to(training_args.device)
        all_x_ref = [all_x]
        all_y_ref = [all_y]

    if overrode_max_train_steps:
        training_args.max_train_steps = math.ceil(training_args.num_train_epochs * num_update_steps_per_epoch)
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)
    # Figure out how many steps we should save the Accelerator states
    training_args.checkpointing_steps = 'epoch'
    checkpointing_steps = training_args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)
    if data_args.with_tracking:
        experiment_config = vars(training_args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("glue_no_trainer", experiment_config)
    # Train!
    total_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {training_args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {training_args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(training_args.max_train_steps))
    completed_steps = 0
    starting_epoch = 0
    resume_step = None
    # Potentially load in the weights and states from a previous save
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint is not None or training_args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {training_args.resume_from_checkpoint}")
            accelerator.load_state(training_args.resume_from_checkpoint)
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
    performace_dict = {}
    selection_counter = defaultdict(int)
    # tracker = eco2ai.Tracker(
    #     project_name=f"{model_args.lora_config}_finetune",
    #     experiment_description=f"Fine-tuning with {model_args.lora_config} at small model",
    #     file_name=f"./{training_args.output_dir}/emission.csv"
    # )
    if model_args.lora_config == 'lorasync':
        random.seed(training_args.seed)
        num_meta_model = [100]
    for epoch in range(starting_epoch, training_args.num_train_epochs):
        if training_args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            train_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        total_loss = 0.0
        if model_args.lora_config in ["lora", "lora-lpq", "FullFinetuning"]:
            model.train()
            if data_args.with_tracking:
                lora_utils.train_one_epoch(model, optimizer, train_dataloader, eval_dataloader, accelerator, training_args,
                                            data_args, metric, run_eval, logger, task_to_metrics, epoch, starting_epoch,
                                           resume_step, checkpointing_steps, performace_dict, total_loss, is_regression)
        elif model_args.lora_config == 'lorasync':
            lora_utils.train_one_epoch_lora_generate(train_dataloader, model, param_shapes, dummy_model_plus_lora, num_meta_model,
                                                                 eval_dataloader, accelerator, training_args, data_args,
                                                                 model_args, metric, trainable_modules, optimizer,
                                                                 run_eval, logger, task_to_metrics, epoch, starting_epoch,
                                                                 resume_step, checkpointing_steps, performace_dict,
                                                                 total_loss, is_regression, base_path, value_dict, meta_data_numbers, meta_data_numbers_select,
                                                                 all_x_ref, all_y_ref)
        elif model_args.lora_config == 'lorashare':
            lora_utils.train_one_epoch_lora_shared_with_tracking(model_set, share_lora, train_dataloader, eval_dataloader, accelerator, training_args, data_args, metric,
                                                      run_eval, logger, task_to_metrics, epoch, starting_epoch, optimizer_dict,
                                                      resume_step, checkpointing_steps, performace_dict, total_loss, is_regression)



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
