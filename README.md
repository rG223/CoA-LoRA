# CoA-LoRA Training

This project demonstrates how to generate LoRA adjustments.

## Project Structure

```bash
├── optimize_config_lora.py
├── README.md
├── requirements.txt
├── run_clm.py
├── run_glue.py
│
├── models/
│   ├── allocation_utils.py
│   ├── allocation_utils_2.py
│   ├── distributed_utils.py
│   ├── factorizations_utils.py
│   ├── lora_utils.py
│   ├── lq_utils.py
│   ├── misc_utils.py
│   ├── packbits_utils.py
│   ├── quantization_utils.py
│   ├── quantization_utils_2.py
│   ├── ray_utils.py
│   ├── tensor_container_utils.py
│   ├── utils.py
│   └── __init__.py
│
└── scripts/
    ├── glue_lorasync.sh
    ├── glue_lorasync_llama.sh
    ├── glue_lora_lpq.sh
    ├── glue_lora_share.sh
    └── glue_lqlora.sh
```

## Generating the Training Configuration Set

To generate the training configuration set, follow these steps:

### Prepare the Model for LoRA Classification

Use the `prepare_model_for_lora_classification` function to modify your base model for LoRA adaptation. This sets up the model to accept LoRA adapters for classification tasks.

```python
 for idx, budget in enumerate(np.arange(2.5, 7.3, 0.05)):
     model = init_base_model(model_args, config, tokenizer, training_args, label2id, id2label)
     model_plus_lora = lora_utils.prepare_model_for_lora_classification(
         model=model,
         num_ranks=model_args.lora_num_ranks,
         lora_dropout=model_args.lora_dropout)
     lora_utils.transform_lora_layers(
         lpq=(model_args.lora_config == "lora-lpq"),
         model=model_plus_lora,
         model_name=model_args.lora_model_name,
         device="cuda",
         given_budget=budget,
         idx=idx)
```

## Training configuration-aware model
```python
bash scripts/glue_lorasync.sh
```
