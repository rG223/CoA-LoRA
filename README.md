# CoA-LoRA Training

This project demonstrates how to generate LoRA adjustments.

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
