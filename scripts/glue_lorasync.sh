task="sst2"
index=3
alg="lorasync"
for data in "c4"
do
  for budget in "2.25"
  do
    echo "GPU ${index}: data=${data}, budget=${budget}, alg=${alg}"
    WANDB_PROJECT="202509" CUDA_VISIBLE_DEVICES="${index}" python run_glue.py \
        --model_name_or_path roberta-large \
        --output_dir _${task}_${data}_${alg}_${budget}_bay_global_select_128 \
        --task_name ${task} \
        \
        --bf16 True \
        --tf32 True \
        --pad_to_max_length True \
        --do_train \
        --do_eval \
        --use_bay True \
        --config_data_dir "/mnt/data1/big_file/yerg/quant_conf_sst2_128" \
        --max_seq_length 128 \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 64 \
        --learning_rate 1e-4\
        --num_train_epochs 20 \
        --with_tracking \
        --report_to tensorboard \
        --save_strategy "steps" \
        --save_steps 0.5 \
        --logging_steps 1 \
        \
        --lora_num_ranks 64 \
        --lora_model_name "roberta-large/lora/${data},budget=${budget}" \
        --lora_dropout 0.0 \
        --lora_config "${alg}" &
  done
done

wait

