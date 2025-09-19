index=0
hf_quantization_method="gptq-3bit"
alg="lorasync"
model="#meta-llama/Llama-2-7b-hf"
for data in "c4"
do
  for budget in "3.5"
  do
    model_tag=$(echo $model | tr '/' '_')
    echo "GPU ${index}: data=${data}, budget=${budget}, alg=${alg}, model=${model_tag}"
    WANDB_PROJECT="202509" CUDA_VISIBLE_DEVICES="${index}" python run_clm.py \
        --model_name_or_path "${model}" \
        --output_dir /mnt/data1/big_file/yerg/llama_results/_${model_tag}_${data}_${alg}_${budget}_3 \
        --dataset_name "c4" \
        --block_size 1024 \
        --bf16 True \
        --tf32 True \
        --num_train_epochs 0.5 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --eval_steps 0.34 \
        --max_eval_samples 30 \
        --do_train \
        --do_eval \
        --use_bay False \
        --config_data_dir "/mnt/data1/big_file/yerg/quant_conf_llama2-7" \
        --max_seq_length 128 \
        --learning_rate 1e-4\
        --num_train_epochs 3 \
        --report_to tensorboard \
        --save_strategy "steps" \
        --save_steps 0.5 \
        --logging_steps 1 \
        \
        --lora_num_ranks 64 \
        --lora_model_name "llama-2-7b/lora/${data},budget=${budget}" \
        --lora_dropout 0.0 \
        --low_cpu_mem_usage True \
        --hf_quantization_method ${hf_quantization_method} \
        --lora_config "${alg}" &
  done
done

wait
