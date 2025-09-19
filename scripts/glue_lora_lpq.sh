#task="qnli"
#index=0
#
#echo "GPU ${index}"
#
#WANDB_PROJECT="20221007" CUDA_VISIBLE_DEVICES="${index}" python run_glue.py \
#    --model_name_or_path roberta-large \
#    --output_dir output_glue_dense_20230924_ranks64_${task} \
#    --task_name ${task} \
#    \
#    --bf16 True \
#    --tf32 True \
#    --do_train \
#    --do_eval \
#    --max_seq_length 128 \
#    --per_device_train_batch_size 32 \
#    --learning_rate 2e-5 \
#    --num_train_epochs 10 \
#    --evaluation_strategy "epoch" \
#    --save_strategy "steps" \
#    --save_steps 0.5 \
#    --logging_steps 1 &
#
#sleep 20m


# --------------------------------------------------------------------------------


#task="qnli"
#index=1
#
#for lora_model_name in "nf3"
#do
#
#for learning_rate in 2e-4 2e-5
#do
#
#echo "GPU ${index}: lora_model_name=${lora_model_name} learning_rate=${learning_rate}"
#
#WANDB_PROJECT="20221007" CUDA_VISIBLE_DEVICES="${index}" python run_glue.py \
#    --model_name_or_path roberta-large \
#    --output_dir output_glue_lora_20230924_ranks64_${task}_${lora_model_name}_${learning_rate} \
#    --task_name ${task} \
#    \
#    --bf16 True \
#    --tf32 True \
#    --do_train \
#    --do_eval \
#    --max_seq_length 128 \
#    --per_device_train_batch_size 32 \
#    --learning_rate ${learning_rate} \
#    --num_train_epochs 10 \
#    --evaluation_strategy "epoch" \
#    --save_strategy "steps" \
#    --save_steps 0.5 \
#    --logging_steps 1 \
#    \
#    --lora_num_ranks 64 \
#    --lora_model_name ${lora_model_name} \
#    --lora_dropout 0.0 \
#    --lora_config "lora" &
#
#index=$(($index+1))
#
#done
#
#wait
#
#done


# --------------------------------------------------------------------------------

task="sst2"
index=3
alg='lora-lpq'
for data in "c4"
do
  for budget in "4"
  do
    echo "GPU ${index}: data=${data}, budget=${budget}, alg=${alg}"
    WANDB_PROJECT="202509" CUDA_VISIBLE_DEVICES="${index}" python run_glue.py \
        --model_name_or_path roberta-large \
        --output_dir _${task}_${data}_${alg}_${budget}_32 \
        --task_name ${task} \
        --bf16 True \
        --tf32 True \
        --pad_to_max_length True \
        --do_train \
        --do_eval \
        --use_bay False \
        --config_data_dir "/mnt/data1/big_file/yerg/quant_conf_mnli_128" \
        --max_seq_length 128 \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 64 \
        --learning_rate 2e-5 \
        --num_train_epochs 20 \
        --with_tracking \
        --report_to tensorboard \
        --save_strategy "steps" \
        --save_steps 0.5 \
        --logging_steps 1 \
        \
        --lora_num_ranks 32 \
        --lora_model_name "roberta-large/lora/${data},budget=${budget}" \
        --lora_dropout 0.0 \
        --lora_config "${alg}" &
  done
done

wait


# --------------------------------------------------------------------------------


#task="qnli"
#index=0
#
#for data in "None" "c4"
#do
#
#index=0
#for budget in "3.25" "3" "2.75" "2.5"
#do
#
#echo "GPU ${index}: data=${data}, budget=${budget}"
#
#WANDB_PROJECT="20221007" CUDA_VISIBLE_DEVICES="${index}" python run_glue.py \
#    --model_name_or_path roberta-large \
#    --output_dir output_glue_lpq_20230924_ranks64_${task}_${data}_${budget} \
#    --task_name ${task} \
#    \
#    --bf16 True \
#    --tf32 True \
#    --do_train \
#    --do_eval \
#    --max_seq_length 128 \
#    --per_device_train_batch_size 32 \
#    --learning_rate 2e-5 \
#    --num_train_epochs 10 \
#    --evaluation_strategy "epoch" \
#    --save_strategy "steps" \
#    --save_steps 0.5 \
#    --logging_steps 1 \
#    \
#    --lora_num_ranks 64 \
#    --lora_model_name "roberta-large/lpq-64/${data},budget=${budget}" \
#    --lora_dropout 0.0 \
#    --lora_config "lora-lpq" &
#
#index=$(($index+1))
#
#done
#
#wait
#
#done
