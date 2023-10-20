export GLUE_DIR=/datasets/
export TASK_NAME=rte
export MODEL_NAME_OR_PATH=roberta-large
NUMCLIENTS=100
NUMEPOCHS=100
SAMPLERATE=0.1
LOCALEPS=2
OPTIM=fadamw
MOM=0.9
SLR=0.05
LRs="4e-2"


# We first calculate the important prompt layers for partial prompt tuning.

# Step 1: Use the following code to obtain the importance ranking of prompt layers:
export CUDA_VISIBLE_DEVICES=0
python encoder-only-roberta-large/run_getscore.py \
      --model_type roberta \
      --model_name_or_path $MODEL_NAME_OR_PATH \
      --task_name $TASK_NAME \
      --do_train \
      --do_eval \
      --do_infer \
      --do_lower_case \
      --data_dir "$GLUE_DIR" \
      --log_dir ./logs/ \
      --output_dir ./ckpts/ \
      --max_seq_length 512 \
      --per_gpu_train_batch_size 8 \
      --per_gpu_eval_batch_size 8 \
      --learning_rate $LR \
      --personalization \
      --server_lr $SLR \
      --dirichlet_alpha 1.0 \
      --optim $OPTIM \
      --momentum $MOM \
      --weight_decay 0.1 \
      --logging_steps 100 \
      --num_train_epochs $NUMEPOCHS  \
      --warmup_rate 0.06 \
      --num_prompt_tokens 20 \
      --prompt_layer_list None \
      --evaluate_during_training \
      --overwrite_output_dir \
      --data_partition_method "dirichlet_label" \
      --seed 42 \
      --num_clients $NUMCLIENTS \
      --sample_rate $SAMPLERATE \
      --local_eps $LOCALEPS &

# Step 2: Use the following code to calculate the minimum remaining prompt parameter ratio:
export CUDA_VISIBLE_DEVICES=0
python encoder-only-roberta-large/run_getpr_perclient.py \
      --model_type roberta \
      --model_name_or_path $MODEL_NAME_OR_PATH \
      --task_name $TASK_NAME \
      --do_train \
      --do_eval \
      --do_infer \
      --do_lower_case \
      --data_dir "$GLUE_DIR" \
      --log_dir ./logs/ \
      --output_dir ./ckpts/ \
      --max_seq_length 512 \
      --per_gpu_train_batch_size 8 \
      --per_gpu_eval_batch_size 8 \
      --learning_rate $LR \
      --personalization \
      --server_lr $SLR \
      --dirichlet_alpha 1.0 \
      --optim $OPTIM \
      --momentum $MOM \
      --weight_decay 0.1 \
      --logging_steps 100 \
      --num_train_epochs $NUMEPOCHS  \
      --warmup_rate 0.06 \
      --num_prompt_tokens 20 \
      --prompt_layer_list None \
      --evaluate_during_training \
      --overwrite_output_dir \
      --data_partition_method "dirichlet_label" \
      --seed 42 \
      --num_clients $NUMCLIENTS \
      --sample_rate $SAMPLERATE \
      --local_eps $LOCALEPS &

# By combining the sorted layer indexes from Step 1 and the ratio calculated in Step 2, selected layers for efficient prompt tuning can be obtained
# e.g.: 24 layers of roberta-large in sorted order: [9, 8, 11, 7, 10, 17, 18, 15, 16, 22, 12, 21, 13, 14, 6, 19, 20, 2, 5, 1, 4, 3, 0]    
#       ratio calculated in Step 2: 50%
#       efficient prompt tuning layers for roberta-large model on RTE dataset: [9, 8, 11, 7, 10, 17, 18, 15, 16, 22, 12, 21]



# Step 3: Specify efficient prompt tuning layers with parameter 'prompt_layer_list', and start tuning with the follwoing code
export CUDA_VISIBLE_DEVICES=0
python encoder-only-roberta-large/run_fed_pers_with_optim_v2.py \
    --model_type roberta \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_infer \
    --do_lower_case \
    --data_dir "$GLUE_DIR" \
    --log_dir ./logs/ \
    --output_dir ./ckpts/ \
    --max_seq_length 512 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --learning_rate $LR \
    --personalization \
    --server_lr $SLR \
    --dirichlet_alpha 1.0 \
    --optim $OPTIM \
    --momentum $MOM \
    --weight_decay 0.1 \
    --logging_steps 100 \
    --num_train_epochs $NUMEPOCHS  \
    --warmup_rate 0.06 \
    --num_prompt_tokens 20 \
    --prompt_layer_list "[9, 8, 11, 7, 10, 17, 18, 15, 16, 22, 12, 21]" \
    --evaluate_during_training \
    --overwrite_output_dir \
    --data_partition_method "dirichlet_label" \
    --seed 42 \
    --num_clients $NUMCLIENTS \
    --sample_rate $SAMPLERATE \
    --local_eps $LOCALEPS &

# The above steps are the same for all models and datasets; you just need to replace the corresponding parameters, such as "TASK_NAME", "MODEL_NAME_OR_PATH", and the code's directory, i.e., "encoder-only-roberta-large."



