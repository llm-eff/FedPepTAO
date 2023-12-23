export GLUE_DIR=/datasets/
export TASK_NAME=mrpc # options: boolq, cola, mpqa, mr, mrpc, qnli, rte, sst-2, subj, trec
export MODEL_NAME_OR_PATH=gpt2-large # options: roberta-large, gpt2-large, openlm-research/open_llama_3b, openlm-research/open_llama_7b
NUMCLIENTS=100
NUMEPOCHS=100
SAMPLERATE=0.1
LOCALEPS=2
OPTIM=fadamw
MOM=0.9
SLR=0.05
LR=4e-2


# We first calculate the important prompt layers for partial prompt tuning.

# Step 1: Use the following code to obtain the importance ranking of prompt layers:
export CUDA_VISIBLE_DEVICES=0
python decoder-only-gpt2/run_getscore.py \
      --model_type gpt2 \
      --model_name_or_path $MODEL_NAME_OR_PATH \
      --task_name $TASK_NAME \
      --do_train \
      --do_eval \
      --do_infer \
      --do_lower_case \
      --data_dir "$GLUE_DIR" \
      --log_dir ./logs/ \
      --output_dir ./ckpts/ \
      --max_seq_length 256 \
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
      --num_prompt_tokens 5 \
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
python decoder-only-gpt2/run_getpr_perclient.py \
      --model_type gpt2 \
      --model_name_or_path $MODEL_NAME_OR_PATH \
      --task_name $TASK_NAME \
      --do_train \
      --do_eval \
      --do_infer \
      --do_lower_case \
      --data_dir "$GLUE_DIR" \
      --log_dir ./logs/ \
      --output_dir ./ckpts/ \
      --max_seq_length 256 \
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
      --num_prompt_tokens 5 \
      --prompt_layer_list None \
      --evaluate_during_training \
      --overwrite_output_dir \
      --data_partition_method "dirichlet_label" \
      --seed 42 \
      --num_clients $NUMCLIENTS \
      --sample_rate $SAMPLERATE \
      --local_eps $LOCALEPS &

# By combining the sorted layer indexes from Step 1 and the ratio calculated in Step 2, selected layers for efficient prompt tuning can be obtained
# e.g.: efficient prompt tuning layers for gpt2 model on MRPC dataset: [2, 1, 3, 34, 33, 32, 31, 29, 30, 28, 0, 27, 4, 26, 25, 24, 23, 22]



# Step 3: Specify efficient prompt tuning layers with parameter 'prompt_layer_list', and start tuning with the follwoing code
export CUDA_VISIBLE_DEVICES=0
python decoder-only-gpt2/run_fed_pers_with_optim_v2.py \
    --model_type gpt2 \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_infer \
    --do_lower_case \
    --data_dir "$GLUE_DIR" \
    --log_dir ./logs/ \
    --output_dir ./ckpts/ \
    --max_seq_length 256 \
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
    --num_prompt_tokens 5 \
    --prompt_layer_list "[2, 1, 3, 34, 33, 32, 31, 29, 30, 28, 0, 27, 4, 26, 25, 24, 23, 22]" \
    --evaluate_during_training \
    --overwrite_output_dir \
    --data_partition_method "dirichlet_label" \
    --seed 42 \
    --num_clients $NUMCLIENTS \
    --sample_rate $SAMPLERATE \
    --local_eps $LOCALEPS &

# The above steps are the same for all models and datasets; you just need to replace the corresponding parameters, 
# such as "TASK_NAME", "MODEL_NAME_OR_PATH", and the code's directory, i.e., "encoder-only-roberta-large."



