import argparse
import json
import numpy as np
import ast
import sys
import random


def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        '--model_type',
        default='roberta',
        type=str,
        
        choices=['bert', 'roberta', 'deberta']
    )
    parser.add_argument(
        "--data_dir",
        default='/home/tzc0029/emnlp2023/LPT/datasets/late-prompt/few-shot/100-samples/seed-13',
        type=str,
        
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default='roberta-large',
        type=str,
        
        help="Path to pre-trained model or shortcut name.",
    )    
    parser.add_argument(
        "--output_dir",
        default='./ckpts/roberta-large/few-shot',
        type=str,
        
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--log_dir",
        default='./logs/roberta-large/few-shot',
        type=str,
        
        help="The output directory where the logs will be written.",
    )
    # Other parameters
    parser.add_argument(
        "--task_name",
        default="rte",
        type=str,
        help="Task name for finetuning.",
    )    
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--debug", action="store_true", help="Whether to use debug mode.")
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_infer", action="store_true", help="Whether to run infer on the dev set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=1,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )

    parser.add_argument(
        '--num_prompt_tokens',
        default=20,
        type=int,
        help='The number of tokens in prompt.'
    )   
    parser.add_argument(
        '--add_prompt_layer',
        default=0,
        type=int,
        help='The layer number to add prompt.'
    )
    parser.add_argument(
        '--proj_down_size',
        default=256,
        type=int,
        help='The intermediate size in prompt generator.'
    ) 
    parser.add_argument(
        '--generator_type',
        default=None,
        type=str,
        help='The type of prompt generator to be used.'
    )
    parser.add_argument(
        '--initialize_from_vocab',
        action="store_true",
        help='Whether or not to initialize the prompt embeddings using vocab tokens.'
    ) 

    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_rate", default=0, type=float, help="Linear warmup over warmup_rate.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.",
    )

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--not_save_model", action="store_true", help="Do not save model checkpoints"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument( "--fp16_opt_level", type=str, default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank",
    )
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features") 

    parser.add_argument("--json_fname", type=str, default="", help="json file name")
    parser.add_argument("--prompt_layer_list", default=None, help="a list of prompt layers")
    parser.add_argument("--ft_idx_list", type=list, default=None, help="indexes of encoder layers for finetuning") 
    parser.add_argument("--mode", type=str, default='pkv_prompt', choices=['vanilla_pt','generator_pt','pkv_prompt'], help="modes") 
    parser.add_argument("--optim", type=str, default='adamw', help="optimizer")
    parser.add_argument("--momentum", type=float, default=0.0, help="momentum in sgd")

    # FL related args
    parser.add_argument("--num_clients", type=int, default=100, help="number of clients in FL") 
    parser.add_argument("--sample_rate", default=0.1, type=float, help="The rate of client sampling")
    parser.add_argument("--local_eps", type=int, default=1, help="number of local epochs in FL") 
    parser.add_argument("--data_partition_method", type=str, default="dirichlet_label", help="data partition method")
    parser.add_argument("--dirichlet_alpha", default=1.0, type=float, help="dirichlet alpha")
    parser.add_argument("--server_optimizer", default="adam", type=str, help="optimizer on server")
    parser.add_argument("--server_lr", default=0.01, type=float, help="server optimizer learining rate")
    parser.add_argument("--personalization", default=False, action="store_true", help="personalization")
    parser.add_argument("--scaff", default=False, help="whether to use scaffold")
    parser.add_argument("--mom", default=False, help="whether to use mom")
    parser.add_argument("--load_from_cache", default=False, help="load from cache or create from scratch")



    args = parser.parse_args()

    for arg in vars(args):
        value = getattr(args, arg)
        if value == "None":
            setattr(args, arg, None)
    
    # read json
    if sys.argv[2].endswith(".json"):
        args = read_from_json(args, args.json_fname)
    if args.prompt_layer_list is None or (args.prompt_layer_list == "None"):
        args.prompt_layer_list = list(range(24)) # make sure to double check the number of layers

    elif args.prompt_layer_list is None or (args.prompt_layer_list == "random"):
        given_number = 24
        half_count = given_number // 2
        all_numbers = list(range(given_number))
        random_numbers = random.sample(all_numbers, half_count)
        args.prompt_layer_list = random_numbers
    else:
        args.prompt_layer_list = ast.literal_eval(args.prompt_layer_list)
    print("tuning layers:")
    for idx in args.prompt_layer_list:
        print(idx)

    if args.prompt_layer_list is not None:
        pass
    else:
        raise ValueError()
    return args    

def read_from_json(args, file_name):
    with open(file_name, 'r') as file:
        json_data = json.load(file)
        for key, value in json_data.items():
            if hasattr(args, key):
                setattr(args, key, value)
    return args