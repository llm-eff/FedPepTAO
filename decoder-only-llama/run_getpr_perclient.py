import json
import logging
import os
import random
import sys
import copy
import fitlog

import numpy as np
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler, Subset, random_split, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import is_main_process
import matplotlib.pyplot as plt
import time
from models.modeling_llama import LlamaForCausalLM
from transformers import LlamaConfig, LlamaTokenizer


from arguments import get_args, read_from_json
from data_loader import load_and_cache_data
from data.process import processors, output_modes
from data.partition import partition_gpt2
from evaluations import aggregate_plus, predict, aggregate, quiet_evaluate, ratio_minus, update_client_c, average_weights_plus
from get_score import get_score
from fsgdm import Fsgdm
from fadamw import fAdamW
from get_ratio import get_ratio
from gpt2_evaluations import get_loss, evaluate, predict, personalized_evaluate

logger = logging.getLogger(__name__)

def get_metric_key(task_name):
    if task_name == "cola":
        return "mcc"
    elif task_name == "sst-2":
        return "acc"
    elif task_name == "mrpc":
        return "acc_and_f1"
    elif task_name == "sts-b":
        return "corr"
    elif task_name == "qqp":
        return "acc_and_f1"
    elif task_name == "mnli":
        return "mnli/acc"
    elif task_name == "mnli-mm":
        return "mnli-mm/acc"
    elif task_name == "qnli":
        return "acc"
    elif task_name == "rte":
        return "acc"
    elif task_name == "wnli":
        return "acc"
    elif task_name == "hans":
        return "acc"
    elif task_name == "mpqa":
        return "acc"
    elif task_name == "mr":
        return "acc"
    elif task_name == "subj":
        return "acc"
    elif task_name == "trec":
        return "acc"
    elif task_name == "snli":
        return "acc"
    elif task_name == "wnli":
        return "acc"
    elif task_name == "boolq":
        return "acc"
    else:
        raise KeyError(task_name)

        
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, logger):
    if not args.not_save_model and not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir) 
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size) 
    eval_dataset = load_and_cache_data(args, tokenizer, data_type='dev')

    train_dataloader_list, eval_dataloader_list, n_sample_list = partition_gpt2(args, train_dataset, eval_dataset, logger)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    optimizer_list = []
    scheduler_list = []
    for _ in range(args.num_clients):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]  

        
        if args.warmup_steps > 0:
            num_warmup_steps = args.warmup_steps
        else:
            num_warmup_steps = args.warmup_rate * t_total
        num_warmup_steps = 0


        global_v = {}
    
        for k, v in model.named_parameters():
            if v.requires_grad == True:
                global_v[k] = torch.zeros_like(v.data)


        if args.optim == "fsgdm":
            optimizer = Fsgdm(model, args.learning_rate, args.momentum)
        elif args.optim == "fadamw":
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            # optimizer = fAdamW(model, v=copy.deepcopy(global_v), lr=args.learning_rate, eps=args.adam_epsilon)
        elif args.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
        elif args.optim == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum)
        elif args.optim == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        else:
            raise NotImplementedError()

        optimizer_list.append(optimizer)
        scheduler_list.append(get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=t_total
        ))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level) 

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)


    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0


    best_all_metric = {}
    tr_loss, logging_loss, best = 0.0, 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )

    set_seed(args)  # Added here for reproductibility
    metric_key = get_metric_key(args.task_name)
    if args.task_name == 'mnli':
        metric_key = 'avg_acc'

    global_generator_param = copy.deepcopy(model.get_generator_param())
    local_generator_param_list = [copy.deepcopy(global_generator_param) for _ in range(args.num_clients)]
    global_c = {}
    for k in global_generator_param:
        global_c[k] = torch.zeros_like(global_generator_param[k].data).cpu().detach()

    clients_c = [copy.deepcopy(global_c) for i in range(args.num_clients)]

    plot_acc=[]
    plot_loss=[]
    eval_loss=[]
    time_list_per_epoch=[]
    # initialize the previsous grad
    pre_grad = None
    save_epoch = 5

    for epoch in train_iterator:
        if epoch == save_epoch:
            rate = []
            for client_idx in tqdm(selected_clients):
                for batch_idx, batch in enumerate(train_dataloader_list[client_idx]):
                    start = time.time()
                    r = get_ratio(model, batch, device='cpu')
                    print(r)
                    rate.append(r)
                    print(f"time used in rate calculation {time.time() - start}")
                    logger.info(f"time used in rate calculation {time.time() - start}")
                    break
            average_rate = sum(rate) / len(rate)
            print(f"averaged pruning rate: {average_rate}")
            logger.info(f"averaged pruning rate: {average_rate}")
            print(f"all pruning rates: {rate}")
            logger.info(f"all pruning rates: {rate}")
            return "done"

        m = max(int(args.sample_rate * args.num_clients), 1)
        selected_clients = np.random.choice(range(args.num_clients), m, replace=False)
        
        
        generator_param_list = []
        weight_list = []        
        local_steps = []
        
        original_global_w = copy.deepcopy(global_generator_param)
        time_list_this_epoch = [] # time used for each client
        for client_idx in tqdm(selected_clients):
            weight_list.append(n_sample_list[client_idx])
            if args.personalization:
                model.load_local_param(copy.deepcopy(local_generator_param_list[client_idx]))
                model.load_transferable_param(copy.deepcopy(global_generator_param))
            else:
                model.load_generator_param(global_generator_param)

            client_step = 0
            start = time.time()
            for _ in range(args.local_eps):
                for batch_idx, batch in enumerate(train_dataloader_list[client_idx]):
                    model.train()
                    batch = tuple(t.to(args.device) for t in batch)
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                    }
                    outputs = model(**inputs, return_dict=True)

                    loss = get_loss(args, outputs.logits, batch)
                    
                    if args.method_type != 'discriminative':
                        loss = loss.mean()
                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()    

                    tr_loss += loss.item()
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer_list[client_idx].step()
                    scheduler_list[client_idx].step()
                    client_step += 1                    
                    global_step += 1
            time_list_this_epoch.append(time.time() - start)
            local_steps.append(client_step)

            local_generator_param = model.get_generator_param()
            local_generator_param_list[client_idx] = copy.deepcopy(local_generator_param)
            generator_param_list.append(copy.deepcopy(local_generator_param))
            
        time_list_per_epoch.append(np.max(time_list_this_epoch))
        global_generator_param = aggregate(generator_param_list, weight_list)
        
        if args.personalization:
            model.load_transferable_param(copy.deepcopy(global_generator_param))
        else:
            model.load_generator_param(global_generator_param)
        delta = ratio_minus(original_global_w, global_generator_param, 1)
        delta_c_list = []
        for client_index, idx in enumerate(selected_clients):
            delta_local = ratio_minus(original_global_w, generator_param_list[client_index], 1)
            original_client_c = copy.deepcopy(clients_c[idx])
            clients_c[idx] = update_client_c(global_c, clients_c[idx], delta_local, local_steps[client_index] * args.learning_rate)
            delta_c_list.append((local_steps[client_index], ratio_minus(clients_c[idx], original_client_c, 1)))

        global_c = average_weights_plus(delta_c_list, global_c, ratio=(args.sample_rate))

        args.optim = None
        if args.optim == "fadamw":
            if args.server_optimizer == 'adam':
                if not pre_grad:
                    pre_grad = copy.deepcopy(delta)
                else:
                    local_state = []
                    local_v = []
                    delta_g = ratio_minus(pre_grad, delta, 1)
                    pre_grad = copy.deepcopy(delta)
                    original_w = model.get_generator_param()
                    args.server_lr = max(args.server_lr * 0.99, 0.0001)
                    for client_idx, idx in enumerate(selected_clients):
                        model.load_transferable_param(copy.deepcopy(original_w))
                        server_optimizer = Fsgdm(model, lr=args.server_lr, local_c=clients_c[idx], \
                            global_c=global_c, v=global_v, momentum=0.9)

                        model.zero_grad()
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                param.grad.data.copy_(delta_g["embedding.weight"])

                        
                        server_optimizer.step()
                        model.zero_grad()

                        local_state.append(copy.deepcopy(model.get_generator_param()))
                        local_v.append(copy.deepcopy(server_optimizer.v))
                    
                    updated_global_weights = aggregate(local_state, weight_list)
                    global_v = aggregate(local_v, weight_list)
                    model.load_transferable_param(copy.deepcopy(updated_global_weights))
                    global_generator_param = copy.deepcopy(model.get_generator_param())
                
            elif args.server_optimizer == 'sgdm':
                raise NotImplementedError()
            else:
                pass
        

        do_test = True
        if do_test and (epoch == (save_epoch-1)):
            logs = {}
            if (
                    args.local_rank == -1 and args.evaluate_during_training
            ):  # Only evaluate when single GPU otherwise metrics may not average well
                if args.personalization:
                    results = personalized_evaluate(args, model, 
                                                    tokenizer, 
                                                    eval_dataloader_list,
                                                    local_generator_param_list, 
                                                    global_generator_param
                                                    )
                else:
                    results = evaluate(args, model, tokenizer)
                
                plot_acc.append(results[metric_key])
                
                eval_loss.append(results['loss'])
                res_for_display = {}
                num_metric = 0
                avg_metric = 0
                for k, v in results.items():
                    num_metric += 1
                    avg_metric += v
                    res_for_display[k.replace("-", "_")] = v
                if args.task_name == 'mnli':
                    results[metric_key] = avg_metric / num_metric
                    res_for_display[metric_key] = avg_metric / num_metric
                fitlog.add_metric({"dev": res_for_display}, step=global_step)
                if results[metric_key] > best:
                    best = results[metric_key]
                    best_all_metric.update(results)
                    fitlog.add_best_metric({"dev": {metric_key.replace("-", "_"): best}})
                    # save the best model
                    if not args.not_save_model:
                        output_dir = os.path.join(args.output_dir, "best_model")
                        model_to_save = OrderedDict()
                        for n, p in model.named_parameters():
                            if p.requires_grad:
                                model_to_save[n] = p
                        model.save_pretrained(output_dir, state_dict=model_to_save)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                for key, value in results.items():
                    eval_key = "eval_{}".format(key)
                    logs[eval_key] = value


            loss_scalar = (tr_loss - logging_loss) / np.sum(local_steps)
            learning_rate_scalar = scheduler_list[client_idx].get_last_lr()[0]
            logs["learning_rate"] = learning_rate_scalar
            logs["loss"] = loss_scalar
            plot_loss.append(loss_scalar)
            logging_loss = tr_loss
            fitlog.add_loss(loss_scalar, name="Loss", step=global_step)
            print(json.dumps({**logs, **{"step": global_step}}))

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    
    e = np.arange(len(plot_acc))
    plt.plot(e, plot_acc)
    plt.savefig(args.log_dir + "/acc.png")
    plt.clf()
    plt.plot(e, plot_loss)
    plt.savefig(args.log_dir + "/loss.png")

    num_params = args.num_prompt_tokens * len(args.prompt_layer_list) * 2 * model.config.hidden_size
    num_transfer_params = [num_params for _ in range(len(plot_acc))]

    result_dict = {
        "acc": plot_acc,
        "best_acc": best,
        "training_loss": plot_loss,
        "eval_loss": eval_loss,
        "training_time": time_list_per_epoch,
        "num_transfer_params": num_transfer_params
    }

    logger.info(result_dict)

    logs = {}
    if (
            args.local_rank == -1 and args.evaluate_during_training and
            args.logging_steps > 0 and (global_step-1) % args.logging_steps != 0
    ):
        results = evaluate(args, model, tokenizer)
        res_for_display = {}
        num_metric = 0
        avg_metric = 0
        for k, v in results.items():
            num_metric += 1
            avg_metric += v
            res_for_display[k.replace("-", "_")] = v
        if args.task_name == 'mnli':
            results[metric_key] = avg_metric / num_metric
            res_for_display[metric_key] = avg_metric / num_metric
        fitlog.add_metric({"dev": res_for_display}, step=global_step)
        if results[metric_key] > best:
            best = results[metric_key]
            best_all_metric.update(results)
            fitlog.add_best_metric({"dev": {metric_key.replace("-", "_"): best}}) 
            # save the best model
            if not args.not_save_model:
                output_dir = os.path.join(args.output_dir, "best_model")
                model_to_save = OrderedDict()
                for n, p in model.named_parameters():
                    if p.requires_grad:
                        model_to_save[n] = p
                model.save_pretrained(output_dir, state_dict=model_to_save)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

        for key, value in results.items():
            eval_key = "eval_{}".format(key)
            logs[eval_key] = value

        learning_rate_scalar = scheduler_list[0].get_lr()[0]
        logs["learning_rate"] = learning_rate_scalar

        print(json.dumps({**logs, **{"step": global_step}}))

    if args.local_rank in [-1, 0]:
        fitlog.finish()

    return global_step, tr_loss / global_step, best_all_metric


def main():
    args = get_args()
    # SEARCH
    args.data_dir = os.path.join(args.data_dir, args.task_name)
    args.output_dir = os.path.join(
        args.output_dir, 
        str(args.generator_type),
        str(args.add_prompt_layer),
        str(args.num_prompt_tokens),
        str(args.per_gpu_train_batch_size),
        str(args.learning_rate),
        str(args.warmup_rate),
        str(args.proj_down_size),
        args.task_name
    )


    if not os.path.exists(args.log_dir):
        try:
            os.makedirs(args.log_dir)
        except:
            pass


    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach() 

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device


    if args.debug:
        fitlog.debug()
    if args.local_rank in [-1, 0]:
        fitlog.set_log_dir(args.log_dir)
        fitlog.add_hyper(args)
    if not args.debug:
        fitlog.create_log_folder()
        dir = fitlog.get_log_folder(absolute=True)
        args.log_dir = dir
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    if not args.debug:
        logger = logging.getLogger('my_logger')
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(args.log_dir + '/log_file.log')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)


    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    config = LlamaConfig.from_pretrained(        
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
        # output_hidden_states=True,
        # output_attentions=True
    )
    # prompt related configs
    config.generator_type = args.generator_type
    config.add_prompt_layer = args.add_prompt_layer
    config.prompt_layer_list = args.prompt_layer_list
    config.ft_idx_list = args.ft_idx_list
    config.mode = args.mode
    config.classifier_dropout = 0.1
    config.num_prompt_tokens = args.num_prompt_tokens
    config.proj_down_size = args.proj_down_size
    config.prefix_projection = False
    config.pre_seq_len = config.num_prompt_tokens
    config.personalization = args.personalization

    tokenizer = LlamaTokenizer.from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,        
    )
    model = LlamaForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=config,
        )

    total_model_params = 0
    num_trained_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_trained_params += p.numel()
        else:
            total_model_params += p.numel()

    logger.info("Total Model Parameters: {}, Trainable Parameters: {}".format(total_model_params, num_trained_params))

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)  

    logger.info("Training/evaluation parameters %s", args)

    train_dataset = None
    best_all_metric = None
    if args.do_train:
        train_dataset = load_and_cache_data(args, tokenizer, data_type='train')
        _ = train(args, train_dataset, model, tokenizer, logger)
        return "done"
    if args.do_infer:
        best_model = os.path.join(args.output_dir, "best_model", "pytorch_model.bin")
        model.load_state_dict(torch.load(best_model), strict=False)
        model.to(args.device) 
        results = evaluate(args, model, tokenizer)
        logger.info(results)
    return best_all_metric


if __name__ == "__main__":
    best = main()
    print(best)