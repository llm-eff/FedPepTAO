import csv
import logging
import os
import sys
sys.path.append('../')

from collections import OrderedDict
import numpy as np

import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from transformers import glue_compute_metrics

from data_loader import PromptDataset
from data.process import processors
from data.partition import partition
import copy

logger = logging.getLogger(__name__)


task_mappings = {
    'sst-2': 'sst-2',
    'cola': 'cola',
    'mnli': 'mnli',
    'mnli-mm': 'mnli-mm',
    'qqp': 'qqp',
    'qnli': 'qnli',
    'rte': 'rte',
    'mrpc': 'mrpc',
    'mpqa': 'sst-2',
    'mr': 'sst-2',
    'subj': 'sst-2',
    'trec': 'sst-2',
    'snli': 'qnli',
    'wnli': 'wnli',
    'boolq': 'sst-2',
}


def evaluate(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        processor = processors[eval_task]()
        label_ids = []
        label_map = processor.get_label_map()
        for k, v in label_map.items():
            label_id = tokenizer(' ' + v, add_special_tokens=False)['input_ids']
            assert len(label_id) == 1
            label_ids.append(label_id[0])
            
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        eval_dataset = PromptDataset(args, eval_task, tokenizer, data_type='dev')
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=eval_dataset.collate_fn)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[-1],
                }
                inputs["token_type_ids"] = batch[2]
                inputs["mask_pos"] = batch[-2]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                logits = logits[:, label_ids]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                batch_labels = inputs["labels"]
                for i, label_id in enumerate(label_ids):
                    batch_labels[batch_labels == label_id] = i
                out_label_ids = batch_labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                batch_labels = inputs["labels"]
                for i, label_id in enumerate(label_ids):
                    batch_labels[batch_labels == label_id] = i
                out_label_ids = np.append(out_label_ids, batch_labels.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        loss_dict = {'loss':eval_loss}
        results.update(loss_dict)
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = glue_compute_metrics(task_mappings[eval_task], preds, out_label_ids)
        results.update(result)
        

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            print("  %s = %s" % (key, str(result[key])))

    return results


def personalized_evaluate(args, model, tokenizer, eval_dataloader_list, local_generator_param_list, global_generator_param):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        processor = processors[eval_task]()
        label_ids = []
        label_map = processor.get_label_map()
        for k, v in label_map.items():
            label_id = tokenizer(' ' + v, add_special_tokens=False)['input_ids']
            assert len(label_id) == 1
            label_ids.append(label_id[0])
            
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            raise NotImplementedError()
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation *****")
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        
        for client_idx in tqdm(range(args.num_clients)):
            model.load_local_param(copy.deepcopy(local_generator_param_list[client_idx]))
            model.load_transferable_param(copy.deepcopy(global_generator_param))
            

            for batch in eval_dataloader_list[client_idx]:
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": batch[-1],
                    }
                    inputs["token_type_ids"] = batch[2]
                    inputs["mask_pos"] = batch[-2]
                    outputs = model(**inputs)
                    tmp_eval_loss, logits = outputs[:2]
                    logits = logits[:, label_ids]

                    eval_loss += tmp_eval_loss.mean().item()

                nb_eval_steps += 1
                if preds is None:
                    preds = logits.detach().cpu().numpy()
                    batch_labels = inputs["labels"]
                    for i, label_id in enumerate(label_ids):
                        batch_labels[batch_labels == label_id] = i
                    out_label_ids = batch_labels.detach().cpu().numpy()
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                    batch_labels = inputs["labels"]
                    for i, label_id in enumerate(label_ids):
                        batch_labels[batch_labels == label_id] = i
                    out_label_ids = np.append(out_label_ids, batch_labels.detach().cpu().numpy(), axis=0)

        logger.info("  Num examples = %d", len(preds))
        eval_loss = eval_loss / nb_eval_steps
        loss_dict = {'loss':eval_loss}
        results.update(loss_dict)
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = glue_compute_metrics(task_mappings[eval_task], preds, out_label_ids)
        results.update(result)
        

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            print("  %s = %s" % (key, str(result[key])))

    return results




def predict(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        processor = processors[eval_task]()
        label_ids = []
        label_list = processor.get_labels()
        label_map = processor.get_label_map()
        for k, v in label_map.items():
            label_id = tokenizer(' ' + v, add_special_tokens=False)['input_ids']
            assert len(label_id) == 1
            label_ids.append(label_id[0])
            
        args.eval_batch_size = 32
        eval_dataset = PromptDataset(args, eval_task, tokenizer, data_type='test')
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=eval_dataset.collate_fn)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running inference *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        nb_eval_steps = 0
        preds = None

        for batch in tqdm(eval_dataloader, desc="Infering"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                }
                inputs["token_type_ids"] = batch[2]
                inputs["mask_pos"] = batch[3]
                outputs = model(**inputs)
                logits = outputs[0]
                logits = logits[:, label_ids]

            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        output_infer_file = os.path.join(
            eval_output_dir, 
            "{}_{}_{}_{}_{}_{}_{}.tsv".format(
                eval_task, 
                args.generator_type,
                args.add_prompt_layer, 
                args.num_prompt_tokens, 
                args.proj_down_size,
                args.per_gpu_train_batch_size,
                args.learning_rate,
                args.warmup_rate,
            )
        )
        with open(output_infer_file, "w", encoding='utf-8') as fout:
            writer = csv.writer(fout, delimiter='\t', quotechar=None)
            writer.writerow(["index", "prediction"])
            for i, pred in enumerate(preds):
                if args.output_mode == "classification":
                    prediction = label_list[pred]
                elif args.output_mode == "regression":
                    prediction = str(pred)
                writer.writerow([i, prediction]) 

def aggregate(param_list, n_sample_list):
    # convert number of samples into weights
    total_samples = sum(n_sample_list)
    weight_list = [x / total_samples for x in n_sample_list]
    assert np.isclose(sum(weight_list), 1), f"sum of weights is {sum(weight_list)}"

    if isinstance(param_list[0], list):
        raise NotImplementedError()
    else:
        averaged_dict = OrderedDict()
        first_dict = param_list[0]
        for key in first_dict.keys():
            averaged_dict[key] = 0
        for ordered_dict, weight in zip(param_list, weight_list):
            for key, value in ordered_dict.items():
                averaged_dict[key] += value * weight
        return averaged_dict
    
def aggregate_plus(local_param_list, global_param, n_sample_list, ratio):
    # do weighted aggregtation on local_param_list, then do weighted sum into global_param
    total_samples = sum(n_sample_list)
    weight_list = [x / total_samples for x in n_sample_list]
    assert np.isclose(sum(weight_list), 1), f"sum of weights is {sum(weight_list)}"

    if isinstance(local_param_list[0], list):
        raise NotImplementedError()    
    else:
        averaged_dict = OrderedDict()
        first_dict = local_param_list[0]
        for key in first_dict.keys():
            averaged_dict[key] = 0
        for ordered_dict, weight in zip(local_param_list, weight_list):
            for key, value in ordered_dict.items():
                averaged_dict[key] += value * weight
        for key in averaged_dict:
            averaged_dict[key] = ratio * averaged_dict[key] + global_param[key]
        return averaged_dict

    

def quiet_evaluate(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}

    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        processor = processors[eval_task]()
        label_ids = []
        label_map = processor.get_label_map()
        for k, v in label_map.items():
            label_id = tokenizer(' ' + v, add_special_tokens=False)['input_ids']
            assert len(label_id) == 1
            label_ids.append(label_id[0])
            
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

        eval_dataset = PromptDataset(args, eval_task, tokenizer, data_type='dev')
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=eval_dataset.collate_fn)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        # for batch in tqdm(eval_dataloader, desc="Evaluating"):
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[-1],
                }
                inputs["token_type_ids"] = batch[2]
                inputs["mask_pos"] = batch[-2]
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
                logits = logits[:, label_ids]

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                batch_labels = inputs["labels"]
                for i, label_id in enumerate(label_ids):
                    batch_labels[batch_labels == label_id] = i
                out_label_ids = batch_labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                batch_labels = inputs["labels"]
                for i, label_id in enumerate(label_ids):
                    batch_labels[batch_labels == label_id] = i
                out_label_ids = np.append(out_label_ids, batch_labels.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = glue_compute_metrics(task_mappings[eval_task], preds, out_label_ids)
        results.update(result)

    return results

def ratio_minus(w1, P, ratio=0):
    w = {}
    for key in w1.keys():
        if key in P:
            w[key] = w1[key] - P[key] * ratio
    return w

def update_client_c(global_c, client_c, delta, local_steps):
    res = {}
    for k in global_c:
        res[k] = client_c[k] - global_c[k] + delta[k].cpu().detach() / local_steps
    return res

def average_weights_plus(w, v, ratio=0):
    """
    Returns the average of the weights w and plus v.
    """
    w_avg = copy.deepcopy(w[0][1])
    total = 0
    for i in range(0, len(w)):
        total += w[i][0]
    for key in w_avg.keys():
        w_avg[key] *= w[0][0]
        for i in range(1, len(w)):
            w_avg[key] += w[i][1][key] * w[i][0]
        w_avg[key] = ratio * torch.div(w_avg[key], total) + v[key]
    return w_avg