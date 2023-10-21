import os
import csv
import sys
import logging
sys.path.append('../')

import numpy as np

from tqdm import tqdm

import torch
from statistics import mean
from sklearn.metrics import f1_score, matthews_corrcoef
from torch import nn
from torch.utils.data import SequentialSampler, DataLoader

from data import processors
from data_loader import load_and_cache_data

import copy


logger = logging.getLogger(__name__)


def get_loss(args, logits, batch_data):
    loss = None

    if args.method_type == 'discriminative':
        seq_lengths = batch_data[-2]
        labels = batch_data[-1]

        pooled_logits = logits[range(len(seq_lengths)), seq_lengths]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(pooled_logits.view(-1, logits.size(-1)), labels.view(-1))
    else:
        logits = logits[..., :-1, :].contiguous()
        labels = batch_data[0]
        labels = labels[..., 1:].contiguous()
        loss_mask = batch_data[2][..., 1:]

        loss_fct = nn.CrossEntropyLoss(reduction="none")

        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) # [batch_size, length]
        loss = loss.view(logits.size(0), logits.size(1)) * loss_mask.view(logits.size(0), logits.size(1))
        loss = torch.sum(loss, axis=1) / torch.sum(loss_mask, axis=1)   

    return loss


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels), f"Predictions and labels have mismatched lengths {len(preds)} and {len(labels)}"
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mpqa":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mr":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "subj":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "trec":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def evaluate(args, model, tokenizer):
    eval_task = args.task_name

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    results = {}

    if args.method_type == 'discriminative':
        eval_dataset = load_and_cache_data(args, tokenizer, data_type='dev')
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

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

            seq_lengths = batch[-2]
            labels = batch[-1]

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                }
                outputs = model(**inputs, return_dict=True)
                logits = outputs.logits
                tmp_eval_loss = get_loss(args, logits, batch)

                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1

            if preds is None:
                preds = logits[range(len(seq_lengths)), seq_lengths].detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits[range(len(seq_lengths)), seq_lengths].detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)
    else:
        eval_datasets = load_and_cache_data(args, tokenizer, data_type='dev')

        preds = []
        out_label_ids = None
        eval_loss = []
        nb_eval_steps = 0
        for i, eval_dataset in enumerate(eval_datasets):
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

            # Eval!
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_dataset))
            logger.info("  Batch size = %d", args.eval_batch_size)

            losses = []

            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                labels = batch[-1]

                with torch.no_grad():
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                    }
                    outputs = model(**inputs, return_dict=True)
                    logits = outputs.logits
                    loss = get_loss(args, logits, batch)

                eval_loss += loss[labels == i].cpu().detach().numpy().tolist()
                losses += loss.cpu().detach().numpy().tolist()

                if i == 0:
                    if out_label_ids is None:
                        out_label_ids = labels.detach().cpu().numpy()
                    else:
                        out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

            preds.append(losses)

        eval_loss = mean(eval_loss)
        loss_dict = {'loss':eval_loss}
        results.update(loss_dict)
        preds = np.array(preds)
        preds = preds.transpose()

        preds = np.argmin(preds, axis=1)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        print("  %s = %s" % (key, str(result[key])))

    return results     





def personalized_evaluate(args, model, tokenizer, eval_dataloader_lists, local_generator_param_list, global_generator_param):
    eval_task = args.task_name

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    results = {}

    eval_datasets = load_and_cache_data(args, tokenizer, data_type='dev')

    preds = []
    out_label_ids = None
    eval_loss = []
    nb_eval_steps = 0


    for i, eval_dataloaders in enumerate(eval_dataloader_lists):

        losses = []
        for client_idx in tqdm(range(args.num_clients)):
            model.load_local_param(copy.deepcopy(local_generator_param_list[client_idx]))
            model.load_transferable_param(copy.deepcopy(global_generator_param))

        
            eval_dataloader = eval_dataloaders[client_idx]

            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                labels = batch[-1]

                with torch.no_grad():
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                    }
                    outputs = model(**inputs, return_dict=True)
                    logits = outputs.logits
                    loss = get_loss(args, logits, batch)

                eval_loss += loss[labels == i].cpu().detach().numpy().tolist()
                losses += loss.cpu().detach().numpy().tolist()

                if i == 0:
                    if out_label_ids is None:
                        out_label_ids = labels.detach().cpu().numpy()
                    else:
                        out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

        preds.append(losses)

    eval_loss = mean(eval_loss)
    loss_dict = {'loss':eval_loss}
    results.update(loss_dict)
    preds = np.array(preds)
    preds = preds.transpose()

    preds = np.argmin(preds, axis=1)
    result = compute_metrics(eval_task, preds, out_label_ids)
    results.update(result)

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
        print("  %s = %s" % (key, str(result[key])))

    return results    

def predict(args, model, tokenizer):
    eval_task = args.task_name

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    processor = processors[eval_task]()
    label_list = processor.get_labels()

    if args.method_type == 'discriminative':
        eval_dataset = load_and_cache_data(args, tokenizer, data_type='test')
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running inference *****")
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Infering"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            seq_lengths = batch[-2]

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                }
                outputs = model(**inputs, return_dict=True)
                logits = outputs.logits

            nb_eval_steps += 1

            if preds is None:
                preds = logits[range(len(seq_lengths)), seq_lengths].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits[range(len(seq_lengths)), seq_lengths].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds, axis=1)
    else:
        eval_datasets = load_and_cache_data(args, tokenizer, data_type='test')

        preds = []
        out_label_ids = None
        eval_loss = []
        nb_eval_steps = 0
        for i, eval_dataset in enumerate(eval_datasets):
            eval_sampler = SequentialSampler(eval_dataset)
            eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

            # Eval!
            logger.info("***** Running inference *****")
            logger.info("  Num examples = %d", len(eval_dataset))
            logger.info("  Batch size = %d", args.eval_batch_size)

            losses = []

            for batch in tqdm(eval_dataloader, desc="Infering"):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    inputs = {
                        "input_ids": batch[0],
                        "attention_mask": batch[1],
                        "seq_lengths": batch[3],
                    }
                    outputs = model(**inputs, return_dict=True)
                    logits = outputs.logits
                    loss = get_loss(args, logits, batch)

                losses += loss.cpu().detach().numpy().tolist()

            preds.append(losses)

        preds = np.array(preds)
        preds = preds.transpose()
        preds = np.argmin(preds, axis=1)

    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
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
            prediction = label_list[pred]
            writer.writerow([i, prediction]) 
              