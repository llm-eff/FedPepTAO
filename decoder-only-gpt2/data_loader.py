import os
import sys
import logging

import numpy as np

from typing import List, Optional, Union
sys.path.append('../')

import torch
from torch.utils.data import TensorDataset
from transformers import InputExample, PreTrainedTokenizer

from data import (
    processors,
    get_templates,
    convert_to_tensor,
    prepend_task_tokens,
    truncate_and_padding_discriminative,
)

import pickle

logger = logging.getLogger(__name__)


def prepare_data(
    args,
    examples: List[InputExample] = None,
    tokenizer: PreTrainedTokenizer = None,
    max_length: Optional[int] = None,
    labels=None,
    num_labels=None,
    templates=None,
    is_training=False,
    cache_data_file=None,
):

    if max_length is None:
        max_length = tokenizer.model_max_length

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
        
    if args.method_type == 'discriminative':

        all_input_ids, all_attention_mask, all_seq_length = [], [], []

        if args.task_name in ['sst-2', 'subj', 'trec']:
            if os.path.exists(cache_data_file) and not args.overwrite_cache:
                logger.info("Loading features from cached file %s", cache_data_file)
                inputs_a = torch.load(cache_data_file)
            else:
                inputs_a = tokenizer(
                    [example.text_a for example in examples], 
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )['input_ids']

                if args.local_rank in [-1, 0]:
                    logger.info("Saving input ids into cached file %s", cache_data_file)
                    torch.save(inputs_a, cache_data_file)

            if labels is not None:
                assert len(inputs_a) == len(labels)

            for input_ids in inputs_a:
                input_ids, attention_mask, seq_length = truncate_and_padding_discriminative(
                    input_ids, max_length=max_length, 
                    bos_token_id=bos_token_id, eos_token_id=eos_token_id
                )

                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_seq_length.append(seq_length)

        elif args.task_name in ['rte', 'qnli', 'mrpc']:
            if os.path.exists(cache_data_file) and not args.overwrite_cache:
                logger.info("Loading features from cached file %s", cache_data_file)
                inputs_a, inputs_b = torch.load(cache_data_file)
            else:
                inputs_a = tokenizer(
                    [example.text_a for example in examples], 
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )['input_ids']

                inputs_b = tokenizer(
                    [example.text_b for example in examples], 
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )['input_ids']

                if args.local_rank in [-1, 0]:
                    logger.info("Saving input ids into cached file %s", cache_data_file)
                    torch.save((inputs_a, inputs_b), cache_data_file)

            if labels is None:
                assert len(inputs_a) == len(inputs_b)
            else:
                assert len(inputs_a) == len(inputs_b) == len(labels)

            for input_ids_a, input_ids_b in zip(inputs_a, inputs_b):
                input_ids, attention_mask, seq_length = truncate_and_padding_discriminative(
                    input_ids_a, input_ids_b, max_length=max_length,
                    bos_token_id=bos_token_id, eos_token_id=eos_token_id,
                )

                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_seq_length.append(seq_length)
        
        if labels is not None:
            return {
                'input_ids': torch.tensor(all_input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(all_attention_mask, dtype=torch.long),
                'seq_length': torch.tensor(all_seq_length, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
            }
        else:             
            return {
                'input_ids': torch.tensor(all_input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(all_attention_mask, dtype=torch.long),
                'seq_length': torch.tensor(all_seq_length, dtype=torch.long),
            }

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    templates = [template.strip() for template in templates]
    templates = [" "+template for template in templates]

    label_descs = [tokenizer(template)["input_ids"] for template in templates]
    start_idx_for_label = [idx for idx, _label_descs in enumerate(zip(*label_descs))
            if not np.all([_label_descs[0]==_label_desc for _label_desc in _label_descs])][0]

    for i in range(num_labels):
        for j in range(i+1, num_labels):
            assert label_descs[i][:start_idx_for_label]==label_descs[j][:start_idx_for_label]
            assert label_descs[i][start_idx_for_label]!=label_descs[j][start_idx_for_label]

    inputs_a, inputs_b = None, None
    if args.task_name in ['sst-2', 'mpqa', 'mr', 'subj', 'trec']:
        if os.path.exists(cache_data_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cache_data_file)
            inputs_a = torch.load(cache_data_file)
        else:
            inputs_a = tokenizer(
                [example.text_a for example in examples], 
                return_attention_mask=False,
                return_token_type_ids=False,
            )['input_ids']

            if args.local_rank in [-1, 0]:
                logger.info("Saving input ids into cached file %s", cache_data_file)
                torch.save(inputs_a, cache_data_file)

    elif args.task_name in ['rte', 'qnli', 'mrpc']:
        if os.path.exists(cache_data_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cache_data_file)
            inputs_a, inputs_b = torch.load(cache_data_file)
        else:
            inputs_a = tokenizer(
                [example.text_a for example in examples], 
                return_attention_mask=False,
                return_token_type_ids=False,
            )['input_ids']

            inputs_b = tokenizer(
                [example.text_b for example in examples], 
                return_attention_mask=False,
                return_token_type_ids=False,
            )['input_ids']

            if args.local_rank in [-1, 0]:
                logger.info("Saving input ids into cached file %s", cache_data_file)
                torch.save((inputs_a, inputs_b), cache_data_file)  
    
    if is_training:

        return convert_to_tensor(
            inputs_a=inputs_a, inputs_b=inputs_b,
            task=args.task_name, labels=labels, label_descs=label_descs,
            max_length=max_length, bos_token_id=bos_token_id,
            eos_token_id=eos_token_id, method_type=args.method_type, 
            is_training=is_training, start_idx_for_label=start_idx_for_label,
        )

    input_tensors = []

    for i in range(num_labels):
        label_desc = label_descs[i]

        input_tensors.append(convert_to_tensor(
            inputs_a=inputs_a, inputs_b=inputs_b,
            task=args.task_name, labels=labels, label_descs=[label_desc],
            max_length=max_length, bos_token_id=bos_token_id,
            eos_token_id=eos_token_id, method_type=args.method_type, 
            is_training=is_training, start_idx_for_label=start_idx_for_label,
        ))

    return input_tensors


def load_and_cache_data(args, tokenizer, data_type="train"):

    if args.local_rank not in [-1, 0] and data_type == "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[args.task_name]()
    # Load data features from cache or dataset file
    cached_examples_file = os.path.join(
        args.data_dir,
        "cached_examples_{}_{}_{}_{}_{}_{}".format(
            data_type,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(args.task_name),
            str(args.method_type),
            str(args.template_idx),
        ),
    )

    cached_input_ids_file = os.path.join(
        args.data_dir,
        "cached_input_ids_{}_{}_{}_{}_{}_{}".format(
            data_type,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(args.task_name),
            str(args.method_type),
            str(args.template_idx),
        ),
    )

    pkl_file = os.path.join(
        args.data_dir, f"{str(args.task_name)}_data.pkl"
    )

    if os.path.exists(cached_examples_file) and not args.overwrite_cache:
        examples = torch.load(cached_examples_file)
    

    else:
        if data_type == "train":
            examples = processor.get_train_examples(args.data_dir)
            
        elif data_type == "dev":
            examples = processor.get_dev_examples(args.data_dir)
        elif data_type == "test":
            examples = processor.get_test_examples(args.data_dir)
        else:
            raise NotImplementedError

        if args.local_rank in [-1, 0]:
            torch.save(examples, cached_examples_file)

    is_training = True if data_type == 'train' else False

    label_list = processor.get_labels()
    num_labels=len(label_list)
    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample):
        return label_map[example.label]

    labels = [label_from_example(example) for example in examples] if examples[0].label is not None else None

    templates = None
    if args.method_type != 'discriminative':
        templates = get_templates(args.task_name, args.template_idx)

    if os.path.exists(cached_input_ids_file):
        del examples
        input_tensors = prepare_data(
            args, tokenizer=tokenizer, labels=labels, num_labels=num_labels, templates=templates, 
            max_length=args.max_seq_length, is_training=is_training, cache_data_file=cached_input_ids_file,
        )
    else:  
        input_tensors = prepare_data(
            args, examples=examples, tokenizer=tokenizer, max_length=args.max_seq_length, num_labels=num_labels, 
            labels=labels, templates=templates, is_training=is_training, cache_data_file=cached_input_ids_file,
        )
    
    if args.method_type != 'discriminative' and not is_training:
        datasets = []
        for input_tensor in input_tensors: 
            if 'labels' in input_tensor:
                datasets.append(
                    TensorDataset(
                        input_tensor['input_ids'],
                        input_tensor['attention_mask'],
                        input_tensor['loss_mask'],
                        input_tensor['seq_lengths'],
                        input_tensor['labels'],
                    )
                )
            else:
                datasets.append(
                    TensorDataset(
                        input_tensor['input_ids'],
                        input_tensor['attention_mask'],
                        input_tensor['loss_mask'],
                        input_tensor['seq_lengths'],
                    )
                )
    elif args.method_type != 'discriminative' and is_training:
        datasets = TensorDataset(
                    input_tensors['input_ids'],
                    input_tensors['attention_mask'],
                    input_tensors['loss_mask'],
                    input_tensors['seq_lengths'],
                    input_tensors['labels'],            
                )
    elif args.method_type == 'discriminative':
        if 'labels' in input_tensors:
            datasets = TensorDataset(
                        input_tensors['input_ids'],
                        input_tensors['attention_mask'],
                        input_tensors['seq_length'],
                        input_tensors['labels'],            
                    )
        else:
            datasets = TensorDataset(
                        input_tensors['input_ids'],
                        input_tensors['attention_mask'],
                        input_tensors['seq_length'],            
                    )                                        

    if args.local_rank == 0 and not data_type == "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    return datasets