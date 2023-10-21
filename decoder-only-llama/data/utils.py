import os
import json
import warnings
import dataclasses
import numpy as np

from dataclasses import dataclass
from dataclasses import asdict
from typing import List, Optional, Union
import torch

from transformers.utils import logging


logger = logging.get_logger(__name__)

DEPRECATION_WARNING = (
    "This {0} will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
    "library. You can have a look at this example script for pointers: "
    "https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py"
)


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    seq_lengths: Optional[int] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def truncate_and_padding_discriminative(
    input_a, input_b=None,
    max_length=None, bos_token_id=None, eos_token_id=None,    
):
    if input_b is None:
        if len(input_a) <= max_length - 2:
            seq_length = len(input_a) + 2
            mask_length = max_length - seq_length

            input_ids = [bos_token_id] + input_a + [eos_token_id] + [0 for _ in range(mask_length)]
            attention_mask = [1 for _ in range(seq_length)] + [0 for _ in range(mask_length)]
        else:
            seq_length = max_length
            truncate_length = len(input_a) - max_length + 2

            assert truncate_length <= len(input_a) / 2
            
            input_ids = [bos_token_id] + input_a[0:max_length-2] + [eos_token_id]
            attention_mask = [1 for _ in range(max_length)]
    else:
        if len(input_a) + len(input_b) <= max_length - 2:
            seq_length = len(input_a) + len(input_b) + 2
            mask_length = max_length - seq_length

            input_ids = [bos_token_id] + input_a + input_b + [eos_token_id] + [0 for _ in range(mask_length)]
            attention_mask = [1 for _ in range(seq_length)] + [0 for _ in range(mask_length)]
        else:
            seq_length = max_length
            truncate_length = len(input_a) + len(input_b) - max_length + 2

            truncate_length_a = round(truncate_length * len(input_a) / (len(input_a) + len(input_b)))
            truncate_length_b = truncate_length - truncate_length_a

            assert truncate_length_a <= len(input_a) / 2 and truncate_length_b <= len(input_b) / 2

            input_a = input_a[0:(len(input_a)-truncate_length_a)]
            input_b = input_b[0:(len(input_b)-truncate_length_b)]

            input_ids = [bos_token_id] + input_a + input_b + [eos_token_id]
            attention_mask = [1 for _ in range(max_length)]

    if len(input_ids) != max_length:
        print(len(input_ids))
    assert len(input_ids) == max_length

    return input_ids, attention_mask, seq_length - 1


def truncate_and_padding_generative_single(
    input_a, input_b,
    max_length=None, bos_token_id=None, 
    eos_token_id=None, method_type=None,    
):
    assert method_type is not None

    if len(input_a) + len((input_b)) + 2 > max_length:
        seq_length = len(input_a) + 1
        truncate_length = seq_length + len(input_b) + 1 - max_length
        assert truncate_length < seq_length / 2

        input_a = [bos_token_id] + input_a
        input_b = input_b + [eos_token_id]

        input_a = input_a[0:seq_length-truncate_length]

        input_ids = input_a + input_b
        loss_mask = [0 for _ in input_a] + [1 for _ in input_b]
        
        attention_mask = [1 for _ in input_a + input_b]
    else:
        mask_length = max_length - len(input_a) - len(input_b) - 2

        input_a = [bos_token_id] + input_a
        input_b = input_b + [eos_token_id]            
    
        input_ids = input_a + input_b + [0 for _ in range(mask_length)]
        loss_mask = [0 for _ in input_a] + [1 for _ in input_b] + [0 for _ in range(mask_length)]        
        
        attention_mask = [1 for _ in input_a + input_b] + [0 for _ in range(mask_length)]

    assert len(input_ids) == max_length
    seq_length = len(input_a) - 1

    return input_ids, attention_mask, loss_mask, seq_length


def truncate_and_padding_generative_pair(
    input_a, input_b, input_c,
    max_length=None, bos_token_id=None, 
    eos_token_id=None, method_type=None,    
):
    assert method_type is not None

    if len(input_a) + len((input_b)) + len(input_c) + 2 > max_length:
        seq_length = len(input_a) + len(input_b) + 1
        truncate_length = seq_length + len(input_c) + 1 - max_length
        truncate_length_a = round(truncate_length * len(input_a) / (len(input_a) + len(input_b)))
        truncate_length_b = truncate_length - truncate_length_a

        assert truncate_length_a < len(input_a) / 2 and truncate_length_b < len(input_b) / 2

        input_a = [bos_token_id] + input_a
        input_c = input_c + [eos_token_id]

        input_a = input_a[0:len(input_a)-truncate_length_a]
        input_b = input_b[0:len(input_b)-truncate_length_b]
        
        input_ids = input_a + input_b + input_c
        loss_mask = [0 for _ in input_a+input_b] + [1 for _ in input_c]

        attention_mask = [1 for _ in input_a + input_b + input_c]
    else:
        mask_length = max_length - len(input_a) - len(input_b) - len(input_c) - 2

        input_a = [bos_token_id] + input_a
        input_c = input_c + [eos_token_id]

        input_ids = input_a + input_b + input_c + [0 for _ in range(mask_length)]
        loss_mask = [0 for _ in input_a+input_b] + [1 for _ in input_c] + [0 for _ in range(mask_length)]        
        
        attention_mask = [1 for _ in input_a + input_b + input_c] + [0 for _ in range(mask_length)]

    assert len(input_ids) == max_length
    seq_length = len(input_a) + len(input_b) - 1

    return input_ids, attention_mask, loss_mask, seq_length 


def convert_to_tensor(
    inputs_a=None, inputs_b=None,
    task=None, labels=None, label_descs=None, 
    max_length=None, bos_token_id=None, eos_token_id=None,
    method_type='direct', is_training=False, start_idx_for_label=None
):

    if is_training:
        assert labels is not None
        assert len(label_descs) > 1
    
    all_input_ids, all_attention_mask, all_loss_mask, all_seq_lengths = [], [], [], []

    if task in ['sst-2', 'mpqa', 'mr', 'subj', 'trec']:
        assert inputs_a is not None and inputs_b is None

        for i, input_ids_a in enumerate(inputs_a):
            if is_training:
                label = labels[i]
                label_desc = label_descs[label].copy()
            else:
                label_desc = label_descs[0].copy()

            input_ids_a = input_ids_a + label_desc[:start_idx_for_label]
            input_ids_b = label_desc[start_idx_for_label:]
            
            input_ids, attention_mask, loss_mask, seq_length = truncate_and_padding_generative_single(
                input_ids_a, input_ids_b, max_length=max_length,
                bos_token_id=bos_token_id, eos_token_id=eos_token_id, method_type=method_type
            )
            
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_loss_mask.append(loss_mask)
            all_seq_lengths.append(seq_length)
    elif task in ['rte', 'qnli', 'mrpc']:
        assert inputs_a is not None and inputs_b is not None

        for i, (input_ids_a, input_ids_b) in enumerate(zip(inputs_a, inputs_b)):
            if is_training:
                label = labels[i]
                label_desc = label_descs[label].copy()
            else:
                label_desc = label_descs[0].copy()

            input_ids_b = input_ids_b + label_desc[:start_idx_for_label]
            input_ids_c = label_desc[start_idx_for_label:]

            input_ids, attention_mask, loss_mask, seq_length = truncate_and_padding_generative_pair(
                input_ids_a, input_ids_b, input_ids_c, max_length=max_length,
                bos_token_id=bos_token_id, eos_token_id=eos_token_id, method_type=method_type
            )                
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_loss_mask.append(loss_mask)
            all_seq_lengths.append(seq_length)

    if labels is not None:
        return {
            'input_ids': torch.tensor(all_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(all_attention_mask, dtype=torch.long),
            'loss_mask': torch.tensor(all_loss_mask, dtype=torch.long),
            'seq_lengths': torch.tensor(all_seq_lengths, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }
    else:
        return {
            'input_ids': torch.tensor(all_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(all_attention_mask, dtype=torch.long),
            'loss_mask': torch.tensor(all_loss_mask, dtype=torch.long),
            'seq_lengths': torch.tensor(all_seq_lengths, dtype=torch.long),
        }   


# get templates + verbalizers
def get_templates(task, idx):

    if task in ["sst-2", 'mpqa', 'mr', 'subj', 'trec', 'cola']:
        templates = ["It was %s . "]  
    elif task in ["mrpc", "rte", "qnli"]:
        templates = ["They are %s . "]              
    else:
        raise NotImplementedError(task)

    if task in ["sst-2", 'mpqa', 'mr']:
        label_words = ["terrible", "great"]
    elif task == 'subj':
        label_words = ["subjective", "objective"]
    elif task == 'trec':
        label_words = ["Description", "Entity", "Expression", "Human", "Location", "Number"]
    elif task in ["rte", "qnli"]:
        label_words = ["Yes", "No"]
    elif task == "mrpc":
        label_words = ["No", "Yes"]
    elif task == "cola":
        label_words = ["unacceptable", "acceptable"]
    else:
        raise NotImplementedError(task)

    return [templates[idx] % label_word for label_word in label_words]


def prepend_task_tokens(tokenizer, inputs, n_prefix, method_type):
    task_tokens = ["<TASK{}>".format(str(i).zfill(2)) for i in range(n_prefix)]
    tokenizer.add_tokens(task_tokens)
    task_token_ids = tokenizer(" ".join(task_tokens), return_tensors="pt")["input_ids"]
    assert task_token_ids.shape[-1]==n_prefix

    def convert(inputs):
        n_train = inputs["input_ids"].shape[0]

        new_input_ids=torch.cat([
                task_token_ids.repeat(n_train, 1),
                inputs["input_ids"][:,1:]], 1)
        inputs["input_ids"] = new_input_ids

        new_attention_mask=torch.cat([
            torch.ones((n_train, n_prefix-1), dtype=torch.long),
            inputs["attention_mask"]], 1)
        inputs["attention_mask"] = new_attention_mask

        if method_type == 'discriminative':
            inputs["seq_length"] = inputs["seq_length"] + n_prefix - 1
        else:
            inputs["loss_mask"] = torch.cat([
                torch.zeros((n_train, n_prefix-1), dtype=torch.long),
                inputs["loss_mask"]], 1)

        return inputs

    if type(inputs)==list:
        return [convert(_inputs) for _inputs in inputs]

    return convert(inputs)