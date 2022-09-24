---
layout: single
title: "RBERT for Relation Extraction: full code & Wandb log"
---

## Model Performance and log on `weights and biases(wandb)`

[📈 Train and evaluation logs are visualized in this wandb link](https://wandb.ai/snoop2head/KLUE-RE/runs/htwirbkj?workspace=user-snoop2head)

![Screen Shot 2021-10-08 at 9.08.50 AM](../assets/images/2021-10-09-Relation-Extraction-Code/Screen Shot 2021-10-08 at 9.08.50 AM.png)

![image-20211008090942237](../assets/images/2021-10-09-Relation-Extraction-Code/image-20211008090942237.png)

Evaluation loss is constantly lowered as steps progresses, until 4000 steps(or 5 epochs). This was different from concat baseline model where its' evaluation loss sparks around 2000 steps(or 3 epochs). We concluded that providing more information of `[CLS]`, subject entity, object entity's hidden features enabled models to find deeper local minimum.

## RBERT explanation

![image-20211008224657844](../assets/images/2021-10-09-Relation-Extraction-Code/image-20211008224657844.png)

RBERT extracts hidden features wrapped around the entity markers commences average pooling. **Key is to concatenate 1) [CLS], 2) Subject entity tokens’ hidden features' average, 3) Object entity tokens’ hidden features' average.** Furthermore, in this competition, I customized to include entity wrappers around target entities and pooled those tokens also. This was to include more contextual information which divided target entities and other tokens. This is different from the original proposal at [Enriching Pre-trained Language Model with Entity Information for Relation Classification, Wu et al. 2019](https://arxiv.org/pdf/1905.08284v1.pdf).

## Config

```python
""" define train data and test data path """
import os
from glob import glob

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# root path
ROOT_PATH = os.path.abspath(".") # this makes compatible absolute path both for local and server

# directory paths
ROOT_DATA_DIR = os.path.join(os.path.dirname(ROOT_PATH), 'dataset')
TRAIN_DATA_DIR = os.path.join(ROOT_DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(ROOT_DATA_DIR, 'test')
PRIDICTION_DIR = os.path.join(os.path.dirname(ROOT_PATH), 'prediction')
BASELINE_DIR = os.path.join(os.path.dirname(ROOT_PATH), 'baseline-code')

# files paths
TRAIN_FILE_PATH = glob(os.path.join(TRAIN_DATA_DIR, '*'))[0]
TEST_FILE_PATH = glob(os.path.join(TEST_DATA_DIR, '*'))[0]
SAMPLE_SUBMISSION_PATH = os.path.join(PRIDICTION_DIR, 'sample_submission.csv')
PORORO_TRAIN_PATH = os.path.join(ROOT_DATA_DIR, 'train_pororo_sub.csv')
PORORO_TEST_PATH = os.path.join(ROOT_DATA_DIR, 'test_pororo_sub.csv')
PORORO_SPECIAL_TOKEN_PATH = os.path.join(ROOT_DATA_DIR, 'pororo_special_token.txt')

print(TRAIN_FILE_PATH, TEST_FILE_PATH,SAMPLE_SUBMISSION_PATH)
```

    /opt/ml/klue-level2-nlp-15/dataset/train/train.csv /opt/ml/klue-level2-nlp-15/dataset/test/test_data.csv /opt/ml/klue-level2-nlp-15/prediction/sample_submission.csv

```python
""" Set configuration as dictionary format """

import wandb
from datetime import datetime
from easydict import EasyDict

# login wandb and get today's date until hour and minute
wandb.login()
today = datetime.now().strftime("%m%d_%H:%M")

# Debug set to true in order to debug high-layer code.
# CFG Configuration
CFG = wandb.config # wandb.config provides functionality of easydict.EasyDict
CFG.DEBUG = False

# Dataset Config as constants
CFG.num_labels = 30
CFG.num_workers = 4
CFG.batch_size = 40

# Train configuration
CFG.user_name = "snoop2head"
CFG.file_base_name = f"{CFG.user_name}_{today}"
CFG.model_name = "klue/roberta-large"
CFG.num_folds = 5 # 5 Fold as default
CFG.num_epochs = 5 # loss is increasing after 5 epochs for xlm-roberta-large and 3 epochs for klue/roberta-large
CFG.max_token_length = 128 + 4 # refer to EDA where Q3 is 119, there are only 460 sentences out of 32k train set
CFG.stopwords = []
CFG.learning_rate = 5e-5
CFG.weight_decay = 1e-2 # https://paperswithcode.com/method/weight-decay
CFG.input_size = 768 # 768 for klue/roberta-large, 1024 for klue/roberta-large
CFG.output_size = 768 # 768 for klue/roberta-large, 1024 for klue/roberta-large
CFG.num_rnn_layers = 3
CFG.dropout_rate = 0.1

# training steps configurations
CFG.save_steps = 500
CFG.early_stopping_patience = 5
CFG.warmup_steps = 500
CFG.logging_steps = 100
CFG.evaluation_strategy = 'epoch'
CFG.evaluation_steps = 500

# Directory configuration
CFG.result_dir = os.path.join(os.path.dirname(ROOT_PATH), "results")
CFG.saved_model_dir = os.path.join(os.path.dirname(ROOT_PATH), "best_models")
CFG.logging_dir = os.path.join(os.path.dirname(ROOT_PATH), "logs")
CFG.baseline_dir = os.path.join(os.path.dirname(ROOT_PATH), 'baseline-code')

# file configuration
CFG.result_file = os.path.join(CFG.result_dir, f"{CFG.file_base_name}.csv")
CFG.saved_model_file = os.path.join(CFG.saved_model_dir, f"{CFG.file_base_name}.pth")
CFG.logging_file = os.path.join(CFG.logging_dir, f"{CFG.file_base_name}.log")
CFG.label_to_num_file = os.path.join(CFG.baseline_dir, 'dict_label_to_num.pkl')
CFG.num_to_label_file = os.path.join(CFG.baseline_dir, "dict_num_to_label.pkl")
CFG.train_file_path = TRAIN_FILE_PATH
CFG.test_file_path = TEST_FILE_PATH
CFG.sample_submission_file_path = SAMPLE_SUBMISSION_PATH

# Other configurations
CFG.load_best_model_at_end = True
```

## Dataset

```python
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import StratifiedKFold

# read pororo dataset
df_pororo_dataset = pd.read_csv(PORORO_TRAIN_PATH)

# remove the first index column
df_pororo_dataset = df_pororo_dataset.drop(df_pororo_dataset.columns[0], axis=1)
```

```python
import random

def seed_everything(seed) :
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
seed_everything(42)
```

```python
import pickle as pickle

def label_to_num(label=None):
  """ change the string labels into numerics """
  num_label = []
  with open(os.path.join(BASELINE_DIR, "dict_label_to_num.pkl"), 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])

  return num_label
```

```python
class CustomDataset(Dataset):

    def __init__(self, dataset, tokenizer, is_training: bool = True):

        # pandas.Dataframe dataset
        self.dataset = dataset
        self.sentence = self.dataset["sentence"]
        self.subject_entity = self.dataset["subject_entity"]
        self.object_entity = self.dataset["object_entity"]
        if is_training:
            self.train_label = label_to_num(self.dataset["label"].values)
        if not is_training:
            self.train_label = self.dataset["label"].values
        self.label = torch.tensor(self.train_label)

        # tokenizer and etc
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        sentence = self.sentence[idx]
        subject_entity = self.subject_entity[idx]
        object_entity = self.object_entity[idx]
        label = self.label[idx]

        # concat entity in the beginning
        concat_entity = subject_entity + "[SEP]" + object_entity

        # tokenize
        encoded_dict = self.tokenizer(
            concat_entity,
            sentence,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=RBERT_CFG.max_token_length,
            add_special_tokens=True,
            return_token_type_ids=False,  # for RoBERTa
        )

        # RoBERTa's provided masks (do not include token_type_ids for RoBERTa)
        encoded_dict["input_ids"] = encoded_dict["input_ids"].squeeze(0)
        encoded_dict["attention_mask"] = encoded_dict["attention_mask"].squeeze(0)

        # add subject and object entity masks where masks notate where the entity is
        subject_entity_mask, object_entity_mask = self.add_entity_mask(
            encoded_dict, subject_entity, object_entity
        )
        encoded_dict["subject_mask"] = subject_entity_mask
        encoded_dict["object_mask"] = object_entity_mask

        # fill label
        encoded_dict["label"] = label
        return encoded_dict

    def __len__(self):
        return len(self.dataset)

    def add_entity_mask(self, encoded_dict, subject_entity, object_entity):
        """add entity token to input_ids"""
        # print("tokenized input ids: \n",encoded_dict['input_ids'])

        # initialize entity masks
        subject_entity_mask = np.zeros(CFG.max_token_length, dtype=int)
        object_entity_mask = np.zeros(CFG.max_token_length, dtype=int)

        # get token_id from encoding subject_entity and object_entity
        subject_entity_token_ids = self.tokenizer.encode(
            subject_entity, add_special_tokens=False
        )
        object_entity_token_ids = self.tokenizer.encode(
            object_entity, add_special_tokens=False
        )
        # print("entity token's input ids: ",subject_entity_token_ids, object_entity_token_ids)

        # get the length of subject_entity and object_entity
        subject_entity_length = len(subject_entity_token_ids)
        object_entity_length = len(object_entity_token_ids)

        # find coordinates of subject_entity_token_ids inside the encoded_dict["input_ids"]
        subject_coordinates = np.where(encoded_dict["input_ids"] == subject_entity_token_ids[0])
        subject_coordinates = list(
            map(int, subject_coordinates[0])
        )  # change the subject_coordinates into int type
        for subject_index in subject_coordinates:
            subject_entity_mask[
                subject_index : subject_index + subject_entity_length
            ] = 1

        # find coordinates of object_entity_token_ids inside the encoded_dict["input_ids"]
        object_coordinates = np.where(encoded_dict["input_ids"] == object_entity_token_ids[0])
        object_coordinates = list(
            map(int, object_coordinates[0])
        )  # change the object_coordinates into int type
        for object_index in object_coordinates:
            object_entity_mask[object_index : object_index + object_entity_length] = 1

        # print(subject_entity_mask)
        # print(object_entity_mask)

        return torch.Tensor(subject_entity_mask), torch.Tensor(object_entity_mask)

```

```python
"""Debugging add_entity_mask_sample function with sample code"""

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

special_token_list = []
with open(PORORO_SPECIAL_TOKEN_PATH, 'r', encoding = 'UTF-8') as f :
    for token in f :
        special_token_list.append(token.split('\n')[0])

added_token_num = tokenizer.add_special_tokens({"additional_special_tokens":list(set(special_token_list))})

sample_item = CustomDataset(df_pororo_dataset, tokenizer)[0]
decoded_item = tokenizer.decode(sample_item['input_ids'])

def add_entity_mask_sample(item, subject_entity, object_entity):
    # add entity token to input_ids
    print("tokenized input ids: \n",item['input_ids'])

    # initialize entity masks
    subject_entity_mask = np.zeros(CFG.max_token_length, dtype=int)
    object_entity_mask = np.zeros(CFG.max_token_length, dtype=int)

    # get token_id from encoding subject_entity and object_entity
    subject_entity_token_ids = tokenizer.encode(subject_entity, add_special_tokens=False)
    object_entity_token_ids = tokenizer.encode(object_entity, add_special_tokens=False)
    # print("entity token's input ids: ",subject_entity_token_ids, object_entity_token_ids)

    # get the length of subject_entity and object_entity
    subject_entity_length = len(subject_entity_token_ids)
    object_entity_length = len(object_entity_token_ids)

    # find coordinates of subject_entity_token_ids inside the item["input_ids"]
    subject_coordinate = np.where(item['input_ids'] == subject_entity_token_ids[0])
    subject_coordinate = list(map(int, subject_coordinate[0])) # change the subject_coordinate into int type
    for subject_index in subject_coordinate:
        subject_entity_mask[subject_index:subject_index+subject_entity_length] = 1

    # find coordinates of object_entity_token_ids inside the item["input_ids"]
    object_coordinate = np.where(item['input_ids'] == object_entity_token_ids[0])
    object_coordinate = list(map(int, object_coordinate[0])) # change the object_coordinate into int type
    for object_index in object_coordinate:
        object_entity_mask[object_index:object_index+object_entity_length] = 1

    # print(subject_entity_mask)
    # print(object_entity_mask)

    return subject_entity_mask, object_entity_mask

# print(sample_item)
print(decoded_item)
subject_entity_mask, object_entity_mask = add_entity_mask_sample(sample_item, '[SUB-PERSON]비틀즈[/SUB-PERSON]'	, '[OBJ-PERSON]조지 해리슨[/OBJ-PERSON]')

```

```
[CLS]'[SUB-PERSON] 비틀즈 [/SUB-PERSON]'[SEP]'[OBJ-PERSON] 조지 해리슨 [/OBJ-PERSON]'[SEP] 〈 Something 〉 는 [OBJ-PERSON] 조지 해리슨 [/OBJ-PERSON] 이 쓰고 [SUB-PERSON] 비틀즈 [/SUB-PERSON] 가 1969년 앨범 《 Abbey Road 》 에 담은 노래다. [SEP] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]

(array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
 array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
```

- `[SUB-PERSON] 비틀즈 [/SUB-PERSON]` is given example's subject entity.
- `[OBJ-PERSON] 조지 해리슨 [/OBJ-PERSON]` is given example's object entity.
- As shown on the result above, we can see that entity masks annotates the location of the corresponding entities with 1 and rest of tokens as 0.
- Rather than using entities’ token only, surrounding entity markers were also pooled. This was to include more contextual information which divided target entities and other tokens. This is different from the original proposal at [Enriching Pre-trained Language Model with Entity Information for Relation Classification, Wu et al. 2019](https://arxiv.org/pdf/1905.08284v1.pdf).

## Evaluation function

```python
class Metrics(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
```

## Custom Model RBERT and Loss

![img](https://user-images.githubusercontent.com/28896432/68673458-1b090d00-0597-11ea-96b1-7c1453e6edbb.png)

```python
import torch
from torch import nn
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer


class FCLayer(nn.Module):
    """R-BERT: https://github.com/monologg/R-BERT/blob/master/model.py"""

    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(nn.Module):
    """R-BERT: https://github.com/monologg/R-BERT/blob/master/model.py"""

    def __init__(
        self,
        model_name: str = "klue/roberta-large",
        num_labels: int = 30,
        dropout_rate: float = 0.1,
        special_tokens_dict: dict = None,
        is_train: bool = True,
    ):
        super(RBERT, self).__init__()

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.backbone_model = AutoModel.from_pretrained(model_name, config=config)
        self.dropout_rate = dropout_rate
        self.num_labels = num_labels
        config.num_labels = num_labels

        # add special tokens
        self.special_tokens_dict = special_tokens_dict
        self.backbone_model.resize_token_embeddings(len(self.tokenizer))

        self.cls_fc_layer = FCLayer(
            config.hidden_size, config.hidden_size, self.dropout_rate
        )
        self.entity_fc_layer = FCLayer(
            config.hidden_size, config.hidden_size, self.dropout_rate
        )
        self.label_classifier = FCLayer(
            config.hidden_size * 3,
            self.num_labels,
            self.dropout_rate,
            use_activation=False,
        )

    def entity_average(self, hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(
        self,
        input_ids,
        attention_mask,
        subject_mask=None,
        object_mask=None,
        labels=None,
    ):

        outputs = self.backbone_model(
            input_ids=input_ids, attention_mask=attention_mask
        )

        sequence_output = outputs["last_hidden_state"]
        pooled_output = outputs[
            "pooler_output"
        ]  # [CLS] token's hidden featrues(hidden state)

        # hidden state's average in between entities
        # print(sequence_output.shape, subject_mask.shape)
        e1_h = self.entity_average(
            sequence_output, subject_mask
        )  # token in between subject entities ->
        e2_h = self.entity_average(
            sequence_output, object_mask
        )  # token in between object entities

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(
            pooled_output
        )  # [CLS] token -> hidden state | green on diagram
        e1_h = self.entity_fc_layer(
            e1_h
        )  # subject entity's fully connected layer | yellow on diagram
        e2_h = self.entity_fc_layer(
            e2_h
        )  # object entity's fully connected layer | red on diagram

        # Concat -> fc_layer / [CLS], subject_average, object_average
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)
        return logits

        # WILL USE FOCAL LOSS INSTEAD OF MSELoss and CrossEntropyLoss
        # outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        # # Softmax
        # if labels is not None:
        #     if self.num_labels == 1:
        #         loss_fct = nn.MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss_fct = nn.CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        #     outputs = (loss,) + outputs

        #  return outputs  # (loss), logits, (hidden_states), (attentions)
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction,
        )
```

## Train

```python
device = torch.device('cuda:0' if torch.cuda.is_available() and CFG.DEBUG == False else 'cpu')
print(device)
```

```
cuda:0
```

```python
# fetch tokenizer
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

# fetch special tokens annotated with ner task
special_token_list = []
with open(PORORO_SPECIAL_TOKEN_PATH, 'r', encoding = 'UTF-8') as f :
    for token in f :
        special_token_list.append(token.split('\n')[0])

# add special pororo ner tokens to the tokenizer
added_token_num = tokenizer.add_special_tokens({"additional_special_tokens":list(set(special_token_list))})
added_token_num
```

    74

```python
criterion = FocalLoss(gamma=1.0) # 0.0 equals to CrossEntropy
```

```python
# read pororo ner labeled dataset
df_pororo_dataset = pd.read_csv(PORORO_TRAIN_PATH)

# remove the first index column
df_pororo_dataset = df_pororo_dataset.drop(df_pororo_dataset.columns[0], axis=1)
df_pororo_dataset.head(2)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>sentence</th>
      <th>label</th>
      <th>subject_entity</th>
      <th>object_entity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>〈Something〉는 [OBJ-PERSON]조지 해리슨[/OBJ-PERSON]이 ...</td>
      <td>no_relation</td>
      <td>'[SUB-PERSON]비틀즈[/SUB-PERSON]'</td>
      <td>'[OBJ-PERSON]조지 해리슨[/OBJ-PERSON]'</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>호남이 기반인 바른미래당·[OBJ-ORGANIZATION]대안신당[/OBJ-ORGA...</td>
      <td>no_relation</td>
      <td>'[SUB-ORGANIZATION]민주평화당[/SUB-ORGANIZATION]'</td>
      <td>'[OBJ-ORGANIZATION]대안신당[/OBJ-ORGANIZATION]'</td>
    </tr>
  </tbody>
</table>
</div>

```python
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from adamp import AdamP
from transformers import AdamW
from tqdm import tqdm
from easydict import EasyDict


# read pororo dataset
df_pororo_dataset = pd.read_csv(DATA_CFG.pororo_train_path)

# remove the first index column
df_pororo_dataset = df_pororo_dataset.drop(df_pororo_dataset.columns[0], axis=1)

# fetch tokenizer
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

# fetch special tokens annotated with ner task
special_token_list = []
with open(DATA_CFG.pororo_special_token_path, "r", encoding="UTF-8") as f:
  for token in f:
    special_token_list.append(token.split("\n")[0])

    if torch.cuda.is_available() and CFG.debug == False:
      device = torch.device("cuda")
      print(f"There are {torch.cuda.device_count()} GPU(s) available.")
      print("GPU Name:", torch.cuda.get_device_name(0))
      else:
        print("No GPU, using CPU.")
        device = torch.device("cpu")

        criterion = FocalLoss(gamma=CFG.gamma)  # 0.0 equals to CrossEntropy

        train_data = RBERT_Dataset(df_pororo_dataset, tokenizer)
        dev_data = RBERT_Dataset(df_pororo_dataset, tokenizer)

        stf = StratifiedKFold(
          n_splits=CFG.num_folds, shuffle=True, random_state=seed_everything(42)
        )

        for fold_num, (train_idx, dev_idx) in enumerate(
          stf.split(df_pororo_dataset, list(df_pororo_dataset["label"]))
        ):

          print(f"#################### Fold: {fold_num + 1} ######################")

          train_set = Subset(train_data, train_idx)
          dev_set = Subset(dev_data, dev_idx)

          train_loader = DataLoader(
            train_set,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
          )
          dev_loader = DataLoader(
            dev_set,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
          )

          # fetch model
          model = RBERT(CFG.model_name)
          model.to(device)

          # fetch loss function, optimizer, scheduler outside of torch library
          # https://github.com/clovaai/AdamP
          optimizer = AdamP(
            model.parameters(),
            lr=CFG.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=CFG.weight_decay,
          )
          # optimizer = AdamW(model.parameters(), lr=CFG.learning_rate, betas=(0.9, 0.999), weight_decay=CFG.weight_decay) # AdamP is better

          # https://huggingface.co/transformers/main_classes/optimizer_schedules.html
          scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=CFG.warmup_steps,
            num_training_steps=len(train_loader) * CFG.num_epochs,
          )

          best_eval_loss = 1.0
          steps = 0

          # fetch training loop
          for epoch in tqdm(range(CFG.num_epochs)):
            train_loss = Metrics()
            dev_loss = Metrics()
            for _, batch in enumerate(train_loader):
              optimizer.zero_grad()
              # print(item)
              # assign forward() arguments to the device

              label = batch[4].to(device)
              inputs = {
                "input_ids": batch[0].to(device),
                "attention_mask": batch[1].to(device),
                "subject_mask": batch[2].to(device),
                # 'token_type_ids' # NOT FOR ROBERTA!
                "object_mask": batch[3].to(device),
                "label": label,
              }

              # model to training mode
              model.train()
              pred_logits = model(**inputs)
              loss = criterion(pred_logits, label)

              # backward
              loss.backward()
              optimizer.step()
              scheduler.step()

              # update metrics
              train_loss.update(loss.item(), len(label))

              steps += 1
              # for every 100 steps
              if steps % 100 == 0:
                print(
                  "Epoch: {}/{}".format(epoch + 1, CFG.num_epochs),
                  "Step: {}".format(steps),
                  "Train Loss: {:.4f}".format(train_loss.avg),
                )
                for dev_batch in dev_loader:
                  dev_label = dev_batch[4].to(device)
                  dev_inputs = {
                    "input_ids": dev_batch[0].to(device),
                    "attention_mask": dev_batch[1].to(device),
                    "subject_mask": dev_batch[2].to(device),
                    # 'token_type_ids' # NOT FOR ROBERTA!
                    "object_mask": dev_batch[3].to(device),
                    "label": dev_label,
                  }

                  # switch model to eval mode
                  model.eval()
                  dev_pred_logits = model(**dev_inputs)
                  loss = criterion(dev_pred_logits, dev_label)

                  # update metrics
                  dev_loss.update(loss.item(), len(dev_label))

                  # print metrics
                  print(
                    "Epoch: {}/{}".format(epoch + 1, CFG.num_epochs),
                    "Step: {}".format(steps),
                    "Dev Loss: {:.4f}".format(dev_loss.avg),
                  )

                  if best_eval_loss > dev_loss.avg:
                    best_eval_loss = dev_loss.avg
                    torch.save(
                      model.state_dict(),
                      "./results/{}-fold-{}-best-eval-loss-model.pt".format(
                        fold_num + 1, CFG.num_folds
                      ),
                    )
                    print(
                      "Saved model with lowest validation loss: {:.4f}".format(
                        best_eval_loss
                      )
                    )
                    early_stop = 0
                    else:
                      early_stop += 1
                      if early_stop > 2:
                        break

                        # Prevent OOM error
                        model.cpu()
                        del model
                        torch.cuda.empty_cache()


print("#################### TRAIN IS OVER! ######################")
```

    #################### Fold: 1 ######################
    Epoch: 1/5 Step: 100 Train Loss: 2.6070 Train Acc: 0.2928
    Epoch: 1/5 Step: 100 Dev Loss: 1.6841 Dev Acc: 0.4740
    Epoch: 1/5 Step: 200 Train Loss: 1.1751 Train Acc: 0.6205
    Epoch: 1/5 Step: 200 Dev Loss: 0.7841 Dev Acc: 0.7094
    Saved model with lowest validation loss: 0.7841
    Epoch: 1/5 Step: 300 Train Loss: 0.7315 Train Acc: 0.7128
    Epoch: 1/5 Step: 300 Dev Loss: 0.6191 Dev Acc: 0.7390
    Saved model with lowest validation loss: 0.6191
    Epoch: 1/5 Step: 400 Train Loss: 0.6154 Train Acc: 0.7340
    Epoch: 1/5 Step: 400 Dev Loss: 0.5609 Dev Acc: 0.7464
    Saved model with lowest validation loss: 0.5609
    Epoch: 1/5 Step: 500 Train Loss: 0.5744 Train Acc: 0.7425
    Epoch: 1/5 Step: 500 Dev Loss: 0.5403 Dev Acc: 0.7556
    Saved model with lowest validation loss: 0.5403
    Epoch: 1/5 Step: 600 Train Loss: 0.5574 Train Acc: 0.7448
    Epoch: 1/5 Step: 600 Dev Loss: 0.4909 Dev Acc: 0.7582
    Saved model with lowest validation loss: 0.4909
    ...
    100%|██████████| 5/5 [1:45:35<00:00, 1267.16s/it]


    #################### Fold: 2 ######################
    ...
    100%|██████████| 5/5 [1:44:46<00:00, 1257.26s/it]

    #################### Fold: 3 ######################
    ...
    100%|██████████| 5/5 [1:44:38<00:00, 1255.74s/it]

    #################### Fold: 4 ######################
    ...
    100%|██████████| 5/5 [1:46:05<00:00, 1273.11s/it]

    #################### Fold: 5 ######################
    ...
    100%|██████████| 5/5 [1:47:18<00:00, 1287.77s/it]
    #################### TRAIN IS OVER! ######################

[📈 Train and evaluation logs are visualized in wandb](https://wandb.ai/snoop2head/KLUE-RE/runs/htwirbkj?workspace=user-snoop2head)

## Inference for the Test Dataset

```python
def num_to_label(label):
  """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
  """
  origin_label = []
  with open(os.path.join(BASELINE_DIR, "dict_num_to_label.pkl"), 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])

  return origin_label

```

```python
from torch.utils.data import DataLoader, Dataset, Subset

test_dataset = pd.read_csv(PORORO_TEST_PATH)
display(test_dataset.head(2))
# test_dataset = test_dataset.drop(test_dataset.columns[0], axis=1)
test_dataset['label'] = 100
print(len(test_dataset))
test_set = CustomDataset(test_dataset, tokenizer, is_training=False)
print(len(test_set))
test_data_loader = DataLoader(test_set, batch_size=CFG.batch_size, num_workers=CFG.num_workers, shuffle=False)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>sentence</th>
      <th>label</th>
      <th>subject_entity</th>
      <th>object_entity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>지난 15일 [SUB-ORGANIZATION]MBC[/SUB-ORGANIZATION...</td>
      <td>100</td>
      <td>'[SUB-ORGANIZATION]MBC[/SUB-ORGANIZATION]'</td>
      <td>'[OBJ-O]탐사기획 스트레이트[/OBJ-O]'</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>사랑스러운 ‘[SUB-ARTIFACT]프린세스 프링[/SUB-ARTIFACT]’의 ...</td>
      <td>100</td>
      <td>'[SUB-ARTIFACT]프린세스 프링[/SUB-ARTIFACT]'</td>
      <td>'[OBJ-LOCATION]공주[/OBJ-LOCATION]'</td>
    </tr>
  </tbody>
</table>
</div>

    7765

```python
""" check whether test dataset is correctly mapped """
from tqdm import tqdm
for i, data in enumerate(tqdm(test_data_loader)) :
    print(data)
    break
```

      0%|          | 0/195 [00:00<?, ?it/s]

    {'input_ids': tensor([[    0,    11, 32067,  ...,     1,     1,     1],
            [    0,    11, 32001,  ...,     1,     1,     1],
            [    0,    11, 32067,  ...,  4271,  2145,     2],
            ...,
            [    0,    11, 32067,  ...,     1,     1,     1],
            [    0,    11, 32056,  ...,     1,     1,     1],
            [    0,    11, 32006,  ...,     1,     1,     1]]), 'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 1, 1, 1],
            ...,
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0],
            [1, 1, 1,  ..., 0, 0, 0]]), 'subject_mask': tensor([[0., 1., 1.,  ..., 0., 0., 0.],
            [0., 1., 1.,  ..., 0., 0., 0.],
            [0., 1., 1.,  ..., 0., 0., 0.],
            ...,
            [0., 1., 1.,  ..., 0., 0., 0.],
            [0., 1., 1.,  ..., 0., 0., 0.],
            [0., 1., 1.,  ..., 0., 0., 0.]]), 'object_mask': tensor([[0., 1., 1.,  ..., 0., 0., 0.],
            [0., 1., 1.,  ..., 0., 0., 0.],
            [0., 1., 1.,  ..., 0., 0., 0.],
            ...,
            [0., 1., 1.,  ..., 0., 0., 0.],
            [0., 1., 1.,  ..., 0., 0., 0.],
            [0., 1., 1.,  ..., 0., 0., 0.]]), 'label': tensor([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])}


      0%|          | 0/195 [00:00<?, ?it/s]

```python
oof_pred = []  # out of fold prediction list
for i in range(5):
  model_path = "/opt/ml/klue-level2-nlp-15/notebooks/results/{}-fold-5-best-eval-loss-model.pt".format(
    i + 1
  )
  model = RBERT(CFG["pretrained_model_name"], dropout_rate=CFG["dropout_rate"])
  model.load_state_dict(torch.load(model_path))
  model.to(device)
  model.eval()

  output_pred = []
  for i, data in enumerate(tqdm(test_data_loader)):
    with torch.no_grad():
      outputs = model(
        input_ids=data["input_ids"].to(device),
        attention_mask=data["attention_mask"].to(device),
        subject_mask=data["subject_mask"].to(device),
        object_mask=data["object_mask"].to(device),
        # token_type_ids=data['token_type_ids'].to(device) # RoBERTa does not use token_type_ids.
      )
      output_pred.extend(outputs.cpu().detach().numpy())
      output_pred = F.softmax(torch.Tensor(output_pred), dim=1)
      oof_pred.append(np.array(output_pred)[:, np.newaxis])

      # Prevent OOM error
      model.cpu()
      del model
      torch.cuda.empty_cache()

models_prob = np.mean(
        np.concatenate(oof_pred, axis=2), axis=2
    )  # probability of each class
result = np.argmax(models_prob, axis=-1)  # label
# print(result, type(result))
# print(models_prob.shape, list_prob)
```

    [ 3 12  0 ...  1  0  0] <class 'numpy.ndarray'>

```python
# save for submission
list_prob = models_prob.tolist()
test_id = test_dataset['id']
df_submission = pd.DataFrame({'id':test_id,'pred_label':num_to_label(result),'probs':list_prob,})
df_submission.to_csv('./prediction/submission_RBERT.csv', index=False)
```

```python

```
