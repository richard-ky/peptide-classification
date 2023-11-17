from pickle import load
from datasets import Dataset

with open("peptides_dataset", "rb") as f:
  data = load(f)
train_dataset = data['train']
valid_dataset = data['valid']

from transformers import AutoModelForSequenceClassification

import torch

#model_ckpt="Rostlab/prot_bert"
model_ckpt="Rostlab/prot_t5_xl_half_uniref50-enc"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))

num_labels = 2
model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=num_labels)
         .to(device))

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
  return tokenizer(batch["text"], padding=True, truncation=True)

from sklearn.metrics import matthews_corrcoef

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  mcc = matthews_corrcoef(labels, preds)
  return {"mcc": mcc}

from transformers import Trainer, TrainingArguments

batch_size = 64
logging_steps = len(train_dataset) // batch_size
model_name = f"{model_ckpt}-finetuned"
training_args = TrainingArguments(output_dir=model_name,
                                  num_train_epochs=10,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,
                                  push_to_hub=False,
                                  log_level="error")

from transformers import Trainer

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=train_dataset,
                  eval_dataset=valid_dataset,
                  tokenizer=tokenizer)
trainer.train();

res = trainer.state.log_history

import numpy as np

preds_output = trainer.predict(train_dataset)
y_preds = np.argmax(preds_output.predictions, axis=1)
train_mcc = matthews_corrcoef(y_preds, train_dataset['label'])

preds_output = trainer.predict(valid_dataset)
y_preds = np.argmax(preds_output.predictions, axis=1)
val_mcc = matthews_corrcoef(y_preds, valid_dataset['label'])

res.append({"train_mcc": train_mcc, "val_mcc": val_mcc})

from pickle import dump
import pandas as pd

with open(str(pd.to_datetime('today').date()) + "_xl_" + str(batch_size), "wb") as f:
  dump(res, f)
