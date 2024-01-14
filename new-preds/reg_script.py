# ProtT5 model

from transformers import T5Tokenizer, T5EncoderModel
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))

transformer_link = "Rostlab/prot_t5_xl_half_uniref50-enc"
print("Loading: {}".format(transformer_link))
model = T5EncoderModel.from_pretrained(transformer_link)
model.full() if device=='cpu' else model.half() # only cast to full-precision if no GPU is available
model = model.to(device)
model = model.eval()
tokenizer = T5Tokenizer.from_pretrained(transformer_link, do_lower_case=False)

# Create ProtT5 embeddings for FASTA data

import re

def sentence_embedding(sentence):
  sentence = " ".join(list(re.sub(r"[UZOB]", "X", sentence)))
  inputs = tokenizer(sentence, padding='longest', return_tensors="pt").to(device)
  input_ids = inputs['input_ids'].clone().detach().to(device)
  attention_mask = inputs['attention_mask'].clone().detach().to(device)
  with torch.no_grad():
      outputs = model(input_ids=input_ids,attention_mask=attention_mask)
      embeddings = torch.mean(outputs.last_hidden_state, dim=1) # Get the CLS token embedding
  return embeddings

# Load model

from sklearn.svm import SVC
import pickle
import lzma
import os

with open('svc60.pkl', 'rb') as f:
  sv_model = pickle.load(f)

for filename in os.listdir('../compressed'):
  if 'bacteria' in filename:
  #if any([x in filename for x in ['viruses', 'leech', 'mouse', 'longhits', 'snake', 'Small']]):
    # Import data
    
    with lzma.open('../compressed/' + filename, 'rb') as f:
      data = pickle.load(f)

    embed = [sentence_embedding(peptide) for peptide in data]
    embed = torch.cat(embed).to('cpu')

    # Make predictions

    preds = sv_model.predict(embed)

    # Save embeddings and predictions

    with open('/preds/' + re.sub('compressed', 'preds.pkl', filename), 'wb') as f:
      pickle.dump(preds, f)

    with open('/embed/' + re.sub('compressed', 'embed.pkl', filename), 'wb') as f:
      pickle.dump(embed, f)