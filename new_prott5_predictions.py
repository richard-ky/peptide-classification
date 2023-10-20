# Import data

import pickle
import lzma
import os

for filename in os.listdir("compressed"):
  with lzma.open(filename) as f:
    fasta = pickle.load(f)

  # Import libraries

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
    input_ids = torch.tensor(inputs['input_ids']).to(device)
    attention_mask = torch.tensor(inputs['attention_mask']).to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids,attention_mask=attention_mask)
        embeddings = torch.mean(outputs.last_hidden_state, dim=1) # Get the CLS token embedding
    return embeddings

  fasta_embeddings = [sentence_embedding(peptide) for peptide in fasta]
  fasta_embeddings = torch.cat(fasta_embeddings).to('cpu')

  # Load model

  from sklearn.svm import SVC
  from joblib import load

  sv_model = SVC()
  sv_model = load('sv_model')

  # Make predictions

  predictions = sv_model.predict(fasta_embeddings)

  # Serialize predictions

  with open('predictions/' + re.sub('compressed', 'predictions', filename), 'wb') as fp:
    pickle.dump(predictions, fp)