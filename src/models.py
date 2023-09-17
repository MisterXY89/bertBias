from src.setup import logger

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig


class Encoder(object):
    """
    wrapper for bert model, enabling encoding of sentences        
    """

    def __init__(self, hf_model_name='bert-base-uncased', load=False):
        self.hf_model_name = hf_model_name
        self.model = None        
        self.tokenizer = None        
        if load:            
            self.load_model(self.hf_model_name)    
                

    def load_model(self, model_name):
        model_config = AutoConfig.from_pretrained(
            model_name, 
            output_hidden_states=True,
            return_dict=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, config=model_config)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()

    
    def encode(self, texts, label):
        ''' Use tokenizer and model to encode texts '''
        logger.info(f"Encoding {label}...")
        encs = {}
        for text in tqdm(texts):
            tokenized = self.tokenizer.tokenize(text)
            indexed = self.tokenizer.convert_tokens_to_ids(tokenized)
            segment_idxs = [0] * len(tokenized)
            tokens_tensor = torch.tensor([indexed])
            segments_tensor = torch.tensor([segment_idxs])
            output = self.model(tokens_tensor, segments_tensor)
            
            # hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when config.output_hidden_states=True) — Tuple of torch.FloatTensor (one for the output of the embeddings, if the model has an embedding layer, + one for the output of each layer) of shape (batch_size, sequence_length, hidden_size).
            hidden_states = output["hidden_states"]            

            # extract the last rep of the first input
            embeds = hidden_states[-1][:, 0, :]            

            encs["text"] = embeds.detach().view(-1).numpy()

        return encs


# def encode(model, tokenizer, texts):
#     ''' Use tokenizer and model to encode texts '''0
#     embeddings = self.model.encode([sentence1, sentence2], convert_to_tensor=True)
        
#     embeddings1 = embeddings[0].unsqueeze(0)
#     embeddings2 = embeddings[1].unsqueeze(0)
#     encs = {}
#     for text in texts:
#         tokenized = tokenizer.tokenize(text)
#         indexed = tokenizer.convert_tokens_to_ids(tokenized)
#         segment_idxs = [0] * len(tokenized)
#         tokens_tensor = torch.tensor([indexed])
#         segments_tensor = torch.tensor([segment_idxs])
#         enc, _ = model(tokens_tensor, segments_tensor, output_all_encoded_layers=False)

#         enc = enc[:, 0, :]  # extract the last rep of the first input
#         encs[text] = enc.detach().view(-1).numpy()
#     return encs