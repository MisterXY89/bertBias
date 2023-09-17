from src.setup import logger

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModel, AutoTokenizer, BertTokenizer, BertModel, AutoModelForMaskedLM, AutoConfig


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
        
        if self.hf_model_name.startswith("bert"):
            self.model = BertModel.from_pretrained(model_name, config=model_config)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.model = AutoModel.from_pretrained(model_name, config=model_config)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.model.eval()


    def encode(self, texts, label):
        """
        Use tokenizer and model to encode texts
        see also: https://github.com/huggingface/transformers/issues/1950
        """
        logger.info(f"Encoding {label}...")        
        encs = {}
        for text in tqdm(texts):
            input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0)  # Batch size 1            
            outputs = self.model(input_ids)
            last_hidden_states = outputs[0][:, 0, :]  # The last hidden-state is the first element of the output tuple            
            encs[text] = last_hidden_states.detach().view(-1).numpy()

        return encs
