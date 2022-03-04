import pickle
import torch
import numpy as np
from torch import nn
from transformers import AutoTokenizer, AutoModel
import os

# Natural Language -> VAD model

class LanguageModel():
    def __init__(self, config):
        '''
        Sets up language model based on StyleNetCOnfig
        '''
        self.style_latent_dim = config.STYLE_LATENT_DIM
        self.no_style = config.NO_STYLE
        self.shared_fp = config.SHARED_FP

        # Saved word -> normalized VAD reference
        self.vad_ref = pickle.load(open(f"{self.shared_fp}/vad_ref.pkl", 'rb'))
        self.vad_ref[self.no_style] = np.zeros(self.style_latent_dim)
        self.vocab = list(self.vad_ref.keys())

        # Whether we have initialized sentence model or not
        self.sentence_model_init = False

    def init_sentence_model(self):
        print("Setting up sentence model")
        self.sentence_model = SentenceModel(self.shared_fp)
        self.sentence_model_init = True

    def project_ann(self, ann, fmt = 'np'):
        '''
        ann: a natural language string we'd like to project into the style latent space
        fmt: 'np' or 'torch'. what data format we want the projection to be

        Returns a projection of the annotation into the latent space.
        '''
        if ann in self.vocab:
            # Word model, just look it up
            proj = self.vad_ref[ann]
        else:
            # Sentence Model, initialize if needed, then run inference
            if not self.sentence_model_init:
                self.init_sentence_model()
            proj = self.sentence_model.predict(ann)

        # Tanh to get into -1 to 1 range
        proj = np.tanh(proj)
        if (fmt == 'torch'):
            proj = torch.tensor(proj, dtype = torch.float32)
        return proj


    def in_vocab(self, ann):
        '''
        ann: a natural language string we'd like to project into the style latent space
        
        Returns whether the annotation is entirely in vocabulary or not. Currently the
        model can handle any language but if you wanted to restrict that could add logic
        here.
        '''
        return True

    def random_vocab_word(self):
        # Returns a random word from the vocabulary
        return np.random.choice(self.vocab)

    def style_latent(self, ann, noise_amt = 0):
        '''
        ann: a natural language string we'd like to project into the style latent space
        noise_amt: standard deviation of gaussian noise we'd like to add to our projection

        return noisy projection of the annotation into the latent space.
        '''
        proj = self.project_ann(ann, fmt = 'torch')
        if (noise_amt == 0):
            return proj
        noise = torch.normal(mean = torch.zeros([self.style_latent_dim]), std = noise_amt)
        return proj + noise

# Sentence Model hyperparameters
SM_HPARAMS = {'dropout_p':0.5, 'hidden_sizes': [], 'BERT_MODEL': 'bert-base-uncased',
              'BERT_DIM': 768, 'VECTORS': ['Valence', 'Arousal', 'Dominance']}
class SentenceModel(nn.Module):
    def __init__(self, shared_fp):
        super().__init__()
        # Setup BERT model
        self.tokenizer = AutoTokenizer.from_pretrained(SM_HPARAMS['BERT_MODEL'])
        self.transformer = AutoModel.from_pretrained(SM_HPARAMS['BERT_MODEL'])

        LAYER_DIMS = [SM_HPARAMS['BERT_DIM']] + SM_HPARAMS['hidden_sizes'] \
                        + [len(SM_HPARAMS['VECTORS'])]
        layers = []
        for i in range(len(LAYER_DIMS) - 1):
            layers.append(nn.Dropout(p=SM_HPARAMS['dropout_p']))
            layers.append(nn.Linear(LAYER_DIMS[i], LAYER_DIMS[i+1])) 
            if (i < len(LAYER_DIMS) - 2):
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(p=SM_HPARAMS['dropout_p']))
            
        self.VAD_head = nn.Sequential(*layers)

        # Load finetuned parameters
        self.load_state_dict(torch.load( f'{shared_fp}/sm_params.pt',
                                    map_location=torch.device('cpu')))
        self.eval()
            
    def predict(self, text):
        # Basic inference
        encoding = self.tokenizer([text], padding = True, truncation = True, 
                                  return_tensors='pt')
        trans_output = self.transformer(**encoding)['pooler_output']
        head_output = self.VAD_head(trans_output)
        return head_output.squeeze().detach().numpy()

