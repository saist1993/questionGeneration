
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn 
import numpy as np
from functools import partial

import sys
sys.path.append('OpenNMT')

import OpenNMT

import OpenNMT.onmt
import OpenNMT.onmt.inputters
import OpenNMT.onmt.modules
import OpenNMT.onmt.utils
from OpenNMT.onmt.models.model import NMTModel
from OpenNMT.onmt.encoders.rnn_encoder import RNNEncoder
from OpenNMT.onmt.decoders.decoder import InputFeedRNNDecoder
from OpenNMT.onmt.utils.misc import tile
import OpenNMT.onmt.translate
import traceback

from tqdm import tqdm_notebook as tqdm
from typing import Callable
from utils.goodies import *
import time
import utils.tensor_utils as tu
import options
from layers import CustomEncoder

# from models import CustomModel

device = torch.device('cuda')

debug = 1


# In[2]:


opt_preprocess = options.OptionsPreProcess()
opt_special_token = options.OptionsSpecialToke()
opt_model_options = options.ModelOptions()


# In[3]:


embeddings = torch.load(opt_preprocess.EmbeddingSave)


# In[4]:


vectors = np.asarray(embeddings.tolist())


# In[5]:


train_data = np.load(opt_preprocess.TrainSave+opt_preprocess.FileSuffix).tolist()
valid_data = np.load(opt_preprocess.ValidSave+opt_preprocess.FileSuffix).tolist()
test_data = np.load(opt_preprocess.TestSave+opt_preprocess.FileSuffix).tolist()
vocab = np.load(opt_preprocess.VocabSave+opt_preprocess.FileSuffix).tolist()

#Insert the code for embeddings here

#string to id and id to string
stoi = {word:index for index,word in enumerate(vocab)}
itos = {index:word for index,word in enumerate(vocab)}


# In[6]:


class PrepareData:

    def __init__(self, data, padding_idx: int, pad_len: int, vocab, bos=None,eos=None):
        '''
            data
            bos - begning of sentence
            eos - end of sentence
                If one sends bos and eos, idfy appends these tags in question
        '''

        self.data = data
        self.pad_idx = padding_idx
        self.vocab = vocab
        self.pad_len = pad_len
        self.bos = bos
        self.eos = eos

    def idfy(self):

        idfy_data = []

        for d in self.data:
            if self.bos:
                d = [self.bos] + d
            if self.eos:
                d = d + [self.eos]
            idfy_data.append([self.vocab[i] for i in d])

        return idfy_data

    def pad_data(self, idfy_data: list):

        padded_data = np.full((len(idfy_data), self.pad_len), self.pad_idx, dtype=np.int)
        for i, node in enumerate(idfy_data):
            padded_data[i][:min(len(node), self.pad_len)] = node[:min(len(node), self.pad_len)]

        return padded_data


# In[7]:


pd  = PrepareData([t[0] for t in train_data],stoi[opt_special_token.pad],opt_preprocess.MaxSeqLen,stoi,
                  opt_special_token.start,opt_special_token.end)
train_src = pd.pad_data(pd.idfy())
pd  = PrepareData([t[1] for t in train_data],stoi[opt_special_token.pad],opt_preprocess.MaxSeqLen,stoi,
                  opt_special_token.start,opt_special_token.end)
train_tgt = pd.pad_data(pd.idfy())
pd  = PrepareData([t[0] for t in valid_data],stoi[opt_special_token.pad],opt_preprocess.MaxSeqLen,stoi,
                  opt_special_token.start,opt_special_token.end)
valid_src = pd.pad_data(pd.idfy())
pd  = PrepareData([t[1] for t in valid_data],stoi[opt_special_token.pad],opt_preprocess.MaxSeqLen,stoi,
                  opt_special_token.start,opt_special_token.end)
valid_tgt = pd.pad_data(pd.idfy())


# In[8]:


class SimpleSampler():
    def __init__(self,data,bs:int = 64):
        try:
            assert len(data['src']) == len(data['tgt'])
        except:
            raise MismatchedDataError(f"Length of x is {len(data['src'])} while of y is {len(data['tgt'])}")
        
        self.x = data['src']
        self.y = data['tgt']
        self.n = len(self.x)
        self.bs = bs
        
    def __len__(self):
        return self.n//self.bs -(1 if self.n % self.bs else 0)
    
    def __iter__(self):
        self.i,self.iter=0,0
        return self
    
    def __next__(self):
        if self.i + self.bs >= self.n:
            raise StopIteration
        
        _x, _y = self.x[self.i:self.i + self.bs], self.y[self.i:self.i + self.bs]
        self.i += self.bs
        return _x, _y


# In[9]:


train_data = {
        'src' : train_src,
        'tgt' : train_tgt
}


valid_data = {
        'src' : valid_src,
        'tgt' : valid_tgt
}

train_iter_fct = SimpleSampler(data=train_data,bs=opt_model_options.batch_size)
valid_iter_fct = SimpleSampler(data=valid_data,bs=opt_model_options.batch_size)


# In[10]:


# class CustomModel(OpenNMT.onmt.models.model.NMTModel):

#     def forward(self,_x,_y):
#         '''
#             Explicitly passes it through generator
#             I don't know why
#         '''

#         h_unsort, o_unsort, lengths = self.encoder(_x.transpose(1,0).unsqueeze(-1))
#         if True:
#             print(f"h_unsort0 and 1 shape is, {h_unsort[0].shape,h_unsort[1].shape}, o unsort is {o_unsort.shape}")
#             print(f"_x.transpose(1,0).unsqueeze(-1) shape is {_x.transpose(1,0).unsqueeze(-1).shape}")
        
#         self.decoder.init_state(_x.transpose(1,0).unsqueeze(-1), o_unsort, h_unsort)
        
#         if True:
#             print(f"_y.transpose(1,0).unsqueeze(-1) shape is {_y.transpose(1,0).unsqueeze(-1).shape}")
        
        
#         dec_out, attns = self.decoder(_y.transpose(1,0).unsqueeze(-1), o_unsort,
#                                   memory_lengths=lengths)
#         dec_out = dec_out.view(-1, dec_out.size(2))
#         generated_output = self.generator(dec_out)
        
#         raise IOError
#         return generated_output


# def build_model(opt_model,opt_preprocess,opt_special_token,parameter_dict,device):
#     '''
    
#         parameter_dict = {
#             'length_of_vocab':300,
#             'itos':itos,
#             'stoi':stoi
#         }
    
#     '''
#         #preparing embeddings 
#     encoder_embeddings = OpenNMT.onmt.modules.Embeddings(opt_preprocess.dim, parameter_dict['length_of_vocab'],
#                                                  word_padding_idx=stoi[opt_special_token.pad])

#     decoder_embeddings = OpenNMT.onmt.modules.Embeddings(opt_preprocess.dim, parameter_dict['length_of_vocab'],
#                                                  word_padding_idx=stoi[opt_special_token.pad])

#         #Encoder and Decoder
#     encoder = CustomEncoder(hidden_size=opt_model.rnn_size, num_layers=opt_model.num_layers, 
#                                      rnn_type=opt_model.rnn_type, bidirectional=opt_model.bidirectional,
#                                      embeddings=encoder_embeddings,device=device,
#                             padding_idx = stoi[opt_special_token.pad])

#     decoder = OpenNMT.onmt.decoders.decoder.InputFeedRNNDecoder(hidden_size=opt_model.rnn_size*(1+opt_model.bidirectional),
#                                                         num_layers=opt_model.num_layers, 
#                                                bidirectional_encoder=opt_model.bidirectional,
#                                                rnn_type="LSTM", embeddings=decoder_embeddings)
#     model = CustomModel(encoder, decoder)
    
    
#     if True:
#         model.encoder.embeddings.load_pretrained_vectors(
#                     emb_file = opt_preprocess.EmbeddingSave)
#         model.decoder.embeddings.load_pretrained_vectors(emb_file = opt_preprocess.EmbeddingSave)

#         # Build Generator.
#     if not opt_model.copy_attn:
#         if opt_model.generator_function == "sparsemax":
#             gen_func = OpenNMT.onmt.modules.sparse_activations.LogSparsemax(dim=-1)
#         else:
#             gen_func = nn.LogSoftmax(dim=-1)
#         generator = nn.Sequential(
#             nn.Linear(opt_model.rnn_size*(1+opt_model.bidirectional),  parameter_dict['length_of_vocab']),
#             gen_func
#         )
#     #     if opt.share_decoder_embeddings:
#     #         generator[0].weight = decoder.embeddings.word_lut.weight
#     else:
#         vocab_size = len(vocab["tgt"])
#         pad_idx = vocab["tgt"].stoi[onmt.inputters.PAD_WORD]
#         generator = CopyGenerator(opt.rnn_size, vocab_size, pad_idx)
#     model.generator = generator
#     model.to(device)
#     return model


# In[11]:


class NotSuchABetterEncoder(nn.Module):
    def __init__(self, max_length, hidden_dim, number_of_layer,
                 embedding_dim, vocab_size, bidirectional,
                 padding_idx=0,
                 dropout=0.0, mode='LSTM', enable_layer_norm=False,
                 vectors=None, debug=False, residual=False):
        '''
            :param max_length: Max length of the sequence.
            :param hidden_dim: dimension of the output of the LSTM.
            :param number_of_layer: Number of LSTM to be stacked.
            :param embedding_dim: The output dimension of the embedding layer/ important only if vectors=none
            :param vocab_size: Size of vocab / number of rows in embedding matrix
            :param bidirectional: boolean - if true creates BIdir LStm
            :param vectors: embedding matrix
            :param debug: Bool/ prints shapes and some other meta data.
            :param enable_layer_norm: Bool/ layer normalization.
            :param mode: LSTM/GRU.
            :param residual: Bool/ return embedded state of the input.

        TODO: Implement multilayered shit someday.
        '''
        super(NotSuchABetterEncoder, self).__init__()

        self.max_length, self.hidden_dim, self.embedding_dim, self.vocab_size = int(max_length), int(hidden_dim), int(embedding_dim), int(vocab_size)
        self.enable_layer_norm = enable_layer_norm
        self.number_of_layer = number_of_layer
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.debug = debug
        self.mode = mode
        self.residual = residual
        self.padding_idx = padding_idx


        assert self.mode in ['LSTM', 'GRU']

        if vectors is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(vectors))
            self.embedding_layer.weight.requires_grad = True
        else:
            # Embedding layer
            self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Mode
        if self.mode == 'LSTM':
            self.rnn = torch.nn.LSTM(input_size=self.embedding_dim,
                                     hidden_size=self.hidden_dim,
                                     num_layers=1,
                                     bidirectional=self.bidirectional)
        elif self.mode == 'GRU':
            self.rnn = torch.nn.GRU(input_size=self.embedding_dim,
                                    hidden_size=self.hidden_dim,
                                    num_layers=1,
                                    bidirectional=self.bidirectional)
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.reset_parameters()

    def init_hidden(self, batch_size, device):
        """
            Hidden states to be put in the model as needed.
        :param batch_size: desired batchsize for the hidden
        :param device: torch device
        :return:
        """
        if self.mode == 'LSTM':
            return (torch.ones((1+self.bidirectional , batch_size, self.hidden_dim), device=device),
                    torch.ones((1+self.bidirectional, batch_size, self.hidden_dim), device=device))
        else:
            return torch.ones((1+self.bidirectional, batch_size, self.hidden_dim), device=device)

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, x, h):
        """

        :param x: input (batch, seq)
        :param h: hiddenstate (depends on mode. see init hidden)
        :param device: torch device
        :return: depends on booleans passed @ init.
        """

        if self.debug:
            print ("\tx:\t", x.shape)
            if self.mode is "LSTM":
                print ("\th[0]:\t", h[0].shape)
            else:
                print ("\th:\t", h.shape)

        mask = tu.compute_mask(x, padding_idx=self.padding_idx)

        x = self.embedding_layer(x).transpose(0, 1)

        if self.debug: print ("x_emb:\t\t", x.shape)

        if self.enable_layer_norm:
            seq_len, batch, input_size = x.shape
            x = x.view(-1, input_size)
            x = self.layer_norm(x)
            x = x.view(seq_len, batch, input_size)

        if self.debug: print("x_emb bn:\t", x.shape)

        # get sorted v
        lengths = mask.eq(1).long().sum(1)
        lengths_sort, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        x_sort = x.index_select(1, idx_sort)
        h_sort = (h[0].index_select(1, idx_sort), h[1].index_select(1, idx_sort))             if self.mode is "LSTM" else h.index_select(1, idx_sort)

        x_pack = torch.nn.utils.rnn.pack_padded_sequence(x_sort, lengths_sort)
        x_dropout = self.dropout.forward(x_pack.data)
        x_pack_dropout = torch.nn.utils.rnn.PackedSequence(x_dropout, x_pack.batch_sizes)

        if self.debug:
            print("\nidx_sort:", idx_sort.shape)
            print("idx_unsort:", idx_unsort.shape)
            print("x_sort:", x_sort.shape)
            if self.mode is "LSTM":
                print ("h_sort[0]:\t\t", h_sort[0].shape)
            else:
                print ("h_sort:\t\t", h_sort.shape)


        o_pack_dropout, h_sort = self.rnn.forward(x_pack_dropout, h_sort)
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack_dropout)

        # Unsort o based ont the unsort index we made
        o_unsort = o.index_select(1, idx_unsort)  # Note that here first dim is seq_len
        h_unsort = (h_sort[0].index_select(1, idx_unsort), h_sort[1].index_select(1, idx_unsort))             if self.mode is "LSTM" else h_sort.index_select(1, idx_unsort)


        # @TODO: Do we also unsort h? Does h not change based on the sort?

        if self.debug:
            if self.mode is "LSTM":
                print("h_sort\t\t", h_sort[0].shape)
            else:
                print("h_sort\t\t", h_sort.shape)
            print("o_unsort\t\t", o_unsort.shape)
            if self.mode is "LSTM":
                print("h_unsort\t\t", h_unsort[0].shape)
            else:
                print("h_unsort\t\t", h_unsort.shape)

        len_idx = (lengths - 1).view(-1, 1).expand(-1, o_unsort.size(2)).unsqueeze(0)

        if self.debug:
            print("len_idx:\t", len_idx.shape)

        # Need to also return the last embedded state. Wtf. How?
        if self.residual:
            len_idx = (lengths - 1).view(-1, 1).expand(-1, x.size(2)).unsqueeze(0)
            x_last = x.gather(0, len_idx)
            x_last = x_last.squeeze(0)
            return o_unsort, h_unsort[0].transpose(1,0).contiguous().view(h_unsort[0].shape[1], -1) , h_unsort, mask, x, x_last
        else:
            return o_unsort, h_unsort[0].transpose(1,0).contiguous().view(h_unsort[0].shape[1], -1) , h_unsort, mask

    @property
    def layers(self):
        return torch.nn.ModuleList([
            torch.nn.ModuleList([self.embedding_layer, self.rnn, self.dropout]),
        ])


# In[12]:


class Decoder(nn.Module):
    '''
        TODO: Base paper (https://arxiv.org/pdf/1704.04368.pdf) might want us to use same embedding in enc, dec.
        Tie them up in that case.
    '''
    
    def __init__(self, vocab_size, inputsize=opt_model_options.rnn_size,
                 hidden_dim=opt_model_options.rnn_size,
                 embedding_dim = 300,
                 vectors = None):
        
        super(Decoder, self).__init__()
        
        self.hidden_dim, self.embedding_dim, self.vocab_size = int(hidden_dim), int(embedding_dim), int(vocab_size)
        self.inputsize = inputsize
        self.bidirectional = False
        self.mode = 'LSTM'
        if vectors is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(vectors))
            self.embedding_layer.weight.requires_grad = True
        else:
            # Embedding layer
            self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
       
        self.rnn = nn.LSTM(self.inputsize+self.embedding_dim, self.hidden_dim)
        
    def forward(self,y_prev,h_prev):
        
        y_emb = self.embedding_layer(y_prev).transpose(0, 1)
        if False:
            print(y_emb.shape)
            print(h_prev[0].shape)
#             print(h[0].shape)
            print(torch.cat([y_emb,h_prev[0]],dim = -1).shape)
        o,h =  self.rnn.forward(torch.cat([y_emb,h_prev[0]],dim = -1), h_prev)
        return o,h


# In[13]:


class Attention(nn.Module):
    def __init__(self,enc_hdim,dec_hdim,odim):
        super(Attention,self).__init__()
        
        self.W_h = nn.Linear(enc_hdim,odim, bias=True)
        self.W_s = nn.Linear(dec_hdim,odim, bias=False)
        self.v_t = nn.Linear(odim,1,bias=False)
#         self.b_attn = nn.Parameter(torch.randn(odim))
    
    def forward(self,h_all,s_t, mask):
        
        mask = mask.transpose(1, 0) # should now be sl, bs
        
        temp = self.W_h(h_all) + (self.W_s(s_t)).repeat(h_all.shape[0],1,1)
#                 temp = self.W_h(h_all) + (self.W_s(s_t) + self.b_attn).repeat(h_all.shape[0],1,1)
        temp = nn.functional.tanh(temp).squeeze()
        energy = self.v_t(temp)
        
        if False:
            # energy.shape should be sl, bs
            print(f"Energy shape is {energy.shape}")
        
        attn = tu.masked_softmax(energy.squeeze(), mask, dim=0)
        return attn        


# In[14]:


class LinearDecoder(nn.Module):
    
    def __init__(self, vocab_size, hdim):
        super(LinearDecoder,self).__init__()
        self.lina = nn.Linear(hdim, int(hdim/2))
        self.linb = nn.Linear(int(hdim/2), vocab_size)
        
    def forward(self, s_t, h_new):
        if False:
            print(f"shape of s_t is {s_t.shape}, h_new shape is {h_new.shape},and cat shape is {torch.cat([s_t, h_new], dim=-1).shape}")
#         return torch.softmax(self.linb(self.lina(torch.cat([s_t, h_new], dim=-1))), dim=-1)
        return torch.log_softmax(self.linb(self.lina(torch.cat([s_t, h_new], dim=-1))),dim=-1)


# In[15]:


class Model(nn.Module):
    
    def __init__(self,vocab,device,max_output_len,start_token=2,vectors=None):
        super().__init__()
        args = {'max_length' : opt_preprocess.MaxSeqLen,
            'hidden_dim' : opt_model_options.rnn_size,
            'number_of_layer' : 1,
            'embedding_dim' : opt_preprocess.dim,
            'vocab_size' : len(vocab),
            'bidirectional' : True,
                     'dropout' : 0.2,
               'vectors':vectors}
        
        self.encoder = NotSuchABetterEncoder(**args)
        self.encoder.to(device)

        args = {
            'hidden_dim' : opt_model_options.rnn_size*2,
            'embedding_dim' : opt_preprocess.dim,
            'vocab_size' : len(vocab),
            'inputsize' : opt_model_options.rnn_size*2,
            'vectors' : vectors
                }
        self.decoder = Decoder(**args)
        self.decoder.to(device)

        args = {
            'enc_hdim' : opt_model_options.rnn_size*2,
            'dec_hdim' : opt_model_options.rnn_size*2,
            'odim': 25
        }

        self.attention = Attention(**args)
        self.attention.to(device)
    
        args = {
            'vocab_size' : len(vocab),
            'hdim' : int(opt_model_options.rnn_size*4),
        }

        self.linear_decoder = LinearDecoder(**args)
        self.linear_decoder.to(device)
        
        self.out_seqlen = max_output_len
        
        self.teacher_force = True
        self.start_token = start_token
        
        
#     def eval(self):
#         self.teacher_force = False
#         super().eval()
        
        
#     def train(self):
#         super().train()
#         self.teacher_force = True
    
    def run_model(self,x,y=None,teacher_forcing=True):
        '''
        
            x --> bs*sl
        '''
        
        if teacher_forcing == False:
            self.teacher_force = False
        else:
            self.teacher_forcing = True

        h = self.encoder.init_hidden(opt_model_options.batch_size,device)
        
        
#         x = torch.randint(0,100, (64,25), device = device).long()
        
        output = self.encoder(tu.trim(x),h)
        
        if False:
            print(f"x shape is {x.shape}")
            print(f"y shape is {y.shape}")
            print(f"h shape is {h[0].shape}")
            print(f"output[0],h_all shape is {output[0].shape}")
            print(f"dec_h[0] shape is {output[2][0].shape}")
            print(f"shape of mask is {output[3].shape}")
            
        
        
        dec_h = output[2]
        dec_h_prev = (dec_h[0].transpose(1,0).contiguous().view(dec_h[0].shape[1], -1).unsqueeze(0),                  dec_h[1].transpose(1,0).contiguous().view(dec_h[1].shape[1], -1).unsqueeze(0))
        h_all = output[0] #bascially h_i's
#         y_prev = torch.randint(0,len(vocab), (opt_model_options.batch_size,1), device = device).long()
        # Need to check this  --> recheck the logic
        h_prev = dec_h_prev
        
        if self.teacher_force:
            y_prev = y[0]
        else:
            y_prev = torch.tensor(torch.Tensor(64),dtype=torch.long, device=device)
            y_prev = y_prev.fill_(self.start_token).unsqueeze(-1)
        preds = [] 
        for t in range(self.out_seqlen-1):
        
            # The killer loop
            
            #since teacher forcing --> recheck the logic
            if self.teacher_force:
                y_prev = y[:,t].unsqueeze(-1)
#             if True:
#                 print(y_prev)
#                 print(f"y_prev shape is {y_prev.shape}")
            s_t, h_prev = self.decoder(y_prev,h_prev)
            # Compute attention
            
            if False:
                print(f"shape of s_t is {s_t.shape}")
            attn = self.attention(h_all=h_all, s_t=s_t, mask=output[3])
            #update h's with new attention
            h_new_all = torch.einsum('ij,ijk->ijk', [attn,h_all])
            h_new = torch.einsum('ijk->jk', [h_new_all])
            #pass it through linear to finally get the probablity for t'th position
            final_output = self.linear_decoder(s_t=s_t.squeeze(),h_new=h_new)
#             if True:
#                 print(final_output.shape)
#                 print(torch.argmax(final_output,dim=1).unsqueeze(-1))
#                 print(torch.argmax(final_output,dim=1).unsqueeze(-1).shape)
            preds.append(final_output)
            
            if not self.teacher_force:
                y_prev = torch.argmax(final_output,dim=1).unsqueeze(-1)
            
            
        
        return torch.stack(preds).view(-1,preds[0].shape[-1])


# In[16]:


parameter_dict = {
    'length_of_vocab':len(stoi),
            'itos':itos,
            'stoi':stoi
}
# model = build_model(opt_model = opt_model_options,
#                     opt_preprocess = opt_preprocess,
#                     opt_special_token= opt_special_token,
#                     parameter_dict=parameter_dict,
#                     device = device)

model = Model(vocab=stoi, device=device, max_output_len=opt_preprocess.MaxSeqLen,vectors=None)


# In[17]:


for x,y in train_iter_fct:
    break
    
_x = torch.tensor(x, dtype=torch.long, device=device)
_y = torch.tensor(y, dtype=torch.long, device=device)
preds = model.run_model(_x,_y)


# In[24]:


#Helper functions
def evaluate(y_pred,y_true):

#     gtruth = y_true.transpose(1,0).contiguous().view(-1)
#     return torch.mean((torch.argmax(y_pred,dim=1) == gtruth).float())

#     y_pred = y_pred[opt_model_options.batch_size:]
    y_true = y_true[:,1:]
    match = torch.eq(torch.argmax(y_pred, dim=1).reshape(opt_preprocess.MaxSeqLen-1, opt_model_options.batch_size).transpose(1,0), y_true).float()
    acc=[]
    for i,j in (y_true == opt_special_token.endIndex).nonzero():
        acc.append(torch.mean(match[i,:j]).item())
#     acc, np.mean(acc)
    return np.mean(acc)



def translate(y,itos):
    '''
        y --> b*sl
    '''
    
    generated_sentence = []
    for sent in y:
        generated_sentence.append([itos(i) for i in sent])
    
    return generated_sentence

def loss_wrapper(criterion,y_pred,y_true):
#     gtruth = y_true.transpose(1,0).contiguous().view(-1)
#     return criterion(y_pred,gtruth)
#     '''
#     y_pred = y_pred[opt_model_options.batch_size:]
    y_true = y_true[:,1:]
    gtruth = y_true.transpose(1,0).contiguous().view(-1)
    return criterion(y_pred,gtruth)


# In[25]:


#Training loop

def generic_loop(epochs: int,
                 device: torch.device,
                 opt: torch.optim,
                 loss_fn: torch.nn,
                 model: torch.nn.Module,
                 train_fact_iter,
                 valid_fact_iter,
                 weight_decay: float = 0.0,
                 clip_grads_at: float = -1.0,
                 lr_schedule=None,
                 eval_fn: Callable = None) -> (list, list, list):

    train_loss = []
    train_acc = []
    val_acc = []
    lrs = []

    # Epoch level
    for e in range(epochs):

        per_epoch_loss = []
        per_epoch_tr_acc = []

        # Train
        with Timer() as timer:



            # Make data
#             trn_dl, val_dl = data_fn(data['train']), data_fn(data['valid'])
#             trn_dl = train_fact_iter
            for x, y in tqdm(train_fact_iter):

#                 if batch_start_hook: batch_start_hook()
                opt.zero_grad()

                if lr_schedule: lrs.append(update_lr(opt, lr_schedule.get()))

                _x = torch.tensor(x, dtype=torch.long, device=device)
                _y = torch.tensor(y, dtype=torch.long, device=device)
                try:
                    y_pred = model.run_model(_x,_y,True)
                except:
                    print(traceback.print_exc())
                    return (_x,_y)
                try:
                    loss = loss_fn(y_pred = y_pred, y_true = _y)
                except:
                    print(y_pred.shape,_y.shape)
                    return(_x,_y)
                per_epoch_tr_acc.append(eval_fn(y_pred=y_pred, y_true=_y).item())
                per_epoch_loss.append(loss.item())

                loss.backward()

                if clip_grads_at > 0.0: torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grads_at)
#                 for group in opt.param_groups:
#                     for param in group['params']:
#                         param.data = param.data.add(-weight_decay * group['lr'], param.data)

                opt.step()


        # Val
        with torch.no_grad():

            per_epoch_vl_acc = []
            for x, y in tqdm(valid_fact_iter):
                _x = torch.tensor(x, dtype=torch.long, device=device)
                _y = torch.tensor(y, dtype=torch.long, device=device)
                
#                 model.eval()
                
                y_pred = model.run_model(_x,None,False)
                loss = loss_fn(y_pred = y_pred, y_true = _y)
                per_epoch_vl_acc.append(eval_fn(y_pred=y_pred, y_true=_y).item())
#                 model.train()
#                 per_epoch_vl_acc.append(loss.item())

        # Bookkeep
#         per_epoch_vl_acc = [0] # @TODO:Remove this once we start calculating accuracy.
        train_acc.append(np.mean(per_epoch_tr_acc))
        train_loss.append(np.mean(per_epoch_loss))
        val_acc.append(np.mean(per_epoch_vl_acc))

        print("Epoch: %(epo)03d | Loss: %(loss).5f | Tr_c: %(tracc)0.5f | Vl_c: %(vlacc)0.5f | Time: %(time).3f min"
              % {'epo': e,
                 'loss': float(np.mean(per_epoch_loss)),
                 'tracc': float(np.mean(per_epoch_tr_acc)),
                 'vlacc': float(np.mean(per_epoch_vl_acc)),
                 'time': timer.interval / 60.0})

    return train_acc, train_loss, val_acc, lrs


# In[26]:


#Setting up network
criterion = nn.NLLLoss(ignore_index=stoi[opt_special_token.pad], reduction='sum')
loss_fn = partial(loss_wrapper, criterion = criterion)
# params = [p for p in model.parameters() if p.requires_grad]

# optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, model.encoder.parameters())) +
#                                    list(filter(lambda p: p.requires_grad, model.decoder.parameters())) +
#                                 list(filter(lambda p: p.requires_grad, model.attention.parameters())) +
#            list(filter(lambda p: p.requires_grad, model.linear_decoder.parameters())))

optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())))
#  torch.optim.Adam(params, lr=0.001, eps=1e-9)


# In[27]:


model


# In[28]:


# for p in model.parameters():
#     print(p.shape)
#     input()
    


# In[29]:


args = {
        'epochs' :  30,
        'device' :  device,
        'opt' : optimizer,
        'loss_fn' : loss_fn,
        'model' : model,
        'train_fact_iter' : train_iter_fct,
        'valid_fact_iter' : valid_iter_fct,
        'eval_fn':evaluate
}


traces = generic_loop(**args)


# In[30]:


from matplotlib import pyplot as plt
from matplotlib import style as pltstyle
get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (16, 8)

def plot_accs(tra, vla, style='seaborn-deep', _savedir=None):
    pltstyle.use(style)
    fig = plt.figure(figsize = (16,8))
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
#     plt.xticks([])
#     plt.yticks([])
    plt.plot(tra, label=f"Train Acc", linewidth=3)
    plt.plot(vla, label=f"Valid Acc", linewidth=3)
    plt.legend()
    plt.show()

    if _savedir:
        print(f"saving model at {_savedir}")
        plt.savefig(_savedir)

plot_accs(traces[0], traces[2])


# In[ ]:


train_data = [t for t in valid_iter_fct]
data = train_data[0]
x, y = data[0], data[1]
_x = torch.tensor(x, dtype=torch.long, device=device)
_y = torch.tensor(y, dtype=torch.long, device=device)

output = model.run_model(_x, None,False)
o = torch.argmax(output, dim=1).tolist()


# In[ ]:


k = [2,3,7,11]
for j in k:
    temp = []
    for i in range(25):
        temp.append(itos[o[i * 64+j]])
    print(temp)
    count = 0
    for sent in _y.tolist():
        temp = []
        for i in sent:
            temp.append(itos[i])
        if j == count:
            print(temp)
        count = count + 1


# In[ ]:


valid_data = np.load(opt_preprocess.ValidSave+opt_preprocess.FileSuffix).tolist()


# In[ ]:


valid_data

