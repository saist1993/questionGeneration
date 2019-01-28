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

from tqdm import tqdm
from typing import Callable
from utils.goodies import *
import time
import utils.tensor_utils as tu
import options
from layers import CustomEncoder

from models import CustomModel

device = torch.device('cpu')

debug = 1

opt_preprocess = options.OptionsPreProcess()
opt_special_token = options.OptionsSpecialToke()
opt_model_options = options.ModelOptions()

train_data = np.load(opt_preprocess.TrainSave+opt_preprocess.FileSuffix).tolist()
valid_data = np.load(opt_preprocess.ValidSave+opt_preprocess.FileSuffix).tolist()
test_data = np.load(opt_preprocess.TestSave+opt_preprocess.FileSuffix).tolist()


SP_CHARS = [opt_special_token.pad, opt_special_token.unknown, opt_special_token.start, opt_special_token.end]
concated = [x[0] + x[1] for x in train_data + valid_data]
vocab = []
for sent in concated:
    vocab += sent
vocab = SP_CHARS + list(set(vocab))

stoi = {word:index for index,word in enumerate(vocab)}
itos = {index:word for index,word in enumerate(vocab)}


class SimpleSampler():
    def __init__(self, data, bs: int = 64):
        try:
            assert len(data['src']) == len(data['tgt'])
        except:
            raise MismatchedDataError(f"Length of x is {len(data['src'])} while of y is {len(data['tgt'])}")

        self.x = data['src']
        self.y = data['tgt']
        self.n = len(self.x)
        self.bs = bs

    def __len__(self):
        return self.n // self.bs - (1 if self.n % self.bs else 0)

    def __iter__(self):
        self.i, self.iter = 0, 0
        return self

    def __next__(self):
        if self.i + self.bs >= self.n:
            raise StopIteration

        _x, _y = self.x[self.i:self.i + self.bs], self.y[self.i:self.i + self.bs]
        self.i += self.bs
        return _x, _y

add_sp = lambda x: [opt_special_token.start] + x + [opt_special_token.end]
train_x, train_y = [add_sp(x[0]) for x in train_data], [add_sp(x[1]) for x in train_data]
valid_x, valid_y = [add_sp(x[0]) for x in valid_data], [add_sp(x[1]) for x in valid_data]

train_tosample = {'src':train_x, 'tgt':train_y}
valid_tosample = {'src':valid_x, 'tgt':valid_y}

train_iter_fct = SimpleSampler(data=train_tosample,bs=opt_model_options.batch_size)
valid_iter_fct = SimpleSampler(data=valid_tosample,bs=opt_model_options.batch_size)

class NotSuchABetterEncoder(nn.Module):
    def __init__(self, max_length, hidden_dim, number_of_layer,
                 embedding_dim, vocab_size, bidirectional,
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

        mask = tu.compute_mask(x)

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
        h_sort = (h[0].index_select(1, idx_sort), h[1].index_select(1, idx_sort)) \
            if self.mode is "LSTM" else h.index_select(1, idx_sort)

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
        h_unsort = (h_sort[0].index_select(1, idx_unsort), h_sort[1].index_select(1, idx_unsort)) \
            if self.mode is "LSTM" else h_sort.index_select(1, idx_unsort)


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


class Decoder(nn.Module):
    '''
        TODO: Base paper (https://arxiv.org/pdf/1704.04368.pdf) might want us to use same embedding in enc, dec.
        Tie them up in that case.
    '''

    def __init__(self, vocab_size, inputsize=opt_model_options.rnn_size,
                 hidden_dim=opt_model_options.rnn_size,
                 embedding_dim=300,
                 vectors=None):

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

        self.rnn = nn.LSTM(self.inputsize + self.embedding_dim, self.hidden_dim)

    def forward(self, y_prev, h_prev):

        y_emb = self.embedding_layer(y_prev).transpose(0, 1)
        # if False:
        #     print(y_emb.shape)
        #     print(h_prev[0].shape)
        #     #             print(h[0].shape)
        #     print(torch.cat([y_emb, h_prev[0]], dim=-1).shape)
        o, h = self.rnn.forward(torch.cat([y_emb, h_prev[0]], dim=-1), h_prev)
        return o, h


# Code to run encoder
if True:
    args = {'max_length' : opt_preprocess.MaxSeqLen,
        'hidden_dim' : opt_model_options.rnn_size,
        'number_of_layer' : 1,
        'embedding_dim' : opt_preprocess.dim,
        'vocab_size' : len(vocab),
        'bidirectional' : True,
                 'dropout' : 0.0}
    encoder = NotSuchABetterEncoder(**args)
    encoder.to(device)
    h = encoder.init_hidden(opt_model_options.batch_size,device)
    x = torch.randint(0,len(vocab), (opt_model_options.batch_size,opt_preprocess.MaxSeqLen), device = device).long()
    output = encoder(x,h)

# Code to run decoder
if True:

    """
        Shall we put the whole loop thing within the forward?
            NO DUMMY. The attn thing comes in the loop and so does everything else needed to compute y_prev
    """

    args = {
        'hidden_dim': opt_model_options.rnn_size * 2,
        'embedding_dim': opt_preprocess.dim,
        'vocab_size': len(vocab),
        'inputsize': opt_model_options.rnn_size * 2,
    }
    decoder = Decoder(**args)
    seqlen = 5
    decoder.to(device)
    h_enc = output[2]
    h_prev = (h_enc[0].transpose(1, 0).contiguous().view(h_enc[0].shape[1], -1).unsqueeze(0), \
              h_enc[1].transpose(1, 0).contiguous().view(h_enc[1].shape[1], -1).unsqueeze(0))

    y_prev = torch.randint(0, len(vocab), (opt_model_options.batch_size, 1), device=device).long()
    #     h = decoder.init_hidden(opt_model_options.batch_size,device)

    casses = []
    for i in range(seqlen):
        s_t, h_prev = decoder(y_prev, h_prev)
        casses.append(s_t)

    s_ts = torch.cat(casses).shape