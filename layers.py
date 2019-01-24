import torch
import torch.nn as nn
import numpy as np
from functools import partial
import OpenNMT

class CustomEncoder(OpenNMT.onmt.encoders.rnn_encoder.RNNEncoder):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
       embeddings (:obj:`onmt.modules.Embeddings`): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, device, padding_idx=1, dropout=0.0, embeddings=None,
                 use_bridge=False, debug=False):

        '''


        '''
        super(OpenNMT.onmt.encoders.rnn_encoder.RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.device = device
        self.hidden_size = hidden_size
        self.mode = 'LSTM'
        self.debug = debug

        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.embeddings = embeddings

        # No other RNN supported

        self.rnn = torch.nn.LSTM(input_size=embeddings.embedding_size,
                                 hidden_size=self.hidden_size,
                                 num_layers=1,
                                 bidirectional=self.bidirectional)

        self.dropout = torch.nn.Dropout(p=self.dropout)

        #         # Initialize the bridge layer
        #         self.use_bridge = use_bridge
        #         if self.use_bridge:
        #             self._initialize_bridge(rnn_type,
        #                                     hidden_size,
        #                                     num_layers)
        self.reset_parameters()

    def init_hidden(self, batch_size, device):
        """
            Hidden states to be put in the model as needed.
        :param batch_size: desired batchsize for the hidden
        :param device: torch device
        :return:
        """
        if self.mode == 'LSTM':
            return (torch.ones((1 + self.bidirectional, batch_size, self.hidden_size), device=device),
                    torch.ones((1 + self.bidirectional, batch_size, self.hidden_size), device=device))
        else:
            return torch.ones((1 + self.bidirectional, batch_size, self.hidden_size), device=device)

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

    def forward(self, src, lengths=None):
        "See :obj:`EncoderBase.forward()`"
        #         self._check_args(src, lengths)

        mask = tu.compute_mask(src.squeeze().transpose(1, 0), padding_idx=self.padding_idx)
        h = self.init_hidden(src.squeeze().shape[1], self.device)

        x = self.embeddings(src)
        # get sorted v
        #         print(x.shape)
        lengths = mask.eq(self.padding_idx).long().sum(1)
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
                print("h_sort[0]:\t\t", h_sort[0].shape)
            else:
                print("h_sort:\t\t", h_sort.shape)

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

        return h_unsort, o_unsort, lengths