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


#Accessing macros
opt_preprocess = options.OptionsPreProcess()
opt_special_token = options.OptionsSpecialToke()
opt_model_options = options.ModelOptions()


#Loading the data
train_data = np.load(opt_preprocess.TrainSave+opt_preprocess.FileSuffix).tolist()
valid_data = np.load(opt_preprocess.ValidSave+opt_preprocess.FileSuffix).tolist()
test_data = np.load(opt_preprocess.TestSave+opt_preprocess.FileSuffix).tolist()
vocab = np.load(opt_preprocess.VocabSave+opt_preprocess.FileSuffix).tolist()

#Insert the code for embeddings here

#string to id and id to string
stoi = {word:index for index,word in enumerate(vocab)}
itos = {index:word for index,word in enumerate(vocab)}


#Think of adding this somewhere later.
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


#Idfying data
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


#Data iter creation
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





#Final model buildup --> also move it somewhere
def build_model(opt_model, opt_preprocess, opt_special_token, parameter_dict, device):
    '''

        parameter_dict = {
            'length_of_vocab':300,
            'itos':itos,
            'stoi':stoi
        }

    '''
    # preparing embeddings
    encoder_embeddings = OpenNMT.onmt.modules.Embeddings(opt_preprocess.dim, parameter_dict['length_of_vocab'],
                                                         word_padding_idx=stoi[opt_special_token.pad])

    decoder_embeddings = OpenNMT.onmt.modules.Embeddings(opt_preprocess.dim, parameter_dict['length_of_vocab'],
                                                         word_padding_idx=stoi[opt_special_token.pad])

    # Encoder and Decoder
    encoder = CustomEncoder(hidden_size=opt_model.rnn_size, num_layers=opt_model.num_layers,
                            rnn_type=opt_model.rnn_type, bidirectional=opt_model.bidirectional,
                            embeddings=encoder_embeddings, device=device,
                            padding_idx=stoi[opt_special_token.pad])

    decoder = OpenNMT.onmt.decoders.decoder.InputFeedRNNDecoder(
        hidden_size=opt_model.rnn_size * (1 + opt_model.bidirectional),
        num_layers=opt_model.num_layers,
        bidirectional_encoder=opt_model.bidirectional,
        rnn_type="LSTM", embeddings=decoder_embeddings)
    model = CustomModel(encoder, decoder)

    if False:
        model.encoder.embeddings.load_pretrained_vectors(
            emb_file=opt_preprocess.EmbeddingSave, fixed=False)
        model.decoder.embeddings.load_pretrained_vectors(emb_file=opt_preprocess.EmbeddingSave, fixed=False)

        # Build Generator.
    if not opt_model.copy_attn:
        if opt_model.generator_function == "sparsemax":
            gen_func = OpenNMT.onmt.modules.sparse_activations.LogSparsemax(dim=-1)
        else:
            gen_func = nn.LogSoftmax(dim=-1)
        generator = nn.Sequential(
            nn.Linear(opt_model.rnn_size * (1 + opt_model.bidirectional), parameter_dict['length_of_vocab']),
            gen_func
        )
    #     if opt.share_decoder_embeddings:
    #         generator[0].weight = decoder.embeddings.word_lut.weight
    else:
        vocab_size = len(vocab["tgt"])
        pad_idx = vocab["tgt"].stoi[onmt.inputters.PAD_WORD]
        generator = CopyGenerator(opt.rnn_size, vocab_size, pad_idx)
    model.generator = generator
    model.to(device)
    return model



# Helper functions
def evaluate(y_pred, y_true):
    gtruth = y_true.transpose(1, 0).contiguous().view(-1)
    return torch.mean((torch.argmax(y_pred, dim=1) == gtruth).float())


def translate(y, itos):
    '''
        y --> b*sl
    '''

    generated_sentence = []
    for sent in y:
        generated_sentence.append([itos(i) for i in sent])

    return generated_sentence


def loss_wrapper(criterion, y_pred, y_true):
    gtruth = y_true.transpose(1, 0).contiguous().view(-1)
    return criterion(y_pred, gtruth)


# Training loop

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
                    y_pred = model(_x, _y)
                except:
                    print(traceback.print_exc())
                    return (_x, _y)
                try:
                    loss = loss_fn(y_pred=y_pred, y_true=_y)
                except:
                    print(y_pred.shape, _y.shape)
                    return (_x, _y)
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

                model.eval()

                y_pred = model(_x, _y)
                loss = loss_fn(y_pred=y_pred, y_true=_y)
                per_epoch_vl_acc.append(eval_fn(y_pred=y_pred, y_true=_y).item())
                model.train()
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

#data dressup
train_data = {
        'src' : train_src,
        'tgt' : train_tgt
}


valid_data = {
        'src' : valid_src,
        'tgt' : valid_tgt
}

#Final data iters
train_iter_fct = SimpleSampler(data=train_data,bs=opt_model_options.batch_size)
valid_iter_fct = SimpleSampler(data=valid_data,bs=opt_model_options.batch_size)


parameter_dict = {
    'length_of_vocab':len(stoi),
            'itos':itos,
            'stoi':stoi
}
model = build_model(opt_model = opt_model_options,
                    opt_preprocess = opt_preprocess,
                    opt_special_token= opt_special_token,
                    parameter_dict=parameter_dict,
                    device = device)


#Setting up network
criterion = nn.NLLLoss(ignore_index=stoi[opt_special_token.pad], reduction='sum')
loss_fn = partial(loss_wrapper, criterion = criterion)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.001, eps=1e-9)


args = {
        'epochs' :  20,
        'device' :  device,
        'opt' : optimizer,
        'loss_fn' : loss_fn,
        'model' : model,
        'train_fact_iter' : train_iter_fct,
        'valid_fact_iter' : valid_iter_fct,
        'eval_fn':evaluate
}


traces = generic_loop(**args)


#Some backup data. Still need to organize it properly
if False:
    train_data = [t for t in valid_iter_fct]
    data = train_data[0]
    x, y = data[0], data[1]
    _x = torch.tensor(x, dtype=torch.long, device=device)
    _y = torch.tensor(y, dtype=torch.long, device=device)

    output = model(_x, _y)

    o = torch.argmax(output, dim=1).tolist()

    for i in range(25):
        print(vocab['src'].itos[o[i * 64 + 2]], (i * 64) + 1)
    argmax_output = torch.argmax(output, dim=1)
    argmax_output_reshape = argmax_output.view(64, 25)
    print(argmax_output)
    print(argmax_output.view(64, 25))
    for sent in argmax_output_reshape.tolist():
        temp = []
        for i in sent:
            temp.append(vocab['src'].itos[i])
        print(temp)

    for sent in _y.tolist():
        temp = []
        for i in sent:
            temp.append(vocab['src'].itos[i])
        print(temp)
