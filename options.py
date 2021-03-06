'''
    Some options to set things up. Reset to more fine grained control later
'''
class ModelOptions():
    batch_size = 64
    valid_batch_size = 64
    batch_type = 'tokens'
    gpu_ranks = 1
    emb_size = 300
    rnn_size = 256
    rnn_type = 'LSTM'
    bidirectional = True
    num_layers = 1
    copy_attn = False
    generator_function = 'softmax'
    share_decoder_embeddings = True
    coverage_penalty = 'none'
    alpha = 0.0
    beta = 0.0
    length_penalty = 'none'
    n_best = 1

class OptionsPreProcess():
    LCQuADLocation = 'data/lcquad_data_set.json'
    EmbeddingMatrixLocation = 'data/glove.42B.300d.txt'
    dim = 300
    UniqueToken = True
    ReturnEmbeddingMatrix = True
    SaveLocations = 'data/preProcess/'
    ValidSave = SaveLocations+'valid.pt'
    TrainSave = SaveLocations+'train.pt'
    TestSave = SaveLocations+'test.pt'
    VocabSave = SaveLocations+'vocab.pt'
    EmbeddingSave = SaveLocations+'emb.pt'
    FileSuffix = '.npy'
    MaxSeqLen = 25

class OptionsSpecialToke():
    unknown = '<unk>'
    unknownIndex = 1
    start = '<s>'
    startIndex = 2
    end = '</s>'
    endIndex = 3
    pad = '<pad>'
    padIndex = 0