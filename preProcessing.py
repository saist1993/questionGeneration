'''

    This creates the whole pre-processing pipeline.


    Reads LC-QuAD specific data from the folder and creates
        --> Tokenize data
        --> Associated vocabulary
        --> Associated embeddings

'''
#For OpenNMT.
import sys

sys.path.append('OpenNMT')


# External Libraries
import json,os,spacy
from tqdm import tqdm
import numpy as np

'''
    Note that there is some bug is spacy and thus  msgpack needs to be downgraded 
        pip install msgpack==0.5.6
'''

# Internal functions
from utils.goodies import *
import utils.natural_language_utilities as nlutils
import options


def tokenize(text):
    '''


    :return: tokenized sentence as list
    '''

    doc = global_nlp(text)
    tokens = [token.text for token in doc]
    return tokens


def tokenize_updated_vocab(text):
    '''
        Take global variable for vocabulary and then update the vocaulary as it sees and new sentence
    :param text:
    :return:
    '''
    global global_vocab
    tokens = tokenize(text)
    global_vocab = list(set(global_vocab + tokens))
    return tokens


def load_emebedding_matrix_file(preprocess_opt):
    f = open(preprocess_opt.EmbeddingMatrixLocation)
    embeddings_index = {}
    print("parsing the embedding file")
    for line in tqdm(f):
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    return embeddings_index


def create_emebdding_matrix(len_special_tokens,preprocess_opt,special_token_opt,vocab,unique_token=True):

    embeddings_index = load_emebedding_matrix_file(preprocess_opt)
    embedding_matrix = np.zeros((len(vocab) + len_special_tokens, preprocess_opt.dim))

    #adding special token in the dictionary

    for word, i in tqdm(enumerate(vocab)):
        if word == special_token_opt.unknown:
            embedding_matrix[i] = np.zeros(preprocess_opt.dim)
        elif word == special_token_opt.start:
            embedding_matrix[i] = np.ones(preprocess_opt.dim)
        elif word == special_token_opt.end:
            embedding_matrix[i] = np.ones(preprocess_opt.dim)*-1
        elif word == special_token_opt.pad:
            embedding_matrix[i] = np.zeros(preprocess_opt.dim)
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                if unique_token:
                    embedding_matrix[i] = np.random.rand(preprocess_opt.dim)
                else:
                    embedding_matrix[i] = np.zeros(preprocess_opt.dim)
    embeddings_index = None
    return embedding_matrix


def main(return_embedding_matrix=False):

    #global vocab only contains special tokens
    global global_vocab
    special_token_length = len(global_vocab)
    nlutils.create_dir(preprocess_opt.SaveLocations)
    dataset = json.load(open(preprocess_opt.LCQuADLocation))

    # Printing some statistics:
    print(f"the length of the dataset is {len(dataset)}")
    print(f"a data point is {dataset[0]}")

    '''
        The expected keys are 
            'corrected_question' --> Human generated (target label),
            'verbalized_question' --> Computer generated (source label)

        A pair is a list of tuple with each tuple being 
        (verbalized_question, 'corrected_question' )
    '''

    pair = [(tokenize_updated_vocab(d['verbalized_question'].replace('<', '').replace('>', '').replace('?', '')),
             tokenize_updated_vocab(d['corrected_question'].replace('?', ''))) for d in tqdm(dataset)]


    '''
        There is a bug in the code. This assumes ordering already while inserting.
    '''

    global_vocab.insert(special_token_opt.padIndex,special_token_opt.pad)
    global_vocab.insert(special_token_opt.unknownIndex, special_token_opt.unknown)
    global_vocab.insert(special_token_opt.startIndex, special_token_opt.start)
    global_vocab.insert(special_token_opt.endIndex, special_token_opt.end)

    train_pair, val_pair, test_pair = \
        pair[:int(len(pair) * .80)], pair[int(len(pair) * .80):int(len(pair) * .90)], pair[int(len(pair) * .90):]

    print(f"length of train is {len(train_pair)}, test pair {len(test_pair)} and valid is {len(val_pair)}")

    '''
        create an embedding file
    '''

    if return_embedding_matrix:
        #Still need to check this part of the code

        embedding_matrix = create_emebdding_matrix(len_special_tokens=special_token_length,
                                                   preprocess_opt=preprocess_opt,
                                                   special_token_opt=special_token_opt,
                                                   vocab=global_vocab,
                                               unique_token=preprocess_opt.UniqueToken)
        np.save(preprocess_opt.EmbeddingSave,embedding_matrix)
    #Now save them at appropriate location
    np.save(preprocess_opt.TrainSave,train_pair)
    np.save(preprocess_opt.ValidSave,val_pair)
    np.save(preprocess_opt.TestSave,test_pair)
    np.save(preprocess_opt.VocabSave,global_vocab)



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



if __name__ == '__main__':
    # global variable
    global_vocab = []  # a list for the whole vocabulary.

    global_nlp = spacy.load('en')

    preprocess_opt = options.OptionsPreProcess()
    special_token_opt = options.OptionsSpecialToke()





    '''
        In this word embedding schema we are assigning a 
            random vector to each token which does not exists in embedding file
            also unk - 0's
                <s> - 1
                </s> - -1
    
    '''
    main()

    #pad id needs to be zero for easier retrival
    assert global_vocab[special_token_opt.padIndex] == special_token_opt.pad
    assert global_vocab[special_token_opt.unknownIndex] == special_token_opt.unknown
    assert global_vocab[special_token_opt.endIndex] == special_token_opt.end
    assert global_vocab[special_token_opt.startIndex] == special_token_opt.start