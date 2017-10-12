import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import sys
import tarfile
import re
import collections
import string
from operator import itemgetter
from pprint import pprint as pp

batch_size = 50

class Config(object):
    embedding_size = 50
    dictionary_size = 50000
    record_length = 40
    embeddings_file = "glove.6B.50d.txt"
    reviews_file = "reviews.tar.gz"
    reviews_path = os.path.join(os.path.dirname(__file__), 'review_data/')
    debug = 1
    hidden_units_size = 1
    dropout_keep_prob = 1
    cell_size = 128
    learning_rate = 0.001



def printd(print_string, **kwargs):
    c = Config()
    if c.debug > 0:
        print(print_string.format(**kwargs))

# Read the data into a list of strings.
def extract_data(filename, path):
    printd("Extracting zipped data {p}{f}",p=path,f=filename)
    if not os.path.exists(path):
        with tarfile.open(filename, "r") as tarball:
            tarball.extractall(path)
    printd("Extracted zipped data successfully")
    return

def read_data(path):
    printd("Reading data {p}",p=path)
    data = []
    dir = os.path.dirname(__file__)
    file_list = glob.glob(os.path.join(path,'pos/*'))
    file_list.extend(glob.glob(os.path.join(path,'neg/*')))

    printd("Parsing {lf} files", lf=len(file_list))
    for f in file_list:
        with open(f, "r") as openf:
            s = openf.read()
            no_punct = ''.join(c for c in s if c not in string.punctuation).lower()
            data.append(no_punct.split())
    printd("Read data {p} successfully with length {ld}",p=path,ld=len(data))
    for i in range(len(data[:5])):
        print(len(data[i]),data[i][:10])
    return data

def count_words(data):
    printd('Counting words in dataset')
    counts = dict()
    for entry in data:
        for word in entry:
            if word in counts.keys():
                counts[word] += 1
            else:
                counts[word] = 1
    return counts

def build_dictionary(data, counts, glove_dict, dictionary_size):
    dictionary = dict()
    exceptions = list()
    sorted_counts = sorted(counts.items(), key=lambda x:x[1], reverse=True)
    # print('Top 10 words: {}'.format(sorted_counts[:10]))
    # print('Bottom 10 words: {}'.format(sorted_counts[-10:]))
    # print('Vocabulary cuttoff: {}'.format(sorted_counts[dictionary_size]))
    printd('Counted {sv} words, {lc} distinct', sv=np.sum([i for i in counts.values()]), lc=len(counts))
    i = 0
    while i < 100000 and len(dictionary) < dictionary_size:
        key, value = sorted_counts[i]
        if key in glove_dict.keys():
            dictionary[key] = glove_dict[key]
        else:
            exceptions.extend([key,value])
        i += 1

    dictionary['UNK'] = glove_dict['UNK']
    counts['UNK'] = 0
    for key, value in sorted_counts[dictionary_size+1:]:
        counts['UNK'] += value

    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    printd('Dictionary built with {ld} entries, {cu} UNK counted', ld=len(dictionary), cu=counts['UNK'])
    printd('{le} Exceptions - First 5 {e}', le=len(exceptions),e=exceptions[:5])

    return dictionary, reversed_dictionary, counts

def create_capped_dataset(dictionary, raw_data, record_length):
    num_records = len(raw_data)
    # initialise zeros to enforce zero padding requirement
    data = np.zeros(shape=(num_records,record_length), dtype=int)

    validate_short = list()
    validate_long = list()
    printd('Data has {r} records of varied length',r=num_records)
    printd('Encoding records with embedding index')
    printd('Capping dataset records to {rl} words',rl=record_length)

    for r, record in enumerate(raw_data):
        if len(record) < record_length: validate_short.append(r)
        for w, word in enumerate(record):
            # enforce cap of 40 words per record
            if w >= record_length:
                validate_long.append(r)
                break
            # uncommon words are not in the dictionary
            if word not in dictionary.keys():
                word = 'UNK'
            data[r][w] = dictionary[word]

    printd('{ld} records',ld=len(data))
    # validation
    long_count = 0
    short_count = 0
    for r in raw_data:
        if len(r) < record_length: short_count += 1
        if len(r) > record_length: long_count += 1

    printd('Num short records: {ls} / {sc}', ls=len(validate_short), sc=short_count)
    printd('Num long records: {ll} / {sc}', ll=len(validate_long), sc=long_count)

    data = np.array(data)
    printd('Encoded records created successfully')
    printd('Data now has shape {ds}',ds=data.shape)

    return data

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""
    config = Config()

    extract_data(config.reviews_file, config.reviews_path)

    raw_data = read_data(config.reviews_path)

    # count all words
    counts = count_words(raw_data)

    # create dictionary of (word, index) pairs holding uncommon ('UNK') values from the dictionary
    # create a reverse_dictionary with (index,word) pairs
    dictionary, reverse_dictionary, counts = build_dictionary(raw_data, counts, glove_dict, config.dictionary_size)

    # cap each entry to 40 words and pad short entries with 0 ('UNK')
    data = create_capped_dataset(dictionary, raw_data, config.record_length)

    return data

def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    #if you are running on the CSE machines, you can load the glove data from here
    #data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    config = Config()
    datafile = config.embeddings_file
    embedding_size = config.embedding_size

    embeddings = []
    print('Embeding Unit Length %d'%(embedding_size))
    word_index_dict = {}

    # open datafile: a space seperated string of (key,pair) of (word, word_embedings)
    print('Opening %s'%(datafile))
    with open(datafile,'r',encoding="utf-8") as opendata:
        data_string = opendata.read()
        buffer = collections.deque()
        buffer.extend(data_string.split())

        print("Opened %s successfully"%(datafile))
        print('Loading word embeddings')
        embedding_unit = []
        embeddings_index = 0
        # add entry for UNK - uncommon words
        word_index_dict['UNK'] = embeddings_index
        embeddings.append(np.zeros(embedding_size))
        embeddings_index += 1
        for i in range(len(buffer)):
            item = buffer.popleft()
            # entry is a key word
            if i == 0:
                word_index_dict[item] = embeddings_index
            elif i % (embedding_size+1) == 0:
                if len(embedding_unit) != embedding_size:
                    print('current word item ', word_item, '| current item ', item)
                    print('current embedding unit (%d): %s'%(len(embedding_unit),embedding_unit))
                    print('last embedding unit %s'%(embeddings[-1]))
                    raise ValueError('Embedding unit length != %d'%(embedding_size))
                # previous embedding is complete so append to embeddings and reset unit
                embeddings.append(embedding_unit)
                embeddings_index += 1
                embedding_unit = []
                # set (key, value) pair as (word, embeddings_index)
                word_index_dict[item] = embeddings_index
                word_item = item
            # item is an embedding value (float)
            else:
                try:
                    embedding_unit.extend([float(item)])
                except:
                    print('current word item ', word_item, '| current item ', item)
                    print('current embedding unit (%d): %s'%(len(embedding_unit),embedding_unit))
                    print('last embedding unit %s'%(embeddings[-1]))
                    ValueError('Item %s is not convertable to a float'%(item))

        embeddings.append(embedding_unit)

        print('Loaded %d word embeddings with %d index entries successfully'%(len(embeddings),len(word_index_dict)))

    return np.array(embeddings, dtype=float), word_index_dict

def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, dropout_keep_prob, optimizer, accuracy and loss
    tensors"""
    config              = Config()
    input_size          = config.record_length
    embedding_size      = config.embedding_size
    hidden_units_size   = config.hidden_units_size
    dictionary_size     = config.dictionary_size
    cell_size           = config.cell_size
    learning_rate       = config.learning_rate
    dropout_rate        = config.dropout_keep_prob

    with tf.device("/gpu:0"):
        with tf.name_scope("data"):
            # input
            X = tf.placeholder([None,None], dtype=tf.uint16, name="input_data")
            if dropout_rate < 1:
                X = tf.nn.dropout(X, dropout_rate)
            # label
            Y = tf.placeholder(batch_size, dtype=tf.uint16, name="labels")

        with tf.name_scope("embed"):
            embedding = tf.get_variable(glove_embeddings_arr.shape, tf.constant_initializer(glove_embeddings_arr),\
                            dtype=tf.float32, name="embedding", trainable=False)
            X_ = tf.nn.embedding_lookup(embedding, X)

        cell = tf.nn.rnn_cell.GRUCell(cell_size)
        if hidden_units_size > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell]*hidden_units_size)

        with tf.name_scope("output"):
            init_state = cell.zero_state(batch_size, dtype=tf.float32)
            Ycell, final_state = tf.dynamic_rnn(cell, X_, initial_state=initial_state)

            Yflat = tf.reshape(Ycell, cell_size)
            Ylogits = tf.contrib.layers.fully_connected(Yflat,1, activation_fn=None)

            Y_ = tf.nn.sigmoid(Ylogits, name="predictions")

            accuracy = tf.reduce_mean(tf.cast(tf.equal(Y_, Y), tf.float32))

        with tf.name_scope("loss"):
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=Ylogits, labels=Y)
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


    return X, Y, dropout_rate, optimizer, accuracy, loss


# embeddings, index = load_glove_embeddings()
# # np.save('glove_embeddings', embeddings)
# # np.save('glove_index', index)

# index = np.load('glove_index.npy').item()
# glove = np.load('glove_embeddings.npy')
# load_data(index)
