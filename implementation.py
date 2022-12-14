import tensorflow as tf
import numpy as np
import glob #this will be useful when reading reviews from file
import os
import sys
import tarfile
import re
import collections
import string
import random
from sklearn.model_selection import train_test_split
from operator import itemgetter
from pprint import pprint as pp

batch_size = 50
hidden_layers = 3

class Config(object):
    embedding_size = 50
    dictionary_size = 50000
    max_sample = 25000
    record_length = 40
    embeddings_file = "glove.6B.50d.txt"
    reviews_file = "reviews.tar.gz"
    reviews_path = os.path.join(os.path.dirname(__file__), 'review_data/')
    debug = 1
    hidden_layers = 2
    dropout_keep_prob = 1
    cell_size = 128
    learning_rate = 0.001
    classes = 2


def printd(print_string, **kwargs):
    c = Config()
    if c.debug > 0:
        print(print_string.format(**kwargs))

def split_data(X, split):
    config = Config()
    total_pop = config.max_sample
    printd('Splitting data train {trs} / test {ts} from {records} records', trs=1-split, ts=split, records=total_pop)
    y = [ [1,0] if i < total_pop / 2 else [0,1] for i in range(total_pop)]
    Xtr, Xt, Ytr, Yt = train_test_split(X, y, test_size=split, random_state=42)
    Xtr = np.array(Xtr)
    Xt = np.array(Xt)
    Ytr = np.array(Ytr)
    Yt = np.array(Yt)
    printd('Datasets\nXtrain {xtr}\tYtrain {ytr}\nXtest  {xt}\tYtest  {yt}', \
                xtr=Xtr.shape, ytr=Ytr.shape, xt=Xt.shape, yt=Yt.shape)
    return Xtr, Ytr, Xt, Yt

# Read the data into a list of strings.
def extract_data(filename, path):
    printd("Extracting zipped data {p}{f}",p=path,f=filename)
    if not os.path.exists(path):
        with tarfile.open(filename, "r") as tarball:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tarball, path)
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
        printd(len(data[i]),data[i][:10])
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

def define_graph(glove_embeddings_arr, batch_size):
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
    hidden_layers       = config.hidden_layers
    dictionary_size     = config.dictionary_size
    cell_size           = config.cell_size
    learning_rate       = config.learning_rate
#    dropout_keep   = config.dropout_keep_prob
    num_classes         = config.classes

    with tf.device("/cpu:0"):
        # input dropout
        dropout_keep = tf.placeholder_with_default(1.0, shape=(), name="dropout_keep_prob")

    with tf.device("/gpu:0"):
        with tf.name_scope("data"):
            # input
            X = tf.placeholder(tf.int32, [batch_size,input_size], name="input_data")
            printd('input_data\t(X) \t{shape}\tsuccessfully assigned',shape=X.shape)

            # labels
            labels = tf.placeholder(tf.int32, [batch_size,num_classes], name="labels")
            printd('labels\t\t\t{shape}\t\tsuccessfully assigned',shape=labels.shape)

        with tf.name_scope("embed"):
            # set word embeddings
            embedding = tf.get_variable("embedding", glove_embeddings_arr.shape, tf.float32,\
                            tf.constant_initializer(glove_embeddings_arr), trainable=False)
            printd('embedding tensor \t{shape}\tsuccessfully assigned',shape=embedding.shape)
            X_ = tf.nn.embedding_lookup(embedding, X, name="input_embed")
            printd('input_embed \t(X_)\t{shape}\tsuccessfully assigned',shape=X_.shape)

        with tf.name_scope("cell"):
            cells = [ tf.nn.rnn_cell.GRUCell(cell_size) for _ in range(hidden_layers) ]
            dropcells = [ tf.nn.rnn_cell.DropoutWrapper(cell, dropout_keep) for cell in cells ]
            printd('Dropout applied to {ncells} cell/s at keep rate {rate}', \
                    ncells=hidden_layers, rate=dropout_keep)

            multicell = tf.nn.rnn_cell.MultiRNNCell(dropcells)
            multicell = tf.nn.rnn_cell.DropoutWrapper(multicell, dropout_keep)
            printd('{layer} GRU cell layer/s \t{out} outputs | {states} states | {drop} dropout keep rate',\
                        layer=hidden_layers, out=multicell.output_size, \
                        states=multicell.state_size, drop=dropout_keep)

        with tf.name_scope("output"):
            init_state = multicell.zero_state(batch_size, tf.float32)

            Ycell, final_state = tf.nn.dynamic_rnn(multicell, X_, initial_state=init_state)
            printd('Ycell \t\t\t{shape}\tsuccessfully assigned as {type}',shape=Ycell.shape, type=Ycell.dtype)

            Yflat = tf.reshape(Ycell[:,-1], [-1, cell_size])
            Ylogits = tf.contrib.layers.fully_connected(Yflat, 2, activation_fn=None)
            printd('Ylogits -> calculated \t{shape} as {type}', shape=Ylogits.shape, type=Ylogits.dtype)

            Y_ = tf.argmax(Ylogits, 1)
            printd('Y_ -> prediction made \t{ps} as {pt}', ps=Y_.shape, pt=Y_.dtype)
            # printd('prediction examples {pe}', pe=Y_[:5])

        with tf.name_scope("loss"):
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=labels, name="loss")
            printd('loss\t\t\t{shape}', shape=loss.shape)

            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            loss = tf.reduce_mean(loss)
            printd('optimizer\t\t{shape}', shape=optimizer)

    with tf.device('/cpu:0'):
        Y = tf.argmax(labels, 1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(Y,Y_), tf.float32), name="accuracy")
        printd('accuracy\t{detail}\n', detail=accuracy)

        printd('Accuracy, loss & optimizer set')
        printd('Graph definition complete\n\n')

    return X, labels, dropout_keep, optimizer, accuracy, loss


# embeddings, index = load_glove_embeddings()
# # np.save('glove_embeddings', embeddings)
# # np.save('glove_index', index)

# index = np.load('glove_index.npy').item()
# glove = np.load('glove_embeddings.npy')
# load_data(index)
