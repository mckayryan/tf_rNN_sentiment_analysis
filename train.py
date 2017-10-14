"""
You are encouraged to edit this file during development, however your final
model must be trained using the original version of this file. This file
trains the model defined in implementation.py, performs tensorboard logging,
and saves the model to disk every 10000 iterations. It also prints loss
values to stdout every 50 iterations.
"""


import numpy as np
import tensorflow as tf
from random import randint
import datetime
import os
import json

import implementation as imp

batch_size = imp.batch_size
iterations = 16000
seq_length = 40  # Maximum length of sentence
load_model = 0
train_test_split = 0.1
dropout_p = 0.75
hidden_layers = imp.hidden_layers
timestamp = datetime.datetime.now()

checkpoints_dir = "./checkpoints"


def evaluate_model(sess, input_dict, itr, run_type=""):
    loss_value, accuracy_value, summary = sess.run(
        [loss, accuracy, summary_op], input_dict)
    writer.add_summary(summary, itr)
    if run_type != "Test":
        imp.printd("\nIteration:\t{itr}", itr=itr)
        imp.printd("{type} loss:\t\t{loss:.2}", type=run_type, loss=loss_value)
        imp.printd("{type} acc:\t\t{acc:.2}", type=run_type, acc=accuracy_value)

    return loss_value, accuracy_value

def loss_accuracy_model(sess, train_input_dict, Xtest, Ytest, batch_size, itr):

    train_loss, train_accuracy = evaluate_model(sess, train_input_dict, itr, "Train")

    test_iterations = int(Xtest.shape[0] / batch_size)
    test_loss = list()
    test_accuracy = list()
    for i in range(test_iterations):
        min_i = i*batch_size
        max_i = ((i+1)*batch_size)
        Xb = Xtest[min_i:max_i]
        Yb = Ytest[min_i:max_i]
        loss, accuracy = evaluate_model(sess, {input_data: Xb, labels: Yb}, i, "Test")

        test_loss.append(loss)
        test_accuracy.append(accuracy)

    imp.printd("Total Test loss \t{tloss:.2}", tloss=np.mean(test_loss))
    imp.printd("Total Test accuracy \t{tacc:.2}", tacc=np.mean(test_accuracy))

    return np.asscalar(np.mean(test_loss)), np.asscalar(np.mean(test_accuracy)), \
        np.asscalar(train_loss), np.asscalar(train_accuracy)

def write_summary(train_loss, train_accuracy, test_loss, test_accuracy, it, split, dropout_p, hidden_layers):
    model_description = "{itr}:dropout-{do}:inputs-Base:{dt}".format(itr=it, do=dropout_p, dt=timestamp)
    imp.printd('Saving Model Results\t{m}', m=model_description)
    result = {"model":model_description, 'dropkeep':dropout_p, 'hidden_layers':hidden_layers, \
            'test_loss':test_loss, "test_accuracy":test_accuracy, \
            'train_loss':train_loss, "train_accuracy":train_accuracy, "split":split}

    if not os.path.isfile("results.json"):
        with open('results.json', 'w') as fout:
            json.dump([result], fout)
    else:
        with open('results.json', 'r') as feedjson:
            feed = json.load(feedjson)
        feed.append(result)

        with open('results.json', 'w') as fout:
            json.dump(feed, fout)

def getTrainBatch(X, y):
    examples = len(y)-1
    labels = np.ndarray([batch_size,2])
    arr = np.ndarray([batch_size,seq_length])
    #print(np.array(X).shape, np.array(y).shape)
    #print(arr.shape, labels.shape)
    for i in range(batch_size):
        num = randint(0, examples)
        labels[i] = y[num]
        arr[i] = X[num]
    return arr, labels

# Call implementation
# glove_array, glove_dict = imp.load_glove_embeddings()
# training_data = imp.load_data(glove_dict)
# np.save('glove_array', glove_array)
# np.save('glove_dict', glove_dict)
# np.save('training_data',training_data)

glove_array = np.load('glove_array.npy')
glove_dict = np.load('glove_dict.npy').item()
training_data = np.load('training_data.npy')

Xtrain, Ytrain, Xtest, Ytest = imp.split_data(training_data, train_test_split)

input_data, labels, dropout_keep_prob, optimizer, accuracy, loss = \
    imp.define_graph(glove_array, batch_size)

# tensorboard
train_accuracy_op = tf.summary.scalar("training_accuracy", accuracy)
tf.summary.scalar("loss", loss)
summary_op = tf.summary.merge_all()

# saver
all_saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

logdir = "tensorboard/" + datetime.datetime.now().strftime(
    "%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

train_loss_list = list()
train_accuracy_list = list()
test_loss_list = list()
test_accuracy_list = list()
itr_list = list()

for i in range(iterations):
    batch_data, batch_labels = getTrainBatch(Xtrain, Ytrain)

    sess.run(optimizer, {input_data: batch_data, labels: batch_labels, dropout_keep_prob: dropout_p})

    if (i % 2000 == 0):
        train_loss, train_accuracy, test_loss, test_accuracy\
            = loss_accuracy_model(sess, {input_data: batch_data, labels: batch_labels}, Xtest, Ytest,\
                                    batch_size, i)

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_accuracy)
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)
        itr_list.append(i)

        # print(train_loss_list, train_accuracy_list, test_loss_list, test_accuracy_list)

    if (i % 10000 == 0 and i != 0):
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        save_path = all_saver.save(sess, checkpoints_dir +
                                   "/trained_model.ckpt",
                                   global_step=i)
        print("Saved model to %s" % save_path)
    if train_loss == 0: break

train_loss, train_accuracy, test_loss, test_accuracy\
    = loss_accuracy_model(sess, {input_data: batch_data, labels: batch_labels}, Xtest, Ytest,\
                            batch_size, i)

train_loss_list.append(train_loss)
train_accuracy_list.append(train_accuracy)
test_loss_list.append(test_loss)
test_accuracy_list.append(test_accuracy)
itr_list.append(i)

write_summary(train_loss_list, train_accuracy_list,\
                test_loss_list, test_accuracy_list,\
                itr_list.append(i), train_test_split, dropout_p, hidden_layers)

# else:
#     sess=tf.Session()
#
#     saver = tf.train.import_meta_graph(checkpoints_dir + "/trained_model.ckpt-50000.meta")
#     saver.restore(sess, tf.train.latest_checkpoint('./'))


sess.close
