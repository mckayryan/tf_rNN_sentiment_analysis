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

import implementation as imp

batch_size = imp.batch_size
iterations = 30000
seq_length = 40  # Maximum length of sentence
load_model = 0

checkpoints_dir = "./checkpoints"

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

Xtrain, Ytrain, Xtest, Ytest = imp.split_data(training_data, 0.2)

if load_model == 0:
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

    for i in range(iterations):
        batch_data, batch_labels = getTrainBatch(Xtrain, Ytrain)
        #imp.printd('Batch {i} created | data {ds} | labels {ls}', i=i, ds=np.array(batch_data).shape, ls=(len(batch_labels),len(batch_labels[0])))
        sess.run(optimizer, {input_data: batch_data, labels: batch_labels})
        if (i % 5000 == 0):
            loss_value, accuracy_value, summary = sess.run(
                [loss, accuracy, summary_op],
                {input_data: batch_data,
                 labels: batch_labels})
            writer.add_summary(summary, i)
            print("Train Iteration: ", i)
            print("Train loss", loss_value)
            print("Train acc", accuracy_value)
        if loss_value == 0: break
        if (i % 10000 == 0 and i != 0):
            if not os.path.exists(checkpoints_dir):
                os.makedirs(checkpoints_dir)
            save_path = all_saver.save(sess, checkpoints_dir +
                                       "/trained_model.ckpt",
                                       global_step=i)
            print("Saved model to %s" % save_path)

else:
    sess=tf.Session()

    saver = tf.train.import_meta_graph(checkpoints_dir + "/trained_model.ckpt-50000.meta")
    saver.restore(sess, tf.train.latest_checkpoint('./'))

test_iterations = tf.cast(len(Xtest) / batch_size, tf.int32)
total_loss = 0
total_accuracy = 0
for i in range(test_iterations):
    min_i = i*batch_size
    max_i = ((i+1)*batch_size-1)
    Xb = Xtest[min_i:max_i]
    Yb = Ytest[min_i:max_i]
    loss_value, accuracy_value, summary = sess.run(
        [loss, accuracy, summary_op],
        {input_data: Xb,
         labels: Yb})
    writer.add_summary(summary, i)
    total_loss += loss_value
    total_accuracy += accuracy_value
    print("Test Iteration: ", i)
    print("Test loss", loss_value)
    print("Test acc", accuracy_value)

print("Total Test loss", total_loss/(i+1))
print("Test acc", accuracy_value/(i+1))

sess.close
