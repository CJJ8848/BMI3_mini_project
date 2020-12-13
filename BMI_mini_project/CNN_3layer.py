#-*-coding:utf-8-*-
#-*-coding:utf-8-*-

import os
import sys
import tensorflow as tf
from PIL import Image
from os.path import join, dirname
import matplotlib.pyplot as plt
import numpy as np

# settfrecords used to make tfrecord files for trainset and testset
# parameter: cwd = work directory of trainset or testset, outfile = name of output tfrecord file
from sklearn import metrics


def settfrecords(cwd,outfile):
    # set classes of images
    classes = {'TUM', 'STR','NORM','ADI','DEB','LYM','MUS','MUC'}
    #classes = {'testtum', 'teststr'}
    # the output files
    writer = tf.python_io.TFRecordWriter(outfile)
    #
    for index, name in enumerate(classes):
        #each class pathway
        class_path = cwd + name + '/'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name  # pathway of each image
            img = Image.open(img_path)
            print(np.shape(img)) # (224, 224, 3)
            img_raw = img.tobytes()  # Convert images to binary format
            # index 0 1 2 are set for each class as "label", images are saved as binary format in the "img_raw" feature
            feature = {"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}
            # The Example object encapsulates label and image data features
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # the example is serialized as a string
            writer.write(example.SerializeToString())
    writer.close()
# Read tfrecords
def read_and_decode(filename):
    # generate a quene to receive the tfrecords data
    filename_queue = tf.train.string_input_producer([filename], shuffle=True)
    # initialize a reader
    reader = tf.TFRecordReader()
    # get the serialized example object (filename and file)
    _,example = reader.read(filename_queue)
    # receive features
    features = tf.parse_single_example(example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # Pull out the image data and label

    img = tf.decode_raw(features['img_raw'], tf.uint8)# Throws the Img tensor in the flow
    img = tf.reshape(img, [224, 224, 3])  # Reshape the data to a three channels 224 * 224 images
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 图像减去均值处理
    label = tf.cast(features['label'], tf.int32)  # Throws the Label tensor to the flow

    return img, label

epoch = 10
batch_size = 8
learning_rate = 0.0001
def one_hot(labels,Label_class):
    #get one_hot labels (Stores an ordered class variable as a binary vector)
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])
    return one_hot_label

#initial weights
#shape: the dimension of the output tensor
def weight_variable(shape):
    # Generate truncated normally distributed random numbers as a initial weight in the range [mean-2 * stddev, mean + 2 * stddev]
    initial_weight = tf.truncated_normal(shape, stddev = 0.02) # mean = 0, standard deviation = 0.02
    return tf.Variable(initial_weight) # initial the variable
#initial bias
# hape: the dimension of the output tensor
def bias_variable(shape):
    initial_bias = tf.constant(0.0, shape=shape) # Create a constant with dimension shape and value 0 in TensorFlow
    return tf.Variable(initial_bias)

#convolution layer
# x = A tensor, shape is [batch, in_height, in_weight, in_channel]
# W = weight_variable,  [filter_height, filter_weight, in_channel, out_channels]
def conv(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME') # strides = [1,1,1,1], padding = 'SAME' means edge filling

#max_pool layer
#x = feature map [batch, height, width, channels]
def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')# ksize = window size

if __name__ == '__main__':
    ## only first time to initialize the tfrecord data
    #in OS system, .DS_Store files need to be deleted first with the command below.
    #sudo find /Users/cuijiajun/Desktop/BMI3/mini_project/data  -name ".DS_Store" -depth -exec rm {} \;
    # cwd1 = './data/test/'
    # cwd2= './data/train/'
    # cwd3= './data/validate/'
    # outfile1 = "./tfrecords/8class_test.tfrecords"
    # outfile2 = "./tfrecords/8class_train.tfrecords"
    # outfile3 = "./tfrecords/8class_validate.tfrecords"
    # settfrecords(cwd1,outfile1)
    # settfrecords(cwd2,outfile2)
    # settfrecords(cwd3,outfile3)
    # a placeholder in the model when the neural network is building graph
    x = tf.placeholder(tf.float32, [batch_size,224,224,3])
    y_ = tf.placeholder(tf.float32, [batch_size,8])# 8 classes
    #first convolution and max_pool layer
    W_conv1 = weight_variable([5,5,3,32])# 5x5 size window, 3 channels data (RGB), 32 layers
    b_conv1 = bias_variable([32])# 32 layers of window (32 features have 32 bias)
    out_conv1 = tf.nn.relu(conv(x, W_conv1) + b_conv1)# relu as an activity function
    out_pool1 = max_pool_4x4(out_conv1)# 4x4 max pooling layer to reduce the parameters

    #second convolution and max_pool layer
    W_conv2 = weight_variable([5,5,32,64]) # 5x5 size window, 32 in channel data (output of 32 convolution keras), 64 layers
    b_conv2 = bias_variable([64])# 64 layers of window (64 features have 64 bias)
    out_conv2 = tf.nn.relu(conv(out_pool1, W_conv2) + b_conv2)
    out_pool2 = max_pool_4x4(out_conv2)

    # third convolution and max_pool layer
    W_conv3 = weight_variable([5, 5, 64, 128])  # 5x5 size window, 64 in channel data (output of 128 convolution keras), 64 layers
    b_conv3 = bias_variable([128])  # 64 layers of window (64 features have 64 bias)
    out_conv3 = tf.nn.relu(conv(out_pool2, W_conv3) + b_conv3)
    out_pool3 = max_pool_4x4(out_conv3)

    # fully connected layer
    reshape = tf.reshape(out_pool3, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    W_fc1 = weight_variable([dim, 1024])  # 1024 neurons in the fully connected layer
    b_fc1 = bias_variable([1024])
    out_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)  # activate
    #dropout
    keep_prob = tf.placeholder(tf.float32)
    out_fc1_drop = tf.nn.dropout(out_fc1, keep_prob)# keep_prob means the memory keeping probability (used to control the overfit)
    #softmax predict the probability of belonging to each class
    W_fc2 = weight_variable([1024,8])# 8 means 8 classes
    b_fc2 = bias_variable([8])# 8 means 8 classes
    y_conv = tf.nn.softmax(tf.matmul(out_fc1_drop, W_fc2) + b_fc2)
    # softmax loss (cross entropy)
    # the bigger predict error is, the bigger softmax loss is
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    # adam (optimize algorithm)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)# learning rate = 0.001, minimize: Minimize the cross entropy loss by updating the VAR_list add operation
    #accuracy output (for each image, judge whether tf.argmax(y_conv,1) is equal to tf.argmax(y_,1) )
    correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1)) # tf.argmax: axis =1, return the index array of the largest element in each row
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


    # start to implement
    # get img and label of train set and test set
    img, label = read_and_decode("./tfrecords/8class_train.tfrecords")
    img_test, label_test = read_and_decode("./tfrecords/8class_test.tfrecords")
    img_validate, label_validate = read_and_decode("./tfrecords/8class_validate.tfrecords")

    # use shuffle_batch to random the input
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=batch_size, capacity=2000,
                                                    min_after_dequeue=1000)
    img_test, label_test = tf.train.shuffle_batch([img_test, label_test],
                                                  batch_size=batch_size, capacity=2000,
                                                  min_after_dequeue=1000
                                                  )
    # used to initialize all variables
    init = tf.initialize_all_variables()
    # return a list of all variable objects
    t_vars = tf.trainable_variables()
    print(t_vars)
    model_save_path = dirname(__file__) + '/ckpt/'
    txt_save_path = dirname(__file__) + '/3layers_accuracy_for_the_best_model.txt'

    restore_saver = tf.train.Saver()
    # initialize a session to run the tensor
    with tf.Session() as sess:
        sess.run(init)
        # Coordinator is used to control the threads
        coord = tf.train.Coordinator()
        # tensorflow supports mult-thread process
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # the id of batch
        batch_idxs = int(3200 / batch_size)
        # initialize the variable to restore the label, predict probability and the accuracy list of test set

        acclist=[]
        with open(txt_save_path, 'a') as file_handle:
            file_handle.write('\n')
            file_handle.write("learning rate:"+str(learning_rate)+"\n"+"batch_size:"+str(batch_size)+"\n"+"epoch:"+str(epoch))
        for i in range(epoch):
            for j in range(batch_idxs):
                # val is img vector, l is label vector
                val, l = sess.run([img_batch, label_batch])
                # one hot label for 8 classes
                l = one_hot(l, 8)
                # run the training algorithm and output the accuracy, set the memory keeping probability 0.5 to decline the over-fit
                _, acc = sess.run([train_step, accuracy], feed_dict={x: val, y_: l, keep_prob: 0.5})
                # print the accuracy for each batch of images of each epoch
                print("Epoch:[%4d] [%4d/%4d], accuracy:[%.8f]" % (i+1, j, batch_idxs, acc))
                train_accuracy=("Epoch:[%4d] [%4d/%4d], accuracy:[%.8f]" % (i+1, j, batch_idxs, acc))

                with open(txt_save_path, 'a') as file_handle:
                    file_handle.write('\n')
                    file_handle.write(train_accuracy
                        )
        val, l = sess.run([img_test, label_test])
        l = one_hot(l, 8)  # one hot label for testset labels
        labellist = l
        y, acc = sess.run([y_conv, accuracy], feed_dict={x: val, y_: l, keep_prob: 1})
        plist = y
        acclist.append(acc)
        for t in range(1,800//batch_size):
            # val is test img vector, l is test label vector of the last one batch of images
            val, l = sess.run([img_test, label_test])
            l = one_hot(l, 8)  # one hot label for testset labels
            labellist=np.vstack((labellist, l))
            # get the accuracy of the test set
            y, acc = sess.run([y_conv, accuracy], feed_dict={x: val, y_: l, keep_prob: 1})
            plist = np.vstack((plist, y))
            acclist.append(acc)
            print("test set accuracy: [%.8f]" % (acc))
        accmean = np.mean(acclist)
        print("test set mean accuracy: " + str(accmean))
        with open(txt_save_path, 'a') as file_handle:
            file_handle.write('\n')
            file_handle.write(
                "epoch: " + str(epoch) + "\n" + "batch size: " + str(batch_size) + "\n" + "test set accuracy (mean): " + str(accmean) + "\n"+"\n"
            )
        #plot the ROC curve
        for file in os.listdir(dirname(__file__) + '/ROC_curve/'):
            os.remove(dirname(__file__) + '/ROC_curve/' + file)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        classes = ['TUM', 'STR', 'NORM', 'ADI', 'DEB', 'LYM', 'MUS', 'MUC']

        for i in range(len(classes)):
            fpr[i], tpr[i], thresholds = metrics.roc_curve(labellist[:, i], plist[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area（方法二）
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(labellist.ravel(), plist.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

        # Compute macro-average ROC curve and ROC area（方法一）
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(classes)):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= len(classes)
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

        plt.figure()
        lw = 2

        colors = ['royalblue', 'limegreen', 'r', 'darkgray', 'm', 'y', 'k', 'darkorange']
        for i, color in zip(range(len(classes)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (AUC = {1:0.3f})'
                           ''.format(classes[i], roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.savefig(dirname(__file__) + "/ROC_curve/allclass.png")
        plt.figure()
        lw = 2
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.3f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linewidth=3)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='royalblue', linewidth=3)

        for i in range(len(classes)):
            plt.plot(fpr[i], tpr[i], color='darkorange', linestyle=':', lw=lw,
                     label='ROC curve of class {0} (AUC = {1:0.3f})'
                           ''.format(classes[i], roc_auc[i]))
        plt.legend(["micro-average ROC curve (AUC = {0:0.3f})" ''.format(roc_auc['micro']),
                    "macro-average ROC curve (AUC = {0:0.3f})" ''.format(roc_auc['macro']), "ROC curve of each group"])

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig(dirname(__file__)+"/ROC_curve/with_average.png")
        plt.show()
        # #去除省略号 output the whole np array
        # np.set_printoptions(threshold=sys.maxsize)
        # with open("3layers_accuracy_for_ROC.txt", 'a') as file_handle:
        #     file_handle.write('\n')
        #     file_handle.write(
        #         "epoch: " + str(epoch) + "\n" + "batch size: " + str(batch_size) + "\n" + "test set accuracy: " + str(accmean) + "\n"+"\n"
        #     +"labels: " + str(labellist) + "\n" + "predicted probability: " + str(plist))
        for file in os.listdir(dirname(__file__) + '/ckpt/'):
            os.remove(dirname(__file__) + '/ckpt/' + file)
        ckptpath = dirname(__file__) + '/ckpt/model'  + "_acc_" + str(accmean) + ".ckpt"
        restore_saver.save(sess, ckptpath)
        coord.request_stop()
        coord.join(threads)