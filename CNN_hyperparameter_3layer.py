#-*-coding:utf-8-*-
#-*-coding:utf-8-*-

import os
import tensorflow as tf
from PIL import Image
from os.path import join, dirname
import matplotlib.pyplot as plt
import numpy as np

#image_pathway = dirname(__file__)+"/data/"
#/Users/cuijiajun/Desktop/BMI3/BMI_mini_project/data/
#label_list = 'TUM,STR,NORM,ADI,DEB,LYM,MUS,MUC'
prompt = "> "
print(f"please type the pathway of image data")
print("example: /Users/Colorectal_hunter/Desktop/BMI3/BMI_mini_project/data/")
image_pathway = input(prompt)

print("please type a string of labels (8 classes)")
print("example: TUM,STR,NORM,ADI,DEB,LYM,MUS,MUC")

label_list = input(prompt)
print("start...")

# settfrecords used to make tfrecord files for trainset and testset
# parameter: cwd = work directory of trainset or testset, outfile = name of output tfrecord file
def settfrecords(cwd,outfile):
    # set classes of images
    #classes = ['TUM', 'STR','NORM','ADI','DEB','LYM','MUS','MUC']
    classes=label_list.split(",")
    # the output files
    writer = tf.python_io.TFRecordWriter(outfile)
    #
    for index, name in enumerate(classes):
        #each class pathway
        class_path = cwd + name + '/'
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name  # pathway of each image
            img = Image.open(img_path)
            img_raw = img.tobytes()  # Convert images to binary format
            # index 0 1 2 are set for each class as "label", images are saved as binary format in the "img_raw" feature
            feature = {"label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}
            # The Example object encapsulates label and image data features
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # the example is serialized as a string
            writer.write(example.SerializeToString())
    print("shape:"+str(np.shape(img)))# (224, 224, 3)
    print("create tfrecords success:"+str(outfile))
    writer.close()
# Read tfrecords
def read_and_decode(filename):
    # generate a quene to receive the tfrecords data
    filename_queue = tf.train.string_input_producer([filename], shuffle=True)
    # initialize a reader
    reader = tf.TFRecordReader()
    # get the serialized example object (filename and file)
    _,serialized_example = reader.read(filename_queue)
    # receive features
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # Pull out the image data and label

    img = tf.decode_raw(features['img_raw'], tf.uint8)# Throws the Img tensor in the flow
    img = tf.reshape(img, [224, 224, 3])  # Reshape the data to a three channels 224 * 224 images
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # normalization by subtracting the RGB means
    # img = tf.cast(img, tf.float32) * (1. / 255)
    # mean_red = np.sum(img[:, :, 0]) / (224 * 224)
    # mean_green = np.sum(img[:, :, 1]) / (224 * 224)
    # mean_blue = np.sum(img[:, :, 2]) / (224 * 224)
    # tmp_0 = img[:, :, 0]-mean_red
    # tmp_1 = img[:, :, 1]-mean_green
    # tmp_2 = img[:, :, 2]-mean_blue
    # img = tf.stack([tmp_0, tmp_1, tmp_2], 2)
    label = tf.cast(features['label'], tf.int32)  # Throws the Label tensor to the flow

    return img, label

epoch = 10
batch_size = None
learning_rate = None

def one_hot(labels,Label_class):
    #get one_hot labels (Stores an ordered class variable as a binary vector)
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])
    return one_hot_label

#initial weights
#shape: the dimension of the output tensor
def weight_variable(shape):
    # Generate truncated normally distributed random numbers as a initial weight in the range [mean-2 * stddev, mean + 2 * stddev]
    initial = tf.truncated_normal(shape, stddev = 0.02) # mean = 0, standard deviation = 0.02
    return tf.Variable(initial) # initial the variable
#initial bias
# hape: the dimension of the output tensor
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape) # Create a constant with dimension shape and value 0 in TensorFlow
    return tf.Variable(initial)

#convolution layer
# x = A tensor, shape is [batch, in_height, in_weight, in_channel]
# W = weight_variable,  [filter_height, filter_weight, in_channel, out_channels]
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME') # strides = [1,1,1,1], padding = 'SAME' means edge filling

#max_pool layer
#x = feature map [batch, height, width, channels]
def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')# ksize = window size
def CNN_3layers():
    # first convolution and max_pool layer
    W_conv1 = weight_variable([5, 5, 3, 32])  # 5x5 size window, 3 channels data (RGB), 32 layers
    b_conv1 = bias_variable([32])  # 32 layers of window (32 features have 32 bias)
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)  # relu as an activity function
    h_pool1 = max_pool_4x4(h_conv1)  # 4x4 max pooling layer to reduce the parameters

    # second convolution and max_pool layer
    W_conv2 = weight_variable(
        [5, 5, 32, 64])  # 5x5 size window, 32 in channel data (output of 32 convolution keras), 64 layers
    b_conv2 = bias_variable([64])  # 64 layers of window (64 features have 64 bias)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_4x4(h_conv2)

    # third convolution and max_pool layer
    W_conv3 = weight_variable(
        [5, 5, 64, 128])  # 5x5 size window, 64 in channel data (output of 128 convolution keras), 64 layers
    b_conv3 = bias_variable([128])  # 64 layers of window (64 features have 64 bias)
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_4x4(h_conv3)

    # fully connected layer
    reshape = tf.reshape(h_pool3, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    W_fc1 = weight_variable([dim, 1024])  # 1024 neurons in the fully connected layer
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)  # activate

    # dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1,
                               keep_prob)  # keep_prob means the memory keeping probability (used to control the overfit)
    # softmax predict the probability of belonging to each class
    W_fc2 = weight_variable([1024, 8])  # 8 means 8 classes
    b_fc2 = bias_variable([8])  # 8 means 8 classes
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # softmax loss (cross entropy)
    # the bigger predict error is, the bigger softmax loss is
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    # adam (optimize algorithm)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(
        cross_entropy)  # learning rate = 0.001, minimize: Minimize the cross entropy loss by updating the VAR_list add operation
    # accuracy output (for each image, judge whether tf.argmax(y_conv,1) is equal to tf.argmax(y_,1) )
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,
                                                                  1))  # tf.argmax: axis =1, return the index array of the largest element in each row
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return y_conv,accuracy,train_step,keep_prob
def runtensorflow(init, batch_size,txtpath,img_batch,label_batch,img_validate, label_validate,train_step,accuracy,y_conv,keep_prob):
    # initialize a session to run the tensor
    with tf.Session() as sess:
        sess.run(init)
        # Coordinator is used to control the threads
        coord = tf.train.Coordinator()
        # tensorflow supports mult-thread process
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        # the id of batch
        batch_idxs = int(3200 / batch_size)

        for i in range(epoch):
            acclist = []
            for j in range(batch_idxs):
                # val is img vector, l is label vector
                val, l = sess.run([img_batch, label_batch])
                # one hot label for 8 classes
                l = one_hot(l, 8)
                # run the training algorithm and output the accuracy, set the memory keeping probability 0.5 to decline the over-fit
                _, acc = sess.run([train_step, accuracy], feed_dict={x: val, y_: l, keep_prob: 0.5})
                # print the accuracy for each batch of images of each epoch
                print("Epoch:[%4d] [%4d/%4d], accuracy:[%.8f]" % (i + 1, j, batch_idxs, acc))
            valtrain, lt = sess.run([img_batch, label_batch])
            # one hot label for 8 classes
            lt = one_hot(lt, 8)
            # run the training algorithm and output the accuracy, set the memory keeping probability 0.5 to decline the over-fit
            _, acctrain = sess.run([train_step, accuracy], feed_dict={x: valtrain, y_: lt, keep_prob: 0.5})
            # validation
            val, l = sess.run([img_validate, label_validate])
            l = one_hot(l, 8)  # one hot label for testset labels
            print(l)
            labellist = l
            y, acc = sess.run([y_conv, accuracy], feed_dict={x: val, y_: l, keep_prob: 1})
            print(y)
            plist = y
            acclist.append(acc)
            for t in range(1, 800 // batch_size):
                # val is test img vector, l is test label vector of the last one batch of images
                val, l = sess.run([img_validate, label_validate])
                l = one_hot(l, 8)  # one hot label for testset labels
                print(l)
                labellist = np.vstack((labellist, l))
                # get the accuracy of the test set
                y, acc = sess.run([y_conv, accuracy], feed_dict={x: val, y_: l, keep_prob: 1})
                print(y)
                plist = np.vstack((plist, y))
                acclist.append(acc)
            accmean = np.mean(acclist)

            with open(txtpath, 'a') as file_handle:
                file_handle.write('\n')
                file_handle.write("epoch: " + str(i + 1) + "\n" + "batch size: " + str(
                    batch_size) + "\n" + "validation set accuracy: " + str(accmean) + "\n")

        coord.request_stop()
        coord.join(threads)
        return accmean
if __name__ == '__main__':
    # only used at the first time to initialize the tfrecord data from raw data
    # in OS system, .DS_Store files need to be deleted first with the command below.
    # sudo find /Users/cuijiajun/Desktop/BMI3/BMI_mini_project/data  -name ".DS_Store" -depth -exec rm {} \;
    cwd1 = image_pathway + 'test/'
    cwd2 = image_pathway + 'train/'
    cwd3 = image_pathway + 'validate/'
    if not os.path.exists(dirname(__file__) + '/tfrecords/'):
        os.makedirs(dirname(__file__) + '/tfrecords/')
    for file in os.listdir(dirname(__file__) + '/tfrecords/'):
        os.remove(dirname(__file__) + '/tfrecords/' + file)
    outfile1 = dirname(__file__) + "/tfrecords/8class_test.tfrecords"
    outfile2 = dirname(__file__) + "/tfrecords/8class_train.tfrecords"
    outfile3 = dirname(__file__) + "/tfrecords/8class_validate.tfrecords"
    settfrecords(cwd1, outfile1)
    settfrecords(cwd2, outfile2)
    settfrecords(cwd3, outfile3)
    # start to implement
    # get img and label of train set and test set
    img, label = read_and_decode(dirname(__file__) + "/tfrecords/8class_train.tfrecords")
    img_validate, label_validate = read_and_decode(dirname(__file__) + "/tfrecords/8class_validate.tfrecords")

    # batch_choice = [8, 16, 32]
    # learning_rate_choice = [0.0001, 0.00001]
    txtpath = dirname(__file__) + "/3layer_accuracy_for_all.txt"
    batch_choice = [8,16,32]
    learning_rate_choice = [0.0001,0.00001]
    for l in learning_rate_choice:
        #learning_rate = l
        learning_rate = l
        with open(txtpath, 'a') as file_handle:
            file_handle.write('\n')
            file_handle.write("learning rate:" + str(learning_rate)
                              )
        print("learning rate:" + str(learning_rate))
        #for b in [8, 16, 32]:
        for b in batch_choice:
            batch_size = b
            with open(txtpath, 'a') as file_handle:
                file_handle.write('\n')
                file_handle.write("batch_size:" + str(batch_size)
                                  )
            print("batch_size:" + str(batch_size))
            # a placeholder in the model when the neural network is building graph
            x = tf.placeholder(tf.float32, [batch_size,224,224,3])
            y_ = tf.placeholder(tf.float32, [batch_size,8])# 8 classes
            # construct the CNN architecture and receive the y_conv,accuracy, train_step and keep_prob
            y_conv, accuracy, train_step, keep_prob = CNN_3layers()
            #use shuffle_batch to random the input
            img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                            batch_size=batch_size, capacity=2000,
                                                            min_after_dequeue=1000)
            img_validate, label_validate = tf.train.shuffle_batch([img_validate, label_validate],
                                                          batch_size=batch_size, capacity=2000,
                                                          min_after_dequeue=1000)

            # used to initialize all variables
            init = tf.initialize_all_variables()
            # return a list of all variable objects
            t_vars = tf.trainable_variables()
            print(t_vars)
            restore_saver = tf.train.Saver()
            # initialize a session to run the tensor
            accmean = runtensorflow(init, batch_size, txtpath, img_batch, label_batch, img_validate, label_validate,
                              train_step, accuracy, y_conv, keep_prob)
            print("validation set mean accuracy: " + str(accmean))