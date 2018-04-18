##!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt
import matplotlib.image as mping
import pandas as pd
from  scipy import misc
import os
import glob
import tensorflow as tf
import time
import  numpy as np


w=100
h=100
c=3

def readim(path):
    lena = mping.imread(path)
    l_shape=lena.shape
    print(l_shape)
    if len(l_shape)==3:
        lean1 = misc.imresize(lena, [100, 100, 3])
        lean2=True
    else:
        lean1=0
        lean2=False
    return  lean1,lean2


def read_img():
    path = path = 'C:/Users/85242/Desktop/神经网络/图片数据/'
    cate = cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    path1=[]
    for idx, folder in enumerate(cate):
        print(idx, folder)
        for im in glob.glob(folder + '/*.jpg'):
                print('reading the images:%s' % (im))
                img,flag = readim(im)
                if flag==True:
                    imgs.append(img)
                    labels.append(idx)
                    path1.append(im)

    return path1,np.asarray(imgs,np.float32), np.asarray(labels,np.int32)


def data_hand(data,label):
    # 数据打乱
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]
   #数据拆分
    ratio = 0.8
    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]
    return x_train,y_train,x_val,y_val

def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]


if __name__ == '__main__':
	
    x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')
    # 第一个卷积层（200——>100)
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 第二个卷积层(50->25)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # 第三个卷积层(25->12)
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # 第四个卷积层(12->6)

    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

    # 全连接层
    dense1 = tf.layers.dense(inputs=re1,
                             units=1024,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense2 = tf.layers.dense(inputs=dense1,
                             units=512,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense2=tf.nn.dropout(dense2, 0.5)
    logits = tf.layers.dense(inputs=dense2,
                             units=10,
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))	
		 
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    n_epoch = 16
    batch_size = 64
    saver = tf.train.Saver()



    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    path,data, label = read_img()
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]

    # 将所有数据分为训练集和验证集
    ratio = 0.8
    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = label[:s]

    path_val=path[s:]
    x_val = data[s:]
    y_val = label[s:]

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for epoch in range(1):
        start_time = time.time()
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
            #print(x_train_a, y_train_a)
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err;
            train_acc += ac;
            n_batch += 1
        print("   train loss: %f" % (train_loss / n_batch))
        print("   train acc: %f" % (train_acc / n_batch))
        save_path = saver.save(sess, "C:/Users/85242/Desktop/神经网络/结果/save_net.ckpt")
	

        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
            err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
            val_loss += err;
            val_acc += ac;
            n_batch += 1
        print("   validation loss: %f" % (val_loss / n_batch))
        print("   validation acc: %f" % (val_acc / n_batch))
    saver.save(sess, 'C:/Users/85242/Desktop/神经网络/结果/save_net.ckpt')
    sess.close()
