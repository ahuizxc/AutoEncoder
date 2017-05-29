#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 28 11:00:26 2017
tensorflow 实战 project 1 自编码器

@author: pro
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('/tmp/data/',one_hot=True)
train_data=mnist.train.images
train_label=mnist.train.labels
test_data=mnist.test.images
test_label=mnist.test.labels

training_epochs=101#训练次数
batch_size=256
display_step=1
examples_to_show=10



n_input=784# 28*28 图片像素数量
n_hidden_1=500#第一层编码层神经元个数
n_hidden_2=400#第二层编码层神经元的个数
n_hidden_3=200
n_output=784
lr=0.006

X=tf.placeholder('float',[None,n_input])
Y=tf.placeholder('float',[None,n_output])
weights={
        'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),                                
        'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
        'out1':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
        'out2':tf.Variable(tf.random_normal([n_hidden_3,n_output]))      
        }
biases={
        'h1':tf.Variable(tf.random_normal([n_hidden_1])),
        'h2':tf.Variable(tf.random_normal([n_hidden_2])),
        'out1':tf.Variable(tf.random_normal([n_hidden_3])),
        'out2':tf.Variable(tf.random_normal([n_output]))
        }
##过后尝试用dropout

def encoder(x,weights,biases):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['h1']),biases['h1']))
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['h2']),biases['h2']))
    return layer_2

def decoder(x,weights,biases):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['out1']),biases['out1']))
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['out2']),biases['out2']))
    return layer_2

def addnoise(x,power,batch_size):
    return x+power*np.random.randn(batch_size,784)

encoder_op=encoder(X,weights,biases)
decoder_op=decoder(encoder_op,weights,biases)
predict_image=decoder_op
real_image=X

loss=tf.reduce_mean(tf.pow(predict_image-real_image,2))
train_op=tf.train.AdamOptimizer(lr).minimize(loss)


batch_size=100
disp_step=50
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(training_epochs):
        num_batch=int(mnist.train.num_examples/batch_size)
        total_cost=0
        for j in range(num_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            batch_xs_noise=addnoise(batch_xs,0.3,batch_size)
            _,loss_=sess.run([train_op,loss],feed_dict={X:batch_xs_noise,Y:batch_xs})
            total_cost+=loss_
        if i%disp_step==0:
            print('Epoch %02d/%02d average cost: %0.6f'%(i,training_epochs,total_cost/num_batch))
            randidx=np.random.randint(test_data.shape[0],size=1)
            ##从test_data里面随机的选一张图片
            test_vec=test_data[randidx,:]
            test_vec_noise=addnoise(test_vec,0.3,1)
            out_vec=sess.run(predict_image,feed_dict={X:test_vec_noise})
            out_image=np.reshape(out_vec,(28,28))
            org_image=np.reshape(test_vec,(28,28))
            noise_image=np.reshape(test_vec_noise,(28,28))
            plt.figure(1)
            ##选中子图1
            plt.matshow(org_image,cmap=plt.get_cmap('gray'))
            plt.matshow(noise_image,cmap=plt.get_cmap('gray'))
            plt.matshow(out_image,cmap=plt.get_cmap('gray'))            






    


