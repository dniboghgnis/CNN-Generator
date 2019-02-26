#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 22:37:11 2019

@author: Gobind
"""

import tensorflow as tf
import pandas as pd
from skimage import io
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing
import cv2
import glob
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

rand_data=[]
scalar=preprocessing.MinMaxScaler()
url1="https://github.com/dniboghgnis/Linear-regression/blob/master/rgb_hd.jpg?raw=true"
z_image=io.imread(url1)
z_image=cv2.resize(z_image,(50,50))
b,g,r=cv2.split(z_image)
b=scalar.fit_transform(b)
g=scalar.fit_transform(g)
r=scalar.fit_transform(r)
z_image=cv2.merge([r,g,b])
rand_data.append(z_image)
plt.imshow(z_image)
plt.show()


img_array=[]


url = "https://github.com/dniboghgnis/Linear-regression/blob/master/image_0069.jpg?raw=true"
img = io.imread(url)
img=cv2.resize(img,(50,50))
bb,gg,rr=cv2.split(img)
bb=scalar.fit_transform(bb)
gg=scalar.fit_transform(gg)
rr=scalar.fit_transform(rr)
img=cv2.merge([rr,gg,bb])
img_array.append(img)
plt.imshow(img_array[0])
plt.show()

#placeholders
z=tf.placeholder(dtype=tf.float64,shape=[None,50,50,3])
z=tf.reshape(z,[-1,50,50,3])

####################         generator      ########################
def generator(temp_z):
    w1=tf.Variable(tf.random_normal([5,5,3,3],mean=0.01,stddev=0.01,dtype=tf.float64),name="w1")
    b1=tf.Variable(tf.random_normal([3],mean=0.01,stddev=0.01,dtype=tf.float64),name="b1")
    conv1=tf.nn.conv2d(temp_z,w1,strides=[1,1,1,1],padding='SAME')+b1
    layer1=tf.nn.relu(conv1,name="layer1")
#    return layer1
    w2=tf.Variable(tf.random_normal([5,5,3,3],mean=0.01,stddev=0.01,dtype=tf.float64),name="w2")
    b2=tf.Variable(tf.random_normal([3],mean=0.01,stddev=0.01,dtype=tf.float64),name="b2")
    conv2=tf.nn.conv2d(layer1,w2,strides=[1,1,1,1],padding='SAME')+b2
    layer2=tf.nn.relu(conv2,name="generate")
    w3=tf.Variable(tf.random_normal([5,5,3,3],mean=0.0,stddev=0.01,dtype=tf.float64))
    b3=tf.Variable(tf.random_normal([3],mean=0.0,stddev=0.01,dtype=tf.float64))
    conv3=tf.nn.conv2d(layer2,w3,strides=[1,1,1,1],padding='SAME')+b3
    layer3=tf.nn.relu(conv3)
#    return layer3
    # print("Layer 2 shape     " +str(layer2.shape))
    gen_params=[w1,b1,w2,b2,w3,b3]
    return layer3,gen_params

########### loss function       ##############3
    
output_image,params=generator(z)

lossfn=tf.reduce_mean(tf.square(output_image-img))

learningRate = tf.train.exponential_decay(learning_rate=0.05,
                                          global_step= 1,
                                          decay_steps=1,
                                          decay_rate= 0.95,
                                          staircase=True)
optimizer=tf.train.GradientDescentOptimizer(learningRate)
train_op=optimizer.minimize(lossfn,var_list=params)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

for i in range(15000):
#    print(sess.run(output_image.shape))
    sess.run(train_op,feed_dict={z:rand_data})
    loss=sess.run(lossfn,feed_dict={z:rand_data})
    if i%100==0:
      print("Step :"+str(i)+"  |  Loss is "+str(loss))
    if i%500==0:
        temp=sess.run(output_image,feed_dict={z:rand_data})
        plt.imshow(temp[0])
        plt.show()
        