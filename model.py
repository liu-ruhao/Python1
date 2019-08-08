# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 09:45:35 2019

@author: 15250
"""
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

class model(object):
    def __init__(self,sess,config,logging):
        self.sess=sess
        self.config=config
        self.logging=logging
        self._checkpoint_path=self.config["ckpt"]
        self.global_step=tf.Variable(0,trainable=False)
        self.build()
        self.print_var()
        self.loggingAll()
        self._saver=tf.train.Saver(tf.global_variables(),max_to_keep=10)
        self.initialize()
        
    def loggingAll(self):
        for name in dir(self):
            if name.find("_")==0 and name.find("_")==-1:
                self.logging.info("self.%s\t%s"%(name,str(getattr(self,name))))
                
    def _input(self):
        self.x=tf.placeholder(tf.float32,[None,1],name="x")
        self.y=tf.placeholder(tf.float32,[None,1],name="y")
    def _initial_multy(self):
        with tf.variable_scope("initial_multy"):
            w=tf.Variable(tf.random_normal(shape=[1,1000],mean=0,stddev=0.1),name="weight_linear")
            b=tf.Variable(tf.zeros([1000]),name="bias_linear")
            self.out_put=tf.matmul(self.x,w)+b
    def _initial_multy2(self):
        with tf.variable_scope("initial_multy2"):
            w=tf.Variable(tf.random_normal(shape=[1000,1000],mean=0,stddev=0.1),name="weight_linear")
            b=tf.Variable(tf.zeros([1000]),name="bias_linear")
            self.out_put=tf.matmul(self.out_put,w)+b
    def _initial_multy3(self):
        with tf.variable_scope("initial_multy3"):
            w=tf.Variable(tf.random_normal(shape=[1000,1000],mean=0,stddev=0.1),name="weight_linear")
            b=tf.Variable(tf.zeros([1000]),name="bias_linear")
            self.out_put=tf.matmul(self.out_put,w)+b
    def _initial_square(self):
        with tf.variable_scope("square"):
            w=tf.Variable(1.0,name="weight_linear")
            b=tf.Variable(0.0,name="bias_linear")
            self.out_put=tf.multiply(tf.multiply(self.x,self.x),w)+b
            #self.predict=tf.multiply(w,self.out_put)+b
    def _initial_linear(self):
        with tf.variable_scope("linear"):
            w=tf.Variable(1.0,name="weight_linear")
            b=tf.Variable(0.0,name="bias_linear")
            self.out_put=tf.multiply(w,self.x)+b
            #self.predict=tf.multiply(w,self.out_put)+b
    def nonlinear_sigmoid(self):
        with tf.variable_scope("non_linear_sigmoid"):
            self.out_put=tf.nn.sigmoid(self.out_put)    
    def nonlinear_relu(self):
        with tf.variable_scope("non_linear_relu"):
            self.out_put=tf.nn.relu(self.out_put)   
    def _inter_linear(self,i):
        with tf.variable_scope("linear_%s"%i):
            w=tf.Variable(1.0,name="weight_linear")
            b=tf.Variable(0.0,name="bias_linear")
            self.out_put=tf.multiply(w,self.out_put)+b
    def nonlinear_tanh(self):
        with tf.variable_scope("non_linear_tanh"):
            self.out_put=tf.tanh(self.out_put)  
    def loss_wb(self):
        with tf.variable_scope("loss"):
            w=tf.Variable(tf.random_normal(shape=[1000,1],mean=0,stddev=0.1),name="weight_linear")
            b=tf.Variable(tf.zeros([1]),name="bias_linear")
            self.predict=tf.matmul(self.out_put,w)+b
            self.loss=tf.nn.l2_loss(self.predict-self.y)    
    def loss(self):
        self.predict=self.out_put
        self.loss=tf.nn.l2_loss(self.predict-self.y)
                                    
    def print_var(self):
        for item in dir(self):
            type_string=str(type(getattr(self,item)))
            print(item,type_string)
    
    def opt(self):
        self._opt=tf.train.AdadeltaOptimizer(self.config["learn_rate"])
        self._train_opt=self._opt.minimize(self.loss,global_step=self.global_step)
        
    def build(self):
        self._input()
        self._initial_multy()
        self.nonlinear_relu()
        #self._initial_multy2()
       # self.nonlinear_relu()
        self._initial_multy3()
        self.nonlinear_sigmoid()
        self.loss_wb()
        self.opt()
        self.logging.info("model is built")
    def initialize(self):
        self.sess.run(tf.global_variables_initializer())
    def train(self,input_x,input_y,i):
        feed_dict={self.x:input_x,self.y:input_y}
        loss,global_step,_=self.sess.run([self.loss,self.global_step,self._train_opt],feed_dict)
        if i%10==0:
            print("loss is %s,global_step is %s,i is %s"%(loss,global_step,i))
            self.logging.info("loss is %s,global_step is %s,i is %s"%(loss,global_step,i))
            if i%500==0:
                predict=self.sess.run(self.predict,feed_dict)
                plt.plot(predict[90:100])
                plt.plot(input_y[90:100])
                plt.show()
                
                if i%10000==0:
                    self._saver.save(self.sess,self._checkpoint_path+"checkpoint",global_step=global_step)
                    plt.plot(predict[0:100])
                    plt.plot(input_y[0:100])
                    plt.show()
            
            