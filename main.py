# -*- coding: utf-8 -*- cc
"""
Created on Thu Aug  8 09:41:54 2019

@author: 15250
"""

import tensorflow as tf
import numpy as np
import config as CF
import logging
import model as m
import input_data as ID

print(CF.config["batch_size"])
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(filename)s[line:%(lineno)d]%(levelname)s :%(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    filename=CF.config["logging_name"],
                    filemode="w"
                    )
sess=tf.Session()
model_1=m.model(sess=sess,config=CF.config,logging=logging)
model_1.print_var()
for k in range(CF.config["max_step"]):
    x_input,y_input=ID.generate(CF.config["batch_size"])
    model_1.train(x_input,y_input,k)