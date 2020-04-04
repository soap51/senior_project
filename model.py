from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, Conv2D , MaxPool2D , Dropout , GlobalAveragePooling2D ,BatchNormalization 
from tensorflow.keras import Model , Sequential
tf.__version__

class Alexnet(Model):
    def __init__(self):
        super(Alexnet , self).__init__()
        self.conv1 = Conv2D(filters=96 ,kernel_size=(11,11), activation="relu" , strides=(4,4), input_shape=(256,256,3))        
        self.bnr1 = BatchNormalization()
        self.pooling1 = MaxPool2D(pool_size=(3,3),strides=(2,2))        
        self.conv2 = Conv2D(filters=256  ,kernel_size=(5,5), activation="relu")
        self.bnr2 = BatchNormalization()   
        self.pooling2 = MaxPool2D(pool_size=(3,3),strides=(2,2))            
        self.conv3 = Conv2D(filters=384  ,kernel_size=(3,3), activation="relu")
        self.conv4 = Conv2D(filters=384  ,kernel_size=(3,3), activation="relu")
        self.conv5 = Conv2D(filters=256  ,kernel_size=(3,3), activation="relu")
        self.pooling3 = MaxPool2D(pool_size=(3,3), strides=(2,2))
        self.dropout1 = Dropout(0.5)
        self.gloavg = GlobalAveragePooling2D()
        self.flatten =  Flatten()
        self.dense_1 = Dense(1024)   
        self.dense_2 = Dense(1024)  
        self.dense_3 = Dense(1)
    
    def extract(self , slices):
        slices = self.conv1(slices)   
        slices = self.bnr1(slices)
        slices = self.pooling1(slices)        
        slices = self.conv2(slices)
        slices = self.bnr2(slices)
        slices = self.pooling2(slices)        
        slices = self.conv3(slices)
        slices = self.conv4(slices)
        slices = self.conv5(slices)
        slices = self.pooling3(slices) 
        slices = self.dropout1(slices)
        return slices
    
    def call(self,x):                     
        ans = None
        for data in x:    
            feature_extract = None                 
            for idx in range(data.shape[0]):                                        
                extract = self.extract(tf.dtypes.cast(tf.reshape(data[idx] , [1, 256,256,3]),tf.float32))
                
                if feature_extract is None:
                    feature_extract = extract
                else:
                    feature_extract = tf.concat([feature_extract , extract] , 0)
            if ans is None:
                ans = feature_extract
            else:
                ans = tf.concat([ans , feature_extract] , 0)        
        ans = self.gloavg(ans)          
        ans = tf.math.reduce_max(ans , 0 , keepdims=True)             
        x = self.flatten(ans)    
        x = self.dense_1(x)        
        x = self.dense_2(x)        
        return self.dense_3(x)     
    

class Smallnet(Model):
    def __init__(self):
        super(Smallnet , self).__init__()        
        self.conv1 = Conv2D(filters=64 ,kernel_size=(2,2), activation="relu" , strides=(2,2), input_shape=(256,256,3)) 
        self.pool1 = MaxPool2D(pool_size=(2,2),strides=(2,2)) 
        self.conv2 = Conv2D(filters=32,kernel_size=(2,2), activation="relu" , strides=(2,2))
        self.pool2 = MaxPool2D(pool_size=(2,2),strides=(2,2))
        self.conv3 = Conv2D(filters=32,kernel_size=(2,2), activation="relu" , strides=(2,2))
        self.pool3 = MaxPool2D(pool_size=(2,2),strides=(2,2))       
        self.gloavg = GlobalAveragePooling2D()        
        self.flatten =  Flatten()
        self.dense_1 = Dense(128) 
        self.dense_2 = Dense(128)    
        self.dense_3 = Dense(1 )

    def extract(self , slices):        
        slices = self.conv1(slices)        
        slices = self.pool1(slices)        
        slices = self.conv2(slices)
        slices = self.pool2(slices)
        slices = self.conv3(slices)
        slices = self.pool3(slices)
        return slices
    
    def call(self,x):                     
        ans = None
        for data in x:    
            feature_extract = None                 
            for idx in range(data.shape[0]):                                        
                extract = self.extract(tf.dtypes.cast(tf.reshape(data[idx] , [1, 256,256,3]),tf.float32))
                
                if feature_extract is None:
                    feature_extract = extract
                else:
                    feature_extract = tf.concat([feature_extract , extract] , 0)
            if ans is None:
                ans = feature_extract
            else:
                ans = tf.concat([ans , feature_extract] , 0)        
        ans = self.gloavg(ans)          
        ans = tf.math.reduce_max(ans , 0 , keepdims=True)             
        x = self.flatten(ans)    
        x = self.dense_1(x)        
        x = self.dense_2(x)        
        return self.dense_3(x)    
           

class Net(Model):
    def __init__(self):
        super(Net , self).__init__()
        
        self.conv1 = Conv2D(filters=32 ,kernel_size=(2,2), activation="relu" , strides=(2,2), input_shape=(256,256,3)) 
        self.pool1 = MaxPool2D(pool_size=(2,2),strides=(2,2)) 
        self.conv2 = Conv2D(filters=16,kernel_size=(2,2), activation="relu" , strides=(2,2))
        self.pool2 = MaxPool2D(pool_size=(2,2),strides=(2,2))
        self.conv3 = Conv2D(filters=16,kernel_size=(2,2), activation="relu" , strides=(2,2))
        self.pool3 = MaxPool2D(pool_size=(2,2),strides=(2,2))       
        self.gloavg = GlobalAveragePooling2D()        
        self.flatten =  Flatten()
        self.dense_1 = Dense(128) 
        self.dense_2 = Dense(128)    
        self.dense_3 = Dense(1)

    def extract(self , slices):        
        slices = self.conv1(slices)        
        slices = self.pool1(slices)        
        slices = self.conv2(slices)
        slices = self.pool2(slices)
        slices = self.conv3(slices)
        slices = self.pool3(slices)
        return slices

    def call(self,x):                     
        ans = None
        for data in x:    
            feature_extract = None                 
            for idx in range(data.shape[0]):                                        
                extract = self.extract(tf.dtypes.cast(tf.reshape(data[idx] , [1, 256,256,3]),tf.float32))
                
                if feature_extract is None:
                    feature_extract = extract
                else:
                    feature_extract = tf.concat([feature_extract , extract] , 0)
            if ans is None:
                ans = feature_extract
            else:
                ans = tf.concat([ans , feature_extract] , 0)        
        ans = self.gloavg(ans)          
        ans = tf.math.reduce_max(ans , 0 , keepdims=True)             
        x = self.flatten(ans)    
        x = self.dense_1(x)        
        x = self.dense_2(x)        
        return self.dense_3(x)    
           
    


