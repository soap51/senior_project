from model import Smallnet
model_name = "smallnet_1_output_1_axis"
import matplotlib.pyplot as plt
import tensorflow as tf

import pandas as pd 
import numpy as np
import datetime
import time



from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


TRAIN_IMAGE_DIR = "D:\\Dataset\\MRNet-v1.0\\train\\"
VALID_IMAGE_DIR = "D:\\Dataset\\MRNet-v1.0\\valid\\"
TRAIN_SIZE = 1130
VALID_SIZE = 120
IMG_WIDTH = 256
IMG_HEIGHT = 256
EPOCHS = 20
ROUND = "1"
DISEASE = "acl"
model_name = model_name + "_"+ DISEASE +"_" +ROUND + "_" + str(EPOCHS)

label_train = pd.read_csv("D:\\Dataset\\MRNet-v1.0\\train-"+DISEASE+".csv" , header=None).loc[0: ,1:]  
label_valid = pd.read_csv("D:\\Dataset\\MRNet-v1.0\\valid-"+DISEASE+".csv" , header=None).loc[0: ,1:]
label_train_df = pd.DataFrame(columns=["X","y"])
label_valid_df = pd.DataFrame(columns=["X","y"])
train_dataset = None

for index in range(TRAIN_SIZE):   
    label_train_df.loc[index] = [index] + [tf.convert_to_tensor(np.asarray(label_train.iloc[index]).astype('float32').reshape((1,1)), dtype=tf.float32)]

for index in range(VALID_SIZE):
    label_valid_df.loc[index] =  [index] + [tf.convert_to_tensor(np.asarray(label_valid.iloc[index]+1130).astype('float32').reshape((1,1)), dtype=tf.float32)]




current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
TRAIN_LOG_DIR = 'D:\\SeniorProject\\Logs\\gradient_tape\\'+ model_name+ '\\' + current_time + '\\train'
VALID_LOG_DIR = 'D:\\SeniorProject\\Logs\\gradient_tape\\'+ model_name+ '\\' + current_time + '\\test'
train_summary_writer = tf.summary.create_file_writer(TRAIN_LOG_DIR)
test_summary_writer = tf.summary.create_file_writer(VALID_LOG_DIR)

model = Smallnet()

optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

train_false_positive = tf.keras.metrics.FalsePositives()
train_true_negative = tf.keras.metrics.TrueNegatives()
train_false_negative = tf.keras.metrics.FalseNegatives()
train_true_positive = tf.keras.metrics.TruePositives()
train_AUC = tf.keras.metrics.AUC()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
valid_false_positive = tf.keras.metrics.FalsePositives()
valid_true_negative = tf.keras.metrics.TrueNegatives()
valid_false_negative = tf.keras.metrics.FalseNegatives()
valid_true_positive = tf.keras.metrics.TruePositives()
valid_AUC = tf.keras.metrics.AUC()
valid_loss = tf.keras.metrics.Mean(name='test_loss')
valid_accuracy = tf.keras.metrics.Accuracy(name='valid_accuracy')

ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
manager = tf.train.CheckpointManager(ckpt, 'D:\\SeniorProject\\training\\checkpoints\\'+ model_name+ '\\tf_ckpts', max_to_keep=10)

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

for epoch in range(EPOCHS):
    start = time.time()
    label_train_df = label_train_df.sample(frac=1).reset_index(drop=True)      
    for index in range(TRAIN_SIZE):        
        axial = np.load(TRAIN_IMAGE_DIR + "axial" + "\\"+str(label_train_df.iloc[index][0]).zfill(4)+".npy") 
        # coronal = np.load(TRAIN_IMAGE_DIR + "coronal" + "\\"+str(label_train_df.iloc[index][0]).zfill(4)+".npy")  
        # sagittal = np.load(TRAIN_IMAGE_DIR + "sagittal" + "\\"+str(label_train_df.iloc[index][0]).zfill(4)+".npy") 
        label = tf.convert_to_tensor(np.asarray(label_train_df.iloc[index][1]).astype('float32').reshape(1,1), dtype=tf.float32)
        # x = [axial , coronal , sagittal]
        x = [axial]
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_object(label , predictions)
            predictions = tf.math.sigmoid(predictions)
            predictions = tf.math.round(predictions)                 
            print(label , predictions)
            
            print(loss)
            train_true_negative.update_state(label,predictions)
            train_false_positive.update_state(label,predictions)
            train_false_negative.update_state(label,predictions)
            train_true_positive.update_state(label,predictions)
            train_AUC.update_state(label,predictions)
            print("label : " , label)
            print("prediction : " , predictions)
            print("loss : " , loss)            
            print("True Negative : " , train_true_negative.result().numpy())
            print("False Positive : " , train_false_positive.result().numpy())
            print("True Positive : " , train_true_positive.result().numpy())
            print("False Negative : " , train_false_negative.result().numpy())  
            # print(model.trainable_variables)
        gradients = tape.gradient(loss, model.trainable_variables)
        print(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))           
        train_loss(loss)
        train_accuracy(label, predictions)
        template = 'Epoch {},Data Index : {}, Train Loss: {} , Train Accuracy: {}'   
        print(template.format(epoch+1,
                            index,
                            train_loss.result(),
                            train_accuracy.result()*100
                            )
        )
        ckpt.step.assign_add(1)
        if int(ckpt.step) % 100 == 0 :
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
            print("loss {:1.2f}".format(train_loss.result()))     
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        tf.summary.scalar('sensitivity', ( train_true_positive.result().numpy() / (train_true_positive.result().numpy() + train_false_negative.result().numpy() ) ), step=epoch)
        tf.summary.scalar('specificity', ( train_true_negative.result().numpy() / (train_true_negative.result().numpy() + train_false_positive.result().numpy() ) ) , step=epoch)
        tf.summary.scalar('AUC', train_AUC.result(), step=epoch)                                 
    train_loss.reset_states()   
    train_accuracy.reset_states()     
    train_true_negative.reset_states()
    train_false_positive.reset_states()
    train_true_positive.reset_states()
    train_false_negative.reset_states()
    train_AUC.reset_states()
    end = time.time()  
    with train_summary_writer.as_default():
        tf.summary.scalar('time', end - start, step=epoch)    
    label_valid_df = label_valid_df.sample(frac=1).reset_index(drop=True)
    start = time.time()
    for index in range(VALID_SIZE):
        axial = np.load(VALID_IMAGE_DIR + "axial" + "\\"+str(label_valid_df.iloc[index][0] + 1130).zfill(4)+".npy") 
        # coronal = np.load(VALID_IMAGE_DIR + "coronal" + "\\"+str(label_valid_df.iloc[index][0] + 1130).zfill(4)+".npy") 
        # sagittal = np.load(VALID_IMAGE_DIR + "sagittal" + "\\"+str(label_valid_df.iloc[index][0] + 1130).zfill(4)+".npy") 
        label = tf.convert_to_tensor(np.asarray(label_valid.iloc[index]).astype('float32').reshape(1,1), dtype=tf.float32)
        # x = [axial , coronal , sagittal]                
        x = [axial]
        predictions = model(x)
        t_loss = loss_object(label , predictions)
        predictions = tf.math.sigmoid(predictions)
        predictions = tf.math.round(predictions)   
        valid_true_negative.update_state(label,predictions)
        valid_false_positive.update_state(label,predictions)
        valid_false_negative.update_state(label,predictions)
        valid_true_positive.update_state(label,predictions)
        valid_AUC.update_state(label,predictions)
        valid_loss(t_loss)
        valid_accuracy(label, predictions)
        template = 'Epoch {},Index : {}, Test Loss: {} , Test Accuracy: {}'
        print(template.format(epoch+1,
                            index,
                            valid_loss.result(),
                            valid_accuracy.result()*100
                            )
        )
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', valid_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', valid_accuracy.result(), step=epoch)
        tf.summary.scalar('sensitivity', ( valid_true_positive.result().numpy() / (valid_true_positive.result().numpy() + valid_false_negative.result().numpy() ) ), step=epoch)
        tf.summary.scalar('specificity', ( valid_true_negative.result().numpy() / (valid_true_negative.result().numpy() + valid_false_positive.result().numpy() ) ) , step=epoch)
        tf.summary.scalar('AUC', valid_AUC.result(), step=epoch)  
  
    valid_true_negative.reset_states()
    valid_false_positive.reset_states()
    valid_true_positive.reset_states()
    valid_false_negative.reset_states()
    valid_AUC.reset_states()    
    valid_accuracy.reset_states()    
    valid_loss.reset_states()
    end = time.time()  
    with test_summary_writer.as_default():
        tf.summary.scalar('time', end - start, step=epoch)  