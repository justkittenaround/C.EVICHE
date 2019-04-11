# tensorboard --logdir=//home/whale/Desktop/Rachel/CeVICHE/train/logs
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import tensorflow as tf
import keras
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Embedding
from keras.layers import ConvLSTM2D
from keras.layers import Dense
from keras.layers import Activation
from keras import losses
from keras import optimizers
from keras import backend as K
import skimage
from keras.callbacks import TensorBoard
from time import time
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";



batch_size = 32
label_file = '/home/whale/Desktop/Rachel/CeVICHE/Time2Seize - Sheet1 (3).csv'

def get_train():
    os.chdir('/home/whale/Desktop/Rachel/CeVICHE/train')
    train_names = sorted(glob('**/*/*.avi', recursive=True))
    train_labels = np.genfromtxt(label_file, delimiter=',', skip_header=2, usecols=range(0,20))
    print('labels:', train_labels.shape)
    return(train_names, train_labels)

names, labels = get_train()

n = 600

def get_frames(vid):
    cap = cv2.VideoCapture(vid)
    fps = int(cap.get(5))
    total_frames = int(round(cap.get(7)))
    print(total_frames, vid)
    all_frames = np.zeros([n,224,256,3])
    count = 0
    ind = np.random.choice(total_frames, n, replace=False)
    while(True):
         ret, frame = cap.read()
         frame = skimage.transform.resize(frame, (224,256,3))
         print(frame.shape)
         if count in ind:
            all_frames[count, ...] = frame
         count += 1
         if cv2.waitKey(1):
            break
    cap.release()
    cv2.destroyAllWindows()
    return(all_frames, total_frames, ind)

def get_hot(vid, names, labels):

get, total_frames, ind = get_frames(vid)
row = names.index(vid)
seizing_worms = int(labels[row, 12])
bubble = int(labels[row, 13])
w1 = labels[row, 14]
w2 = labels[row, 15]
w3 = labels[row, 16]
w4 = labels[row, 17]
w5 = labels[row, 18]
w6 = labels[row, 19]
worms = [int(w1), int(w2), int(w3), int(w4), int(w5), int(w6)]
hot = np.zeros((5, total_frames))
for idx, worm in enumerate(worms):
    for frame_index in range(bubble, (worm+1)):
        if frame_index  in ind:
            hot[idx, frame_index] = 1
ind = np.sort(ind)
hot = hot[:, ind]
seize_time_frame = np.sum(hot,axis=0)

    return(get, seize_time_frame)

def trans(x):
    x_mean = np.mean(x, 0)
    x_mean = x_mean.astype('uint8', copy=False)
    x -= x_mean
    return(x)

epochs = 10
act = 'softmax'
opt = 'adam'
loss = 'categorical_crossentropy'
filepath = '/home/whale/Desktop/Rachel/CeVICHE/train/checks'

def create_network():
    res = Sequential()
    res.add(ResNet50(weights='imagenet', include_top=False, input_shape=(224,256,3)))
    res.add(keras.layers.GlobalAveragePooling2D(data_format=None))
    res.add(keras.layers.Dense(7, activation=act))
    res.compile(loss=loss,optimizer=opt,metrics=['acc'])
    return(res)

def run_it(train_vid, current):
    x_train, y_train = get_hot(train_vid, train_names, train_labels)
    x_train = trans(x_train)
    if current == 0:
        res = create_network()
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        # check = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        final = res.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[tensorboard], validation_split=0.20)
        res.save('saved_res0.h5')
    else:
        past = current - 1
        load_path = '/home/whale/Desktop/Rachel/CeVICHE/train/models/saved_res' +  str(past) + '.h5'
        save_path = '/home/whale/Desktop/Rachel/CeVICHE/train/models/saved_res' + str(current) + '.h5'
        res = keras.models.load_model(load_path)
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        # check = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        final = res.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[tensorboard], validation_set=0.20)
        # final = res.fit_generator(aug.flow(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[tensorboard, check], validation_split=0.33, steps_per_epoch=len(x_train) // batch_size))
        res.save(save_path)
    return(final)

# # FC
# tf.reset_default_graph()
# network = input_data(shape=[None, 28, 28], data_augmentation=img_aug)
# network = fully_connected(network, 2048 , activation='tanh')
# network = dropout(network, p)
# network = fully_connected(network, 2048, activation='tanh')
# network = dropout(network, p)
# network = fully_connected(network, 2048, activation='tanh')
# network = dropout(network, p)
# network = fully_connected(network, 2048, activation='tanh')
# network = dropout(network, p)
# network = fully_connected(network, 10, activation='softmax')
# network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)
# model = tflearn.DNN(network, tensorboard_verbose=2, tensorboard_dir='./Graph')
# model.fit(X,Y, n_epoch=100, validation_set=0.25, snapshot_step=10, batch_size=32, show_metric=True, run_id=(dataset))

def train_loop():
    for current, train_vid in enumerate(train_names):
        go = run_it(train_vid, current)
        print(go)
        return(go)

training = train_loop()
