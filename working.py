
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




label_file = '/home/whale/Desktop/Rachel/CeVICHE/Time2Seize - Sheet1 (2).csv'

def get_train():
    os.chdir('/home/whale/Desktop/Rachel/CeVICHE/train')
    train_names = sorted(glob('**/*/*.avi', recursive=True))
    train_labels = np.genfromtxt(label_file, delimiter=',', skip_header=2, usecols=range(0,20))
    return(train_names, train_labels)

def get_test():
    os.chdir('/home/whale/Desktop/Rachel/CeVICHE/test')
    test_names = sorted(glob('**/*/*.avi', recursive=True))
    return(test_names)

train_names, train_labels = get_train()
first = train_names[0]
# test_names = get_test()

n = 900

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
    seizing_worms = int(labels[row, 13])
    bubble = int(labels[row, 14])
    w1 = labels[row, 15]
    w2 = labels[row, 16]
    w3 = labels[row, 17]
    w4 = labels[row, 18]
    w5 = labels[row, 19]
    w6 = labels[row, -1]
    worms = [int(w1), int(w2), int(w3), int(w4), int(w5), int(w6)]
    hot = np.zeros((7, total_frames))
    for idx, worm in enumerate(worms):
        for i in range(bubble, worm):
            if i  in ind:
                hot[idx, i] = 1
                if hot[idx, i] ==1:
                    hot[idx, 7] = 1
    hot = hot[:, ind]
    seize_time_frame = np.sum(hot,axis=0)

    # if seizing_worms > 0:
    #     seize_time_frame = seize_time_frame/seizing_worms
    return(get, seize_time_frame)

def trans(x):
    x_mean = np.mean(x, 0)
    x_mean = x_mean.astype('uint8', copy=False)
    x -= x_mean
    return(x)

epochs = 10
batch_size = 32
opt = 'adam'
act = 'elu'
soft = 'softmax'
loss = 'mean_squared_error'
dialation_rate = (1, 1)

def create_network():
    res = Sequential()
    res.add(ResNet50(weights='imagenet', include_top=False, input_shape=(224,256,3)))
    res.add(keras.layers.GlobalAveragePooling2D(data_format=None))
    res.add(keras.layers.Dense(1, activation='linear'))
    res.compile(loss=loss,optimizer=opt)
    return(res)

def run_it(train_vid, current):
    x_train, y_train = get_hot(train_vid, train_names, train_labels)
    x_train = trans(x_train)
    if current == 0:
        res = create_network()
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        final = res.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[tensorboard], validation_split=.20)
        score = res.evaluate(x_train, y_train, batch_size=batch_size)
        res.save('saved_res0.h5')
    else:
        past = current - 1
        load_path = '/home/whale/Desktop/Rachel/CeVICHE/train/saved_res' +  str(past) + '.h5'
        save_path = 'saved_res' + str(current) + '.h5'
        del_path = '/home/whale/Desktop/Rachel/CeVICHE/train/saved_res' +  str(past) + '.h5'
        res = load_model(load_path)
        print("Loaded model from disk")
        res.compile(loss=loss,optimizer=opt)
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        final = res.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[tensorboard], validation_split=.20
        score = res.evaluate(x_train, y_train, batch_size=batch_size)
        os.remove(del_path)
        res.save(save_path)
        print("Saved model to disk")
    return(final)

def test_it(train_vid, current):
    x_test = get_frames(test_vid)
    x_test = trans(x_test)
    res = load_model('/home/whale/Desktop/Rachel/CeVICHE/train/saved_res68.h5')
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    test = res.predict(x_test, batch_size=batch_size, callbacks=[tensorboard])

def train_loop():
    for current, train_vid in enumerate(train_names):
        final = run_it(train_vid, current)
    return(win, score)

win, score, loop = train_loop()

# def test_loop():
#     for current, test_vid in enumerate(test_names):
#         go = test_it(test_vid)
#         return(go, current)

# def visualize():
#     show = np.squeeze(x_test, axis=0)
#     print(x_test.shape)
#     pltimshow(show)
