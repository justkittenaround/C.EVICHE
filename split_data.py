

import os
from glob import glob
import numpy as np
import shutil


train_path = '/home/blu/C.EVICHE/data/conv_ceviche_data/train/storage/'
val_path = '/home/blu/C.EVICHE/data/conv_ceviche_data/val/'
storage = '/home/blu/C.EVICHE/data/conv_ceviche_data/storage/'

#split excess training data into storage and oversample underrepresented data###
def overflow(folder):
    os.chdir(train_path)
    train = glob(folder + '/**.jpg')
    amount = (len(train)-5783)
    if amount > 0:
        samples = np.random.choice(train, amount, replace=False)
        for sample in samples:
            src = train_path + sample
            dst = storage + sample
            shutil.move(src, dst)
    elif amount < 0:
        make = abs(amount)
        samples = np.random.choice(train, make, replace=True)
        for idx, file in enumerate(samples):
            src = train_path + file
            file = str(file)
            name = file.split('/')
            name = ('6w/copy' + str(idx) + name[1])
            dst = train_path  + name
            shutil.copyfile(src, dst)

    print('finished modifying:', folder)

for bit in ['0w', '1w', '2w', '3w', '4w', '5w', '6w']:
    overflow(bit)


#split training and validation into seperate folders############################
def get_some(folder):
    os.chdir(train_path)
    train = glob(folder + '/**.jpg')
    amount = int(.2*(len(train)))
    val_sample = np.random.choice(train, amount, replace=False)
    for sample in val_sample:
        src = train_path + sample
        dst = val_path + sample
        shutil.move(src, dst)
    print('finished moving:', folder)


for bit in ['0w', '1w', '2w', '3w', '4w', '5w', '6w']:
   get_some(bit)
