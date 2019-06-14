import os
from glob import glob
import numpy as np
import shutil

#specify locations##############################################################
master = '/home/whale/Desktop/Rachel/CeVICHE/conv_ceviche_data/'
evaluate_path = '/home/whale/Desktop/Rachel/CeVICHE/conv_ceviche_data/Evaluate'
train_path = '/home/whale/Desktop/Rachel/CeVICHE/conv_ceviche_data/train/'
val_path = '/home/whale/Desktop/Rachel/CeVICHE/conv_ceviche_data/val/'

#commands#######################################################################
#make folders for [train, val]
def make_folders():
    for type in [train_path, val_path]:
        folders = os.listdir(type)
        for bit in ['0w', '1w', '2w', '3w', '4w', '5w', '6w']:
            if bit not in folders:
                os.mkdir(type + bit)
make_folders()

#combine '6w' imagefolders from multiple videos
def combine():
    os.chdir(evaluate_path)
    fives = glob('*/5w/**.jpg')
    r = 0
    for vid in fives:
        x = vid.split('/')
        name = str(r) + x[-1]
        r += 1
        src = evaluate_path + '/' + vid
        dst = train_path + '5w/' + name
        shutil.copyfile(src, dst)
    max5 = len(os.listdir(train_path + '5w/'))
    max_all = len(os.listdir(evaluate_path))
    dividens = int(round(max5/max_all))
    return dividens, max5

dividens, max5 = combine()

#copy training data into imagefolder
def split(bit, max5):
    os.chdir(evaluate_path)
    train = glob('*/' + bit + '/**.jpg')
    if len(train) >= max5:
        samples = np.random.choice(train, max5, replace=False)
    else:
        samples = train
    r = 0
    for vid in samples:
        x = vid.split('/')
        name = str(r) + x[-1]
        r += 1
        src = evaluate_path + '/' + vid
        dst = train_path + bit + name
        shutil.copyfile(src, dst)
    print('finished modifying:', bit)

for bit in ['0w/', '1w/', '2w/', '3w/', '4w/', '6w/']:
    split(bit, max5)

#split training and validation into seperate folders
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

#oversample underrepresented dataset

def oversample(max5):
    amount = len(os.listdir(os.path.join(train_path, '6w/')))
    make = int(abs(max5-amount))
    os.chdir(train_path)
    train = glob(bit + '/**.jpg')
    samples = np.random.choice(train, make, replace=True)
    for idx, file in enumerate(samples):
        src = train_path + file
        file = str(file)
        name = file.split('/')
        name = ('6w/copy' + str(idx) + name[1])
        dst = train_path  + name
        shutil.copyfile(src, dst)

oversample(max5)
