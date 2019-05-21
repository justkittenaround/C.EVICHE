from torchvision import datasets, models, transforms
import torch
import torch.multiprocessing as mp
import torch.utils.data as data_utils
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
import visdom
vis = visdom.Visdom()


PATH = '/home/whale/Desktop/Rachel/CeVICHE/models/modelsvgg.pt'
data_dir = '/home/whale/Desktop/Rachel/CeVICHE/Data/train/11-14 WORMS/023/'
batch_size = 1
num_classes = 7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 224
thresh = 15
folder = 'test'

##initialize_model#############################################################
test_model= torch.load(PATH)
test_model.eval()

##TESTING PROCEDURE AND PRINTS################################################
def testing_model(model, vid):
##DATASETS AND DATALOADERS################################################
    data_transforms = {transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    print("Initializing Video as Frames...")

    cap = cv2.VideoCapture(vid)
    total_frames = int(round(cap.get(7)))

    since = time.time()
    worms_seizing_pred = []
    worms_seizing_smoothed = []
    worms_seizing_smoothed_1 = []
    worms_seizing_smoothed_2 = []
    worms_seizing_smoothed_3 = []
    bin = np.array([3,1])
    init = 2
    count_1 = thresh + init
    count_2 = thresh + init
    idx = 0
    for idx in range(0, total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
        ret, frame = cap.read()
        if type(frame) is not type(worms_count):
            print('skipped', idx)
            break
        data_transforms[x]
        input = frame.to(device)
        outputs = test_model(input)
        _, predicts = torch.max(outputs, 1)
        preds = predicts.cpu().numpy()
        seizing_preds = np.asarray(worms_seizing_pred)
        worms_seizing_pred.append(preds)
        vis.line(worms_seizing_pred, win='worms_seizing_predict', opts=dict(title= 'trainedVGG-Predictions'))
        if len(seizing_preds) == count_1:
            tally_1 = np.bincount(np.squeeze(seizing_preds[(count_1-thresh):]))
            max_1 = np.argmax(tally_1)
            worms_seizing_smoothed_1.append(max_1)
            count_1 += thresh
            bin.append(worms_seizing_smoothed_1)
            # vis.line(worms_seizing_smoothed_1, win='worms_seizing_smoothed_1', opts=dict(title= 'smoothed_predictions_block'))

        if len(seizing_preds) == count_2:
            tally_2 = np.bincount(np.squeeze(seizing_preds[idx:]))
            max_2 = np.argmax(tally_2)
            worms_seizing_smoothed_2.append(max_2)
            # vis.line(worms_seizing_smoothed_2, win='worms_seizing_smoothed_2', opts=dict(title= 'smoothed_predictions_after'))
            idx += 1
            count_2 += 1
        if idx >= (thresh*6):
            position = idx - (thresh*6)
            tally_3 = np.bincount(np.squeeze(seizing_preds[position:]))
            max_3 = np.argmax(tally_3)
            worms_seizing_smoothed_3.append(max_3)
            # vis.line(worms_seizing_smoothed_3, win='worms_seizing_smoothed_3', opts=dict(title= 'smoothed_predictions_surround'))


        # for idx in enumerate(seizing_preds):
        #     a = np.append([worms_seizing_smoothed_1[idx]], [worms_seizing_smoothed_2[idx]], axis=0)
        #     a = np.append([a], [worms_seizing_smoothed_3[idx]], axis=0)
        #     tally = np.bincount(np.squeeze(a[:, idx]), axis=1)
        #     max = np.argmax(tally)
        #     worms_seizing_smoothed.append(max)
        #     vis.line(worms_seizing_smoothed, win='worms_seizing_smoothed', opts=dict(title= 'smoothed_predictions_combined'))




        time_elapsed = time.time() - since
        print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        return test_model
##INITIALIZE AND RESHAPE MODEL################################################
def initialize_model(PATH):
    test_model= torch.load(PATH)
    test_model.eval()
    return test_model

##multiprocessing##############################################################
# if __name__ == '__main__':
#     nump_processes = 4
#     test_model = test_model()
#     test_model.share_memory()
#     for rank in range(num_processes):
#         p = mp.Process(target=testing_model, args=(model,))
#         p.start()
#         processes.append(p)
#     for p in processes:
#         p.join()



# Initialize the test_modelfor this run
test_model = initialize_model(PATH)
#Send the model to GPU
test_model= test_model.to(device)
#Run test model
test_model = testing_model(testing_model, data_dir)
