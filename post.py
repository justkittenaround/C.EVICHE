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
from PIL import Image as pil
import visdom
vis = visdom.Visdom()


PATH = '/home/whale/Desktop/Rachel/CeVICHE/models/modelsvgg.pt'
vid_dir = '/home/whale/Desktop/Rachel/CeVICHE/Data/train/11-14 WORMS/023/ceviche-023.avi'
root = '/home/whale/Desktop/Rachel/CeVICHE/Data/test/'
batch_size = 1
num_classes = 7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 224
thresh = 15
folder = 'test'

##initialize_model#############################################################
test_model= torch.load(PATH)
test_model.eval()
test_model= test_model.to(device)

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


##TESTING PROCEDURE AND PRINTS################################################
def GetFrames(dir):
    images = []
    #images = ['pil_image', 'frame_position', 'video_number']
    dir = os.path.expanduser(dir)
    for target in sorted(file_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isfile(d):
            continue
        cap = cv2.VideoCapture(d)
        total_frames = int(round(cap.get(7)))
        print('Total frames in video is: ', total_frames)
        for idx in range(0, total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
            ret, frame = cap.read()
            if type(frame) is not type(bin):
                print('skipped', idx)
                break
            item = (pil.fromarray(frame), idx, file_to_idx[target])
            images.append(item)
    return images

class Dataset():
    def __init__(self, root, transform=None):
        super(Dataset, self).__init__(root)
        self.transform = transform
        files, file_to_idx = self._find_files(self.root)
        samples = GetFrames(self.root)
        self.loader = loader
        self.files = files
        self.file_to_idx = file_to_idx
        self.samples = samples
    def _find_files(self, dir):
        files = [d for d in os.listdir(root) if os.path.isfile(os.path.join(dir, d))]
        files.sort()
        file_to_idx = {files[i]: i for i in range(len(files))}
        return files, file_to_idx
################################################################################
    def __getitem__(self, index):
        vid = self.samples[index]
        sample = vid[index])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
###############################################################################
    def __len__(self):
        return len(self.samples)


image_datasets = {x: Dataset(root, data_transforms}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in [folder]}






print("Initializing Video as Frames...")

data_transforms = {transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]),
}

def testing_model(model, vid):



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
    if type(frame) is not type(bin):
        print('skipped', idx)
        break
    frame = pil.fromarray(frame)
    data_transforms[frame]
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
        vis.line(worms_seizing_smoothed_1, win='worms_seizing_smoothed_1', opts=dict(title= 'smoothed_predictions_block'))

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


#Run test model
test_model = testing_model(testing_model, vid_dir)
