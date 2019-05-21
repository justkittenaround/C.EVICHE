from torchvision import datasets, models, transforms
import torch
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import visdom
vis = visdom.Visdom()


PATH = '/home/whale/Desktop/Rachel/CeVICHE/models/modelsvgg.pt'
data_dir = '/home/whale/Desktop/Rachel/CeVICHE/conv_ceviche_data/test/'
batch_size = 1
num_classes = 7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 224
phase = 'test'
thresh = 15
more = 60

##DATASETS AND DATALOADERS################################################
data_transforms = {
    phase: transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
print("Initializing Datasets and Dataloaders...")
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [phase]}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4) for x in [phase]}

##TESTING PROCEDURE AND PRINTS################################################
def testing_model(model, dataloaders):
    since = time.time()

    worms_seizing_pred = []
    worms_seizing_smoothed = []
    worms_seizing_smoothed_1 = []
    worms_seizing_smoothed_2 = []
    worms_seizing_smoothed_3 = []
    worms_seizing_act = []
    testing_acc = []
    confusion = np.zeros([8, 8])
    running_corrects = 0
    init = 2
    count_1 = thresh + init
    count_2 = thresh + init
    idx = 0

    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        bin  = [np.zeros([3,1])]
        outputs = test_model(inputs)
        _, predicts = torch.max(outputs, 1)
        running_corrects += torch.sum(predicts == labels.data)
        preds = predicts.cpu().numpy()
        worms_seizing_pred.append(preds)
        worms_seizing_act.append(labels.data.cpu().numpy())
        seizing_preds = np.asarray(worms_seizing_pred)

        if len(seizing_preds) == count_1:
            tally_1 = np.bincount(np.squeeze(seizing_preds[(count_1-thresh):]))
            max_1 = np.argmax(tally_1)
            worms_seizing_smoothed_1.append(max_1)
            count_1 += thresh
            vis.line(worms_seizing_smoothed_1, win='worms_seizing_smoothed_1', opts=dict(title= 'smoothed_predictions_block'))

        if len(seizing_preds) == count_2:
            tally_2 = np.bincount(np.squeeze(seizing_preds[idx:]))
            max_2 = np.argmax(tally_2)
            worms_seizing_smoothed_2.append(max_2)
            vis.line(worms_seizing_smoothed_2, win='worms_seizing_smoothed_2', opts=dict(title= 'smoothed_predictions_after'))
            idx += 1
            count_2 += 1

        if idx >= (thresh*6):
            position = idx - (thresh*6)
            tally_3 = np.bincount(np.squeeze(seizing_preds[position:]))
            max_3 = np.argmax(tally_3)
            worms_seizing_smoothed_3.append(max_3)
            vis.line(worms_seizing_smoothed_3, win='worms_seizing_smoothed_3', opts=dict(title= 'smoothed_predictions_surround'))
#        if idx >= len(worms_seizing_smoothed_3):
#            for id, number in enumerate(seizing_preds):
#                a = np.append([worms_seizing_smoothed_1[id]], [worms_seizing_smoothed_2[id]], axis=0)
#                a = np.append([a], [worms_seizing_smoothed_3[id]], axis=0)
#                tally = np.bincount(np.squeeze(a[:, id]))
#                max = np.argmax(tally)
#                worms_seizing_smoothed.append(max)
#            vis.line(worms_seizingg_smoothed, win='worms_seing_smoothed', opts=dict(title= 'smoothed_predicitons_combined'))



        print(len(worms_seizing_smoothed_1), len(worms_seizing_smoothed_2), len(worms_seizing_smoothed_3))
        vis.line(worms_seizing_pred, win='worms_seizing_predict', opts=dict(title= 'trainedVGG-Predictions'))
        vis.line(worms_seizing_act, win='worms_seizing_target', opts=dict(title= 'trainedVGG-Target'))
        confusion[worms_seizing_pred[-1], worms_seizing_act[-1]] += 1
        vis.heatmap(confusion, win='confusion_matrix', opts=dict(ylabel= 'predicted', xlabel= 'target', colormap= 'Electric'))
    testing_acc = (running_corrects/len(labels.data))
    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Testing_acc:', testing_acc.cpu().numpy())
    return test_model, testing_acc

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
test_model, testing_acc = testing_model(testing_model, dataloaders_dict)
