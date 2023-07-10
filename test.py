import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import Dataset,DataLoader
from model import AlexNet
from torch import nn, optim



def main():
    #CPU或GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "test": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    
    net = AlexNet()

    net.to(device)
    #data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    #image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    #image_path = "/D:\catOrDogHomework\catDogRecognition\catdog_data/" 
    data_root = os.getcwd()
    image_path = data_root + "/catdog_data/"
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                         transform=data_transform["test"])

    if os.path.exists("./AlexNet.pth"):
        net.load_state_dict(torch.load("./AlexNet.pth"))
        print("Load Success!")
    else:
        print("No Params!")
        exit()
    
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    net.eval()
    test_acc = 0
    test_total = 0
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)
            output2 = net(img)
            _, predicted = torch.max(output2.data, 1)
            test_acc += (predicted == label).sum().item()
            test_total += label.size(0)
    test_accuracy = test_acc / test_total
    print('Test Acc: {:.3f}'.format(test_accuracy))

if __name__ == '__main__':
    main()