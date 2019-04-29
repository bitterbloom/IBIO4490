import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data.dataset import Dataset
import torch.utils.model_zoo as model_zoo
import numpy as np
import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CelebADataset(Dataset):
    """FER2013 dataset.""" 
    def __init__(self, img_path, labels_path, partition_path, train=True, val=False, transform=None):

        self.img_path = img_path
        self.labels_path = labels_path
        self.partition_path = partition_path
        self.transform = transform
        self.train = train
        self.val = val

        import os
        img_list = os.listdir(img_path)
        img_list.sort()

        partition_file = open(partition_path,"r")
        partitions = partition_file.readlines()

        labels_file = open(labels_path,"r")
        labels = labels_file.readlines()

        train_images, val_images, test_images, train_labels, val_labels = [], [], [], [], []
        for i in range(len(img_list)):
            set_id = int(partitions[i+1].split(",")[1].split("\n")[0])
            img = img_list[i]
            if set_id == 0:
                label_string = labels[i+1].split("jpg")[1][1:].split("\n")[0].split(",")
                label = []
                for lab in range(len(label_string)):
                    if lab in [15,5,8,9,11,17,20,26,31,39]:
                        label.append(int(label_string[lab]))
                train_images.append(img)
                train_labels.append(label)
            elif set_id == 1:
                label_string = labels[i+1].split("jpg")[1][1:].split("\n")[0].split(",")
                label = []
                for lab in range(len(label_string)):
                    if lab in [15,5,8,9,11,17,20,26,31,39]:
                        label.append(int(label_string[lab]))
                val_images.append(img)
                val_labels.append(label)
            elif set_id == 2:
                test_images.append(img)
        
        self.train_dict = {
            'images':train_images,
            'labels':train_labels
        }
        self.val_dict = {
            'images':val_images,
            'labels':val_labels
        }
        self.test_dict = {
            'images':test_images,
        }

    def __len__(self):
        if self.train == True:
            return len(self.train_dict["images"])
        elif self.val == True:
            return len(self.val_dict["images"])
        else:
            return len(self.test_dict["images"])
            
    def __getitem__(self,idx):
        if self.train == True:
            im_file = self.train_dict["images"][idx]
            img = self.transform(Image.open(self.img_path+im_file))
            label = self.train_dict["labels"][idx]
            label = torch.LongTensor(label)
        elif self.val == True:
            im_file = self.val_dict["images"][idx]
            img = self.transform(Image.open(self.img_path+im_file))
            label = self.val_dict["labels"][idx]
            label = torch.LongTensor(label)
        else:
            im_file = self.test_dict["images"][idx]
            img = self.transform(Image.open(self.img_path+im_file))
            label = []
        
        return (img,label)

def test_model(model, sample):
    attributes = ['Eyeglasses', 'Bangs', 'Black_Hair', 'Blond_Hair',
            'Brown_Hair', 'Gray_Hair', 'Male', 'Pale_Skin', 'Smiling',
            'Young']
    import matplotlib.pyplot as plt
    sample = sample[0]
    sample = sample.unsqueeze(0)
    data = sample.to(device).requires_grad_(False)
    outputs = model(data)
    temps = []
    for i in range(outputs.size()[1]):
        if outputs[0][i]>0:
            lab_temp = 1
        else:
            lab_temp = 0
        temps.append(lab_temp)
    for i in range(len(temps)):
        if temps[i] == 1:
            ans = "yes"
        else:
            ans = "no"

        print(f"{attributes[i]} : {ans}")
    sample = sample.numpy()
    sample = np.resize(sample,(224,224,3))
    plt.imshow(np.asarray(sample))
def get_data():
    transform_train = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_path = './CelebA/img_align_celeba/'
    partition_path = './CelebA/train_val_test.txt'
    labels_path = './CelebA/list_attr_celeba.txt'
    
    data_test = CelebADataset(img_path = img_path, labels_path = labels_path, partition_path=partition_path, train = False, val = False, transform = transform_train)
    return data_test

def print_network(model, name):
    num_params=0
    for p in model.parameters():
        num_params+=p.numel()
    print(name)
    print(model)
    print("The number of parameters {}".format(num_params))

if __name__=='__main__':
    url = "https://www.dropbox.com/s/r210mjgow17w6nd/best_model?dl=1"
    import urllib.request
    u = urllib.request.urlopen(url)
    data = u.read()
    u.close()

    with open('best_model', "wb") as f :
        f.write(data)

    data_test = get_data()
    num_samples = data_test.__len__()
    import random
    sample = data_test.__getitem__(random.randrange(num_samples))

    model = torch.load("./best_model")
    print_network(model,"ResNet-50")
    test_model(model,sample)