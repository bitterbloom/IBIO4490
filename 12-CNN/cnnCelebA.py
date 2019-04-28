#!/home/afromero/anaconda3/bin/python

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

def print_network(model, name):
    num_params=0
    for p in model.parameters():
        num_params+=p.numel()
    print(name)
    print(model)
    print("The number of parameters {}".format(num_params))


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


def get_data(batch_size):
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_path = './CelebA/img_align_celeba/'
    partition_path = './CelebA/train_val_test.txt'
    labels_path = './CelebA/list_attr_celeba.txt'

    data_train = CelebADataset(img_path = img_path, labels_path = labels_path, partition_path=partition_path, train = True, val = False, transform = transform_train)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True)

    data_val = CelebADataset(img_path = img_path, labels_path = labels_path, partition_path=partition_path, train = False, val = True, transform = transform_train)
    val_loader = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=True)
    
    data_test = CelebADataset(img_path = img_path, labels_path = labels_path, partition_path=partition_path, train = False, val = False, transform = transform_train)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        best_performance_val = 100
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            # running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, torch.max(labels, 1)[1])
                        loss2 = criterion(aux_outputs, torch.max(labels, 1)[1])
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, torch.max(labels, 1)[1])

                    # _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if phase == 'val':
                if epoch_loss < best_performance_val:
                    best_performance_val = epoch_loss
                    save_checkpoint(model,True,'./best_model_Fix')

            print('{} Loss: {:.4f} '.format(phase, epoch_loss))

            # # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'val':
            #     val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # load best model weights
    model = torch.load('./best_model_Fix')
    return model, val_acc_history

def save_checkpoint(state, is_best, filename='/output/checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG19_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

def test_model(model, dataloader):
    first = 182638
    with open("testeo.txt","w") as f:
        for batch_idx, (data,target) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc="[TEST] Testing images"):
            data = data.to(device).requires_grad_(False)
            outputs = model(data)
            for i in range(outputs.size()[0]):
                temps = []
                for j in range(outputs.size()[1]):
                    if outputs[i][j]>0:
                        lab_temp = 1
                    else:
                        lab_temp = 0
                    temps.append(lab_temp)
                temp_str = ""
                for k in range(len(temps)):
                    if k == len(temps)-1:
                        temp_str += f"{temps[k]}"
                    else:
                        temp_str += f"{temps[k]},"
                f.write(f"{first}.jpg,{temp_str}\n")
                first += 1

if __name__=='__main__':
    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
    model_name = "resnet"

    # Number of classes in the dataset
    num_classes = 10

    # Batch size for training (change depending on how much memory you have)
    batch_size = 100

    # Number of epochs to train for
    num_epochs = 3

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = True

    train_loader, val_loader, test_loader = get_data(batch_size)
    training_loaders = {
        "train": train_loader,
        "val": val_loader 
    }

   # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, training_loaders, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
    test_model(model_ft,test_loader)
    
