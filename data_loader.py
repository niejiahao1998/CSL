import numpy as np
from PIL import Image
import torch.utils.data as data
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing
import torchvision.transforms as transforms
import random
import math
import cv2
import os

class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    

    ### We modify this class for our ChannelExchange module
    def __init__(self, gray = 3):
        self.gray = gray
        self.channel_idx = [0,1,2]

    def __call__(self, img):
    
        idx = random.randint(0, self.gray)
        
        if idx ==0:
            # random select R Channel
            new_img = img
            new_img[1, :,:] = img[0,:,:]
            new_img[2, :,:] = img[0,:,:]
        elif idx ==1:
            # random select B Channel
            new_img = img
            new_img[0, :,:] = img[1,:,:]
            new_img[2, :,:] = img[1,:,:]
        elif idx ==2:
            # random select G Channel
            new_img = img
            new_img[0, :,:] = img[2,:,:]
            new_img[1, :,:] = img[2,:,:]
        else:
            ### random exchange the order of channel
            random.shuffle(self.channel_idx)
            new_img = img       
            new_img[0,:,:] = img[self.channel_idx[0],:,:]
            new_img[1,:,:] = img[self.channel_idx[1],:,:]
            new_img[2,:,:] = img[self.channel_idx[2],:,:]

        img = (new_img+img)/2

        return img

class SYSUData(data.Dataset):
    ### We add height and width parameters
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None, height = 288, width = 144):

        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        # BGR to RGB
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.transform_thermal = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)])
            
        self.transform_color = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p = 0.1),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5)])
            
        self.transform_color1 = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ### We modify this line for our ChannelExchange module
            ChannelExchange(gray = 3)])
       
    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1_0 = self.transform_color(img1)
        img1_1 = self.transform_color1(img1)

        img2 = self.transform_thermal(img2)

        return img1_0, img1_1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        
class RegDBData(data.Dataset):
    ### We add height and width parameters
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None, height = 288, width = 144):

        # Load training images (path) and labels
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((width, height), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((width, height), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_thermal = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)])
            
        self.transform_color = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p = 0.1),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5)])
            
        self.transform_color1 = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ### We modify this line for our ChannelExchange module
            ChannelExchange(gray = 3)])

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1_0 = self.transform_color(img1)
        img1_1 = self.transform_color1(img1)
        img2 = self.transform_thermal(img2)

        return img1_0, img1_1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        
class TestData(data.Dataset):
    ### We add height and width parameters
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            ### We add this line caused by our saving functions are from cv2
            img = img.convert("RGB") 
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            
            pix_array = np.array(img) 
            test_image.append(pix_array)

        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)
        
class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)        
def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label


class NTUData(data.Dataset):
    ### We add height and width parameters
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None, height = 288, width = 144):

        # Load training and test ids
        train_id_path = os.path.join(data_dir, 'train.txt')
        train_id_list = open(train_id_path, 'r').read().splitlines()
        train_id = [s.split(' ')[0] for s in train_id_list]

        # Load training images (path) and labels
        train_color_path = os.path.join(data_dir, 'RGB')
        color_img_file = []
        train_color_label = []
        
        pid_container = set()
        for id in os.listdir(train_color_path):
            if id in train_id:
                pid_container.add(int(id))
        pid_container = sorted(pid_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        for id in os.listdir(train_color_path):
            if id in train_id:
                id_path = os.path.join(train_color_path, id)
                for img in os.listdir(id_path):
                    img_path = os.path.join(id_path,img)
                    label = pid2label[int(id)]
                    train_color_label.append(label)
                    color_img_file.append(img_path)
        
        train_thermal_path = os.path.join(data_dir, 'IR')
        thermal_img_file = []
        train_thermal_label = []

        for id in os.listdir(train_thermal_path):
            if id in train_id:
                id_path = os.path.join(train_thermal_path, id)
                for img in os.listdir(id_path):
                    img_path = os.path.join(id_path,img)
                    label = pid2label[int(id)]
                    train_thermal_label.append(label)
                    thermal_img_file.append(img_path)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(color_img_file[i])
            img = img.resize((width, height), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(thermal_img_file[i])
            img = img.resize((width, height), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_thermal = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)])
            
        self.transform_color = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p = 0.1),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5)])
            
        self.transform_color1 = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((height, width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelExchange(gray = 3)])

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1_0 = self.transform_color(img1)
        img1_1 = self.transform_color1(img1)
        img2 = self.transform_thermal(img2)

        return img1_0, img1_1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)