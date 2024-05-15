import numpy as np
from PIL import Image
import torch.utils.data as data
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing
import torchvision.transforms as transforms
import random
import math
import os.path as osp
import glob
import re

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
        
class TestData(data.Dataset):
    ### We add height and width parameters ### date-2022/JUN/12
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            ### We add this line caused by our saving functions are from cv2  ### date-2022/MAR/21
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
              
def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label


class PRCCData(data.Dataset):

    def __init__(self, data_dir, height = 384, width = 192, transform = None, rgbIndex = None, sketchIndex= None, **kwargs):
        self.data_dir = data_dir
        self.train_dir = osp.join(self.data_dir, 'rgb/train')
        self.val_dir = osp.join(self.data_dir, 'rgb/val')
        self.test_dir = osp.join(self.data_dir, 'rgb/test')
        self._check_before_run()

        train_rgb_image, train_rgb_label, \
            train_sketch_image, train_sketch_label = self._process_dir_train(self.train_dir, height, width)

        self.train_rgb_image = train_rgb_image
        self.train_rgb_label = train_rgb_label
        self.train_sketch_image = train_sketch_image
        self.train_sketch_label = train_sketch_label

        self.transform = transform
        self.cIndex = rgbIndex
        self.tIndex = sketchIndex

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
            ### We modify this line for our ChannelExchange module ### date-2022/JUL/06
            ChannelExchange(gray = 3)]) ### default: gray=2

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir_train(self, data_dir, height, width):
        pdirs = glob.glob(osp.join(data_dir, '*'))
        pdirs.sort()

        pid_container = set()
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            pid_container.add(pid)
        pid_container = sorted(pid_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        num_pids = len(pid_container)

        train_rgb_image = []
        train_rgb_label = []
        train_sketch_image = []
        train_sketch_label = []
        for pdir in pdirs:
            pid = int(osp.basename(pdir))
            img_dirs = glob.glob(osp.join(pdir, '*.jpg'))
            for img_dir in img_dirs:
                label = pid2label[pid]
                train_rgb_label.append(label)
                train_sketch_label.append(label)
                img = Image.open(img_dir)
                img = img.resize((width, height), Image.ANTIALIAS)
                pix_array = np.array(img)
                train_rgb_image.append(pix_array)
                train_sketch_image.append(pix_array)
        num_imgs = len(train_rgb_image)

        return train_rgb_image, train_rgb_label, train_sketch_image, train_sketch_label

    def __getitem__(self, index):

        img1,  target1 = self.train_rgb_image[self.cIndex[index]],  self.train_rgb_label[self.cIndex[index]]
        img2,  target2 = self.train_sketch_image[self.tIndex[index]], self.train_sketch_label[self.tIndex[index]]
        
        img1_0 = self.transform_color(img1)
        img1_1 = self.transform_color1(img1)
        img2 = self.transform_thermal(img2)

        return img1_0, img1_1, img2, target1, target2


class LTCCData(data.Dataset):

    def __init__(self, data_dir, height = 384, width = 192, transform = None, rgbIndex = None, **kwargs):
        self.data_dir = data_dir
        self.train_dir = osp.join(self.data_dir, 'train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'test')
        self._check_before_run()

        train_image, train_label  = self._process_dir_train(self.train_dir, height, width)

        self.train_image = train_image
        self.train_label = train_label

        self.transform = transform
        self.cIndex = rgbIndex

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
            ### We modify this line for our ChannelExchange module ### date-2022/JUL/06
            ChannelExchange(gray = 3)]) ### default: gray=2

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir_train(self, dir_path, height, width):
        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        img_paths.sort()
        pattern1 = re.compile(r'(\d+)_(\d+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _, _ = map(int, pattern1.search(img_path).groups())
            pid_container.add(pid)
        pid_container = sorted(pid_container)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
        num_pids = len(pid_container)

        train_image = []
        train_label = []
        for img_path in img_paths:
            pid, _, camid = map(int, pattern1.search(img_path).groups())
            pid = pid2label[pid]
            train_label.append(pid)
            img = Image.open(img_path)
            img = img.resize((width, height), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_image.append(pix_array)
        num_imgs = len(train_label)

        return train_image, train_label

    def __getitem__(self, index):

        img1,  target1 = self.train_image[self.cIndex[index]],  self.train_label[self.cIndex[index]]
        img2,  target2 = self.train_image[self.cIndex[index]],  self.train_label[self.cIndex[index]]
        
        img1_0 = self.transform_color(img1)
        img1_1 = self.transform_color1(img1)
        img2 = self.transform_thermal(img2)

        return img1_0, img1_1, img2, target1, target2