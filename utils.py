import os, time, shutil, argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, rgb2gray, lab2rgb
plt.switch_backend('agg')

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models

def save_checkpoint(state, is_best_so_far, filename='checkpoints/checkpoint.pth.tar'):
    '''Saves checkpoint, and replace the old best model if the current model is better'''
    torch.save(state, filename)
    if is_best_so_far:
        shutil.copyfile(filename, 'checkpoints/model_best.pth.tar')

class AverageMeter(object):
    '''An easy way to compute and store both average and current values'''
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def visualize_image(grayscale_input, ab_input=None, show_image=False, save_path=None, save_name=None):
    '''Show or save image given grayscale (and ab color) inputs. Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
    plt.clf() # clear matplotlib plot
    ab_input = ab_input.cpu()
    grayscale_input = grayscale_input.cpu()    
    if ab_input is None:
        grayscale_input = grayscale_input.squeeze().numpy() 
        if save_path is not None and save_name is not None: 
            plt.imsave(grayscale_input, '{}.{}'.format(save_path['grayscale'], save_name) , cmap='gray')
        if show_image: 
            plt.imshow(grayscale_input, cmap='gray')
            plt.show()
    else: 
        color_image = torch.cat((grayscale_input, ab_input), 0).numpy()
        color_image = color_image.transpose((1, 2, 0))  
        color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
        color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
        color_image = lab2rgb(color_image.astype(np.float64))
        grayscale_input = grayscale_input.squeeze().numpy()
        if save_path is not None and save_name is not None:
            plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
            plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))
        if show_image: 
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(grayscale_input, cmap='gray')
            axarr[1].imshow(color_image)
            plt.show()

class GrayscaleImageFolder(datasets.ImageFolder):
    '''Custom images folder, which converts images to grayscale before loading'''
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img_original = self.transform(img)
            img_original = np.asarray(img_original)
            img_lab = rgb2lab(img_original)
            img_lab = (img_lab + 128) / 255
            img_ab = img_lab[:, :, 1:3]
            img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
            img_original = rgb2gray(img_original)
            img_original = torch.from_numpy(img_original).unsqueeze(0).float()
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img_original, img_ab, target
