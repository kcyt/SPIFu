


import os
import random

import numpy as np 
from PIL import Image, ImageOps
import cv2
import torch
import json
import trimesh
import logging

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F





class NormalDataset(Dataset):


    def __init__(self, opt, phase = 'train', evaluation_mode=False, validation_mode=False, frontal_only=False):
        self.opt = opt
        self.training_subject_list = np.loadtxt("/mnt/lustre/kennard.chan/getTestSet/train_set_list_version_2.txt", dtype=str)

        self.validation_mode = validation_mode
        self.phase = phase
        self.is_train = (self.phase == 'train')

        if self.opt.useValidationSet:

            indices = np.arange( len(self.training_subject_list) )
            np.random.seed(10)
            np.random.shuffle(indices)
            lower_split_index = round( len(self.training_subject_list)* 0.1 )
            val_indices = indices[:lower_split_index]
            train_indices = indices[lower_split_index:]

            if self.validation_mode:
                self.training_subject_list = self.training_subject_list[val_indices]
                self.is_train = False
            else:
                self.training_subject_list = self.training_subject_list[train_indices]

        self.training_subject_list = self.training_subject_list.tolist()



        if evaluation_mode:
            print("Overwriting self.training_subject_list!")
            self.training_subject_list = np.loadtxt("/mnt/lustre/kennard.chan/getTestSet/test_set_list_version_2.txt", dtype=str).tolist()
            self.is_train = False





        self.groundtruth_normal_map_directory = "/mnt/lustre/kennard.chan/render_THuman_with_blender/buffer_normal_maps_of_full_mesh_version_2"
        self.root = "/mnt/lustre/kennard.chan/render_THuman_with_blender/buffer_fixed_full_mesh_version_2"


        self.subjects = self.training_subject_list  



        self.img_files = []
        if not frontal_only:
            for training_subject in self.subjects:
                subject_render_folder = os.path.join(self.root, training_subject)
                subject_render_paths_list = [  os.path.join(subject_render_folder,f) for f in os.listdir(subject_render_folder) if "image" in f   ]
                self.img_files = self.img_files + subject_render_paths_list

        else:
            with open('/mnt/lustre/kennard.chan/split_mesh/all_subjects_with_angles_version_2.txt', 'r') as f:
                file_list = f.readlines()
                file_list = [ ( line.split()[0], line.split()[1].replace('\n', '') ) for line in file_list ]

            for subject, frontal_angle in file_list:
                if subject in self.subjects:
                    subject_render_filepath = os.path.join(self.root, subject, 'rendered_image_{0:03d}.png'.format( int(frontal_angle) ) )
                    self.img_files.append(subject_render_filepath)


        self.img_files = sorted(self.img_files)


        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(), #  ToTensor converts input to a shape of (C x H x W) in the range [0.0, 1.0] for each dimension
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalise with mean of 0.5 and std_dev of 0.5 for each dimension. Finally range will be [-1,1] for each dimension
        ])

        # augmentation
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
                                   hue=opt.aug_hue)
        ])


    def __len__(self):
        return len(self.img_files)





    def get_item(self, index):

        img_path = self.img_files[index]
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # get yaw
        yaw = img_name.split("_")[-1]
        yaw = int(yaw)



        # get subject
        subject = img_path.split('/')[-2] # e.g. "0507"

        param_path = os.path.join(self.root, subject , "rendered_params_" + "{0:03d}".format(yaw) + ".npy"  )
        render_path = os.path.join(self.root, subject, "rendered_image_" + "{0:03d}".format(yaw) + ".png"  )
        mask_path = os.path.join(self.root, subject, "rendered_mask_" + "{0:03d}".format(yaw) + ".png"  )
        
        nmlF_high_res_path =  os.path.join(self.groundtruth_normal_map_directory, subject, "rendered_nmlF_" + "{0:03d}".format(yaw) + ".exr"  )
        nmlB_high_res_path =  os.path.join(self.groundtruth_normal_map_directory, subject, "rendered_nmlB_" + "{0:03d}".format(yaw) + ".exr"  )


        load_size_associated_with_scale_factor = 1024

        # get params
        param = np.load(param_path, allow_pickle=True)  # param is a np.array that looks similar to a dict.  # ortho_ratio = 0.4 , e.g. scale or y_scale = 0.961994278, e.g. center or vmed = [-1.0486  92.56105  1.0101 ]
        center = param.item().get('center') # is camera 3D center position in the 3D World point space (without any rotation being applied).
        R = param.item().get('R')   # R is used to rotate the CAD model according to a given pitch and yaw.
        scale_factor = param.item().get('scale_factor') # is camera 3D center position in the 3D World point space (without any rotation being applied).



        mask = Image.open(mask_path).convert('L') # convert to grayscale (it shd already be grayscale)
        render = Image.open(render_path).convert('RGB')

        mask = transforms.ToTensor()(mask).float()

        if self.opt.use_augmentation and self.is_train:
            render = self.aug_trans(render)

        render = self.to_tensor(render)  # normalize render from [0,255] to [-1,1]
        render = mask.expand_as(render) * render



        nmlF_high_res = cv2.imread(nmlF_high_res_path, cv2.IMREAD_UNCHANGED).astype(np.float32) # numpy of [1024,1024,3]
        nmlB_high_res = cv2.imread(nmlB_high_res_path, cv2.IMREAD_UNCHANGED).astype(np.float32) 
        nmlB_high_res = nmlB_high_res[:,::-1,:].copy()
 
        
        nmlF_high_res = np.transpose(nmlF_high_res, [2,0,1]  ) # change to shape of [3,1024,1024]
        nmlF_high_res = torch.Tensor(nmlF_high_res)
        nmlF_high_res = mask.expand_as(nmlF_high_res) * nmlF_high_res

        nmlB_high_res = np.transpose(nmlB_high_res, [2,0,1]  ) # change to shape of [3,1024,1024]
        nmlB_high_res = torch.Tensor(nmlB_high_res)
        nmlB_high_res = mask.expand_as(nmlB_high_res) * nmlB_high_res



        return {
            'name': subject,
            'render_path':render_path,
            'original_high_res_render':render,
            'mask_high_res':mask,
            'nmlB_high_res':nmlB_high_res,
            'nmlF_high_res':nmlF_high_res

                }



    def __getitem__(self, index):
        return self.get_item(index)







    












