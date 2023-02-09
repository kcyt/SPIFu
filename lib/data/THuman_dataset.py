


import os
import random
from math import radians

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

from ..geometry import index, orthogonal

from numpy.linalg import inv
from scipy import sparse  

NUM_OF_NEIGHBORS = 8
NUM_OF_FACES = 6
NUM_OF_INITIAL_FEATURES = 7

GT_OUTPUT_RESOLUTION = 512 
GT_NUM_OF_FACES = 6
GT_NUM_OF_FEATURES = 4 

log = logging.getLogger('trimesh')
log.setLevel(40)



def load_trimesh(root_dir, training_subject_list = None):

    folders = os.listdir(root_dir)
    meshs = {}
    for i, f in enumerate(folders):
        if f == ".DS_Store" or f == "val.txt":
            continue
        sub_name = f

        if sub_name not in training_subject_list: # only load meshes that are in the training set
            continue

        meshs[sub_name] = trimesh.load(os.path.join(root_dir, f, '%s_clean.obj' % sub_name))

    return meshs






def make_rotate(rx, ry, rz):
    # rx is rotation angle about the x-axis
    # ry is rotation angle about the y-axis
    # rz is rotation angle about the z-axis

    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3,3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3,3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3,3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz,Ry),Rx)
    return R

class THumanDataset(Dataset):


    def __init__(self, opt, projection='orthogonal', phase = 'train', evaluation_mode=False, validation_mode=False, frontal_only=False):
        self.opt = opt
        self.projection_mode = projection

        use_fake_training_set = True # set to True to use fake training set

        self.training_subject_list = np.loadtxt("/mnt/lustre/kennard.chan/getTestSet/train_set_list_version_2.txt", dtype=str)


        self.validation_mode = validation_mode
        self.phase = phase
        self.is_train = (self.phase == 'train')

        self.frontal_only = frontal_only

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


        self.evaluation_mode = evaluation_mode

        if evaluation_mode:
            print("Overwriting self.training_subject_list!")
            self.training_subject_list = np.loadtxt("/mnt/lustre/kennard.chan/getTestSet/test_set_list_version_2.txt", dtype=str).tolist()

            self.is_train = False

        

        if use_fake_training_set:
            self.training_subject_list = np.loadtxt("/mnt/lustre/kennard.chan/getTestSet/fake_train_set_list.txt", dtype=str).tolist()
            
            print("using fake training subject list!")



        # use THuman dataset
        self.root = "/mnt/lustre/kennard.chan/render_THuman_with_blender/buffer_fixed_full_mesh_version_2"

        self.mesh_directory = "/mnt/lustre/kennard.chan/split_mesh/results"

        if (  evaluation_mode  or  (self.frontal_only and not self.validation_mode)  ):
            pass 
        else:
            self.mesh_dic = load_trimesh(self.mesh_directory,  training_subject_list = self.training_subject_list)  # a dict containing the meshes of all the CAD models.


        self.normal_directory_high_res = "/mnt/lustre/kennard.chan/SPIFu/trained_normal_maps"

        self.frontal_angle_dict = {}



        if self.opt.use_unrolled_smpl:

            if self.opt.use_gt_smplx :  
                self.unrolled_smpl_features_directory = "/mnt/lustre/kennard.chan/unroll_groundtruth_mesh/unroll_given_gt_smplx_THuman2_meshes_version_2" 
            else:
                self.unrolled_smpl_features_directory = "/mnt/lustre/kennard.chan/smpl_unrolling/unroll_proper_smplx_results_version_2"


            if self.opt.use_unrolled_smpl:
                # compute coordinate_matrix

                XY, step_size = np.linspace(-1,1, self.opt.output_resolution_smpl_map+1, endpoint=True, retstep=True) # XY has shape of (self.opt.output_resolution_smpl_map+1,) 
                XY = XY + 0.5*step_size
                XY = XY[:-1] # remove last element. Shape of (self.opt.output_resolution_smpl_map,) 

                XY = np.meshgrid(XY,XY) # # A list of 2 coordinate matrix
                XY = np.array(XY).T  # Shape of [self.opt.output_resolution_smpl_map, self.opt.output_resolution_smpl_map, 2], where the '2' is y coordinate first then x.


                self.coordinate_matrix = np.repeat(XY[:,:,None,:], repeats=NUM_OF_FACES , axis=2 )  # Shape of [self.opt.output_resolution_smpl_map, self.opt.output_resolution_smpl_map, NUM_OF_FACES , 2]

                self.blend_weights = np.load("SPIFu/RayBasedSampling/PCA_smplx_blend_weights.npy") # Shape of [10475, 24]

                self.front_or_back_facing_at_rest = np.load("SPIFu/RayBasedSampling/smplx_front_or_back_facing_at_rest.npy") # [6890, 1] 
                self.front_or_back_facing_at_rest[self.front_or_back_facing_at_rest<1] = -1 # convert the zeros in front_or_back_facing_at_rest into -1


                if self.opt.use_normals_smpl and self.opt.use_normals_smpl_with_rest_pose_normals:
                    self.normals_at_rest = np.load('SPIFu/RayBasedSampling/rest_pose_smplx_vertex_normals.npy') # [6890, 3]



        self.subjects = self.training_subject_list 

        self.num_sample_inout = self.opt.num_sample_inout 


        
        self.img_files = []
        if not self.frontal_only:
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



    def select_sampling_method(self, subject, calib, b_min, b_max, R = None):


        compensation_factor = 4.0

        mesh = self.mesh_dic[subject] # the mesh of 1 subject/CAD

        # note, this is the solution for when dataset is "THuman"
        # adjust sigma according to the mesh's size (measured using the y-coordinates)
        y_length = np.abs(  np.max(mesh.vertices, axis=0)[1]  -  np.min(mesh.vertices, axis=0)[1]  )
        sigma_multiplier = y_length/188

        surface_points, face_indices = trimesh.sample.sample_surface(mesh, int(compensation_factor * 4 * self.num_sample_inout) )  # self.num_sample_inout is no. of sampling points and is default to 8000.


        # add random points within image space
        length = b_max - b_min # has shape of (3,)
        random_points = np.random.rand( int(compensation_factor * self.num_sample_inout // 4) , 3) * length + b_min # shape of [compensation_factor*num_sample_inout/4, 3]
        surface_points_shape = list(surface_points.shape)
        random_noise = np.random.normal(scale= self.opt.sigma_low_resolution_pifu * sigma_multiplier, size=surface_points_shape)
        sample_points_low_res_pifu = surface_points + random_noise # sample_points are points very near the surface. The sigma represents the std dev of the normal distribution
        sample_points_low_res_pifu = np.concatenate([sample_points_low_res_pifu, random_points], 0) # shape of [compensation_factor*4.25*num_sample_inout, 3]
        np.random.shuffle(sample_points_low_res_pifu)

        inside_low_res_pifu = mesh.contains(sample_points_low_res_pifu) # return a boolean 1D array of size (num of sample points,)
        inside_points_low_res_pifu = sample_points_low_res_pifu[inside_low_res_pifu]

        outside_points_low_res_pifu = sample_points_low_res_pifu[np.logical_not(inside_low_res_pifu)]


        # reduce the number of inside and outside points if there are too many inside points. (it is very likely that "nin > self.num_sample_inout // 2" is true)
        nin = inside_points_low_res_pifu.shape[0]
        inside_points_low_res_pifu = inside_points_low_res_pifu[
                        :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else inside_points_low_res_pifu  # should have shape of [2500, 3]
        outside_points_low_res_pifu = outside_points_low_res_pifu[
                         :self.num_sample_inout // 2] if nin > self.num_sample_inout // 2 else outside_points_low_res_pifu[
                                                                                               :(self.num_sample_inout - nin)]     # should have shape of [2500, 3]

        samples_low_res_pifu = np.concatenate([inside_points_low_res_pifu, outside_points_low_res_pifu], 0).T   # should have shape of [3, 5000]


        labels_low_res_pifu = np.concatenate([np.ones((1, inside_points_low_res_pifu.shape[0])), np.zeros((1, outside_points_low_res_pifu.shape[0]))], 1) # should have shape of [1, 5000]. If element is 1, it means the point is inside. If element is 0, it means the point is outside.

        samples_low_res_pifu = torch.Tensor(samples_low_res_pifu).float()
        labels_low_res_pifu = torch.Tensor(labels_low_res_pifu).float()


        del mesh

        return {
            'samples_low_res_pifu': samples_low_res_pifu,
            'labels_low_res_pifu': labels_low_res_pifu,
            }





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
        
        nmlF_high_res_path =  os.path.join(self.normal_directory_high_res, subject, "rendered_nmlF_" + "{0:03d}".format(yaw) + ".npz"  )
        nmlB_high_res_path =  os.path.join(self.normal_directory_high_res, subject, "rendered_nmlB_" + "{0:03d}".format(yaw) + ".npz"  )


        if self.opt.use_unrolled_smpl:
            unrolled_smpl_features_path = os.path.join(self.unrolled_smpl_features_directory, subject, 'unrolled_smpl_features_sparse_subject_{0}_angle_{1:03d}.npz'.format(subject, yaw) )



        load_size_associated_with_scale_factor = 1024


        # get params
        param = np.load(param_path, allow_pickle=True)  # param is a np.array that looks similar to a dict.  # ortho_ratio = 0.4 , e.g. scale or y_scale = 0.961994278, e.g. center or vmed = [-1.0486  92.56105  1.0101 ]
        center = param.item().get('center') # is camera 3D center position in the 3D World point space (without any rotation being applied).
        R = param.item().get('R')   # R is used to rotate the CAD model according to a given pitch and yaw.
        scale_factor = param.item().get('scale_factor') # is camera 3D center position in the 3D World point space (without any rotation being applied).


        b_range = load_size_associated_with_scale_factor / scale_factor # e.g. 512/scale_factor
        b_center = center
        b_min = b_center - b_range/2
        b_max = b_center + b_range/2


        # the calib matrix is only useful for points of the groundtruth/object mesh. It is not useful for the points under gen_mesh().

        # extrinsic is used to rotate the 3D points according to our specified pitch and yaw
        translate = -center.reshape(3, 1)
        extrinsic = np.concatenate([R, translate], axis=1)  # when applied on the 3D pts, the rotation is done first, then the translation
        extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)

        temp_extrinsic = np.copy(extrinsic)
        
        scale_intrinsic = np.identity(4)
        scale_intrinsic[0, 0] = 1.0 * scale_factor #2.4851518#1.0   
        scale_intrinsic[1, 1] = -1.0 * scale_factor #-2.4851518#-1.0
        scale_intrinsic[2, 2] = 1.0 * scale_factor  #2.4851518#1.0

        uv_intrinsic = np.identity(4)
        uv_intrinsic[0, 0] = 1.0 / float(load_size_associated_with_scale_factor // 2) # self.opt.loadSizeGlobal == 512 by default. This value must be 512 unless you change the "scale_factor"
        uv_intrinsic[1, 1] = 1.0 / float(load_size_associated_with_scale_factor // 2) # uv_intrinsic[1, 1] is equal to 1/256
        uv_intrinsic[2, 2] = 1.0 / float(load_size_associated_with_scale_factor // 2) 

        intrinsic = np.matmul(uv_intrinsic, scale_intrinsic)
        calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float() # calib should still work to transform pts into the [-1,1] range even if the input image size actually changes from 512 to 1024.
        extrinsic = torch.Tensor(extrinsic).float()

        render = Image.open(render_path).convert('RGB')

        mask = Image.open(mask_path).convert('L') # convert to grayscale (it shd already be grayscale)

        # to reload back from sparse matrix to numpy array:
        sparse_matrix = sparse.load_npz(nmlF_high_res_path).todense()
        nmlF_high_res = np.asarray(sparse_matrix) 
        nmlF_high_res = np.reshape(nmlF_high_res, [3, 1024, 1024] )

        sparse_matrix = sparse.load_npz(nmlB_high_res_path).todense()
        nmlB_high_res = np.asarray(sparse_matrix) 
        nmlB_high_res = np.reshape(nmlB_high_res, [3, 1024, 1024] )


        mask = transforms.ToTensor()(mask).float() # *Should* have a shape of (C,H,W)

        if self.opt.use_augmentation and self.is_train:
            render = self.aug_trans(render)


        render = self.to_tensor(render)  # normalize render from [0,255] to [-1,1]
        render = mask.expand_as(render) * render


        
        # resize the 1024 x 1024 image to 512 x 512 for the low-resolution pifu
        render_low_pifu = F.interpolate(torch.unsqueeze(render,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
        mask_low_pifu = F.interpolate(torch.unsqueeze(mask,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
        render_low_pifu = render_low_pifu[0]
        mask_low_pifu = mask_low_pifu[0]
        
        
        nmlF_high_res = torch.Tensor(nmlF_high_res)
        nmlB_high_res = torch.Tensor(nmlB_high_res)
        nmlF_high_res = mask.expand_as(nmlF_high_res) * nmlF_high_res
        nmlB_high_res = mask.expand_as(nmlB_high_res) * nmlB_high_res

        nmlF  = F.interpolate(torch.unsqueeze(nmlF_high_res,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
        nmlF = nmlF[0]
        nmlB  = F.interpolate(torch.unsqueeze(nmlB_high_res,0), size=(self.opt.loadSizeGlobal,self.opt.loadSizeGlobal) )
        nmlB = nmlB[0]




        if self.opt.use_unrolled_smpl:

            loaded_features_array = sparse.load_npz(unrolled_smpl_features_path).todense()
            loaded_features_array = np.asarray(loaded_features_array) # shape of [self.opt.output_resolution_smpl_map*self.opt.output_resolution_smpl_map*NUM_OF_FACES, num_of_features] ]
            loaded_features_array = np.reshape(loaded_features_array, [self.opt.output_resolution_smpl_map,self.opt.output_resolution_smpl_map, NUM_OF_FACES, NUM_OF_INITIAL_FEATURES] ) # Shape of [self.opt.output_resolution_smpl_map,self.opt.output_resolution_smpl_map, NUM_OF_FACES, Num_of_initial_features == 7 ] 
            # initial features are:
            #   1. Point 1 vertices_indices - 1 dim
            #   2. Point 2 vertices_indices - 1 dim
            #   3. Point 3 vertices_indices - 1 dim
            #   4. face_average_depth_value - 1 dim
            #   5. face_average_normals - 3 dims

            #We will be using "self.coordinate_matrix"  # Shape of [self.opt.output_resolution_smpl_map, self.opt.output_resolution_smpl_map, NUM_OF_FACES, 2], where the '2' is y coordinate first then x.
            

            # Now, compute front-facing or back-facing wrt camera (Can just test if the z value of 'smpl_vertex_normals' is positive or negative)
            smpl_vertex_normals = loaded_features_array[:,:,:,4:7] # [self.opt.output_resolution_smpl_map,self.opt.output_resolution_smpl_map, NUM_OF_FACES, 3]
            smpl_vertex_normals_z_values = np.copy(smpl_vertex_normals[:,:,:,-1]) # The z value
            bool_arr_1 = smpl_vertex_normals_z_values<0
            bool_arr_2 = smpl_vertex_normals_z_values>=0
            smpl_vertex_normals_z_values[bool_arr_1] = -1 # points generated from faces that are facing backwards
            smpl_vertex_normals_z_values[bool_arr_2] = 1 # points generated from faces that are facing camera
            front_or_back_facing = smpl_vertex_normals_z_values[..., None] # [self.opt.output_resolution_smpl_map, self.opt.output_resolution_smpl_map, NUM_OF_FACES, 1 ] # Elements are either -1 (back-facing) or 1 (front facing)

            vertices_indices_1 = loaded_features_array[:,:,:,0] # vertices_indices_1, shape of [self.opt.output_resolution_smpl_map, self.opt.output_resolution_smpl_map, NUM_OF_FACES ] 
            vertices_indices_unrolled_1 = np.reshape(vertices_indices_1, [-1]).astype(np.int32) # shape of (Num_of_vertices,)
         
            vertices_indices_2 = loaded_features_array[:,:,:,1] # vertices_indices_2, shape of [self.opt.output_resolution_smpl_map, self.opt.output_resolution_smpl_map, NUM_OF_FACES ] 
            vertices_indices_unrolled_2 = np.reshape(vertices_indices_2, [-1]).astype(np.int32) # shape of (Num_of_vertices,)

            vertices_indices_3 = loaded_features_array[:,:,:,2] # vertices_indices_3, shape of [self.opt.output_resolution_smpl_map, self.opt.output_resolution_smpl_map, NUM_OF_FACES ] 
            vertices_indices_unrolled_3 = np.reshape(vertices_indices_3, [-1]).astype(np.int32) # shape of (Num_of_vertices,)

            vertices_blend_weights = self.blend_weights[vertices_indices_unrolled_1, :] + self.blend_weights[vertices_indices_unrolled_2, :] + self.blend_weights[vertices_indices_unrolled_3, :]  # shape of [Num_of_vertices, 24]
            vertices_blend_weights = vertices_blend_weights / 3
            vertices_blend_weights = np.reshape(vertices_blend_weights, [self.opt.output_resolution_smpl_map, self.opt.output_resolution_smpl_map, NUM_OF_FACES, 24]  )




            vertices_front_or_back_facing_at_rest = self.front_or_back_facing_at_rest[vertices_indices_unrolled_1, 0:1] + self.front_or_back_facing_at_rest[vertices_indices_unrolled_2, 0:1] + self.front_or_back_facing_at_rest[vertices_indices_unrolled_3, 0:1]   # shape of [Num_of_vertices, 1]
            vertices_front_or_back_facing_at_rest = vertices_front_or_back_facing_at_rest / 3
            vertices_front_or_back_facing_at_rest = np.reshape(vertices_front_or_back_facing_at_rest, [self.opt.output_resolution_smpl_map, self.opt.output_resolution_smpl_map, NUM_OF_FACES, 1]  )

            if self.opt.use_normals_smpl_with_rest_pose_normals:
                vertices_normals_at_rest = self.normals_at_rest[vertices_indices_unrolled_1, :] + self.normals_at_rest[vertices_indices_unrolled_2, :] + self.normals_at_rest[vertices_indices_unrolled_3, :] # shape of [Num_of_vertices, 3]
                vertices_normals_at_rest = vertices_normals_at_rest / 3  # shape of [Num_of_vertices, 3]
                vertices_normals_at_rest = np.reshape(vertices_normals_at_rest, [self.opt.output_resolution_smpl_map, self.opt.output_resolution_smpl_map, NUM_OF_FACES, 3]  )


            smpl_depth_values = loaded_features_array[:,:,:,3:4]  # Shape of [self.opt.output_resolution_smpl_map,self.opt.output_resolution_smpl_map, NUM_OF_FACES, 1 ] 

            
            features_array = None 
            if self.opt.use_coordinates_smpl :
                features_array = np.concatenate([self.coordinate_matrix, smpl_depth_values] ,axis=3)


            if self.opt.use_normals_smpl :
                if features_array is None:
                    
                    if not self.opt.use_normals_smpl_with_rest_pose_normals:
                        features_array = np.concatenate( [smpl_vertex_normals, front_or_back_facing] ,axis=3)
                    else:
                        features_array = np.concatenate( [smpl_vertex_normals, front_or_back_facing, vertices_front_or_back_facing_at_rest, vertices_normals_at_rest] ,axis=3)

                else:
                    if not self.opt.use_normals_smpl_with_rest_pose_normals:
                        features_array = np.concatenate( [features_array, smpl_vertex_normals, front_or_back_facing] , axis=3 )
                    else:
                        features_array = np.concatenate( [features_array, smpl_vertex_normals, front_or_back_facing, vertices_front_or_back_facing_at_rest, vertices_normals_at_rest] , axis=3 )


            if self.opt.use_blendweights_smpl:
                if features_array is None:
                    features_array = vertices_blend_weights

                else:
                    features_array = np.concatenate( [features_array, vertices_blend_weights] , axis=3 )


            number_of_final_features = features_array.shape[-1]
            # interpolate to (512,512) size
            interpolation_size = 512
            features_array = np.reshape(features_array, [self.opt.output_resolution_smpl_map, self.opt.output_resolution_smpl_map, NUM_OF_FACES*number_of_final_features ] )

            features_array = torch.Tensor( np.transpose(features_array, [2,0,1] ) ) # [C,H,W]
            
            if self.opt.output_resolution_smpl_map != interpolation_size:
                features_array = F.interpolate(torch.unsqueeze(features_array,0), size=(interpolation_size,interpolation_size) )
                features_array = features_array[0] # [NUM_OF_FACES*32, self.opt.output_resolution_smpl_map, self.opt.output_resolution_smpl_map ] or [C,H,W]
                
        else:
            features_array = 0 





        if self.evaluation_mode or self.frontal_only:
            sample_data = {'samples_low_res_pifu':0, 'labels_low_res_pifu':0 }

        else:

            if self.opt.num_sample_inout:  # opt.num_sample_inout has default of 8000
                sample_data = self.select_sampling_method(subject, calib, b_min = b_min, b_max = b_max, R = R)




        final_dict = {
                'name': subject,
                'render_path':render_path,
                'render_low_pifu': render_low_pifu,
                'mask_low_pifu': mask_low_pifu,
                'original_high_res_render':render,
                'mask':mask,
                'calib': calib,
                'extrinsic': extrinsic,    
                'samples_low_res_pifu': sample_data['samples_low_res_pifu'],
                'labels_low_res_pifu': sample_data['labels_low_res_pifu'],
                'b_min': b_min,
                'b_max': b_max,
                'nmlF': nmlF,
                'nmlB': nmlB,
                'nmlF_high_res':nmlF_high_res,
                'nmlB_high_res':nmlB_high_res,
                'unrolled_smpl_features':features_array
                    }



        return final_dict
                



    def __getitem__(self, index):
        return self.get_item(index)



# for debugging only
if __name__ == "__main__":


    import sys
    from PIL import Image
    import cv2

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    

    from options import BaseOptions

    parser = BaseOptions()
    opt = parser.parse()





    












