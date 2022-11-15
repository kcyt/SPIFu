
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import trimesh
import trimesh.proximity
import trimesh.sample
import numpy as np
import math
import os
from PIL import Image


import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import cv2
import pickle

import logging
log = logging.getLogger('trimesh')
log.setLevel(40)



num_samples_to_use = 10000


# Full Thuman2.0 dataset (version 2.0)
#source_mesh_folder = "/mnt/lustre/kennard.chan/SPIFu/apps/results/Date_23_Aug_22_Time_22_34_49" # SPIFu, Epoch 30, Tested on GT SMPLX, THuman
source_mesh_folder = "/mnt/lustre/kennard.chan/SPIFu/apps/results/Date_27_Aug_22_Time_15_35_22" # SPIFu, Epoch 30, Tested on predicted SMPLX, THuman

use_clean_mesh = True

use_quick_test = False




def run_test_mesh_already_prepared():

        test_subject_list = np.loadtxt("/mnt/lustre/kennard.chan/getTestSet/test_set_list_version_2.txt", dtype=str).tolist()

        total_chamfer_distance = []
        total_point_to_surface_distance = []

        for subject in test_subject_list:

            GT_mesh_path = os.path.join("/mnt/lustre/kennard.chan/split_mesh/results", subject, '%s_clean.obj' % subject)

            source_mesh_path = os.path.join(source_mesh_folder, 'test_%s.obj' % subject)

            GT_mesh = trimesh.load(GT_mesh_path, process=False)
            source_mesh = trimesh.load(source_mesh_path, process=False)


            if use_clean_mesh:
                # clean mesh
                cc = source_mesh.split(only_watertight=False)   
                clean_mesh = cc[0]
                bbox = clean_mesh.bounds
                height = bbox[1,0] - bbox[0,0]
                for c in cc:
                    bbox = c.bounds
                    if height < bbox[1,0] - bbox[0,0]:
                        height = bbox[1,0] - bbox[0,0]
                        clean_mesh = c
                

                print("source_mesh cleaned")
                source_mesh = clean_mesh


            if use_quick_test:
                chamfer_distance, point_to_surface_distance = quick_get_chamfer_and_surface_dist(src_mesh=source_mesh, tgt_mesh=GT_mesh, num_samples=num_samples_to_use )
            else:
                chamfer_distance = get_chamfer_dist(src_mesh=source_mesh, tgt_mesh=GT_mesh, num_samples=num_samples_to_use )
                point_to_surface_distance = get_surface_dist(src_mesh=source_mesh, tgt_mesh=GT_mesh, num_samples=num_samples_to_use )


            print("subject: ", subject)

            print("chamfer_distance: ", chamfer_distance)
            total_chamfer_distance.append(chamfer_distance)

            print("point_to_surface_distance: ", point_to_surface_distance)
            total_point_to_surface_distance.append(point_to_surface_distance)


            print( 'total_chamfer_distance:' , np.sum(total_chamfer_distance) )
            print( 'total_point_to_surface_distance:' , np.sum(total_point_to_surface_distance) )

        average_chamfer_distance = np.mean(total_chamfer_distance) 
        average_point_to_surface_distance = np.mean(total_point_to_surface_distance) 

        print("average_chamfer_distance:",average_chamfer_distance)
        print("average_point_to_surface_distance:",average_point_to_surface_distance)







def get_chamfer_dist(src_mesh, tgt_mesh,  num_samples=10000):
    # Chamfer
    src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_samples) # src_surf_pts  has shape of (num_of_pts, 3)
    tgt_surf_pts, _ = trimesh.sample.sample_surface(tgt_mesh, num_samples)

    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt_mesh, src_surf_pts)
    _, tgt_src_dist, _ = trimesh.proximity.closest_point(src_mesh, tgt_surf_pts)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    tgt_src_dist[np.isnan(tgt_src_dist)] = 0

    #src_tgt_dist = src_tgt_dist.mean()
    #tgt_src_dist = tgt_src_dist.mean()
    src_tgt_dist = np.mean(np.square(src_tgt_dist))
    tgt_src_dist = np.mean(np.square(tgt_src_dist))

    chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2

    return chamfer_dist




def get_surface_dist(src_mesh, tgt_mesh, num_samples=10000):
    # P2S
    src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_samples)

    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt_mesh, src_surf_pts)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0

    #src_tgt_dist = src_tgt_dist.mean()
    src_tgt_dist = np.mean(np.square(src_tgt_dist))

    return src_tgt_dist





def quick_get_chamfer_and_surface_dist(src_mesh, tgt_mesh,  num_samples=10000):
    # Chamfer
    src_surf_pts, _ = trimesh.sample.sample_surface(src_mesh, num_samples) # src_surf_pts  has shape of (num_of_pts, 3)
    tgt_surf_pts, _ = trimesh.sample.sample_surface(tgt_mesh, num_samples)

    _, src_tgt_dist, _ = trimesh.proximity.closest_point(tgt_mesh, src_surf_pts)
    _, tgt_src_dist, _ = trimesh.proximity.closest_point(src_mesh, tgt_surf_pts)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    tgt_src_dist[np.isnan(tgt_src_dist)] = 0

    #src_tgt_dist = src_tgt_dist.mean()
    #tgt_src_dist = tgt_src_dist.mean()
    src_tgt_dist = np.mean(np.square(src_tgt_dist))
    tgt_src_dist = np.mean(np.square(tgt_src_dist))

    chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2
    surface_dist = src_tgt_dist

    return chamfer_dist, surface_dist




if __name__ == "__main__":


    run_test_mesh_already_prepared()









