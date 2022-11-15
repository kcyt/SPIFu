'''
MIT License

Copyright (c) 2019 Shunsuke Saito, Zeng Huang, and Ryota Natsume

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
from skimage import measure
import numpy as np
import torch
from .sdf import create_grid, eval_grid_octree, eval_grid
from skimage import measure

from numpy.linalg import inv

def reconstruction(net, cuda, calib_tensor,
                   resolution, thresh=0.5,
                   use_octree=False, num_samples=10000, transform=None, b_min = np.array([-1,-1,-1]), b_max = np.array([1,1,1]) , using_pretrained_model=False, generate_from_low_res = False ):
    
    
    '''
    Reconstruct meshes from sdf predicted by the network.
    :param net: a BasePixImpNet object. call image filter beforehead.
    :param cuda: cuda device
    :param calib_tensor: calibration tensor
    :param resolution: resolution of the grid cell # is 512 by default
    :param b_min: bounding box corner [x_min, y_min, z_min]
    :param b_max: bounding box corner [x_max, y_max, z_max]
    :param use_octree: whether to use octree acceleration
    :param num_samples: how many points to query each gpu iteration
    :return: marching cubes results.
    '''
    # First we create a grid by resolution
    # and transforming matrix for grid coordinates to real world xyz
    # for create_grid(), the b_min and b_max automatically set to b_min=np.array([-1, -1, -1]), b_max=np.array([1, 1, 1]
    coords, mat = create_grid(resolution, resolution, resolution, b_min=b_min, b_max=b_max)   # coords has shape (3, 512, 512, 512) and is already transformed into the "bounding box" dimension; mat has shape of [4,4] and it transforms 3D points from 'resolution' dimension to 'bounding box' dimension
                              #b_min, b_max, transform=transform)
    # Note that 'coords' are all within the range of [-1,1]
    # Note that 'mat' transform points from 'resolution' dimension to 'bounding box' dimension 




    #calib = calib_tensor[0].cpu().numpy()
    if using_pretrained_model:
        calib = calib_tensor[0].cpu().numpy()
        calib_inv = inv(calib)
        coords = coords.reshape(3,-1).T
        coords = np.matmul(np.concatenate([coords, np.ones((coords.shape[0],1))], 1), calib_inv.T)[:, :3]
        coords = coords.T.reshape(3,resolution,resolution,resolution)

    #calib = calib_tensor.cpu().numpy() # calib merely invert the sign of the y-coordinates
    #calib_inv = inv(calib)


    #coords = np.matmul(np.concatenate([coords, np.ones((coords.shape[0],1))], 1), calib_inv.T)[:, :3] # ownself: think calib_inv should be removed here 


    #coords = coords.reshape(3,-1).T # shape of [Num_of_pts, 3]
    #coords = coords.T.reshape(3,resolution,resolution,resolution) # shape of [3, resolution, resolution, resolution]

    if not generate_from_low_res:
        print("generating using high_res_pifu")
    else: 
        print("generating using low_res_pifu")

    # Then we define the lambda function for cell evaluation
    def eval_func(points):
        points = np.expand_dims(points, axis=0)
        points = np.repeat(points, 1, axis=0) # seems to do nth
        samples = torch.from_numpy(points).to(device=cuda).float() 
        
        if using_pretrained_model:
            net.query(samples, calib_tensor)
            pred = net.get_preds()[0][0]
        else:
            if not generate_from_low_res:
                # high resolution pifu. Low and High res use the same calib during evaluation
                net.new_query(samples, calib_low_res=calib_tensor, calib_high_res = calib_tensor) # the calib_tensor merely flip the sign of the y-coordinates. The 'samples' remain in the range of [-1,1]
                pred = net.get_preds()[0][0]
            else:
                # low resolution pifu
                if net.name == "hg_pifu_high_res":
                    net.netG.query(samples, calib_tensor) # the calib_tensor merely flip the sign of the y-coordinates. The 'samples' remain in the range of [-1,1]
                    pred = net.netG.get_preds()[0][0]
                elif net.name == "hg_pifu_low_res":
                    net.query(samples, calib_tensor)
                    pred = net.get_preds()[0][0]
                else:
                    raise Exception("incorrect net.name !")

        return pred.detach().cpu().numpy()
    # Then we evaluate the grid
    if use_octree:
        sdf = eval_grid_octree(coords, eval_func, num_samples=num_samples) # shape of (256, 256, 256)
    else:
        sdf = eval_grid(coords, eval_func, num_samples=num_samples)






    # Finally we do marching cubes
    try:
        verts, faces, normals, values = measure.marching_cubes_lewiner(sdf, 0.5) # verts should be in resolution dimension
        # transform verts into world coordinate system (ownself:Think its the opposite)
        
        if using_pretrained_model:
            trans_mat = np.matmul(calib_inv, mat) # ownself: think calib_inv should be removed here also
        else:
            trans_mat = mat # own edit, changed from the above line
        verts = np.matmul(trans_mat[:3, :3], verts.T) + trans_mat[:3, 3:4]  # now verts should be in bounding box dimensions of [-1,1]
        verts = verts.T
        # in case mesh has flip transformation
        if np.linalg.det(trans_mat[:3, :3]) < 0.0:
            faces = faces[:,::-1]
        return verts, faces, normals, values
    except:
        print('error cannot marching cubes')
        return -1








def save_obj_mesh(mesh_path, verts, faces=None):
    file = open(mesh_path, 'w')

    for v in verts:
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
    if faces is not None:
        for f in faces:
            if f[0] == f[1] or f[1] == f[2] or f[0] == f[2]:
                continue
            f_plus = f + 1
            file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_color(mesh_path, verts, faces, colors):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        c = colors[idx]
        file.write('v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))
    for f in faces:
        f_plus = f + 1
        file.write('f %d %d %d\n' % (f_plus[0], f_plus[2], f_plus[1]))
    file.close()


def save_obj_mesh_with_uv(mesh_path, verts, faces, uvs):
    file = open(mesh_path, 'w')

    for idx, v in enumerate(verts):
        vt = uvs[idx]
        file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
        file.write('vt %.4f %.4f\n' % (vt[0], vt[1]))

    for f in faces:
        f_plus = f + 1
        file.write('f %d/%d %d/%d %d/%d\n' % (f_plus[0], f_plus[0],
                                              f_plus[2], f_plus[2],
                                              f_plus[1], f_plus[1]))
    file.close()
