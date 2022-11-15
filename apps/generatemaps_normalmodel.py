
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import cv2
import pickle

from PIL import Image

from lib.options import BaseOptions
from lib.networks import define_G

import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
from scipy import sparse

from lib.data.THuman_NormalDataset import NormalDataset



parser = BaseOptions()
opt = parser.parse()
HIGH_RES_SIZE = 1024


opt.useValidationSet = False
opt.use_augmentation = False
print('useValidationSet is set to False')
print('use_augmentation is set to False')




trained_normal_maps_path = "/mnt/lustre/kennard.chan/SPIFu/trained_normal_maps"






def generate_maps(opt):

    if torch.cuda.is_available():
        # set cuda
        device = 'cuda:0'
    else:
        device = 'cpu'

    print("using device {}".format(device) )

    print("changing batch size for normal model!")
    opt.batch_size = 2
    

    train_dataset = NormalDataset(opt, evaluation_mode=False)
    
    test_dataset = NormalDataset(opt, phase = 'test', evaluation_mode=True)
    test_dataset.is_train = False
    test_data_loader = DataLoader(test_dataset, 
                       batch_size=opt.batch_size, shuffle=False,
                       num_workers=opt.num_threads, pin_memory=opt.pin_memory)

    print('test loader size: ', len(test_data_loader))




    train_dataset.is_train = False
    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=opt.batch_size, shuffle=False,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)


    print('train loader size: ', len(train_data_loader))

    
    
    netF = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")

    netB = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")




    # load model
    F_modelnormal_path = "/mnt/lustre/kennard.chan/SPIFu/apps/checkpoints/Date_09_Jul_22_Time_13_56_45/netF_model_state_dict_epoch34.pickle"
    B_modelnormal_path = "/mnt/lustre/kennard.chan/SPIFu/apps/checkpoints/Date_09_Jul_22_Time_13_56_45/netB_model_state_dict_epoch34.pickle"

    print('Resuming from ', F_modelnormal_path)
    print('Resuming from ', B_modelnormal_path)

    if device == 'cpu' :
        import io

        class CPU_Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else:
                    return super().find_class(module, name)

        with open(F_modelnormal_path, 'rb') as handle:
           netF_state_dict = CPU_Unpickler(handle).load()

        with open(B_modelnormal_path, 'rb') as handle:
           netB_state_dict = CPU_Unpickler(handle).load()

    else:

        with open(F_modelnormal_path, 'rb') as handle:
           netF_state_dict = pickle.load(handle)

        with open(B_modelnormal_path, 'rb') as handle:
           netB_state_dict = pickle.load(handle)

    netF.load_state_dict( netF_state_dict , strict = True )
    netB.load_state_dict( netB_state_dict , strict = True )
        
        


    netF = netF.to(device=device)
    netB = netB.to(device=device)

    netF.eval()
    netB.eval()




    data_loader_list = [ ('test data',test_data_loader), ('train data',train_data_loader) ]


    with torch.no_grad():

        for dataset_type, data_loader in data_loader_list:
            print('Working on {0}'.format(dataset_type) )
            for train_idx, train_data in enumerate(data_loader):
                print("batch {}".format(train_idx) )

                # retrieve the data
                subject_list = train_data['name']
                render_filename_list = train_data['render_path']  

                render_tensor = train_data['original_high_res_render'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
                
                mask_high_res_array = train_data['mask_high_res'].cpu().numpy() # Shape of [Batch_size, 1, Height, Width]

                res_netF = netF.forward(render_tensor)
                res_netB = netB.forward(render_tensor)

                res_netF = res_netF.detach().cpu().numpy()
                res_netB = res_netB.detach().cpu().numpy()



                for i in range(opt.batch_size):
                    try:
                        subject = subject_list[i]
                        render_filename = render_filename_list[i]
                    except Exception as e:
                        print(e)
                        print("Last batch of the dataloader")
                        break


                    if not os.path.exists( os.path.join(trained_normal_maps_path,subject) ):
                        os.makedirs(os.path.join(trained_normal_maps_path,subject) )

                    yaw = render_filename.split('_')[-1].split('.')[0]
                    save_normalmapF_path = os.path.join(trained_normal_maps_path, subject, "rendered_nmlF_" + yaw + ".npy"  )
                    save_normalmapB_path = os.path.join(trained_normal_maps_path, subject, "rendered_nmlB_" + yaw + ".npy"  )

                    save_netF_normalmap_path = os.path.join(trained_normal_maps_path, subject, "rendered_nmlF_" + yaw + ".png"  )
                    save_netB_normalmap_path = os.path.join(trained_normal_maps_path, subject, "rendered_nmlB_" + yaw + ".png"  )


                    mask_high_res = mask_high_res_array[i, ...]
                    mask_high_res = np.repeat(mask_high_res, repeats=3, axis=0 )

                    res_netF[i] = res_netF[i] * mask_high_res
                    generated_map = res_netF[i]


                    save_normalmapF_path = save_normalmapF_path.replace('.npy', '.npz')
                    save_normalmapB_path = save_normalmapB_path.replace('.npy', '.npz')

                    generated_map = np.reshape(generated_map, [3, -1]) # shape of [3, HIGH_RES_SIZE*HIGH_RES_SIZE ]
                    sparse_data = sparse.csr_matrix(generated_map)
                    sparse.save_npz(save_normalmapF_path, sparse_data)


                    res_netB[i] = res_netB[i] * mask_high_res
                    generated_map = res_netB[i]


                    generated_map = np.reshape(generated_map, [3, -1]) # shape of [3, HIGH_RES_SIZE*HIGH_RES_SIZE ]
                    sparse_data = sparse.csr_matrix(generated_map)
                    sparse.save_npz(save_normalmapB_path, sparse_data)


                    # save as images
                    save_netF_normalmap = (np.transpose(res_netF[i], (1, 2, 0)) * 0.5 + 0.5) * 255.0
                    save_netF_normalmap = save_netF_normalmap.astype(np.uint8)
                    save_netF_normalmap = Image.fromarray(save_netF_normalmap)
                    save_netF_normalmap.save(save_netF_normalmap_path)

                    save_netB_normalmap = (np.transpose(res_netB[i], (1, 2, 0)) * 0.5 + 0.5) * 255.0
                    save_netB_normalmap = save_netB_normalmap.astype(np.uint8)
                    save_netB_normalmap = Image.fromarray(save_netB_normalmap)
                    save_netB_normalmap.save(save_netB_normalmap_path)



            



if __name__ == '__main__':
    generate_maps(opt)

