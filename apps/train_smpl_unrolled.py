
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
import matplotlib.pyplot as plt

from lib.options import BaseOptions
from lib.model.HGPIFuNetwNML import HGPIFuNetwNML
from lib.data.THuman_dataset import THumanDataset
from lib.mesh_util import save_obj_mesh_with_color, reconstruction
from lib.geometry import index



seed = 10 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


parser = BaseOptions()
opt = parser.parse()
gen_test_counter = 0

# Start of options check

test_script_activate = False # Set to True to generate the test subject meshes


# Whether to load model weights
load_model_weights = False
checkpoint_folder_to_load_low_res = '/mnt/lustre/kennard.chan/SPIFu/apps/checkpoints/Date_23_Aug_22_Time_22_34_49'
epoch_to_load_from_low_res = 66 # use epoch 66 for SPIFu trained with gt smplx

start_epoch = 0 # epoch to start training at, usually is 0 or "epoch_to_load_from_low_res+1"

# End of options check




gen_test_counter = start_epoch 
generate_point_cloud = True









def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob >= 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )







def adjust_learning_rate(optimizer, epoch, lr, schedule, learning_rate_decay):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        lr *= learning_rate_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr





def gen_mesh(resolution, net, device, data, save_path, thresh=0.5, use_octree=True, generate_from_low_res = False):


    calib_tensor = data['calib'].to(device=device)
    calib_tensor = torch.unsqueeze(calib_tensor,0)
    
    b_min = data['b_min']
    b_max = data['b_max']

    # low-resolution image that is required by both models
    image_low_tensor = data['render_low_pifu'].to(device=device)  
    image_low_tensor = image_low_tensor.unsqueeze(0)

    if opt.use_front_normal:
        nmlF_low_tensor = data['nmlF'].to(device=device)
        nmlF_low_tensor = nmlF_low_tensor.unsqueeze(0)
    else:
        nmlF_low_tensor = None


    if opt.use_back_normal:
        nmlB_low_tensor = data['nmlB'].to(device=device)
        nmlB_low_tensor = nmlB_low_tensor.unsqueeze(0)
    else:
        nmlB_low_tensor = None

    if opt.use_unrolled_smpl:
        unrolled_smpl_features = data['unrolled_smpl_features'].to(device=device)
        unrolled_smpl_features = unrolled_smpl_features.unsqueeze(0)
    else:
        unrolled_smpl_features = None


    net.filter( image_low_tensor, nmlF=nmlF_low_tensor, nmlB = nmlB_low_tensor,  unrolled_smpl_features=unrolled_smpl_features ) # forward-pass 
    image_tensor = image_low_tensor

    

    try:
        save_img_path = save_path[:-4] + '.png'
        save_img_list = []
        for v in range(image_tensor.shape[0]):
            save_img = (np.transpose(image_tensor[v].detach().cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
            save_img_list.append(save_img)
        save_img = np.concatenate(save_img_list, axis=1)
        cv2.imwrite(save_img_path, save_img)

        verts, faces, _, _ = reconstruction(
            net, device, calib_tensor, resolution, thresh, use_octree=use_octree, num_samples=50000, b_min=b_min , b_max=b_max , generate_from_low_res = generate_from_low_res)

        verts_tensor = torch.from_numpy(verts.T).unsqueeze(0).to(device=device).float()

        xyz_tensor = net.projection(verts_tensor, calib_tensor) # verts_tensor should have a range of [-1,1]
        uv = xyz_tensor[:, :2, :]
        color = index(image_tensor, uv).detach().cpu().numpy()[0].T
        color = color * 0.5 + 0.5


        save_obj_mesh_with_color(save_path, verts, faces, color)


    except Exception as e:
        print(e)
        print("Cannot create marching cubes at this time.")






def train(opt):
    global gen_test_counter
    currently_epoch_to_update_low_res_pifu = True


    if torch.cuda.is_available():
        # set cuda
        device = 'cuda:0'

    else:
        device = 'cpu'

    print("using device {}".format(device) )





    
    if test_script_activate:
        train_dataset = THumanDataset(opt, projection='orthogonal', phase = 'train', evaluation_mode = True, frontal_only=True)
        
    else:
        train_dataset = THumanDataset(opt, projection='orthogonal', phase = 'train')
        train_dataset_frontal_only = THumanDataset(opt, projection='orthogonal', phase = 'train', frontal_only=True)
    
    projection_mode = train_dataset.projection_mode

    if len(train_dataset) < opt.batch_size:
        batch_size = len(train_dataset)
        print("Change batch_size from {0} to {1}".format(opt.batch_size, batch_size) )
    else:
        batch_size = opt.batch_size
        print("Using batch_size == {0}".format(batch_size) )

    train_data_loader = DataLoader(train_dataset, 
                                   batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                                   num_workers=opt.num_threads, pin_memory=opt.pin_memory)


    print('train loader size: ', len(train_data_loader))

    if opt.useValidationSet:
        validation_dataset_frontal_only = THumanDataset(opt, projection='orthogonal', phase = 'validation', evaluation_mode=False, validation_mode=True, frontal_only=True)

        validation_epoch_cd_dist_list = []
        validation_epoch_p2s_dist_list = []

        validation_graph_path = os.path.join(opt.results_path, opt.name, 'ValidationError_Graph.png')
        validation_results_path = os.path.join(opt.results_path, opt.name, 'validation_results.txt')

    
    
    netG = HGPIFuNetwNML(opt, projection_mode)




    if (not os.path.exists(opt.checkpoints_path) ):
        os.makedirs(opt.checkpoints_path)
    if (not os.path.exists(opt.results_path) ):
        os.makedirs(opt.results_path)
    if (not os.path.exists('%s/%s' % (opt.checkpoints_path, opt.name))  ):
        os.makedirs('%s/%s' % (opt.checkpoints_path, opt.name))
    if (not os.path.exists('%s/%s' % (opt.results_path, opt.name)) ):
        os.makedirs('%s/%s' % (opt.results_path, opt.name))



    if load_model_weights:

        # load weights for low-res model
        modelG_path = os.path.join( checkpoint_folder_to_load_low_res ,"netG_model_state_dict_epoch{0}.pickle".format(epoch_to_load_from_low_res) )

        print('Resuming from ', modelG_path)

        if device == 'cpu' :
            import io

            class CPU_Unpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == 'torch.storage' and name == '_load_from_bytes':
                        return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                    else:
                        return super().find_class(module, name)

            with open(modelG_path, 'rb') as handle:
               netG_state_dict = CPU_Unpickler(handle).load()


        else:
            with open(modelG_path, 'rb') as handle:
               netG_state_dict = pickle.load(handle)




        netG.load_state_dict( netG_state_dict , strict = False )
        
        
        
            



        
    if test_script_activate:
        # testing script
        with torch.no_grad():

            
            train_dataset.is_train = False
            netG = netG.to(device=device)
            netG.eval()


            print('generate mesh (test) ...')
            len_to_iterate = len(train_dataset) #105 #72
            for gen_idx in tqdm(range(len_to_iterate)):

                index_to_use = gen_idx #gen_test_counter % len(train_dataset)
                train_data = train_dataset.get_item(index=index_to_use) 
                save_path = '%s/%s/test_%s.obj' % (
                    opt.results_path, opt.name, train_data['name'])

                generate_from_low_res = True
                gen_mesh(resolution=opt.resolution, net=netG, device = device, data = train_data, save_path = save_path, generate_from_low_res = generate_from_low_res)


        print("Testing is Done! Exiting...")
        return
        
         
    



    opt_log = os.path.join(opt.results_path, opt.name, 'opt.txt')
    with open(opt_log, 'w') as outfile:
        outfile.write(json.dumps(vars(opt), indent=2))




    netG = netG.to(device=device)
    lr_G = opt.learning_rate_G
    optimizerG = torch.optim.RMSprop(netG.parameters(), lr=lr_G, momentum=0, weight_decay=0)


    

    if load_model_weights:
        # load saved weights for optimizerG
        optimizerG_path = os.path.join(checkpoint_folder_to_load_low_res, "optimizerG_epoch{0}.pickle".format(epoch_to_load_from_low_res) )

        if device == 'cpu' :
            with open(optimizerG_path, 'rb') as handle:
                optimizerG_state_dict = CPU_Unpickler(handle).load()

        else:
            with open(optimizerG_path, 'rb') as handle:
                optimizerG_state_dict = pickle.load(handle)



        try:
            optimizerG.load_state_dict( optimizerG_state_dict )
        except Exception as e:
            print(e)
            print("Unable to load optimizerG saved weights!")
        
         

    for epoch in range(start_epoch, opt.num_epoch):


        print("start of epoch {}".format(epoch) )

        netG.train()

        epoch_error = 0
        train_len = len(train_data_loader)
        for train_idx, train_data in enumerate(train_data_loader):
            print("batch {}".format(train_idx) )


            # retrieve the data
            calib_tensor = train_data['calib'].to(device=device) # the calibration matrices for the renders ( is np.matmul(intrinsic, extrinsic)  ). Shape of [Batchsize, 4, 4]


            # low-resolution image that is required by both models
            render_pifu_tensor = train_data['render_low_pifu'].to(device=device)  # the renders. Shape of [Batch_size, Channels, Height, Width]
            
            if opt.use_front_normal:
                nmlF_tensor = train_data['nmlF'].to(device=device)
            else:
                nmlF_tensor = None

            if opt.use_back_normal:
                nmlB_tensor = train_data['nmlB'].to(device=device)
            else:
                nmlB_tensor = None


            if opt.use_unrolled_smpl:
                unrolled_smpl_features = train_data['unrolled_smpl_features'].to(device=device) 
            else:
                unrolled_smpl_features = None

 
            # low-resolution pifu
            samples_pifu_tensor = train_data['samples_low_res_pifu'].to(device=device)  # contain inside and outside points. Shape of [Batch_size, 3, num_of_points]
            labels_pifu_tensor = train_data['labels_low_res_pifu'].to(device=device)  # tell us which points in sample_tensor are inside and outside in the surface. Should have shape of [Batch_size ,1, num_of_points]


            error_low_res_pifu, res_low_res_pifu = netG.forward(images=render_pifu_tensor, points=samples_pifu_tensor, calibs=calib_tensor, labels=labels_pifu_tensor,  points_nml=None, labels_nml=None, nmlF = nmlF_tensor, nmlB = nmlB_tensor, unrolled_smpl_features=unrolled_smpl_features)
            optimizerG.zero_grad()
            error_low_res_pifu['Err(occ)'].backward()
            curr_low_res_loss = error_low_res_pifu['Err(occ)'].item()
            optimizerG.step()

            print(
            'Name: {0} | Epoch: {1} | error_low_res_pifu: {2:.06f} | LR: {3:.06f} '.format(
                opt.name, epoch, curr_low_res_loss, lr_G)
            )

            epoch_error += curr_low_res_loss


        lr_G = adjust_learning_rate(optimizerG, epoch, lr_G, opt.schedule, opt.learning_rate_decay)

        print("Overall Epoch {0} -  Error for network: {1}".format(epoch, epoch_error/train_len) )


        with torch.no_grad():
            if True:

                # save as pickle:
                with open( '%s/%s/netG_model_state_dict_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch) ) , 'wb') as handle:
                    pickle.dump(netG.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open( '%s/%s/optimizerG_epoch%s.pickle' % (opt.checkpoints_path, opt.name, str(epoch)) , 'wb') as handle:
                    pickle.dump(optimizerG.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)
   

                if generate_point_cloud:
                    r = res_low_res_pifu


                print('generate mesh (train) ...')
                train_dataset_frontal_only.is_train = False
                netG.eval()
                for gen_idx in tqdm(range(1)):

                    index_to_use = gen_test_counter % len(train_dataset_frontal_only)
                    gen_test_counter += 1 
                    train_data = train_dataset_frontal_only.get_item(index=index_to_use) 
                    # train_data["img"].shape  has shape of [1, 3, 512, 512]
                    save_path = '%s/%s/train_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, train_data['name'])


                    generate_from_low_res = True

                    gen_mesh(resolution=opt.resolution, net=netG , device = device, data = train_data, save_path = save_path, generate_from_low_res = generate_from_low_res)

                if generate_point_cloud:
                    try:
                        # save visualization of model performance
                        save_path = '%s/%s/pred.ply' % (opt.results_path, opt.name)
                        r = r[0].cpu() # get only the first example in the batch (i.e. 1 CAD model or subject). [1, Num of sampled points]
                        points = samples_pifu_tensor[0].transpose(0, 1).cpu()    # note that similar to res[0], we only take sample_tensor[0] i.e. the first CAD model. Shape of [Num of sampled points, 3] after the transpose. 
                        save_samples_truncted_prob(save_path, points.detach().numpy(), r.detach().numpy())
                    except:
                        print("Unable to save point cloud.")
                    
                train_dataset_frontal_only.is_train = True



            if opt.useValidationSet:
                import trimesh
                from clean_and_evaluate_model import quick_get_chamfer_and_surface_dist

                num_samples_to_use = 4000

                print('Commencing validation..')
                print('generate mesh (validation) ...')

                netG.eval()
                val_len = len(validation_dataset_frontal_only)
                val_mesh_paths = []
                index_to_use_list = []
                num_of_val_examples_to_try = val_len
                for gen_idx in tqdm(range(num_of_val_examples_to_try)):
                    print('[Validation] generating mesh #{0}'.format(gen_idx) )

                    index_to_use = gen_idx

                    val_data = validation_dataset_frontal_only.get_item(index=index_to_use) 

                    save_path = '%s/%s/val_eval_epoch%d_%s.obj' % (
                        opt.results_path, opt.name, epoch, val_data['name'])

                    val_mesh_paths.append(save_path)

                    generate_from_low_res = True
                    gen_mesh(resolution=opt.resolution, net=netG , device = device, data = val_data, save_path = save_path, generate_from_low_res = generate_from_low_res)

                total_chamfer_distance = []
                total_point_to_surface_distance = []
                for val_path in val_mesh_paths:
                    subject = val_path.split('_')[-1]
                    subject = subject.replace('.obj','')
                    GT_mesh = validation_dataset_frontal_only.mesh_dic[subject]
                    

                    try: 
                        print('Computing CD and P2S for {0}'.format( os.path.basename(val_path) ) )
                        source_mesh = trimesh.load(val_path)
                        chamfer_distance, point_to_surface_distance = quick_get_chamfer_and_surface_dist(src_mesh=source_mesh, tgt_mesh=GT_mesh, num_samples=num_samples_to_use )
                        total_chamfer_distance.append(chamfer_distance)
                        total_point_to_surface_distance.append(point_to_surface_distance)
                    except:
                        print('Unable to compute chamfer_distance and/or point_to_surface_distance!')
                
                if len(total_chamfer_distance) == 0:
                    average_chamfer_distance = 0
                else:
                    average_chamfer_distance = np.mean(total_chamfer_distance) 

                if len(total_point_to_surface_distance) == 0:
                    average_point_to_surface_distance = 0 
                else:
                    average_point_to_surface_distance = np.mean(total_point_to_surface_distance) 

                validation_epoch_cd_dist_list.append(average_chamfer_distance)
                validation_epoch_p2s_dist_list.append(average_point_to_surface_distance)

                
                print("[Validation] Overall Epoch {0}- Avg CD: {1}; Avg P2S: {2}".format(epoch, average_chamfer_distance, average_point_to_surface_distance ) )


                # Delete files that are created for validation
                for file_path in val_mesh_paths:
                    mesh_path = file_path
                    image_path = file_path.replace('.obj', '.png')
                    os.remove(mesh_path)
                    os.remove(image_path)


                plt.plot( np.arange( len(validation_epoch_cd_dist_list) ) , np.array(validation_epoch_cd_dist_list) )
                plt.plot( np.arange( len(validation_epoch_p2s_dist_list) ) , np.array(validation_epoch_p2s_dist_list), '-.' )
                plt.xlabel('Epoch')
                plt.ylabel('Validation Error (CD + P2D)')
                plt.title('Epoch Against Validation Error (CD + P2D)')
                plt.savefig(validation_graph_path)

                try:
                    sum_of_cd_and_p2s = np.array(validation_epoch_cd_dist_list) + np.array(validation_epoch_p2s_dist_list)
                    sum_of_cd_and_p2s = sum_of_cd_and_p2s.tolist()
                    epoch_list = np.arange( len(validation_epoch_cd_dist_list) ).tolist()
                    z = zip(epoch_list, validation_epoch_cd_dist_list, validation_epoch_p2s_dist_list, sum_of_cd_and_p2s)
                    z = list(z)
                    z.sort( key= lambda x:x[3] )
                    with open(validation_results_path, 'w') as f:
                        for item in z:
                            f.write( 'Epoch:{0},  CD:{1},  P2S:{2},  Sum:{3} \n'.format( item[0], item[1], item[2], item[3] )  )


                except:
                    print("Unable to save table of validation errors")












if __name__ == '__main__':
    train(opt)

