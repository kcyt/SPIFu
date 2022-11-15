
import argparse
import os
import time 

class BaseOptions():
    def __init__(self):
        self.initialized = False
        self.parser = None

    def initialize(self, parser):



        parser.add_argument('--useValidationSet', default=False)

        parser.add_argument('--num_epoch', default=35) 

        parser.add_argument('--use_unrolled_smpl', default=True) # Must first be true before any of the below sub-options can be true
        parser.add_argument('--output_resolution_smpl_map', default=512)
        parser.add_argument('--use_coordinates_smpl', default=True)  
        parser.add_argument('--use_normals_smpl', default=True) # Must first be true before 'use_normals_smpl_with_rest_pose_normals' can be true
        parser.add_argument('--use_normals_smpl_with_rest_pose_normals', default=True)
        parser.add_argument('--use_blendweights_smpl', default=True)  
        
        parser.add_argument('--use_gt_smplx', default=True)  # use the gt smplx provided by the owner of THuman2.0 dataset.


        parser.add_argument('--num_threads', default=2, type=int, help='# sthreads for loading data')

        parser.add_argument('--use_front_normal', default=True)
        parser.add_argument('--use_back_normal', default=True)

        parser.add_argument('--num_sample_inout', type=int, default=8000, help='# of sampling points')





        # Less commonly used options:
        

        parser.add_argument('--sigma_low_resolution_pifu', type=float, default=3.5, help='sigma for sampling')
        parser.add_argument('--sigma_high_resolution_pifu', type=float, default=2.0, help='sigma for sampling') 

        parser.add_argument('--learning_rate_G', type=float, default=1e-3, help='adam learning rate for low res')
        parser.add_argument('--learning_rate_MR', type=float, default=1e-3, help='adam learning rate for high res')


        parser.add_argument('--loadSize', type=int, default=1024, help='load size of input image')
        parser.add_argument('--loadSizeGlobal', type=int, default=512, help='load size of input image')

        parser.add_argument('--activate_multi_view', default=True)

        # Experiment related
        timestamp = time.strftime('Date_%d_%b_%y_Time_%H_%M_%S')
        parser.add_argument('--name', type=str, default=timestamp,
                           help='name of the experiment. It decides where to store samples and models')

        # Training related
        parser.add_argument('--tmp_id', type=int, default=0, help='tmp_id')
        parser.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda')
        
        parser.add_argument('--batch_size', type=int, default=2, help='input batch size')

        parser.add_argument('--resolution', type=int, default=256, help='# of grid in mesh reconstruction')
        
        parser.add_argument('--serial_batches', action='store_true',
                             help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--pin_memory', action='store_true', help='pin_memory')


        parser.add_argument('--learning_rate_decay', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

        parser.add_argument('--z_size', type=float, default=200.0, help='z normalization factor')


        parser.add_argument('--num_gen_mesh_test', type=int, default=1,
                            help='how many meshes to generate during testing')

        # path
        parser.add_argument('--checkpoints_path', type=str, default='./checkpoints', help='path to save checkpoints')
        parser.add_argument('--results_path', type=str, default='./results', help='path to save results ply')
        
        parser.add_argument('--schedule', default=[99999],
                    help='Decrease learning rate at these epochs.')


        # aug
        group_aug = parser.add_argument_group('aug')
        group_aug.add_argument('--use_augmentation', default=True)
        group_aug.add_argument('--aug_bri', type=float, default=0.2, help='augmentation brightness')
        group_aug.add_argument('--aug_con', type=float, default=0.2, help='augmentation contrast')
        group_aug.add_argument('--aug_sat', type=float, default=0.05, help='augmentation saturation')
        group_aug.add_argument('--aug_hue', type=float, default=0.05, help='augmentation hue')



        parser.add_argument('--num_stack_low_res', type=int, default=4, help='# of hourglass')
        parser.add_argument('--num_stack_high_res', type=int, default=1, help='# of hourglass')

        parser.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        parser.add_argument('--hg_dim_low_res', type=int, default=256, help='256 | 512')
        parser.add_argument('--hg_dim_high_res', type=int, default=16)

        parser.add_argument('--mlp_dim_low_res', nargs='+', default=[257, 1024, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp. no need to put the first channel')

        parser.add_argument('--mlp_dim_normal_pred', nargs='+', default=[257, 1024, 512, 256, 128, 3], type=int,
                             help='# of dimensions of mlp. no need to put the first channel')

        parser.add_argument('--mlp_dim_multiple_angles_pred_low_res', nargs='+', default=[129, 1024, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp. no need to put the first channel')
        
        parser.add_argument('--mlp_dim_multiple_angles_pred_high_res', nargs='+', default=[513, 1024, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp. no need to put the first channel')

        parser.add_argument('--mlp_res_layers_low_res', nargs='+', default=[2,3,4], type=int,
                             help='leyers that has skip connection. use 0 for no residual pass')

        parser.add_argument('--mlp_res_layers_high_res', nargs='+', default=[1,2], type=int,
                             help='leyers that has skip connection. use 0 for no residual pass')

        parser.add_argument('--merge_layer_low_res', type=int, default=2)

        parser.add_argument('--mlp_dim_high_res', nargs='+', default=[272, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp.')

        parser.add_argument('--hg_depth_high_res', type=int, default=2, help='# of stacked layer of hourglass')
        parser.add_argument('--hg_depth_low_res', type=int, default=2, help='# of stacked layer of hourglass')
        parser.add_argument('--mlp_norm', type=str, default='none', help='normalization for volume branch')

        parser.add_argument('--norm', type=str, default='group',
                             help='instance normalization or batch normalization or group normalization')



        # special tasks
        self.initialized = True
        return parser

    def gather_options(self, args=None):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
            self.parser = parser

        if args is None:
            return self.parser.parse_args()
        else:
            return self.parser.parse_args(args)

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self, args=None):
        opt = self.gather_options(args)
             
        return opt
