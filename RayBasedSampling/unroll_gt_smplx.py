
import os 
import joblib
import numpy as np 
import trimesh 
from sklearn.neighbors import NearestNeighbors
import smplx 
from smplx import SMPLX
import torch
import math
from scipy import sparse 
#import cv2


use_barycentric_coords = False
use_default_beta= True




SMPLX_MODEL_DIR = '/mnt/lustre/kennard.chan/smplify-x-master/smplifyx/models/smplx'
Given_GT_THuman2_pkl_file_directory = '/mnt/lustre/kennard.chan/THUman.20 Release Smpl-X Paras'


smplx_model = SMPLX( SMPLX_MODEL_DIR)
smplx_lbs_weights = smplx_model.lbs_weights # torch.Size([10475, 55])

device = torch.device('cpu')
print("Using cpu!")


OUTPUT_RESOLUTION = 512  
NUM_OF_FEATURES = 7
if use_barycentric_coords:
	NUM_OF_FEATURES = 7 + 3 # plus 3 becaause of the barycentric coordinates

# Note, SMPLX has 10475 vertices rather than 6890


image_folder = "/mnt/lustre/kennard.chan/render_THuman_with_blender/buffer_fixed_full_mesh_version_2"

results_folder = 'RBS_GT_Results' 

os.makedirs(results_folder, exist_ok=True)






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





for subject in os.listdir(image_folder):
	if subject == ".DS_Store" or subject == "val.txt":
		continue

	subject_path = os.path.join(image_folder, subject )
	print("Processing subject {0}...".format(subject) )


	package_file_path = os.path.join(Given_GT_THuman2_pkl_file_directory, subject, 'smplx_param.pkl' )
	package_output = joblib.load(package_file_path)


	model_params = dict(model_path='../smplify-x-master/smplifyx/models',
	                    #joint_mapper=joint_mapper,
	                    create_global_orient=True,
	                    create_body_pose=True,
	                    create_betas=True,
	                    create_left_hand_pose=True,
	                    create_right_hand_pose=True,
	                    create_expression=True,
	                    create_jaw_pose=True,
	                    create_leye_pose=True,
	                    create_reye_pose=True,
	                    create_transl=False,
	                    dtype=torch.float32,
	                    use_face_contour = False,
	                    use_pca= True,
	                    flat_hand_mean= False,
	                    num_pca_comps= 12)

	body_model = smplx.create(model_type='smplx', gender='neutral', **model_params)
	body_model = body_model.to(device=device)


	if not use_default_beta:
	    betas = package_output['betas']
	else:
	    betas = np.zeros([1,10])

	body_pose = package_output['body_pose']
	global_orient = package_output['global_orient']
	left_hand_pose = package_output['left_hand_pose']
	right_hand_pose = package_output['right_hand_pose']
	jaw_pose = package_output['jaw_pose']
	leye_pose = package_output['leye_pose']
	reye_pose = package_output['reye_pose']
	expression = package_output['expression']

	betas = torch.tensor(betas).to(device=device).to(torch.float32)
	body_pose = torch.tensor(body_pose).to(device=device)
	global_orient = torch.tensor(global_orient).to(device=device).to(torch.float32)
	left_hand_pose = torch.tensor(left_hand_pose).to(device=device)
	right_hand_pose = torch.tensor(right_hand_pose).to(device=device)
	jaw_pose = torch.tensor(jaw_pose).to(device=device)
	leye_pose = torch.tensor(leye_pose).to(device=device)
	reye_pose = torch.tensor(reye_pose).to(device=device)
	expression = torch.tensor(expression).to(device=device).to(torch.float32)


	model_output = body_model(return_verts=True, body_pose=body_pose, betas=betas, global_orient=global_orient,
	                                left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, 
	                                jaw_pose=jaw_pose, leye_pose=leye_pose, reye_pose=reye_pose, expression=expression)

	gt_smplx_vertices = model_output.vertices[0] * package_output['scale'] + package_output['translation']

	gt_smplx_vertices = gt_smplx_vertices.detach().cpu().numpy().squeeze()
	





	for image in os.listdir(subject_path):
		if 'rendered_image_' not in image:
			continue

		angle = image.split('_')[2]
		angle = int(angle.split('.')[0])

		image_path = os.path.join(subject_path, image)
		param_path = os.path.join(subject_path, 'rendered_params_{0:03d}.npy'.format(angle) )



		gt_smplx_mesh = trimesh.Trimesh( np.copy(gt_smplx_vertices), body_model.faces, process=False)
		
		vertices = np.copy(gt_smplx_mesh.vertices) # shape of [10475,3]
		vertices =  np.concatenate([vertices, np.ones([vertices.shape[0],1]) ], axis=1 ).T   # shape of [4,10475]





		# convert vertices in gt space to the camera space

		# get params
		param = np.load(param_path, allow_pickle=True)  # param is a np.array that looks similar to a dict.  # ortho_ratio = 0.4 , e.g. scale or y_scale = 0.961994278, e.g. center or vmed = [-1.0486  92.56105  1.0101 ]
		center = param.item().get('center') # is camera 3D center position in the 3D World point space (without any rotation being applied).
		R = param.item().get('R')   # R is used to rotate the CAD model according to a given pitch and yaw.
		scale_factor = param.item().get('scale_factor') # is camera 3D center position in the 3D World point space (without any rotation being applied).
		load_size_associated_with_scale_factor = 1024

		translate = -center.reshape(3, 1)
		extrinsic = np.concatenate([R, translate], axis=1)  # when applied on the 3D pts, the rotation is done first, then the translation
		extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)

		scale_intrinsic = np.identity(4)
		scale_intrinsic[0, 0] = 1.0 * scale_factor #2.4851518#1.0   
		scale_intrinsic[1, 1] = -1.0 * scale_factor #-2.4851518#-1.0
		scale_intrinsic[2, 2] = 1.0 * scale_factor  #2.4851518#1.0


		# Match image pixel space to image uv space  (convert a 512x512 image from range of [-256,255] to range of [-1,1] )
		uv_intrinsic = np.identity(4)
		uv_intrinsic[0, 0] = 1.0 / float(load_size_associated_with_scale_factor // 2) # self.opt.loadSizeGlobal == 512 by default. This value must be 512 unless you change the "scale_factor"
		uv_intrinsic[1, 1] = 1.0 / float(load_size_associated_with_scale_factor // 2) # uv_intrinsic[1, 1] is equal to 1/256
		uv_intrinsic[2, 2] = 1.0 / float(load_size_associated_with_scale_factor // 2) 

		intrinsic = np.matmul(uv_intrinsic, scale_intrinsic)
		calib = np.matmul(intrinsic, extrinsic) 

		projected_vertices = np.matmul(calib, vertices) # [4,10475]
		projected_vertices = projected_vertices.T 
		projected_vertices = projected_vertices[:,0:3] # [10475, 3]



		# Can directly use the z-value of projected_vertices as depth. So that is already done. z-value range should be within -1 and 1.
		smpl_depth_values = np.copy(projected_vertices[:,2:3]) # [10475, 1]



		# Now, compute vertex normals:
		gt_smplx_mesh.apply_transform(calib)
		smpl_vertex_normals = gt_smplx_mesh.vertex_normals # [10475, 3]




		from trimesh.ray.ray_pyembree import RayMeshIntersector
		rmi = RayMeshIntersector(gt_smplx_mesh, scale_to_box=False)


		XY, step_size = np.linspace(-1,1, OUTPUT_RESOLUTION+1, endpoint=True, retstep=True) # XY has shape of (OUTPUT_RESOLUTION+1,) 
		XY = XY + 0.5*step_size
		XY = XY[:-1] # remove last element. Shape of (OUTPUT_RESOLUTION,) 

		XY = np.meshgrid(XY,XY) # # A list of 2 coordinate matrix
		XY = np.array(XY).T # Transpose will transform your (2,num_y,num_x) array into the desired (num_x,num_y,2) array. Shape of [OUTPUT_RESOLUTION,OUTPUT_RESOLUTION,2]


		query_list = np.reshape(XY, [-1, 2])
		query_list = np.concatenate( [query_list, np.ones( [query_list.shape[0], 1] ) * 2.0 ], axis=1 ) # [num_of_query_pts, 3] where num_of_query_pts == OUTPUT_RESOLUTION*OUTPUT_RESOLUTION
		

		ray_dirs = np.zeros_like(query_list) # [num_of_query_pts, 3]
		ray_dirs[:,2] = -1.0


		if use_barycentric_coords:
			# below, hit_face_indices is of shape (num_of_faces_that_are_hit, )
			# hit_ray_indices is also of shape (num_of_faces_that_are_hit, ), and it tells you which hit face belongs to which ray
			# intersection_pts is of shape (num_of_faces_that_are_hit,) 
			hit_face_indices, hit_ray_indices, intersection_pts = rmi.intersects_id(ray_origins=query_list, ray_directions=ray_dirs, multiple_hits=True, max_hits=6, return_locations=True )
		else:
			# below, hit_face_indices is of shape (num_of_faces_that_are_hit, )
			# hit_ray_indices is also of shape (num_of_faces_that_are_hit, ), and it tells you which hit face belongs to which ray
			hit_face_indices, hit_ray_indices = rmi.intersects_id(ray_origins=query_list, ray_directions=ray_dirs, multiple_hits=True, max_hits=6, return_locations=False )
			

		hit_vertices_indices = gt_smplx_mesh.faces[hit_face_indices, :] # shape of [num_of_faces_that_are_hit, 3]


		if not use_barycentric_coords: 
			face_depth_value_average = smpl_depth_values[  hit_vertices_indices[:, 0] ,0] + smpl_depth_values[  hit_vertices_indices[:, 1] ,0] + smpl_depth_values[  hit_vertices_indices[:, 2] ,0] # Shape of (num_of_faces_that_are_hit,)
			face_depth_value_average = face_depth_value_average / 3 # since there are 3 vertices in a face (always true in a smpl)

			face_normals_average = smpl_vertex_normals[ hit_vertices_indices[:, 0], :] + smpl_vertex_normals[ hit_vertices_indices[:, 1], :] + smpl_vertex_normals[ hit_vertices_indices[:, 2], :]
			face_normals_average = face_normals_average / 3   # Shape of [num_of_faces_that_are_hit,3]
		
		else:

			# compute the barycentric coordinates of each sample
			bary = trimesh.triangles.points_to_barycentric(
				triangles=gt_smplx_mesh.triangles[ hit_face_indices ], points=intersection_pts ) # bary has Shape of (num_of_faces_that_are_hit, 3)
			
			# interpolate vertex depth from barycentric coordinates
			# first row below has shape of [num_of_faces_that_are_hit, 3== num of face vertices, 1== depth value ]
			face_depth_value_average = (smpl_depth_values[gt_smplx_mesh.faces[hit_face_indices]] *
										  bary.reshape(
										  (-1, 3, 1))).sum(axis=1)   # face_depth_value_average has Shape of [num_of_faces_that_are_hit, 1]

			face_depth_value_average = face_depth_value_average[:,0] # to get shape of (num_of_faces_that_are_hit,)

			# interpolate vertex normals from barycentric coordinates
			# first row below has shape of [num_of_faces, 3== num of face vertices, 3== num of elements in a normal vector ]
			face_normals_average = trimesh.unitize((smpl_vertex_normals[gt_smplx_mesh.faces[hit_face_indices]] *
									  trimesh.unitize(bary).reshape(
										  (-1, 3, 1))).sum(axis=1))   # Shape of [num_of_faces_that_are_hit, 3]
			



		import pandas as pd

		if use_barycentric_coords:
			df = pd.DataFrame({ 'RayIndex': hit_ray_indices, 'Point1': hit_vertices_indices[:, 0], 'Point2': hit_vertices_indices[:, 1],  'Point3': hit_vertices_indices[:, 2], 
								'face_depth_value_average': face_depth_value_average, 'face_normals_average':face_normals_average.tolist(), 'barycentric_coords':bary.tolist()
										} )
		else:
			df = pd.DataFrame({ 'RayIndex': hit_ray_indices, 'Point1': hit_vertices_indices[:, 0], 'Point2': hit_vertices_indices[:, 1],  'Point3': hit_vertices_indices[:, 2], 
								'face_depth_value_average': face_depth_value_average, 'face_normals_average':face_normals_average.tolist()
										} )

		df = df.sort_values(["RayIndex", "face_depth_value_average"], ascending = [True, False])

		df["group_element_index"] = df.groupby(['RayIndex']).cumcount()

		df.reset_index(drop=True, inplace=True)
		pd.set_option('display.max_columns', None) # Just so printing out the df can see all the columns


		Point1_np =  df['Point1'].to_numpy(dtype=np.float32)  # Shape of (num_of_faces_that_are_hit,)
		Point2_np =  df['Point2'].to_numpy(dtype=np.float32)  # Shape of (num_of_faces_that_are_hit,)
		Point3_np =  df['Point3'].to_numpy(dtype=np.float32)  # Shape of (num_of_faces_that_are_hit,)
		face_depth_value_average_np =  df['face_depth_value_average'].to_numpy(dtype=np.float32)  # Shape of (num_of_faces_that_are_hit,)
		face_normals_average_np =  np.array( df['face_normals_average'].values.tolist() )   # Shape of [num_of_faces_that_are_hit, 3]


		features = np.stack( [Point1_np, Point2_np, Point3_np, face_depth_value_average_np  ], axis=-1 ) # Shape of [num_of_faces_that_are_hit, 4]
		features = np.concatenate( [features, face_normals_average_np], axis=1 )  # Shape of [num_of_faces_that_are_hit, 7]

		if use_barycentric_coords:
			face_barycentric_coords_np =  np.array( df['barycentric_coords'].values.tolist() )   # Shape of [num_of_faces_that_are_hit, 3]
			features = np.concatenate( [features, face_barycentric_coords_np], axis=1 )  # Shape of [num_of_faces_that_are_hit, 10]

			


		vertices_map = np.zeros([query_list.shape[0], 6, NUM_OF_FEATURES]) # Shape of [OUTPUT_RESOLUTION*OUTPUT_RESOLUTION, 6, 7] # where 6 refers to the max number of faces that are hit, and 7 refers to the 3 vertices in a face + average face depth value + average face normal value
		vertices_map[ df['RayIndex'].to_numpy(dtype=np.int32) , df["group_element_index"].to_numpy(dtype=np.int32), : ] = features # Shape of [OUTPUT_RESOLUTION*OUTPUT_RESOLUTION, 6, 7]

		vertices_map = np.reshape(vertices_map, [OUTPUT_RESOLUTION,OUTPUT_RESOLUTION, 6, NUM_OF_FEATURES] ) # Shape of [OUTPUT_RESOLUTION, OUTPUT_RESOLUTION, 6, 7]

		# Have to make sure now that the vertices_map is in (H,W) i.e. y coordinates in the first dimension, then the x coordinates. And also, smallest y starts from the top of the image (but this is already done due to the 'calib').
		vertices_map = np.transpose(vertices_map, (1, 0, 2, 3)) # Shape of [OUTPUT_RESOLUTION,OUTPUT_RESOLUTION, 6, 7] still

		num_of_features = vertices_map.shape[-1]

		vertices_map = vertices_map.astype(np.float32)
		vertices_map = np.reshape(vertices_map, [-1, num_of_features])


		vertices_map_sparse = sparse.csr_matrix(vertices_map)

		save_folder = os.path.join(results_folder, subject)
		os.makedirs(save_folder, exist_ok=True)

		save_file_path = os.path.join(results_folder, subject, 'unrolled_smpl_features_sparse_subject_{0}_angle_{1:03d}.npz'.format(subject, angle) )
		sparse.save_npz(save_file_path, vertices_map_sparse)





	






