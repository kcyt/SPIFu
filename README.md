# SPIFu
Official Implementation of SPIFu (NeurIPS 2022 )

## Prerequisites:
### 1) Request permission to use THuman2.0 dataset (https://github.com/ytrock/THuman2.0-Dataset). 
After permission granted, download the dataset (THuman2.0_Release). Put the "THuman2.0_Release" folder inside the "rendering_script" folder. 

### 2) Rendering Images and Normal maps
Please refer to https://github.com/kcyt/IntegratedPIFu. Note that Depth maps are not required here.

### 3) Install smplx (Only if not using groundtruth smplx fittings from THuman2.0 dataset)
Please follow the installation instructions from smplx from https://github.com/vchoutas/smplx#installation. 
Then, use smplify-x (https://github.com/vchoutas/smplify-x) on the rendered RGB images from Step 2. 

### 3) Run Ray-Based Sampling (RBS)
Run the script unroll_gt_smplx.py or unroll_predicted_smplx.py depending on whether you want to apply RBS on the groundtruth smplx fittings or the predicted smplx fittings from smplify-x.

### 4) Train the normal maps predicator, and then use it for generating predicted normal maps.
Run the script train_normalmodel.py. After the model is trained, run the script generatemaps_normalmodel.py to generated the predicted normal maps.


## To Run :
Run the script train_smpl_unrolled.py. Configuration can be set in lib/options.py file.



