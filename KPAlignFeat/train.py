#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim 

import sys
from scene import Scene
import uuid
from tqdm import tqdm

from argparse import ArgumentParser, Namespace
from arguments import ModelParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torch.nn.functional as F
from PIL import Image

#/////////////////////////
import numpy as np
from torch.utils.tensorboard import SummaryWriter
#////////////////////////

from encoders.XFeat.modules.xfeat import XFeat
from utils.pose_utils import getGTXYZ
from scene.feat_pointcloud import FeatPointCloud

from utils.general_utils import image_process, sample_features

"""
python train.py -s ../../GSplatLoc/gsplatloc-main/datasets/wholehead/ -m ../../GSplatLoc/gsplatloc-main/output_wholescene/img_2000_head 

we need to already train a 3DGS with xfeat feature in 15000 iteration and put it into the "output_wholescene/img_2000_head"
Training image must be put in datasets/wholehead/

If we train with disk, we need to point out in the dataset_reader.py where to find the pre-extract disk feature

"""



def training(dataset):
    first_iter = 0

    scene = Scene(dataset, load_iteration=15000)
    featpc = FeatPointCloud()
    featpc.init_feat_pc(dataset.source_path, 64)
    xfeat = XFeat(top_k=4096)
    
    # 2D semantic feature map CNN decoder
    viewpoint_stack = scene.getTrainCameras().copy()
    num_datas = len(viewpoint_stack)
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


    iter_start = torch.cuda.Event(enable_timing = True)


    first_iter += 1
    
    for index in range(num_datas-1):

        iter_start.record()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        try:
            image = Image.open(viewpoint_cam.image_path) 
        except:
            print(f"Error opening image: {viewpoint_cam.image_path}")
            continue
        
        original_image = image_process(image)

        gt_image = original_image.cuda()
        gt_feature_map = xfeat.get_descriptors(gt_image[None])[0]
        print("gt_feature_map shape = ", gt_feature_map.shape)


        feature_map = F.interpolate(gt_feature_map.unsqueeze(0), size=(gt_image.shape[1], gt_image.shape[2]), mode='bilinear', align_corners=True).squeeze(0) #640x480
        print("feature map shape = ", feature_map.shape)
        keypoint_feat = sample_features(torch.tensor(viewpoint_cam.keypoints).to("cuda"), feature_map)
            
        featpc.update_ply(viewpoint_cam.point3d_id, keypoint_feat)
        
        print("index = ", index)

    print("\n[Data : {}] Saving".format(num_datas))

    #save feature point cloud 
    point_cloud_path = os.path.join(scene.model_path, "kpalign_point_cloud/iteration_{}".format(15000))
    featpc.save_ply(os.path.join(point_cloud_path, "kpalign_point_cloud.ply"))

  

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)

    args = parser.parse_args(sys.argv[1:])

    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)

    # Start GUI server, configure and run training
    ###### Fan WU #######
    args.eval = True
    #####################
   
    training(lp.extract(args))

    # All done
    print("\nTraining complete.")
