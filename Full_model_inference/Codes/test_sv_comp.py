# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from spatial_network import build_SpatialNet, SpatialNet
from temporal_network import build_TemporalNet, TemporalNet
from smooth_network import build_SmoothNet, SmoothNet
import os
import numpy as np
import skimage
import cv2
import utils.torch_tps_transform as torch_tps_transform
import utils.torch_tps_transform_point as torch_tps_transform_point
from PIL import Image
import glob
import time
from torchvision.transforms import GaussianBlur

import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False

from sv_comp.dataset import MultiWarpDataset
from omegaconf import OmegaConf
import yaml

last_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目文件夹
MODEL_DIR = os.path.join(last_path, 'Full_model_inference', 'full_model_ssd')



def linear_blender(ref, tgt, ref_m, tgt_m, mask=False):
    blur = GaussianBlur(kernel_size=(21,21), sigma=20)
    r1, c1 = torch.nonzero(ref_m[0, 0], as_tuple=True)
    r2, c2 = torch.nonzero(tgt_m[0, 0], as_tuple=True)

    center1 = (r1.float().mean(), c1.float().mean())
    center2 = (r2.float().mean(), c2.float().mean())

    vec = (center2[0] - center1[0], center2[1] - center1[1])

    ovl = (ref_m * tgt_m).round()[:, 0].unsqueeze(1)
    ref_m_ = ref_m[:, 0].unsqueeze(1) - ovl
    r, c = torch.nonzero(ovl[0, 0], as_tuple=True)

    ovl_mask = torch.zeros_like(ref_m_).cuda()
    proj_val = (r - center1[0]) * vec[0] + (c - center1[1]) * vec[1]
    ovl_mask[ovl.bool()] = (proj_val - proj_val.min()) / (proj_val.max() - proj_val.min() + 1e-3)

    mask1 = (blur(ref_m_ + (1-ovl_mask)*ref_m[:,0].unsqueeze(1)) * ref_m + ref_m_).clamp(0,1)
    if mask: return mask1

    mask2 = (1-mask1) * tgt_m
    stit = ref * mask1 + tgt * mask2

    return stit


def recover_mesh(norm_mesh, height, width):
    #from [bs, pn, 2] to [bs, grid_h+1, grid_w+1, 2]

    batch_size = norm_mesh.size()[0]
    mesh_w = (norm_mesh[...,0]+1) * float(width) / 2.
    mesh_h = (norm_mesh[...,1]+1) * float(height) / 2.
    mesh = torch.stack([mesh_w, mesh_h], 2) # [bs,(grid_h+1)*(grid_w+1),2]

    return mesh.reshape([batch_size, grid_h+1, grid_w+1, 2])

def get_rigid_mesh(batch_size, height, width):


    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt

def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) # bs*(grid_h+1)*(grid_w+1)*2

    return norm_mesh.reshape([batch_size, -1, 2]) # bs*-1*2



# bs, T, h, w, 2  smooth_path
def get_stable_sqe(img1_list, img2_list, smooth_mesh1, smooth_mesh2, warp_mode, fusion_mode):
    batch_size, _, img_h, img_w = img2_list[0].shape
    print(img2_list[0].shape)

    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)

    smooth_mesh1 = torch.stack([smooth_mesh1[...,0]*img_w/480, smooth_mesh1[...,1]*img_h/360], 4)
    smooth_mesh2 = torch.stack([smooth_mesh2[...,0]*img_w/480, smooth_mesh2[...,1]*img_h/360], 4)

    width_max1 = torch.max(torch.max(smooth_mesh1[...,0]))
    width_max2 = torch.max(torch.max(smooth_mesh2[...,0]))
    width_max = torch.maximum(width_max1, width_max2)
    width_min1 = torch.min(smooth_mesh1[...,0])
    width_min2 = torch.min(smooth_mesh2[...,0])
    width_min = torch.minimum(width_min1, width_min2)
    height_max1 = torch.max(smooth_mesh1[...,1])
    height_max2 = torch.max(smooth_mesh2[...,1])
    height_max = torch.maximum(height_max1, height_max2)
    height_min1 = torch.min(smooth_mesh1[...,1])
    height_min2 = torch.min(smooth_mesh2[...,1])
    height_min = torch.minimum(height_min1, height_min2)

    out_width = width_max - width_min
    out_height = height_max - height_min

    print(out_width)
    print(out_height)

    stable_list = []

    for i in range(len(img2_list)):

        mesh1 = smooth_mesh1[:,i,:,:,:]
        mesh_trans1 = torch.stack([mesh1[...,0]-width_min, mesh1[...,1]-height_min], 3)
        norm_mesh1 = get_norm_mesh(mesh_trans1, out_height, out_width)
        img1 = img1_list[i].cuda()

        mesh2 = smooth_mesh2[:,i,:,:,:]
        mesh_trans2 = torch.stack([mesh2[...,0]-width_min, mesh2[...,1]-height_min], 3)
        norm_mesh2 = get_norm_mesh(mesh_trans2, out_height, out_width)
        img2 = img2_list[i].cuda()

        if fusion_mode == 'AVERAGE':
            img_warp = torch_tps_transform.transformer(torch.cat([img1, img2], 0), torch.cat([norm_mesh1, norm_mesh2], 0), torch.cat([norm_rigid_mesh, norm_rigid_mesh], 0), (out_height.int(), out_width.int()), mode = warp_mode)

            fusion = img_warp[0] * (img_warp[0]/ (img_warp[0]+img_warp[1]+1e-6)) + img_warp[1] * (img_warp[1]/ (img_warp[0]+img_warp[1]+1e-6))
        else:
            mask = torch.ones_like(img1[:,0,...].unsqueeze(1)).cuda()
            img1 = torch.cat([img1, mask], 1)
            img2 = torch.cat([img2, mask], 1)
            img_warp = torch_tps_transform.transformer(torch.cat([img1, img2], 0), torch.cat([norm_mesh1, norm_mesh2], 0), torch.cat([norm_rigid_mesh, norm_rigid_mesh], 0), (out_height.int(), out_width.int()), mode = warp_mode)

            fusion = linear_blender(img_warp[0,0:3,...].unsqueeze(0), img_warp[1,0:3,...].unsqueeze(0), img_warp[0,3,...].unsqueeze(0).unsqueeze(0), img_warp[1,3,...].unsqueeze(0).unsqueeze(0))
            fusion = fusion[0]

        stable_list.append(fusion.cpu().numpy().transpose(1,2,0))

    return stable_list, out_width.int(), out_height.int()



def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 加载数据集
    sv_comp_cfg = OmegaConf.load('sv_comp/sv_comp.yaml')
    with open('sv_comp/intrinsics.yaml', 'r', encoding='utf-8') as file:
        intrinsics = yaml.safe_load(file)
    dataset = MultiWarpDataset(config=sv_comp_cfg, intrinsics=intrinsics, is_train=False)
    video_length = 10  # 一张图像复制若干次作为一个视频

    # define the network
    spatial_net = SpatialNet()
    temporal_net = TemporalNet()
    smooth_net = SmoothNet()
    if torch.cuda.is_available():
        spatial_net = spatial_net.cuda()
        temporal_net = temporal_net.cuda()
        smooth_net = smooth_net.cuda()

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) == 3:
        # load spatial warp model
        spatial_model_path = MODEL_DIR + "/spatial_warp.pth"
        spatial_checkpoint = torch.load(spatial_model_path)
        spatial_net.load_state_dict(spatial_checkpoint['model'])
        print('load model from {}!'.format(spatial_model_path))
        # load temporal warp model
        temporal_model_path = MODEL_DIR + "/temporal_warp.pth"
        temporal_checkpoint = torch.load(temporal_model_path)
        temporal_net.load_state_dict(temporal_checkpoint['model'])
        print('load model from {}!'.format(temporal_model_path))
        # load smooth warp model
        smooth_model_path = MODEL_DIR + "/smooth_warp.pth"
        smooth_checkpoint = torch.load(smooth_model_path)
        smooth_net.load_state_dict(smooth_checkpoint['model'])
        print('load model from {}!'.format(smooth_model_path))
    else:
        print('No checkpoint found!')
        exit(0)

    spatial_net.eval()
    temporal_net.eval()
    smooth_net.eval()

    print("##################start testing#######################")

    count = 0

    for idx in range(len(dataset)):
        # if idx > 5:
        #     break
        sample = dataset[idx]
        input_imgs, input_masks = sample[0], sample[1]
        middle_stitch_results = None  # 中间拼接结果
        origin_w, origin_h = input_imgs[0].shape[1], input_imgs[0].shape[0]

        print(f'---------------{idx}---------------')

        for j in range(sv_comp_cfg['input_img_num'] - 1):
            # 创建保存结果文件夹
            batch_path = dataset.get_path(idx)
            path_result = os.path.join(batch_path, 'stabstitch2/')
            os.makedirs(path_result, exist_ok=True)
            path_warp = os.path.join(batch_path, 'warp/')
            os.makedirs(path_warp, exist_ok=True)
            path_mask = os.path.join(batch_path, 'mask/')
            os.makedirs(path_mask, exist_ok=True)

            # 生成待拼接的两路视频
            if j == 0:
                img1_list = [input_imgs[j]] * video_length
                img2_list = [input_imgs[j+1]] * video_length
            else:
                img1_list = middle_stitch_results
                img2_list = [input_imgs[j+1]] * video_length

            #define an online buffer (len == 7)
            buffer_len = 7
            tmotion_tensor_list1 = []
            smotion_tensor_list1 = []
            tmotion_tensor_list2 = []
            smotion_tensor_list2 = []

            img1_tensor_list = []
            img2_tensor_list = []
            img1_hr_tensor_list = []
            img2_hr_tensor_list = []

            img_h = 360
            img_w = 480
            # prepare imgs
            for k in range(0, len(img2_list)):

                img1 = img1_list[k]
                if img1.shape[0] != origin_h or img1.shape[1] != origin_w:
                    img1 = cv2.resize(img1, (origin_w, origin_h))
                # get high-resolution input
                img1_hr = img1.astype(dtype=np.float32)
                img1_hr = np.transpose(img1_hr, [2, 0, 1])
                img1_hr_tensor = torch.tensor(img1_hr).unsqueeze(0)
                img1_hr_tensor_list.append(img1_hr_tensor)
                # get 360x480 input
                img1 = cv2.resize(img1, (img_w, img_h))
                img1 = img1.astype(dtype=np.float32)
                img1 = np.transpose(img1, [2, 0, 1])
                img1 = (img1 / 127.5) - 1.0
                img1_tensor = torch.tensor(img1).unsqueeze(0)
                img1_tensor_list.append(img1_tensor)

                img2 = img2_list[k]
                if img2.shape[0] != origin_h or img2.shape[1] != origin_w:
                    img2 = cv2.resize(img2, (origin_w, origin_h))
                # get high-resolution input
                img2_hr = img2.astype(dtype=np.float32)
                img2_hr = np.transpose(img2_hr, [2, 0, 1])
                img2_hr_tensor = torch.tensor(img2_hr).unsqueeze(0)
                img2_hr_tensor_list.append(img2_hr_tensor)
                # get 360x480 input
                img2 = cv2.resize(img2, (img_w, img_h))
                img2 = img2.astype(dtype=np.float32)
                img2 = np.transpose(img2, [2, 0, 1])
                img2 = (img2 / 127.5) - 1.0
                img2_tensor = torch.tensor(img2).unsqueeze(0)
                img2_tensor_list.append(img2_tensor)


            # motion estimation
            for k in range(0, len(img2_list)):

                # step 1: spatial warp
                with torch.no_grad():
                    spatial_batch_out = build_SpatialNet(spatial_net, img1_tensor_list[k].cuda(), img2_tensor_list[k].cuda())
                smotion1 = spatial_batch_out['motion1']
                smotion2 = spatial_batch_out['motion2']
                smotion_tensor_list1.append(smotion1)
                smotion_tensor_list2.append(smotion2)

            # step 2: temporal warp
            with torch.no_grad():
                temporal_batch_out1 = build_TemporalNet(temporal_net, img1_tensor_list)
                temporal_batch_out2 = build_TemporalNet(temporal_net, img2_tensor_list)
            tmotion_tensor_list1 = temporal_batch_out1['motion_list']
            tmotion_tensor_list2 = temporal_batch_out2['motion_list']


            ##############################################
            #############   data preparation  ############
            # converting tmotion (t-th frame) into tsmotion ( (t-1)-th frame )
            rigid_mesh = get_rigid_mesh(1, img_h, img_w)
            norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
            smesh_list1 = []
            smesh_list2 = []
            tsmotion_list1 = []
            tsmotion_list2 = []
            for k in range(len(tmotion_tensor_list1)):
                smotion1 = smotion_tensor_list1[k]
                smesh1 = rigid_mesh + smotion1
                smotion2 = smotion_tensor_list2[k]
                smesh2 = rigid_mesh + smotion2
                if k == 0:
                    tsmotion1 = smotion1.clone() * 0
                    tsmotion2 = smotion2.clone() * 0
                else:
                    smotion1_1 = smotion_tensor_list1[k-1]
                    smesh1_1 = rigid_mesh + smotion1_1
                    tmotion1 = tmotion_tensor_list1[k]
                    tmesh1 = rigid_mesh + tmotion1
                    norm_smesh1_1 = get_norm_mesh(smesh1_1, img_h, img_w)
                    norm_tmesh1 = get_norm_mesh(tmesh1, img_h, img_w)
                    tsmesh1 = torch_tps_transform_point.transformer(norm_tmesh1, norm_rigid_mesh, norm_smesh1_1)
                    tsmotion1 = recover_mesh(tsmesh1, img_h, img_w) - smesh1

                    smotion2_1 = smotion_tensor_list2[k-1]
                    smesh2_1 = rigid_mesh + smotion2_1
                    tmotion2 = tmotion_tensor_list2[k]
                    tmesh2 = rigid_mesh + tmotion2
                    norm_smesh2_1 = get_norm_mesh(smesh2_1, img_h, img_w)
                    norm_tmesh2 = get_norm_mesh(tmesh2, img_h, img_w)
                    tsmesh2 = torch_tps_transform_point.transformer(norm_tmesh2, norm_rigid_mesh, norm_smesh2_1)
                    tsmotion2 = recover_mesh(tsmesh2, img_h, img_w) - smesh2


                # append
                smesh_list1.append(smesh1)
                smesh_list2.append(smesh2)
                tsmotion_list1.append(tsmotion1)
                tsmotion_list2.append(tsmotion2)


            # step 3: smooth warp
            ori_mesh1 = 0
            smooth_mesh1 = 0
            delta_motion1 = 0

            ori_mesh2 = 0
            smooth_mesh2 = 0
            delta_motion2 = 0

            for k in range(len(tmotion_tensor_list1)-6):

                tsmotion_sublist1 = tsmotion_list1[k:k+7]
                tsmotion_sublist1[0] = tsmotion_sublist1[0] * 0

                tsmotion_sublist2 = tsmotion_list2[k:k+7]
                tsmotion_sublist2[0] = tsmotion_sublist2[0] * 0


                with torch.no_grad():
                    smooth_batch_out = build_SmoothNet(smooth_net, tsmotion_sublist1, tsmotion_sublist2, smesh_list1[k:k+7], smesh_list2[k:k+7])

                _ori_mesh1 = smooth_batch_out["ori_mesh1"]
                _smooth_mesh1 = smooth_batch_out["smooth_mesh1"]

                _ori_mesh2 = smooth_batch_out["ori_mesh2"]
                _smooth_mesh2 = smooth_batch_out["smooth_mesh2"]


                if k == 0:
                    ori_mesh1 = _ori_mesh1
                    smooth_mesh1 = _smooth_mesh1

                    ori_mesh2 = _ori_mesh2
                    smooth_mesh2 = _smooth_mesh2

                else:
                    # for ref
                    ori_mesh1 = torch.cat((ori_mesh1, _ori_mesh1[:,-1,...].unsqueeze(1)), 1)
                    smooth_mesh1 = torch.cat((smooth_mesh1, _smooth_mesh1[:,-1,...].unsqueeze(1)), 1)

                    # for tgt
                    ori_mesh2 = torch.cat((ori_mesh2, _ori_mesh2[:,-1,...].unsqueeze(1)), 1)
                    smooth_mesh2 = torch.cat((smooth_mesh2, _smooth_mesh2[:,-1,...].unsqueeze(1)), 1)

            stable_list, out_width, out_height = get_stable_sqe(img1_hr_tensor_list, img2_hr_tensor_list, smooth_mesh1, smooth_mesh2, args.warp_mode, args.fusion_mode)

            print(out_width, out_height)

            # 处理中间拼接结果
            middle_stitch_results = stable_list

            # 保存结果
            cv2.imwrite(os.path.join(path_result, f'{sv_comp_cfg.input_img_num}_{j+2}.jpg'), stable_list[video_length//2].astype(np.uint8))






    print("##################end testing#######################")


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--test_path', type=str, default=os.path.join(last_path, 'Datasets/StabStitch-D/testing/'))
    parser.add_argument('--output_path', type=str, default='../results_ssd/')

    # optional parameter: 'NORMAL' or 'FAST'
    # FAST: use F.grid_sample to interpolate. It's fast, but may produce thin black boundary.
    # NORMAL: use our implemented interpolation function. It's a bit slower, but avoid the black boundary.
    parser.add_argument('--warp_mode', type=str, default='NORMAL') # optional parameter: 'Normal' or 'Fast'
    # optional parameter: 'AVERAGE' or 'LINEAR'
    # AVERAGE: faster but more artifacts
    # LINEAR: slower but less artifacts
    parser.add_argument('--fusion_mode', type=str, default='AVERAGE')

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)
