# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import pprint
import os

import clip
import torch
import torchvision
import numpy as np
import torch.nn as nn
import bitsandbytes as bnb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.spatial.transform import Rotation
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR

import sam2act.utils.peract_utils as peract_utils
import sam2act.mvt.utils as mvt_utils
import sam2act.utils.rvt_utils as rvt_utils
import peract_colab.arm.utils as arm_utils

from sam2act.mvt.augmentation import apply_se3_aug_con, apply_se3_aug_con_same, apply_se3_aug_con_sequence, apply_se3_aug_given_matrix, aug_utils
from peract_colab.arm.optim.lamb import Lamb
from yarr.agents.agent import ActResult
from sam2act.utils.dataset import _clip_encode_text
from sam2act.utils.lr_sched_utils import GradualWarmupScheduler


def eval_con(gt, pred):
    assert gt.shape == pred.shape, print(f"{gt.shape} {pred.shape}")
    assert len(gt.shape) == 2
    dist = torch.linalg.vector_norm(gt - pred, dim=1)
    return {"avg err": dist.mean()}


def eval_con_cls(gt, pred, num_bin=72, res=5, symmetry=1):
    """
    Evaluate continuous classification where floating point values are put into
    discrete bins
    :param gt: (bs,)
    :param pred: (bs,)
    :param num_bin: int for the number of rotation bins
    :param res: float to specify the resolution of each rotation bin
    :param symmetry: degrees of symmetry; 2 is 180 degree symmetry, 4 is 90
        degree symmetry
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) in [0, 1], gt
    assert num_bin % symmetry == 0, (num_bin, symmetry)
    gt = torch.tensor(gt)
    pred = torch.tensor(pred)
    num_bin //= symmetry
    pred %= num_bin
    gt %= num_bin
    dist = torch.abs(pred - gt)
    dist = torch.min(dist, num_bin - dist)
    dist_con = dist.float() * res
    return {"avg err": dist_con.mean()}


def eval_cls(gt, pred):
    """
    Evaluate classification performance
    :param gt_coll: (bs,)
    :param pred: (bs,)
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) == 1
    return {"per err": (gt != pred).float().mean()}


def eval_all(
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    eval_trans = []
    eval_rot_x = []
    eval_rot_y = []
    eval_rot_z = []
    eval_grip = []
    eval_coll = []

    for i in range(bs):
        eval_trans.append(
            eval_con(wpt[i : i + 1], pred_wpt[i : i + 1])["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        euler_gt = Rotation.from_quat(action_rot[i]).as_euler("xyz", degrees=True)
        euler_pred = Rotation.from_quat(pred_rot_quat[i]).as_euler("xyz", degrees=True)

        eval_rot_x.append(
            eval_con_cls(euler_gt[0], euler_pred[0], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_y.append(
            eval_con_cls(euler_gt[1], euler_pred[1], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_z.append(
            eval_con_cls(euler_gt[2], euler_pred[2], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_grip.append(
            eval_cls(
                action_grip_one_hot[i : i + 1].argmax(-1),
                grip_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_coll.append(
            eval_cls(
                action_collision_one_hot[i : i + 1].argmax(-1),
                collision_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
        )

    return eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll


def manage_eval_log(
    self,
    tasks,
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
    reset_log=False,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    if not hasattr(self, "eval_trans") or reset_log:
        self.eval_trans = {}
        self.eval_rot_x = {}
        self.eval_rot_y = {}
        self.eval_rot_z = {}
        self.eval_grip = {}
        self.eval_coll = {}

    (eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll,) = eval_all(
        wpt=wpt,
        pred_wpt=pred_wpt,
        action_rot=action_rot,
        pred_rot_quat=pred_rot_quat,
        action_grip_one_hot=action_grip_one_hot,
        grip_q=grip_q,
        action_collision_one_hot=action_collision_one_hot,
        collision_q=collision_q,
    )

    for idx, task in enumerate(tasks):
        if not (task in self.eval_trans):
            self.eval_trans[task] = []
            self.eval_rot_x[task] = []
            self.eval_rot_y[task] = []
            self.eval_rot_z[task] = []
            self.eval_grip[task] = []
            self.eval_coll[task] = []
        self.eval_trans[task].append(eval_trans[idx])
        self.eval_rot_x[task].append(eval_rot_x[idx])
        self.eval_rot_y[task].append(eval_rot_y[idx])
        self.eval_rot_z[task].append(eval_rot_z[idx])
        self.eval_grip[task].append(eval_grip[idx])
        self.eval_coll[task].append(eval_coll[idx])

    return {
        "eval_trans": eval_trans,
        "eval_rot_x": eval_rot_x,
        "eval_rot_y": eval_rot_y,
        "eval_rot_z": eval_rot_z,
    }


def print_eval_log(self):
    logs = {
        "trans": self.eval_trans,
        "rot_x": self.eval_rot_x,
        "rot_y": self.eval_rot_y,
        "rot_z": self.eval_rot_z,
        "grip": self.eval_grip,
        "coll": self.eval_coll,
    }

    out = {}
    for name, log in logs.items():
        for task, task_log in log.items():
            task_log_np = np.array(task_log)
            mean, std, median = (
                np.mean(task_log_np),
                np.std(task_log_np),
                np.median(task_log_np),
            )
            out[f"{task}/{name}_mean"] = mean
            out[f"{task}/{name}_std"] = std
            out[f"{task}/{name}_median"] = median

    pprint.pprint(out)

    return out


def manage_loss_log(
    agent,
    loss_log,
    reset_log,
):
    if not hasattr(agent, "loss_log") or reset_log:
        agent.loss_log = {}

    for key, val in loss_log.items():
        if key in agent.loss_log:
            agent.loss_log[key].append(val)
        else:
            agent.loss_log[key] = [val]


def print_loss_log(agent):
    out = {}
    for key, val in agent.loss_log.items():
        if val is not None:
            filtered_val = [v for v in val if v is not None]
            mean_value = np.mean(filtered_val) if filtered_val else None  # Avoid empty list error
        else:
            mean_value = None
        if mean_value is not None:
            out[key] = mean_value
    pprint.pprint(out)
    return out


def horizon_loss_cal(output, data_horizon):

    shape = output.shape
    bs, ah = shape[0], shape[1]
    mask = torch.arange(ah).expand(bs, ah).to(output.device) < data_horizon.view(bs, 1)
    if len(shape) == 3:
        mask = mask.unsqueeze(-1).expand_as(output)

        output_mask = output * mask
        ah_ave = output_mask.sum(dim=(0,2))/(mask.sum(dim=(0,2)).float() + 1e-5)
        
    else:
        output_mask = output * mask
        ah_ave = output_mask.sum(dim=0)/(mask.sum(dim=0).float() + 1e-5)
        
    total_ave = output_mask.sum()/(mask.sum().float())
    
    return total_ave,ah_ave


class SAM2Act_Agent:
    def __init__(
        self,
        network: nn.Module,
        num_rotation_classes: int,
        stage_two: bool,
        add_lang: bool,
        amp: bool,
        bnb: bool,
        move_pc_in_bound: bool,
        lr: float = 0.0001,
        lr_cos_dec: bool = False,
        cos_dec_max_step: int = 60000,
        warmup_steps: int = 0,
        image_resolution: list = None,
        lambda_weight_l2: float = 0.0,
        transform_augmentation: bool = True,
        transform_augmentation_xyz: list = [0.1, 0.1, 0.1],
        transform_augmentation_rpy: list = [0.0, 0.0, 20.0],
        place_with_mean: bool = True,
        transform_augmentation_rot_resolution: int = 5,
        optimizer_type: str = "lamb",
        gt_hm_sigma: float = 1.5,
        img_aug: bool = False,
        add_rgc_loss: bool = False,
        scene_bounds: list = peract_utils.SCENE_BOUNDS,
        cameras: list = peract_utils.CAMERAS,
        rot_ver: int = 0,
        rot_x_y_aug: int = 2,
        log_dir="",
        action_horizon=None,
        same_trans_aug_per_seq: bool = False,
        use_memory: bool = False,
        num_maskmem: int = 7,
    ):
        """
        :param gt_hm_sigma: the std of the groundtruth hm, currently for for
            2d, if -1 then only single point is considered
        :type gt_hm_sigma: float
        :param rot_ver: version of the rotation prediction network
            Either:
                0: same as peract, independent discrete xyz predictions
                1: xyz prediction dependent on one another
        :param rot_x_y_aug: only applicable when rot_ver is 1, it specifies how
            much error we should add to groundtruth rotation while training
        :param log_dir: a folder location for saving some intermediate data
        """

        self._network = network
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = 360 / self._num_rotation_classes
        self._lr = lr
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._place_with_mean = place_with_mean
        self._transform_augmentation_xyz = torch.from_numpy(
            np.array(transform_augmentation_xyz)
        )
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = (
            transform_augmentation_rot_resolution
        )
        self._optimizer_type = optimizer_type
        self.gt_hm_sigma = gt_hm_sigma
        self.img_aug = img_aug
        self.add_rgc_loss = add_rgc_loss
        self.amp = amp
        self.bnb = bnb
        self.stage_two = stage_two
        self.add_lang = add_lang
        self.log_dir = log_dir
        self.warmup_steps = warmup_steps
        self.lr_cos_dec = lr_cos_dec
        self.cos_dec_max_step = cos_dec_max_step
        self.scene_bounds = scene_bounds
        self.cameras = cameras
        self.move_pc_in_bound = move_pc_in_bound
        self.rot_ver = rot_ver
        self.rot_x_y_aug = rot_x_y_aug

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        if isinstance(self._network, DistributedDataParallel):
            self._net_mod = self._network.module
        else:
            self._net_mod = self._network

        self.num_all_rot = self._num_rotation_classes * 3

        self.scaler = GradScaler(enabled=self.amp)

        self.action_horizon = action_horizon

        self._same_trans_aug_per_seq = same_trans_aug_per_seq

        self.use_memory = use_memory
        self._num_maskmem = num_maskmem

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        sam_mem_params = []
        upsample_params = []
        
        for name, param in self._network.named_parameters():
            if "sam" in name and "image encoder" not in name:
                sam_mem_params.append(param)
            elif "up0" in name:
                upsample_params.append(param)

        param_groups = [
            {"params": sam_mem_params, "lr": self._lr * 2},  # SAM2 (memory)
            {"params": upsample_params, "lr": self._lr},  # upsample layers
        ]

        if self._optimizer_type == "lamb":
            if self.bnb:
                print("Using 8-Bit Optimizer")
                self._optimizer = bnb.optim.LAMB(
                    self._network.parameters(),
                    # param_groups,
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                )
            else:
                # From: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
                self._optimizer = Lamb(
                    self._network.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )
        elif self._optimizer_type == "adam":
            self._optimizer = torch.optim.Adam(
                self._network.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
            )
        else:
            raise Exception("Unknown optimizer")

        if self.lr_cos_dec:
            after_scheduler = CosineAnnealingLR(
                self._optimizer,
                T_max=self.cos_dec_max_step,
                eta_min=self._lr / 100,  # mininum lr
            )
        else:
            after_scheduler = None
        self._lr_sched = GradualWarmupScheduler(
            self._optimizer,
            multiplier=1,
            total_epoch=self.warmup_steps,
            after_scheduler=after_scheduler,
        )

    def load_clip(self):
        self.clip_model, self.clip_preprocess = clip.load("RN50", device=self._device)
        self.clip_model.eval()

    def unload_clip(self):
        del self.clip_model
        del self.clip_preprocess
        with torch.cuda.device(self._device):
            torch.cuda.empty_cache()

    # copied from per-act and removed the translation part
    def _get_one_hot_expert_actions(
        self,
        batch_size,
        action_rot,
        action_grip,
        action_ignore_collisions,
        device,
    ):
        """_get_one_hot_expert_actions.

        :param batch_size: int
        :param action_rot: np.array of shape (bs, 4), quternion xyzw format
        :param action_grip: torch.tensor of shape (bs)
        :param action_ignore_collisions: torch.tensor of shape (bs)
        :param device:
        """
        bs = batch_size
        assert action_rot.shape == (bs, 4)
        assert action_grip.shape == (bs,), (action_grip, bs)

        action_rot_x_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_y_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_z_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
            gt_rot = action_rot[b]
            gt_rot = aug_utils.quaternion_to_discrete_euler(
                gt_rot, self._rotation_resolution
            )
            action_rot_x_one_hot[b, gt_rot[0]] = 1
            action_rot_y_one_hot[b, gt_rot[1]] = 1
            action_rot_z_one_hot[b, gt_rot[2]] = 1

            # grip
            gt_grip = action_grip[b]
            action_grip_one_hot[b, gt_grip] = 1

            # ignore collision
            gt_ignore_collisions = action_ignore_collisions[b, :]
            action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        return (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        )

    def get_q(self, out, dims, only_pred=False, get_q_trans=True):
        """
        :param out: output of mvt
        :param dims: tensor dimensions (bs, nc, h, w)
        :param only_pred: some speedupds if the q values are meant only for
            prediction
        :return: tuple of trans_q, rot_q, grip_q and coll_q that is used for
            training and preduction
        """
        bs, nc, h, w = dims
        assert isinstance(only_pred, bool)

        if get_q_trans:
            pts = None
            # (bs, h*w, nc)
            q_trans = out["trans"].view(bs, nc, h * w).transpose(1, 2)
            if not only_pred:
                q_trans = q_trans.clone()

            # if two stages, we concatenate the q_trans, and replace all other
            # q
            if self.stage_two and not self.use_memory:
                out = out["mvt2"]
                q_trans2 = out["trans"].view(bs, nc, h * w).transpose(1, 2)
                if not only_pred:
                    q_trans2 = q_trans2.clone()
                q_trans = torch.cat((q_trans, q_trans2), dim=2)
        else:
            pts = None
            q_trans = None
            if self.stage_two:
                out = out["mvt2"]

        if not self.use_memory:

            if self.rot_ver == 0:
                # (bs, 218)
                rot_q = out["feat"].view(bs, -1)[:, 0 : self.num_all_rot]
                grip_q = out["feat"].view(bs, -1)[:, self.num_all_rot : self.num_all_rot + 2]
                # (bs, 2)
                collision_q = out["feat"].view(bs, -1)[
                    :, self.num_all_rot + 2 : self.num_all_rot + 4
                ]
            elif self.rot_ver == 1:
                rot_q = torch.cat((out["feat_x"], out["feat_y"], out["feat_z"]),
                                dim=-1).view(bs, -1)
                grip_q = out["feat_ex_rot"].view(bs, -1)[:, :2]
                collision_q = out["feat_ex_rot"].view(bs, -1)[:, 2:]
            else:
                assert False

            y_q = None

            return q_trans, rot_q, grip_q, collision_q, y_q, pts
        
        else:
            return q_trans, None, None, None, None, None

    def _visualize_pointcloud_to_html(self, pc, step_name, step_num, waypoint=None):
        """
        将点云列表可视化并保存为一个HTML文件（包含多个子图）
        
        Args:
            pc: 点云数据，可能是：
                - tensor，形状为(bs, num_points, 3)
                - 列表，每个元素是一个tensor或numpy数组，形状为(N, 3)
                - numpy数组，形状为(num_points, 3)
            step_name: 步骤名称（如"第五步前"）
            step_num: 步骤编号（用于文件名）
            waypoint: 可选的waypoint数据，可以是：
                - tensor，形状为(bs, 3)或(3,)
                - 列表，每个元素是一个tensor或numpy数组，形状为(3,)
                - numpy数组，形状为(3,)
                如果pc是列表，waypoint也应该是对应的列表
        """
        try:
            # 创建输出目录
            output_dir = "sam2act/visualizations"
            os.makedirs(output_dir, exist_ok=True)
            
            # 将点云数据转换为点云列表格式
            pc_list = []
            if torch.is_tensor(pc):
                # 如果是tensor，形状为(bs, num_points, 3)或(num_points, 3)
                if len(pc.shape) == 3:
                    # (bs, num_points, 3) - 转换为列表
                    for i in range(pc.shape[0]):
                        pc_list.append(pc[i].detach().cpu().numpy())
                elif len(pc.shape) == 2 and pc.shape[1] == 3:
                    # (num_points, 3) - 单个点云
                    pc_list.append(pc.detach().cpu().numpy())
                else:
                    return
            elif isinstance(pc, list) and len(pc) > 0:
                # 如果是列表，处理每个元素
                for pc_item in pc:
                    if torch.is_tensor(pc_item):
                        pc_list.append(pc_item.detach().cpu().numpy())
                    elif isinstance(pc_item, np.ndarray):
                        pc_list.append(pc_item)
                    else:
                        continue
            elif isinstance(pc, np.ndarray):
                # 单个numpy数组
                if len(pc.shape) == 2 and pc.shape[1] == 3:
                    pc_list.append(pc)
                elif len(pc.shape) == 3:
                    # (bs, num_points, 3) - 转换为列表
                    for i in range(pc.shape[0]):
                        pc_list.append(pc[i])
                else:
                    return
            else:
                return
            
            if len(pc_list) == 0:
                return
            
            # 处理waypoint数据，转换为列表格式
            wpt_list = []
            if waypoint is not None:
                if torch.is_tensor(waypoint):
                    if len(waypoint.shape) == 2:
                        # (bs, 3) - 转换为列表
                        for i in range(waypoint.shape[0]):
                            wpt_list.append(waypoint[i].detach().cpu().numpy())
                    elif len(waypoint.shape) == 1 and waypoint.shape[0] == 3:
                        # (3,) - 单个waypoint
                        wpt_list.append(waypoint.detach().cpu().numpy())
                elif isinstance(waypoint, list) and len(waypoint) > 0:
                    # 如果是列表，处理每个元素
                    for wpt_item in waypoint:
                        if torch.is_tensor(wpt_item):
                            wpt_list.append(wpt_item.detach().cpu().numpy())
                        elif isinstance(wpt_item, np.ndarray):
                            wpt_list.append(wpt_item)
                        else:
                            continue
                elif isinstance(waypoint, np.ndarray):
                    if len(waypoint.shape) == 1 and waypoint.shape[0] == 3:
                        # (3,) - 单个waypoint
                        wpt_list.append(waypoint)
                    elif len(waypoint.shape) == 2:
                        # (bs, 3) - 转换为列表
                        for i in range(waypoint.shape[0]):
                            wpt_list.append(waypoint[i])
            
            # 确保waypoint列表长度与点云列表长度匹配（如果waypoint不为空）
            if len(wpt_list) > 0 and len(wpt_list) != len(pc_list):
                # 如果不匹配，只使用第一个waypoint或重复最后一个
                if len(wpt_list) == 1:
                    wpt_list = wpt_list * len(pc_list)
                else:
                    # 如果waypoint数量少于点云，用最后一个填充；如果多于，截断
                    while len(wpt_list) < len(pc_list):
                        wpt_list.append(wpt_list[-1] if len(wpt_list) > 0 else None)
                    wpt_list = wpt_list[:len(pc_list)]
            
            # 计算子图布局（rows和cols）
            num_plots = len(pc_list)
            if num_plots == 1:
                rows, cols = 1, 1
            elif num_plots == 2:
                rows, cols = 1, 2
            elif num_plots <= 4:
                rows, cols = 2, 2
            elif num_plots <= 6:
                rows, cols = 2, 3
            elif num_plots <= 9:
                rows, cols = 3, 3
            else:
                # 对于更多点云，使用网格布局
                cols = int(np.ceil(np.sqrt(num_plots)))
                rows = int(np.ceil(num_plots / cols))
            
            # 创建3D子图，使用specs指定每个子图为scene类型
            specs = [[{"type": "scene"} for _ in range(cols)] for _ in range(rows)]
            fig = make_subplots(
                rows=rows,
                cols=cols,
                specs=specs,
                subplot_titles=[f'点云 #{i}' for i in range(num_plots)],
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            # 为每个点云添加子图
            for idx, pc_np in enumerate(pc_list):
                # 确保是numpy数组且形状正确
                if not isinstance(pc_np, np.ndarray):
                    continue
                
                # 如果形状不是(N, 3)，跳过
                if len(pc_np.shape) != 2 or pc_np.shape[1] != 3:
                    continue
                
                # 计算子图位置（1-indexed）
                row = idx // cols + 1
                col = idx % cols + 1
                
                # 提取xyz坐标
                x = pc_np[:, 0]
                y = pc_np[:, 1]
                z = pc_np[:, 2]
                
                # 添加点云trace
                fig.add_trace(
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        mode='markers',
                        marker=dict(
                            size=1,
                            color=z,  # 使用z坐标作为颜色
                            colorscale='Viridis',
                            opacity=0.8,
                            showscale=(idx == 0),  # 只在第一个子图显示colorbar
                            colorbar=dict(title="Z坐标", len=0.8) if idx == 0 else None
                        ),
                        name=f'点云 #{idx}',
                        showlegend=False
                    ),
                    row=row,
                    col=col
                )
                
                # 处理对应的waypoint数据并添加到图中
                if len(wpt_list) > idx and wpt_list[idx] is not None:
                    wpt_np = wpt_list[idx]
                    # 确保waypoint形状正确
                    if isinstance(wpt_np, np.ndarray) and len(wpt_np.shape) == 1 and wpt_np.shape[0] == 3:
                        fig.add_trace(
                            go.Scatter3d(
                                x=[wpt_np[0]],
                                y=[wpt_np[1]],
                                z=[wpt_np[2]],
                                mode='markers',
                                marker=dict(
                                    size=15,
                                    color='red',
                                    symbol='diamond',
                                    opacity=1.0,
                                    line=dict(width=2, color='darkred')
                                ),
                                name=f'Waypoint #{idx}',
                                showlegend=False
                            ),
                            row=row,
                            col=col
                        )
            
            # 更新每个scene的布局
            for idx in range(num_plots):
                if idx == 0:
                    scene_name = 'scene'
                else:
                    scene_name = f'scene{idx+1}'
                
                fig.update_layout(**{
                    scene_name: dict(
                        xaxis_title='X (米)',
                        yaxis_title='Y (米)',
                        zaxis_title='Z (米)',
                        aspectmode='data'
                    )
                })
            
            # 设置整体布局
            fig.update_layout(
                title_text=f'点云可视化 - {step_name} (共{num_plots}个点云)',
                height=400 * rows,  # 根据行数调整高度
                width=600 * cols if cols > 1 else 1200  # 根据列数调整宽度
            )
            
            # 保存为HTML文件
            output_file = os.path.join(output_dir, f'pc_step{step_num}_{step_name.replace(" ", "_")}.html')
            fig.write_html(output_file)
            print(f"✓ 点云可视化已保存: {output_file} (共{num_plots}个点云)")
            
        except Exception as e:
            print(f"⚠ 点云可视化失败 ({step_name}): {e}")
            import traceback
            traceback.print_exc()

    def update(
        self,
        step: int,
        replay_sample: dict,
        backprop: bool = True,
        eval_log: bool = False,
        reset_log: bool = False,
    ) -> dict:
        """
        模型训练更新方法，执行一个训练步骤
        
        Args:
            step: 当前训练步数
            replay_sample: 从replay buffer中采样的数据字典，包含：
                - rot_grip_action_indicies: 旋转和抓取动作索引 (b, 1, 4)
                - ignore_collisions: 是否忽略碰撞标志 (b, 1, 1)
                - gripper_pose: 机械臂末端位姿 (b, 1, 7) [x, y, z, qx, qy, qz, qw]
                - lang_goal_embs: 语言目标嵌入 (b, 1, 77, 512)
                - low_dim_state: 低维状态信息 (b, 1, proprio_dim)
            backprop: 是否执行反向传播（训练模式）
            eval_log: 是否记录评估日志
            reset_log: 是否重置日志记录器
            
        Returns:
            return_out: 包含损失和评估指标的字典
        """
        # 验证输入数据的形状是否符合预期
        assert replay_sample["rot_grip_action_indicies"].shape[1:] == (1, 4)
        assert replay_sample["ignore_collisions"].shape[1:] == (1, 1)
        assert replay_sample["gripper_pose"].shape[1:] == (1, 7)
        assert replay_sample["lang_goal_embs"].shape[1:] == (1, 77, 512)
        assert replay_sample["low_dim_state"].shape[1:] == (
            1,
            self._net_mod.proprio_dim,
        )

        # ========== 第一步：从replay buffer采样数据并提取动作信息 ==========
        # 提取最后一个时间步的动作数据（时间维度为1，取索引-1）
        action_rot_grip = replay_sample["rot_grip_action_indicies"][
            :, -1
        ].int()  # (b, 4) 旋转和抓取动作的离散索引
        action_ignore_collisions = replay_sample["ignore_collisions"][
            :, -1
        ].int()  # (b, 1) 是否忽略碰撞的标志
        action_gripper_pose = replay_sample["gripper_pose"][:, -1]  # (b, 7) 机械臂末端位姿
        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3) 平移部分 [x, y, z]
        # 旋转部分，四元数格式为 xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4) 旋转四元数 [qx, qy, qz, qw]
        action_grip = action_rot_grip[:, -1]  # (b,) 抓取动作（开/关）
        lang_goal_embs = replay_sample["lang_goal_embs"][:, -1].float()  # (b, 77, 512) 语言目标嵌入
        tasks = replay_sample["tasks"]  # 任务名称列表

        # 处理本体感觉信息（低维状态，如关节角度等）
        proprio = arm_utils.stack_on_channel(replay_sample["low_dim_state"])  # (b, 4)
        return_out = {}  # 用于存储返回的日志信息

        # ========== 第二步：预处理输入数据（图像和点云） ==========
        obs, pcd = peract_utils._preprocess_inputs(replay_sample, self.cameras)

        # 在torch.no_grad()上下文中进行数据预处理和增强（不需要梯度）
        with torch.no_grad():
            # 从观测数据中提取点云和图像特征
            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs,
                pcd,
            )

            # ========== 第三步：应用SE(3)变换数据增强 ==========
            # 如果启用变换增强且处于训练模式，对点云和动作进行随机SE(3)变换
            if self._transform_augmentation and backprop:
                # 判断是否为同一序列使用相同变换
                if not self._same_trans_aug_per_seq:
                    # 每个样本独立应用不同的SE(3)变换增强
                    action_trans_con, action_rot, pc = apply_se3_aug_con(
                        pcd=pc,
                        action_gripper_pose=action_gripper_pose,
                        bounds=torch.tensor(self.scene_bounds),  # 场景边界
                        trans_aug_range=torch.tensor(self._transform_augmentation_xyz),  # 平移增强范围
                        rot_aug_range=torch.tensor(self._transform_augmentation_rpy),  # 旋转增强范围
                    )
                    # 将numpy数组转换回tensor并移到正确的设备
                    action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                    action_rot = torch.tensor(action_rot).to(pc.device)
                else:
                    # 同一序列内的所有观测使用相同的SE(3)变换（保持时序一致性）
                    bs = pc.shape[0]  # batch size
                    num_obs = self._num_maskmem + 1  # 每个序列的观测数量
                    num_seq = bs // num_obs  # 序列数量

                    action_trans_con_after = []
                    action_rot_after = []
                    pc_after = []
                    # 对每个序列分别处理
                    for seq_idx in range(num_seq):
                        # 提取当前序列的观测
                        pc_i = pc[seq_idx*num_obs:seq_idx*num_obs+num_obs]
                        action_gripper_pose_i = action_gripper_pose[seq_idx*num_obs:seq_idx*num_obs+num_obs]
                        # 对同一序列应用相同的SE(3)变换
                        action_trans_con_i, action_rot_i, pc_i = apply_se3_aug_con_same(
                            pcd=pc_i,
                            action_gripper_pose=action_gripper_pose_i,
                            bounds=torch.tensor(self.scene_bounds),
                            trans_aug_range=torch.tensor(self._transform_augmentation_xyz),
                            rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                        )
                        action_trans_con_i = torch.tensor(action_trans_con_i).to(pc.device)
                        action_rot_i = torch.tensor(action_rot_i).to(pc.device)

                        action_trans_con_after.append(action_trans_con_i)
                        action_rot_after.append(action_rot_i)
                        pc_after.append(pc_i)

                    # 拼接所有序列的结果
                    action_trans_con = torch.cat(action_trans_con_after, dim=0)
                    action_rot = torch.cat(action_rot_after, dim=0)
                    pc = torch.cat(pc_after, dim=0)

            # ========== 第四步：归一化四元数 ==========
            # TODO: 向量化优化
            # 将四元数转换为numpy格式进行归一化处理
            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                # 归一化四元数使其为单位四元数
                _action_rot = aug_utils.normalize_quaternion(_action_rot)
                # 确保四元数的w分量为正（q和-q表示相同旋转，统一选择w>0的表示）
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot

            # ========== 保存第五步前的点云 ==========
            # self._visualize_pointcloud_to_html(pc, "第五步前", 5)

            # ========== 第五步：将点云移动到场景边界内 ==========
            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )
            # 提取waypoint位置（目标位置的前3维）
            wpt = [x[:3] for x in action_trans_con]

            # ========== 保存第六步前的点云 ==========
            # self._visualize_pointcloud_to_html(pc, "第六步前", 6)

            # ========== 第六步：将点云和waypoint放置到标准立方体中 ==========
            # 将点云和waypoint转换到局部坐标系（相对于场景中心或边界）
            wpt_local = []  # 局部坐标系下的waypoint
            rev_trans = []  # 反向变换矩阵（用于后续将预测结果转换回全局坐标系）
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,  # 是否使用均值或边界进行归一化
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))  # 局部waypoint坐标
                rev_trans.append(b)  # 反向变换

            wpt_local = torch.cat(wpt_local, axis=0)

            # TODO: 向量化优化
            # 将点云也转换到局部坐标系
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            # ========== 保存第七步前的点云 ==========
            # self._visualize_pointcloud_to_html(pc, "第七步前", 7, waypoint=wpt_local)

            # import pdb; pdb.set_trace()

            # ========== 第七步：设置网络输入参数 ==========
            bs = len(pc)  # batch size
            nc = self._net_mod.num_img  # 相机数量
            h = w = self._net_mod.img_size  # 图像高度和宽度

            # 根据是否训练决定是否应用图像增强
            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None  # 动态相机信息（当前未使用）

        
        # ========== 第八步：前向传播 ==========
        with autocast(enabled=self.amp):  # 混合精度训练
            # 将连续动作转换为离散的one-hot编码
            (
                action_rot_x_one_hot,  # (bs, num_rotation_classes) X轴旋转的one-hot
                action_rot_y_one_hot,  # (bs, num_rotation_classes) Y轴旋转的one-hot
                action_rot_z_one_hot,  # (bs, num_rotation_classes) Z轴旋转的one-hot
                action_grip_one_hot,  # (bs, 2) 抓取动作的one-hot [关闭, 打开]
                action_collision_one_hot,  # (bs, 2) 碰撞标志的one-hot [不忽略, 忽略]
            ) = self._get_one_hot_expert_actions(
                bs, action_rot, action_grip, action_ignore_collisions, device=self._device
            )

            # 如果使用旋转版本1，处理X和Y轴的旋转增强
            if self.rot_ver == 1:
                # 获取X和Y轴旋转的离散类别索引
                rot_x_y = torch.cat(
                    [
                        action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                        action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                    ],
                    dim=-1,
                )
                # 如果启用旋转增强，添加随机偏移
                if self.rot_x_y_aug != 0:
                    # 在[-rot_x_y_aug, rot_x_y_aug]范围内添加随机整数偏移
                    rot_x_y += torch.randint(
                        -self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape
                    ).to(rot_x_y.device)
                    # 取模确保索引在有效范围内
                    rot_x_y %= self._num_rotation_classes

            # 注释掉的代码：生成ground truth热力图（如果使用的话）
            # hm_gt = self.get_gt_hm(
            #     wpt_local, dyn_cam_info, dims=(bs, nc, h, w)
            # )

            # 网络前向传播
            out = self._network(
                pc=pc,  # 点云数据
                img_feat=img_feat,  # 图像特征
                proprio=proprio,  # 本体感觉信息
                lang_emb=lang_goal_embs,  # 语言目标嵌入
                img_aug=img_aug,  # 图像增强强度
                wpt_local=wpt_local if self._network.training else None,  # 训练时提供waypoint用于监督学习
                rot_x_y=rot_x_y if self.rot_ver == 1 else None,  # X和Y轴旋转信息
                # hm_gt=hm_gt,  # ground truth热力图
            )

            # 从网络输出中提取Q值（动作价值函数）
            q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
                out, dims=(bs, nc, h, w)
            )
            
            # 根据waypoint和网络输出计算平移动作的离散索引
            action_trans = self.get_action_trans(
                wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w)
            )

        # ========== 第九步：计算损失并反向传播 ==========
        loss_log = {}
        if backprop:
            with autocast(enabled=self.amp):
                # 计算平移损失（交叉熵损失）
                trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()
                # 初始化其他损失为0
                rot_loss_x = rot_loss_y = rot_loss_z = 0.0
                grip_loss = 0.0
                collision_loss = 0.0
                
                if not self.use_memory:
                    # 如果不使用memory机制，计算旋转、抓取和碰撞损失
                    if self.add_rgc_loss:
                        # 计算X轴旋转损失
                        rot_loss_x = self._cross_entropy_loss(
                            rot_q[
                                :,
                                0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                            ],
                            action_rot_x_one_hot.argmax(-1),
                        ).mean()

                        # 计算Y轴旋转损失
                        rot_loss_y = self._cross_entropy_loss(
                            rot_q[
                                :,
                                1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                            ],
                            action_rot_y_one_hot.argmax(-1),
                        ).mean()

                        # 计算Z轴旋转损失
                        rot_loss_z = self._cross_entropy_loss(
                            rot_q[
                                :,
                                2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                            ],
                            action_rot_z_one_hot.argmax(-1),
                        ).mean()

                        # 计算抓取动作损失
                        grip_loss = self._cross_entropy_loss(
                            grip_q,
                            action_grip_one_hot.argmax(-1),
                        ).mean()

                        # 计算碰撞标志损失
                        collision_loss = self._cross_entropy_loss(
                            collision_q, action_collision_one_hot.argmax(-1)
                        ).mean()

                    # 总损失 = 平移损失 + 旋转损失 + 抓取损失 + 碰撞损失
                    total_loss = (
                        trans_loss
                        + rot_loss_x
                        + rot_loss_y
                        + rot_loss_z
                        + grip_loss
                        + collision_loss
                    )

                else:
                    # 如果使用memory机制，只计算平移损失
                    total_loss = trans_loss

            # 反向传播和优化器更新
            self._optimizer.zero_grad(set_to_none=True)  # 清零梯度
            self.scaler.scale(total_loss).backward()  # 缩放损失并反向传播（混合精度训练）

            # 注释掉的代码：梯度裁剪（如果需要可以取消注释）
            # self.scaler.unscale_(self._optimizer)
            # torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=0.1)

            self.scaler.step(self._optimizer)  # 更新参数
            self.scaler.update()  # 更新scaler
            self._lr_sched.step()  # 更新学习率

            # 记录损失信息
            loss_log = {
                "total_loss": total_loss.item(),  # 总损失
                "trans_loss": trans_loss.item(),  # 平移损失
                "rot_loss_x": rot_loss_x.item() if not self.use_memory else None,  # X轴旋转损失
                "rot_loss_y": rot_loss_y.item() if not self.use_memory else None,  # Y轴旋转损失
                "rot_loss_z": rot_loss_z.item() if not self.use_memory else None,  # Z轴旋转损失
                "grip_loss": grip_loss.item() if not self.use_memory else None,  # 抓取损失
                "collision_loss": collision_loss.item() if not self.use_memory else None,  # 碰撞损失
                "lr": self._optimizer.param_groups[0]["lr"],  # 当前学习率
            }
            manage_loss_log(self, loss_log, reset_log=reset_log)  # 管理损失日志
            return_out.update(loss_log)

        # ========== 第十步：评估日志记录 ==========
        if eval_log:
            with torch.no_grad():  # 评估时不需要梯度
                # 将waypoint列表转换为tensor
                wpt = torch.cat([x.unsqueeze(0) for x in wpt])
                # 从网络输出中获取预测结果
                pred_wpt, pred_rot_quat, _, _ = self.get_pred(
                    out,
                    rot_q,
                    grip_q,
                    collision_q,
                    y_q,
                    rev_trans,  # 使用反向变换将局部坐标转换回全局坐标
                    dyn_cam_info=dyn_cam_info,
                )

                # 管理评估日志（计算预测误差等指标）
                return_log = manage_eval_log(
                    self=self,
                    tasks=tasks,
                    wpt=wpt,  # ground truth waypoint
                    pred_wpt=pred_wpt,  # 预测的waypoint
                    action_rot=action_rot,  # ground truth旋转
                    pred_rot_quat=pred_rot_quat,  # 预测的旋转
                    action_grip_one_hot=action_grip_one_hot,  # ground truth抓取动作
                    grip_q=grip_q,  # 预测的抓取Q值
                    action_collision_one_hot=action_collision_one_hot,  # ground truth碰撞标志
                    collision_q=collision_q,  # 预测的碰撞Q值
                    reset_log=reset_log,
                )

                return_out.update(return_log)

        return return_out

    @torch.no_grad()  # 推理时不计算梯度，节省内存并加速
    def act(
        self, step: int, observation: dict, deterministic=True, pred_distri=False
    ) -> ActResult:
        """
        执行动作推理方法，根据观察生成机器人动作
        
        Args:
            step: 当前步数
            observation: 观察字典，包含图像、点云、本体感觉等信息
            deterministic: 是否使用确定性策略（默认True）
            pred_distri: 是否返回预测分布（默认False）
        
        Returns:
            ActResult: 包含连续动作的结果对象，如果pred_distri=True则额外返回旋转分布
        """
        # 处理语言目标嵌入
        # 如果启用语言模态，使用CLIP编码语言目标token为嵌入向量
        if self.add_lang:
            lang_goal_tokens = observation.get("lang_goal_tokens", None).long()
            _, lang_goal_embs = _clip_encode_text(self.clip_model, lang_goal_tokens[0])
            lang_goal_embs = lang_goal_embs.float()
        else:
            # 否则创建零向量作为占位符
            lang_goal_embs = (
                torch.zeros(observation["lang_goal_embs"].shape)
                .float()
                .to(self._device)
            )

        # 处理本体感觉信息（机械臂关节状态等低维状态）
        proprio = arm_utils.stack_on_channel(observation["low_dim_state"])

        # 预处理观察数据：提取图像和点云
        obs, pcd = peract_utils._preprocess_inputs(observation, self.cameras)
        # 从观察和点云中提取点云特征和图像特征
        pc, img_feat = rvt_utils.get_pc_img_feat(
            obs,
            pcd,
        )

        # 将点云和图像特征移动到场景边界内
        pc, img_feat = rvt_utils.move_pc_in_bound(
            pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
        )

        # 将点云放置在标准立方体中（用于归一化）
        # TODO: Vectorize
        pc_new = []
        rev_trans = []  # 存储反向变换，用于后续将预测结果转换回原始坐标系
        for _pc in pc:
            # 将每个点云放置到立方体中，返回变换后的点云和反向变换函数
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            pc_new.append(a)
            rev_trans.append(b)
        pc = pc_new

        # 准备网络输入维度参数
        bs = len(pc)  # 批次大小
        nc = self._net_mod.num_img  # 相机数量
        h = w = self._net_mod.img_size  # 图像高度和宽度
        dyn_cam_info = None  # 动态相机信息（当前未使用）

        # 网络前向传播：使用点云、图像特征、本体感觉和语言嵌入进行推理
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            proprio=proprio,
            lang_emb=lang_goal_embs,
            img_aug=0,  # 推理时不进行图像增强
        )
        # 从网络输出中提取Q值分布（旋转、抓取、碰撞等动作的离散化分布）
        _, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
            out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=False
        )
        # 从Q值分布中提取预测的动作：路径点、旋转四元数、抓取状态、碰撞状态
        pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
            out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info
        )

        # 将预测的各个动作组成部分拼接成连续的动作向量
        # 包含：路径点坐标(3维) + 旋转四元数(4维) + 抓取状态(1维) + 碰撞状态(1维)
        continuous_action = np.concatenate(
            (
                pred_wpt[0].cpu().numpy(),  # 路径点位置 (x, y, z)
                pred_rot_quat[0],  # 旋转四元数 (w, x, y, z)
                pred_grip[0].cpu().numpy(),  # 抓取状态 (开/关)
                pred_coll[0].cpu().numpy(),  # 碰撞状态
            )
        )
        # 如果需要返回预测分布（用于不确定性分析或可视化）
        if pred_distri:
            # 提取X、Y、Z三个轴的旋转分布
            x_distri = rot_grip_q[
                0,
                0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
            ]
            y_distri = rot_grip_q[
                0,
                1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
            ]
            z_distri = rot_grip_q[
                0,
                2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
            ]
            # 返回动作结果和旋转分布
            return ActResult(continuous_action), (
                x_distri.cpu().numpy(),
                y_distri.cpu().numpy(),
                z_distri.cpu().numpy(),
            )
        else:
            # 正常情况下只返回动作结果
            return ActResult(continuous_action)

    def get_pred(
        self,
        out,
        rot_q,
        grip_q,
        collision_q,
        y_q,
        rev_trans,
        dyn_cam_info,
    ):
        if self.stage_two:
            assert y_q is None
            mvt1_or_mvt2 = False
        else:
            mvt1_or_mvt2 = True

        pred_wpt_local = self._net_mod.get_wpt(
            out, mvt1_or_mvt2, dyn_cam_info, y_q
        )

        pred_wpt = []
        for _pred_wpt_local, _rev_trans in zip(pred_wpt_local, rev_trans):
            pred_wpt.append(_rev_trans(_pred_wpt_local))
        pred_wpt = torch.cat([x.unsqueeze(0) for x in pred_wpt])

        pred_rot = torch.cat(
            (
                rot_q[
                    :,
                    0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
            ),
            dim=-1,
        )
        pred_rot_quat = aug_utils.discrete_euler_to_quaternion(
            pred_rot.cpu(), self._rotation_resolution
        )
        pred_grip = grip_q.argmax(1, keepdim=True)
        pred_coll = collision_q.argmax(1, keepdim=True)

        return pred_wpt, pred_rot_quat, pred_grip, pred_coll

    @torch.no_grad()
    def get_action_trans(
        self,
        wpt_local,
        pts,
        out,
        dyn_cam_info,
        dims,
    ):
        bs, nc, h, w = dims
        wpt_img = self._net_mod.get_pt_loc_on_img(
            wpt_local.unsqueeze(1),
            mvt1_or_mvt2=True,
            dyn_cam_info=dyn_cam_info,
            out=None
        )
        assert wpt_img.shape[1] == 1
        if self.stage_two and not self.use_memory:
            wpt_img2 = self._net_mod.get_pt_loc_on_img(
                wpt_local.unsqueeze(1),
                mvt1_or_mvt2=False,
                dyn_cam_info=dyn_cam_info,
                out=out,
            )
            assert wpt_img2.shape[1] == 1

            # (bs, 1, 2 * num_img, 2)
            wpt_img = torch.cat((wpt_img, wpt_img2), dim=-2)
            nc = nc * 2

        # (bs, num_img, 2)
        wpt_img = wpt_img.squeeze(1)

        action_trans = mvt_utils.generate_hm_from_pt(
            wpt_img.reshape(-1, 2),
            (h, w),
            sigma=self.gt_hm_sigma,
            thres_sigma_times=3,
        )
        action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()

        return action_trans


    def get_gt_hm(
        self,
        wpt_local,
        dyn_cam_info,
        dims,
    ):
        bs, nc, h, w = dims
        wpt_img = self._net_mod.get_pt_loc_on_img(
            wpt_local.unsqueeze(1),
            mvt1_or_mvt2=True,
            dyn_cam_info=dyn_cam_info,
            out=None
        )
        assert wpt_img.shape[1] == 1
        # if self.stage_two and not self.use_memory:
        #     wpt_img2 = self._net_mod.get_pt_loc_on_img(
        #         wpt_local.unsqueeze(1),
        #         mvt1_or_mvt2=False,
        #         dyn_cam_info=dyn_cam_info,
        #         out=out,
        #     )
        #     assert wpt_img2.shape[1] == 1

        #     # (bs, 1, 2 * num_img, 2)
        #     wpt_img = torch.cat((wpt_img, wpt_img2), dim=-2)
        #     nc = nc * 2

        # (bs, num_img, 2)
        wpt_img = wpt_img.squeeze(1)

        action_trans = mvt_utils.generate_hm_from_pt(
            wpt_img.reshape(-1, 2),
            (h, w),
            sigma=self.gt_hm_sigma,
            thres_sigma_times=3,
        )
        action_trans_hm = action_trans.view(bs, nc, h, w).clone()

        return action_trans_hm

    def reset(self):
        pass

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()
