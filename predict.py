""" Image forgery detection using Cog """
# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import os
import sys
import tempfile

from cog import BasePredictor, Input, Path
from PIL import Image

sys.path.insert(0, "./")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import copy
import shutil
from test import Detector, Model, MyDataset, gkern, metric, rm_and_make_dir

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.scse import SCSEUnet

gpu_ids = "0, 1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.model = Model().cuda()
        self.model.load()
        self.model.eval()

    def predict(self, image: Path = Input(description="Grayscale input image")) -> Path:
        """Run a single prediction on the model"""

        print("Saving output image........")
        image = str(image)
        test_size = "896"
        test_path = "cog_data/input/"
        rm_and_make_dir("cog_data/input")
        rm_and_make_dir("cog_data/output")

        im = Image.open(image).convert("RGB")
        im.save(os.path.join(test_path, "image.jpg"))

        print("Performing decomposition.......")
        decompose(test_path, test_size)
        print("Decomposition complete.")
        test_dataset = MyDataset(
            test_path="cog_temp/input_decompose_" + test_size + "/", size=int(test_size)
        )
        path_out = "cog_temp/input_decompose_" + test_size + "_pred/"
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1
        )
        rm_and_make_dir(path_out)

        print("Performing model inference.......")
        for items in test_loader:
            Ii, Mg = (item.cuda() for item in items[:-1])
            filename = items[-1]
            Mo = self.model(Ii)
            Mo = Mo * 255.0
            Mo = Mo.permute(0, 2, 3, 1).cpu().detach().numpy()
            for i in range(len(Mo)):
                Mo_tmp = Mo[i][..., ::-1]
                cv2.imwrite(path_out + filename[i][:-4] + ".png", Mo_tmp)

        print("Prediction complete.")
        if os.path.exists("cog_temp/input_decompose_" + test_size + "/"):
            shutil.rmtree("cog_temp/input_decompose_" + test_size + "/")
        path_pre = merge(test_path, test_size)
        print("Merging complete.")

        path_gt = "cog_data/mask/"
        if os.path.exists(path_gt):
            flist = sorted(os.listdir(path_pre))
            auc, f1, iou = [], [], []
            for file in flist:
                pre = cv2.imread(path_pre + file)
                gt = cv2.imread(path_gt + file[:-4] + ".png")
                H, W, C = pre.shape
                Hg, Wg, C = gt.shape
                if H != Hg or W != Wg:
                    gt = cv2.resize(gt, (W, H))
                    gt[gt > 127] = 255
                    gt[gt <= 127] = 0
                if np.max(gt) != np.min(gt):
                    auc.append(
                        roc_auc_score(
                            (gt.reshape(H * W * C) / 255).astype("int"),
                            pre.reshape(H * W * C) / 255.0,
                        )
                    )
                pre[pre > 127] = 255
                pre[pre <= 127] = 0
                a, b = metric(pre / 255, gt / 255)
                f1.append(a)
                iou.append(b)
            print(
                "Evaluation: AUC: %5.4f, F1: %5.4f, IOU: %5.4f"
                % (np.mean(auc), np.mean(f1), np.mean(iou))
            )

        output_path = path_out + filename[i][:-4] + ".png"
        print(f"Saving output image to {output_path}")
        return Path(output_path)


def decompose(test_path, test_size):
    flist = sorted(os.listdir(test_path))
    size_list = [int(test_size)]
    for size in size_list:
        path_out = "cog_temp/input_decompose_" + str(size) + "/"
        rm_and_make_dir(path_out)
    rtn_list = [[]]
    for file in flist:
        img = cv2.imread(test_path + file)
        # img = cv2.rotate(img, cv2.cv2.ROTATE_180)
        H, W, _ = img.shape
        size_idx = 0
        while size_idx < len(size_list) - 1:
            if H < size_list[size_idx + 1] or W < size_list[size_idx + 1]:
                break
            size_idx += 1
        rtn_list[size_idx].append(file)
        size = size_list[size_idx]
        path_out = "cog_temp/input_decompose_" + str(size) + "/"
        X, Y = H // (size // 2) + 1, W // (size // 2) + 1
        idx = 0
        for x in range(X - 1):
            if x * size // 2 + size > H:
                break
            for y in range(Y - 1):
                if y * size // 2 + size > W:
                    break
                img_tmp = img[
                    x * size // 2 : x * size // 2 + size,
                    y * size // 2 : y * size // 2 + size,
                    :,
                ]
                cv2.imwrite(path_out + file[:-4] + "_%03d.png" % idx, img_tmp)
                idx += 1
            img_tmp = img[x * size // 2 : x * size // 2 + size, -size:, :]
            cv2.imwrite(path_out + file[:-4] + "_%03d.png" % idx, img_tmp)
            idx += 1
        for y in range(Y - 1):
            if y * size // 2 + size > W:
                break
            img_tmp = img[-size:, y * size // 2 : y * size // 2 + size, :]
            cv2.imwrite(path_out + file[:-4] + "_%03d.png" % idx, img_tmp)
            idx += 1
        img_tmp = img[-size:, -size:, :]
        cv2.imwrite(path_out + file[:-4] + "_%03d.png" % idx, img_tmp)
        idx += 1
    return rtn_list


def merge(path, test_size):
    path_d = "cog_temp/input_decompose_" + test_size + "_pred/"
    path_r = "cog_data/output/"
    rm_and_make_dir(path_r)
    size = int(test_size)

    gk = gkern(size)
    gk = 1 - gk

    for file in sorted(os.listdir(path)):
        img = cv2.imread(path + file)
        H, W, _ = img.shape
        X, Y = H // (size // 2) + 1, W // (size // 2) + 1
        idx = 0
        rtn = np.ones((H, W, 3), dtype=np.float32) * -1
        for x in range(X - 1):
            if x * size // 2 + size > H:
                break
            for y in range(Y - 1):
                if y * size // 2 + size > W:
                    break
                img_tmp = cv2.imread(path_d + file[:-4] + "_%03d.png" % idx)
                weight_cur = copy.deepcopy(
                    rtn[
                        x * size // 2 : x * size // 2 + size,
                        y * size // 2 : y * size // 2 + size,
                        :,
                    ]
                )
                h1, w1, _ = weight_cur.shape
                gk_tmp = cv2.resize(gk, (w1, h1))
                weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
                weight_cur[weight_cur == -1] = 0
                weight_tmp = copy.deepcopy(weight_cur)
                weight_tmp = 1 - weight_tmp
                rtn[
                    x * size // 2 : x * size // 2 + size,
                    y * size // 2 : y * size // 2 + size,
                    :,
                ] = (
                    weight_cur
                    * rtn[
                        x * size // 2 : x * size // 2 + size,
                        y * size // 2 : y * size // 2 + size,
                        :,
                    ]
                    + weight_tmp * img_tmp
                )
                idx += 1
            img_tmp = cv2.imread(path_d + file[:-4] + "_%03d.png" % idx)
            weight_cur = copy.deepcopy(
                rtn[x * size // 2 : x * size // 2 + size, -size:, :]
            )
            h1, w1, _ = weight_cur.shape
            gk_tmp = cv2.resize(gk, (w1, h1))
            weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
            weight_cur[weight_cur == -1] = 0
            weight_tmp = copy.deepcopy(weight_cur)
            weight_tmp = 1 - weight_tmp
            rtn[x * size // 2 : x * size // 2 + size, -size:, :] = (
                weight_cur * rtn[x * size // 2 : x * size // 2 + size, -size:, :]
                + weight_tmp * img_tmp
            )
            idx += 1
        for y in range(Y - 1):
            if y * size // 2 + size > W:
                break
            img_tmp = cv2.imread(path_d + file[:-4] + "_%03d.png" % idx)
            weight_cur = copy.deepcopy(
                rtn[-size:, y * size // 2 : y * size // 2 + size, :]
            )
            h1, w1, _ = weight_cur.shape
            gk_tmp = cv2.resize(gk, (w1, h1))
            weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
            weight_cur[weight_cur == -1] = 0
            weight_tmp = copy.deepcopy(weight_cur)
            weight_tmp = 1 - weight_tmp
            rtn[-size:, y * size // 2 : y * size // 2 + size, :] = (
                weight_cur * rtn[-size:, y * size // 2 : y * size // 2 + size, :]
                + weight_tmp * img_tmp
            )
            idx += 1
        img_tmp = cv2.imread(path_d + file[:-4] + "_%03d.png" % idx)
        weight_cur = copy.deepcopy(rtn[-size:, -size:, :])
        h1, w1, _ = weight_cur.shape
        gk_tmp = cv2.resize(gk, (w1, h1))
        weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
        weight_cur[weight_cur == -1] = 0
        weight_tmp = copy.deepcopy(weight_cur)
        weight_tmp = 1 - weight_tmp
        rtn[-size:, -size:, :] = (
            weight_cur * rtn[-size:, -size:, :] + weight_tmp * img_tmp
        )
        idx += 1
        rtn[rtn < 127] = 0
        rtn[rtn >= 127] = 255
        cv2.imwrite(path_r + file[:-4] + ".png", np.uint8(rtn))
    return path_r
