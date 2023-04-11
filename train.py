import os
import cv2
import copy
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import roc_auc_score
from models.scse import SCSEUnet

gpu_ids = '0, 1'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids


class MyDataset(Dataset):
    def __init__(self, test_path='', size=896):
        self.test_path = test_path
        self.size = size
        self.filelist = sorted(os.listdir(self.test_path))
        self.transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        return self.load_item(idx)

    def __len__(self):
        return len(self.filelist)

    def load_item(self, idx):
        if self.choice != 'test':
            fname1, fname2 = self.filelist[idx]
        else:
            fname1, fname2 = self.test_path + self.filelist[idx], ''

        img = cv2.imread(fname1)[..., ::-1]
        H, W, _ = img.shape
        mask = np.zeros([H, W, 3])

        H, W, _ = img.shape
        img = img.astype('float') / 255.
        mask = mask.astype('float') / 255.
        return self.transform(img), self.tensor(mask[:, :, :1]), fname1.split('/')[-1]

    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.name = 'detector'
        self.det_net = SCSEUnet(backbone_arch='senet154', num_channels=3)

    def forward(self, Ii):
        Mo = self.det_net(Ii)
        return Mo


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.save_dir = 'weights/'
        self.networks = Detector()
        self.gen = nn.DataParallel(self.networks).cuda()
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.bce_loss = nn.BCELoss()
        self.global_xi = [torch.zeros([batch_size, 3, int(patch_size), int(patch_size)])]

    def process(self, Ii, Mg, adv=False, eva=False):
        if adv:  # Modeling the noise Tau and Xi
            self.gen_optimizer.zero_grad()
            Ii.sub_(mean).div_(std)
            Ii_tau = self(Ii, isOSN=True)

            xi = self.sampling_xi(size=Ii.size(0))

            Ii_xi = Ii_tau + xi
            Ii_xi.clamp_(-1, 1)

            self.gen_optimizer.zero_grad()
            Mo = self(Ii_xi)
            gen_loss = self.bce_loss(Mo.view(Mo.size(0), -1), Mg.view(Mg.size(0), -1))
            self.backward(gen_loss)
            self.global_xi.append(torch.sign(xi.grad).detach().cpu())
        else:  # Without noise modeling
            self.gen_optimizer.zero_grad()
            Ii.sub_(mean).div_(std)
            Mo = self(Ii)
            gen_loss = self.bce_loss(Mo.view(Mo.size(0), -1), Mg.view(Mg.size(0), -1))
            if not eva:
                self.backward(gen_loss)
        return Mo, gen_loss

    def sampling_xi(self, size):
        if len(self.global_xi) < 5:
            idx = range(len(self.global_xi))
        else:
            idx = np.random.choice(range(len(self.global_xi)), 5, replace=False)
        rtn = self.global_xi[idx[0]][0:size]
        rtn.clamp_(-4 / 255, 4 / 255)
        for i in idx[1:]:
            rtn[0:size] += 4 / 255 * self.global_xi[i][0:size]
            rtn.clamp_(-4 / 255, 4 / 255)
        rtn = Variable(rtn.cuda(), requires_grad=True)
        return rtn

    def forward(self, Ii, isOSN=False):
        return self.gen(Ii, isOSN)

    def backward(self, gen_loss=None):
        if gen_loss:
            gen_loss.backward(retain_graph=False)
            self.gen_optimizer.step()

    def save(self, path=''):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        torch.save(self.gen.state_dict(), self.save_dir + path + '%s_weights.pth' % self.networks.name)

    def load(self, path=''):
        self.gen.load_state_dict(torch.load(self.save_dir + path + '%s_weights.pth' % self.networks.name))


class ForgeryForensics():
    def __init__(self):
        self.batch_size = 16  # modify according to the resources

        self.train_npy = 'train.npy'  # Need to update
        self.train_file = np.load('data/' + self.train_npy)

        self.val_npy = 'train.npy'  # Need to update
        self.val_file = np.load('data/' + self.val_npy)[:4]

        '''
        train.npy should have the format:
        [
            (path_of_input1, path_of_ground_truth1),
            (path_of_input2, path_of_ground_truth2),
            ...
        ]
        '''

        self.train_num = len(self.train_file)
        self.val_num = len(self.val_file)
        train_dataset = MyDataset(self.train_num, self.train_file, choice='train')
        val_dataset = MyDataset(self.val_num, self.val_file, choice='val')

        self.model = Model(self.batch_size).cuda()
        self.n_epochs = 9999
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                       drop_last=True)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False, num_workers=4)

    def train(self):
        with open('log_' + gpu_ids[0] + '.txt', 'a+') as f:
            f.write('\nTrain/Val with ' + self.train_npy + '/' + self.val_npy + ' with num %d/%d' % (
            self.train_num, self.val_num))
        cnt, gen_losses, auc, f1, iou = 0, [], [], [], []
        best_score = 0
        scheduler = ReduceLROnPlateau(self.model.gen_optimizer, patience=6, factor=0.5, mode='max')
        for epoch in range(1, self.n_epochs + 1):
            for items in self.train_loader:
                cnt += self.batch_size
                self.model.train()
                Ii, Mg = (item.cuda() for item in items[:-1])
                Mo, gen_loss = self.model.process(Ii, Mg, adv=True)
                gen_losses.append(gen_loss.item())
                Mg, Mo = self.convert2(Mg), self.convert2(Mo)
                N, H, W, C = Mg.shape
                auc.append(
                    roc_auc_score((Mg.reshape(N * H * W * C) / 255).astype('int'), Mo.reshape(N * H * W * C) / 255.))
                Mo[Mo < 127.5] = 0
                Mo[Mo >= 127.5] = 255
                a, b = metric(Mo / 255, Mg / 255)
                f1.append(a)
                iou.append(b)
                print('Tra (%d/%d): G:%5.4f AUC:%5.4f F1:%5.4f IOU:%5.4f SUM:%5.4f'
                      % (cnt, self.train_num, np.mean(gen_losses), np.mean(auc), np.mean(f1), np.mean(iou),
                         np.mean(auc) + np.mean(f1) + np.mean(iou)), end='\r')
                if cnt % 20 == 0:
                    val_gen_loss, val_auc, val_f1, val_iou = self.val()
                    tmp_score = val_auc + val_f1 + val_iou
                    scheduler.step(tmp_score)
                    print('Val (%d/%d): G:%5.4f AUC:%5.4f F1:%5.4f IOU:%5.4f SUM:%5.4f'
                          % (cnt, self.train_num, val_gen_loss, val_auc, val_f1, val_iou, val_auc + val_f1 + val_iou))
                    self.model.save('latest_' + gpu_ids[0] + '/')
                    if tmp_score > best_score:
                        best_score = tmp_score
                        self.model.save('best_' + gpu_ids[0] + '/')
                    with open('log_' + gpu_ids[0] + '.txt', 'a+') as f:
                        f.write('\n(%5d/%5d): Tra: G:%5.4f AUC:%5.4f F1:%5.4f IOU:%5.4f SUM:%5.4f   ' % (
                        cnt, self.train_num, np.mean(gen_losses), np.mean(auc), np.mean(f1), np.mean(iou),
                        np.mean(auc) + np.mean(f1) + np.mean(iou)))
                        f.write('Val: G:%5.4f AUC:%5.4f F1:%5.4f IOU:%5.4f SUM:%5.4f ' % (
                        val_gen_loss, val_auc, val_f1, val_iou, val_auc + val_f1 + val_iou))
            cnt, gen_losses, auc, f1, iou = 0, [], [], [], []

    def val(self):
        self.model.eval()
        auc, f1, iou, gen_losses = [], [], [], []
        for cnt, items in enumerate(self.val_loader):
            Ii, Mg = (item.cuda() for item in items[:-1])
            filename = items[-1]
            Mo, gen_loss = self.model.process(Ii, Mg, eva=True)
            gen_losses.append(gen_loss.item())
            Ii, Mg, Mo = self.convert1(Ii), self.convert2(Mg), self.convert2(Mo)
            N, H, W, C = Mg.shape
            auc.append(roc_auc_score((Mg.reshape(N * H * W * C) / 255).astype('int'), Mo.reshape(N * H * W * C)))
            Mo[Mo < 127.5] = 0
            Mo[Mo >= 127.5] = 255
            a, b = metric(Mo / 255, Mg / 255)
            f1.append(a)
            iou.append(b)
            if cnt < 100:  # sample some results
                for i in range(len(Mo)):
                    rtn = np.ones([H, W * 3 + 20, 3], dtype=np.uint8) * 255
                    rtn[:, :W, :] = Ii[i][..., ::-1]
                    rtn[:, W * 1 + 10:W * 2 + 10, :] = np.concatenate([Mg[i], Mg[i], Mg[i]], axis=2)
                    rtn[:, W * 2 + 20:W * 3 + 20, :] = np.concatenate(
                        [Mo[i][:, :, 0:1], Mo[i][:, :, 0:1], Mo[i][:, :, 0:1]], axis=2)
                    cv2.imwrite('res/val_' + gpu_ids[0] + '/' + filename[i], rtn)
        return np.mean(gen_losses), np.mean(auc), np.mean(f1), np.mean(iou)

    def convert1(self, x):
        x = x * 127.5 + 127.5
        x = x.permute(0, 2, 3, 1).cpu().detach().numpy()
        return x

    def convert2(self, x):
        x = x * 255.
        x = x.permute(0, 2, 3, 1).cpu().detach().numpy()
        return x


def forensics_test(model):
    test_size = '896'
    test_path = 'data/input/'
    decompose(test_path, test_size)
    print('Decomposition complete.')
    test_dataset = MyDataset(test_path='temp/input_decompose_' + test_size + '/', size=int(test_size))
    path_out = 'temp/input_decompose_' + test_size + '_pred/'
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)
    rm_and_make_dir(path_out)
    for items in test_loader:
        Ii, Mg = (item.cuda() for item in items[:-1])
        filename = items[-1]
        Mo = model(Ii)
        Mo = Mo * 255.
        Mo = Mo.permute(0, 2, 3, 1).cpu().detach().numpy()
        for i in range(len(Mo)):
            Mo_tmp = Mo[i][..., ::-1]
            cv2.imwrite(path_out + filename[i][:-4] + '.png', Mo_tmp)
    print('Prediction complete.')
    if os.path.exists('temp/input_decompose_' + test_size + '/'):
        shutil.rmtree('temp/input_decompose_' + test_size + '/')
    path_pre = merge(test_path, test_size)
    print('Merging complete.')

    path_gt = 'data/mask/'
    if os.path.exists(path_gt):
        flist = sorted(os.listdir(path_pre))
        auc, f1, iou = [], [], []
        for file in flist:
            pre = cv2.imread(path_pre + file)
            gt = cv2.imread(path_gt + file[:-4] + '.png')
            H, W, C = pre.shape
            Hg, Wg, C = gt.shape
            if H != Hg or W != Wg:
                gt = cv2.resize(gt, (W, H))
                gt[gt > 127] = 255
                gt[gt <= 127] = 0
            if np.max(gt) != np.min(gt):
                auc.append(roc_auc_score((gt.reshape(H * W * C) / 255).astype('int'), pre.reshape(H * W * C) / 255.))
            pre[pre > 127] = 255
            pre[pre <= 127] = 0
            a, b = metric(pre / 255, gt / 255)
            f1.append(a)
            iou.append(b)
        print('Evaluation: AUC: %5.4f, F1: %5.4f, IOU: %5.4f' % (np.mean(auc), np.mean(f1), np.mean(iou)))
    return 0


def decompose(test_path, test_size):
    flist = sorted(os.listdir(test_path))
    size_list = [int(test_size)]
    for size in size_list:
        path_out = 'temp/input_decompose_' + str(size) + '/'
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
        path_out = 'temp/input_decompose_' + str(size) + '/'
        X, Y = H // (size // 2) + 1, W // (size // 2) + 1
        idx = 0
        for x in range(X - 1):
            if x * size // 2 + size > H:
                break
            for y in range(Y - 1):
                if y * size // 2 + size > W:
                    break
                img_tmp = img[x * size // 2: x * size // 2 + size, y * size // 2: y * size // 2 + size, :]
                cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
                idx += 1
            img_tmp = img[x * size // 2: x * size // 2 + size, -size:, :]
            cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
            idx += 1
        for y in range(Y - 1):
            if y * size // 2 + size > W:
                break
            img_tmp = img[-size:, y * size // 2: y * size // 2 + size, :]
            cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
            idx += 1
        img_tmp = img[-size:, -size:, :]
        cv2.imwrite(path_out + file[:-4] + '_%03d.png' % idx, img_tmp)
        idx += 1
    return rtn_list


def merge(path, test_size):
    path_d = 'temp/input_decompose_' + test_size + '_pred/'
    path_r = 'data/output/'
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
                img_tmp = cv2.imread(path_d + file[:-4] + '_%03d.png' % idx)
                weight_cur = copy.deepcopy(
                    rtn[x * size // 2: x * size // 2 + size, y * size // 2: y * size // 2 + size, :])
                h1, w1, _ = weight_cur.shape
                gk_tmp = cv2.resize(gk, (w1, h1))
                weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
                weight_cur[weight_cur == -1] = 0
                weight_tmp = copy.deepcopy(weight_cur)
                weight_tmp = 1 - weight_tmp
                rtn[x * size // 2: x * size // 2 + size, y * size // 2: y * size // 2 + size, :] = weight_cur * rtn[
                                                                                                                x * size // 2: x * size // 2 + size,
                                                                                                                y * size // 2: y * size // 2 + size,
                                                                                                                :] + weight_tmp * img_tmp
                idx += 1
            img_tmp = cv2.imread(path_d + file[:-4] + '_%03d.png' % idx)
            weight_cur = copy.deepcopy(rtn[x * size // 2: x * size // 2 + size, -size:, :])
            h1, w1, _ = weight_cur.shape
            gk_tmp = cv2.resize(gk, (w1, h1))
            weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
            weight_cur[weight_cur == -1] = 0
            weight_tmp = copy.deepcopy(weight_cur)
            weight_tmp = 1 - weight_tmp
            rtn[x * size // 2: x * size // 2 + size, -size:, :] = weight_cur * rtn[x * size // 2: x * size // 2 + size,
                                                                               -size:, :] + weight_tmp * img_tmp
            idx += 1
        for y in range(Y - 1):
            if y * size // 2 + size > W:
                break
            img_tmp = cv2.imread(path_d + file[:-4] + '_%03d.png' % idx)
            weight_cur = copy.deepcopy(rtn[-size:, y * size // 2: y * size // 2 + size, :])
            h1, w1, _ = weight_cur.shape
            gk_tmp = cv2.resize(gk, (w1, h1))
            weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
            weight_cur[weight_cur == -1] = 0
            weight_tmp = copy.deepcopy(weight_cur)
            weight_tmp = 1 - weight_tmp
            rtn[-size:, y * size // 2: y * size // 2 + size, :] = weight_cur * rtn[-size:,
                                                                               y * size // 2: y * size // 2 + size,
                                                                               :] + weight_tmp * img_tmp
            idx += 1
        img_tmp = cv2.imread(path_d + file[:-4] + '_%03d.png' % idx)
        weight_cur = copy.deepcopy(rtn[-size:, -size:, :])
        h1, w1, _ = weight_cur.shape
        gk_tmp = cv2.resize(gk, (w1, h1))
        weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
        weight_cur[weight_cur == -1] = 0
        weight_tmp = copy.deepcopy(weight_cur)
        weight_tmp = 1 - weight_tmp
        rtn[-size:, -size:, :] = weight_cur * rtn[-size:, -size:, :] + weight_tmp * img_tmp
        idx += 1
        # rtn[rtn < 127] = 0
        # rtn[rtn >= 127] = 255
        cv2.imwrite(path_r + file[:-4] + '.png', np.uint8(rtn))
    return path_r


def gkern(kernlen=7, nsig=3):
    """Returns a 2D Gaussian kernel."""
    # x = np.linspace(-nsig, nsig, kernlen+1)
    # kern1d = np.diff(st.norm.cdf(x))
    # kern2d = np.outer(kern1d, kern1d)
    # rtn = kern2d/kern2d.sum()
    # rtn = np.concatenate([rtn[..., None], rtn[..., None], rtn[..., None]], axis=2)
    rtn = [[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]]
    rtn = np.array(rtn, dtype=np.float32)
    rtn = np.concatenate([rtn[..., None], rtn[..., None], rtn[..., None]], axis=2)
    rtn = cv2.resize(rtn, (kernlen, kernlen))
    return rtn


def rm_and_make_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def metric(premask, groundtruth):
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    cross = np.logical_and(premask, groundtruth)
    union = np.logical_or(premask, groundtruth)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)
    if np.sum(cross) + np.sum(union) == 0:
        iou = 1
    return f1, iou


if __name__ == '__main__':
    model = ForgeryForensics()
    model.train()
    # After convergence, the best checkpoint will be saved in 'weights/'
