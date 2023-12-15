import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision.transforms import transforms as T
from torch.utils.data import DataLoader
from torch.optim import AdamW, RMSprop

import os
import matplotlib.pyplot as plt
import matplotlib.image as mi
from tqdm import tqdm
from time import time
import random

from Utils.Early_Stop import EarlyStopping
from timm.scheduler import CosineLRScheduler

from sklearn import metrics
from sklearn.decomposition import PCA

from scipy import ndimage

from Utils.Preprocessing import splitHSI, get_data_set, spilt_dataset
from Utils.Label_to_Colormap import label_to_colormap
from Utils.Seed_Everything import seed_everything, stable

import warnings
warnings.filterwarnings('ignore')

from Compared_Methods.SSFTT import SSFTTnet
from Compared_Methods.SpectralFormer import ViT
from Compared_Methods.HiT import HiT
from Compared_Methods.DBDA import DBDA
from Compared_Methods.RSSAN import RSSAN
from Compared_Methods.FDSSC import FDSSC
from Compared_Methods.SSRN import SSRN
from Compared_Methods.HybridFormer import HybridFormer


class HSIdataset(data.Dataset):
    def __init__(self, data_cubes, gt=None, train=False, device='cuda:0'):
        self.data_cubes = data_cubes
        self.gt = gt
        self.train = train
        self.device = device

    def random_horizontal_filp(self, data, r=0.5):
        if random.random() < r:
            return np.flip(data, 1)
        else:
            return data

    def random_vertical_filp(self, data, r=0.5):
        if random.random() < r:
            return np.flip(data, 0)
        else:
            return data

    def __getitem__(self, index):
        if self.train:
            data = self.data_cubes[index]
            data = self.random_horizontal_filp(data)
            data = self.random_vertical_filp(data)
        else:
            data = self.data_cubes[index]
        data = torch.tensor(data.copy(), dtype=torch.float32)
        data = data.permute(2, 0, 1)
        if self.gt is not None:
            gt = self.gt[index]
            return data, gt
        return data

    def __len__(self):
        return len(self.data_cubes)


def data_trans(data_path, pca_nc=None, whiten=False, norm=None, resize=None):
    HSI_data_raw = np.load(data_path)
    if pca_nc is not None:
        data = applyPCA(HSI_data_raw, pca_nc, whiten=whiten)
    else:
        data = HSI_data_raw
    if norm is not None:
        data = (data - data.min()) / (data.max() - data.min())
        data = data * (norm[0] - norm[1]) + norm[1]
    if resize is not None:
        h, w, c = resize
        data = ndimage.zoom(data, (h, w, c) / np.array(data.shape))
    print(data.max(), data.min())
    return data


def applyPCA(X, numComponents, whiten=True):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=whiten)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX


def training(data_list, gt, model, save_path, model_name, lr=1e-3, wd=0., bs=64, epochs=100, early_stopping=False, pretrained=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    h, w, c = data_list[0].shape

    n_class = np.max(gt) + 1
    print('number of class: ', n_class)

    train_data, train_gt, val_data, val_gt = spilt_dataset(data_list, gt, training_ratio=0.5)

    train_dataset = HSIdataset(train_data, train_gt, train=True)
    val_dataset = HSIdataset(val_data, val_gt)
    print('dataset load finished')
    print('训练集大小：' + str(len(train_dataset)))

    model = model.to(device)

    if pretrained:
        pre_keys = []
        ignore_keys = []
        state_dict = {}
        model_dict = model.state_dict()
        pretrain_model_para = torch.load(pretrained, map_location=device)
        for key, v in pretrain_model_para.items():
            if (key in model_dict.keys()) & (key not in ignore_keys):
                state_dict[key] = v
                pre_keys.append(key)
        print(pre_keys)
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_dataload = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
    val_dataload = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    print('batch load finished')
    print('训练轮次：' + str(len(train_dataload)))

    # optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    optimizer = RMSprop(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

    # scheduler = CosineLRScheduler(optimizer, t_initial=epochs, lr_min=1e-5, warmup_t=int(np.ceil(0.1 * epochs)), warmup_lr_init=1e-6)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 10, gamma=0.9)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    early_stop = EarlyStopping(50, delta=0)

    epoch_loss_list = []
    val_loss_list = []
    epoch_AA_list = []
    val_AA_list = []
    grad_norms = []
    iter_num = 0

    fig = plt.figure()
    ax1 = plt.subplot(111)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Avarage Accuracy')

    for epoch in tqdm(range(epochs)):
        train_loss = 0
        model.train()
        pred = np.zeros(1)
        gt_ = np.zeros(1)
        for i, (x, y) in enumerate(stable(train_dataload, 42 + epoch)):
            inputs = x.to(device)
            outputs = model(inputs)
            targets = y.long().to(device)
            loss = criterion(outputs, targets)

            outputs = outputs.detach().cpu().numpy()
            output = np.argmax(outputs, axis=1)
            pred = np.concatenate([pred, output], axis=0)

            gt = y.numpy()
            gt_ = np.concatenate([gt_, gt], axis=0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            train_loss += loss.item()

        pred = pred[1:]
        gt_ = gt_[1:]
        gt_label = gt_[gt_ != 0] - 1
        pred_label = pred[gt_ != 0] - 1

        oa = metrics.accuracy_score(gt_label, pred_label)
        aa = np.mean(metrics.recall_score(gt_label, pred_label, average=None))
        kappa = metrics.cohen_kappa_score(gt_label, pred_label)
        train_aa = (oa + aa + kappa) / 3
        epoch_AA_list.append(train_aa)

        tloss = train_loss / len(train_dataload)
        epoch_loss_list.append(tloss)

        model.eval()
        pred = np.zeros(1)
        gt_ = np.zeros(1)
        with torch.no_grad():
            val_loss = 0
            for j, (x, y) in enumerate(stable(val_dataload, 42 + epoch)):
                inputs = x.to(device)
                outputs = model(inputs)
                targets = y.long().to(device)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

                outputs = outputs.detach().cpu().numpy()
                output = np.argmax(outputs, axis=1)
                pred = np.concatenate([pred, output], axis=0)

                gt = y.numpy()
                gt_ = np.concatenate([gt_, gt], axis=0)

        pred = pred[1:]
        gt_ = gt_[1:]
        gt_label = gt_[gt_ != 0] - 1
        pred_label = pred[gt_ != 0] - 1

        oa = metrics.accuracy_score(gt_label, pred_label)
        aa = np.mean(metrics.recall_score(gt_label, pred_label, average=None))
        kappa = metrics.cohen_kappa_score(gt_label, pred_label)
        val_value = [oa, aa, kappa]

        val_aa = (oa + aa + kappa) / 3
        vloss = val_loss / len(val_dataload)

        val_AA_list.append(val_aa)
        val_loss_list.append(vloss)

        if early_stopping:
            early_stop(val_aa, val_value, model, os.path.join(save_path, model_name))
            if early_stop.early_stop:
                break
        scheduler.step(val_aa)
    if (epoch + 1) == epochs:
        torch.save(model.state_dict(), os.path.join(save_path, model_name))

    ax1.cla()
    ax2.cla()
    ln1 = ax1.plot(epoch_loss_list, 'b', lw=1, label='train loss')
    ln2 = ax1.plot(val_loss_list, 'g', lw=1, label='val loss')
    ln3 = ax2.plot(epoch_AA_list, 'y', lw=1, label='train aa')
    ln4 = ax2.plot(val_AA_list, 'r', lw=1, label='val aa')
    lns = ln1 + ln2 + ln3 + ln4
    labels = [l.get_label() for l in lns]
    plt.legend(lns, labels, loc='center right')
    plt.pause(0.1)

    plt.savefig(save_path + '/finetune_loss_' + str(lr) + '.png')
    plt.close()
    return model, val_value, epoch_loss_list, val_loss_list


def test_model(data_cubes, test_gt, gt, model, model_path, save_path=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, _, c = data_cubes[0].shape

    dataset = HSIdataset(data_cubes)

    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_dataload = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)

    pred = np.zeros(1)
    latents = np.zeros((1, 64 * 7))
    with torch.no_grad():
        for x in tqdm(test_dataload):
            inputs = x.to(device)
            outputs = model(inputs)
            outputs = outputs.detach().cpu().numpy()
            output = np.argmax(outputs[:, 1:], axis=1)
            pred = np.concatenate([pred, output], axis=0)

    pred = pred[1:] + 1
    pred = pred.reshape(gt.shape)
    colormap_all = label_to_colormap(pred)

    pred[gt == 0] = 0
    colormap = label_to_colormap(pred)

    gt_ = test_gt.reshape(-1)
    gt_label = gt_[gt_ != 0] - 1

    pred_ = pred.reshape(-1)
    pred_label = pred_[gt_ != 0] - 1

    oa = metrics.accuracy_score(gt_label, pred_label)
    aa = np.mean(metrics.recall_score(gt_label, pred_label, average=None))
    kappa = metrics.cohen_kappa_score(gt_label, pred_label)
    ca = metrics.recall_score(gt_label, pred_label, average=None)

    if save_path:
        mi.imsave(os.path.join(save_path, model_name.replace('.pkl', '_all_oa_' + str(np.around(oa * 100, 2)) + '.png')), colormap_all)
        mi.imsave(os.path.join(save_path, model_name.replace('.pkl', '_oa_' + str(np.around(oa * 100, 2)) + '.png')), colormap)
    return oa, aa, kappa, ca


def get_metrics(predict_label, true_label, nclass):
    confusion_matrix = metrics.confusion_matrix(true_label, predict_label)
    overall_accuracy = metrics.accuracy_score(true_label, predict_label)

    true_cla = np.zeros(nclass, dtype=np.int64)
    for i in range(nclass):
        true_cla[i] = confusion_matrix[i, i]
    test_num_class = np.sum(confusion_matrix, 1)
    test_num = np.sum(test_num_class)
    num1 = np.sum(confusion_matrix, 0)
    po = overall_accuracy
    pe = np.sum(test_num_class * num1) / (test_num * test_num)
    kappa = (po - pe) / (1 - pe) * 100
    true_cla = np.true_divide(true_cla, test_num_class) * 100
    average_accuracy = np.average(true_cla)
    return overall_accuracy, average_accuracy, kappa


def get_data_path(name):
    if name == 'Salinas':
        data_path = r"D:\dataset\HSIMAE\Dataset\Salinas\data.npy"
        gt_path = r"D:\dataset\HSIMAE\Dataset\Salinas\gt.npy"
    elif name == 'PaviaU':
        data_path = r"D:\dataset\HSIMAE\Dataset\PaviaU\data.npy"
        gt_path = r"D:\dataset\HSIMAE\Dataset\PaviaU\gt.npy"
    elif name == 'Houston2013':
        data_path = r"D:\dataset\HSIMAE\Dataset\Houston2013\data.npy"
        gt_path = r"D:\dataset\HSIMAE\Dataset\Houston2013\gt.npy"
    elif name == 'LongKou':
        data_path = r"D:\dataset\HSIMAE\Dataset\WHU-Hi-LongKou\data.npy"
        gt_path = r"D:\dataset\HSIMAE\Dataset\WHU-Hi-LongKou\gt.npy"
    return data_path, gt_path


def get_model(model_name, dataset):
    if model_name == 'HybridFormer':
        if dataset == 'Salinas':
            model = HybridFormer(image_size=15, patch_size=(3, 5), num_classes=17, dim=100, depth=2, heads=4,
                                 mlp_dim=2048, channels=204, dropout=0.2, emb_dropout=0.2)
        elif dataset == 'PaviaU':
            model = HybridFormer(image_size=15, patch_size=(3, 5), num_classes=10, dim=100, depth=2, heads=4,
                                 mlp_dim=2048, channels=103, dropout=0.2, emb_dropout=0.2)
        elif dataset == 'Houston2013':
            model = HybridFormer(image_size=15, patch_size=(3, 5), num_classes=16, dim=100, depth=2, heads=4,
                                 mlp_dim=2048, channels=144, dropout=0.2, emb_dropout=0.2)
        elif dataset == 'LongKou':
            model = HybridFormer(image_size=15, patch_size=(3, 5), num_classes=10, dim=100, depth=2, heads=4,
                                 mlp_dim=2048, channels=270, dropout=0.2, emb_dropout=0.2)

    if model_name == 'SSFTT':
        if dataset == 'Salinas':
            model = SSFTTnet(1, 17, kennel_3D=8, kennel_2D=64, num_tokens=4, dim=64, heads=4)
        elif dataset == 'PaviaU':
            model = SSFTTnet(1, 10, kennel_3D=8, kennel_2D=32, num_tokens=4, dim=64, heads=4)
        elif dataset == 'Houston2013':
            model = SSFTTnet(1, 16, kennel_3D=24, kennel_2D=48, num_tokens=4, dim=64, heads=4)
        elif dataset == 'LongKou':
            model = SSFTTnet(1, 10, kennel_3D=8, kennel_2D=64, num_tokens=4, dim=64, heads=4)

    elif model_name == 'HiT':
        if dataset == 'Salinas':
            model = HiT(img_size=15, in_chans=204, num_classes=17, embed_dims=[408, 408, 512, 512])
        elif dataset == 'PaviaU':
            model = HiT(img_size=15, in_chans=103, num_classes=10, embed_dims=[208, 208, 512, 512])
        elif dataset == 'Houston2013':
            model = HiT(img_size=15, in_chans=144, num_classes=16, embed_dims=[288, 288, 512, 512])
        elif dataset == 'LongKou':
            model = HiT(img_size=15, in_chans=270, num_classes=10, embed_dims=[544, 544, 512, 512])

    elif model_name == 'SF':
        if dataset == 'Salinas':
            model = ViT(7, 3, 204, 17, dim=64, depth=5, heads=4, mlp_dim=8, dropout=0.1, emb_dropout=0.1, mode='CAF')
        elif dataset == 'PaviaU':
            model = ViT(7, 3, 103, 10, dim=64, depth=5, heads=4, mlp_dim=8, dropout=0.1, emb_dropout=0.1, mode='CAF')
        elif dataset == 'Houston2013':
            model = ViT(7, 3, 144, 16, dim=64, depth=5, heads=4, mlp_dim=8, dropout=0.1, emb_dropout=0.1, mode='CAF')
        elif dataset == 'LongKou':
            model = ViT(7, 3, 270, 10, dim=64, depth=5, heads=4, mlp_dim=8, dropout=0.1, emb_dropout=0.1, mode='CAF')

    elif model_name == 'RSSAN':
        # wd 1e-4, epoch 200
        if dataset == 'Salinas':
            model = RSSAN(17, 204, 3, 16)
        elif dataset == 'PaviaU':
            model = RSSAN(10, 103, 3, 16)
        elif dataset == 'Houston2013':
            model = RSSAN(16, 144, 3, 16)
        elif dataset == 'LongKou':
            model = RSSAN(10, 270, 3, 16)

    elif model_name == 'DBDA':
        if dataset == 'Salinas':
            model = DBDA(204, 17)
        elif dataset == 'PaviaU':
            model = DBDA(103, 10)
        elif dataset == 'Houston2013':
            model = DBDA(144, 16)
        elif dataset == 'LongKou':
            model = DBDA(270, 10)

    elif model_name == 'FDSSC':
        if dataset == 'Salinas':
            model = FDSSC(204, 17)
        elif dataset == 'PaviaU':
            model = FDSSC(103, 10)
        elif dataset == 'Houston2013':
            model = FDSSC(144, 16)
        elif dataset == 'LongKou':
            model = FDSSC(270, 10)

    elif model_name == 'SSRN':
        if dataset == 'Salinas':
            model = SSRN(204, 17)
        elif dataset == 'PaviaU':
            model = SSRN(103, 10)
        elif dataset == 'Houston2013':
            model = SSRN(144, 16)
        elif dataset == 'LongKou':
            model = SSRN(270, 10)

    return model


if __name__ == "__main__":
    seeds = [3407, 3408, 3409, 3410, 3411]
    seed_everything(seeds[0])

    datasets = ['Salinas']
    model_ = 'SSRN'

    label_nums = [100]
    patch_sizes = [9, 9, 9, 9]
    results = [[], [], [], []]

    for k, dataset in enumerate(datasets):
        print('current dataset: ', dataset)
        data_path, gt_path = get_data_path(dataset)

        save_path_1 = r'D:\results/' + model_ + '/' + dataset
        save_path = r'D:\results\compared_results/' + '/' + dataset
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_name = model_ + '_' + dataset + '.pkl'

        HSI_data = data_trans(data_path, norm=(1, 0))
        # HSI_data = data_trans(data_path, pca_nc=30, whiten=True)

        wd = 0
        bs = 16
        epochs = 200

        for l_num in label_nums:

            seed_everything(seeds[0])
            train_set, train_gt, test_set, test_gt, all_gt = get_data_set(HSI_data,
                                                                          gt_path,
                                                                          patch_size=patch_sizes[k],
                                                                          num=l_num,)
                                                                          # mask=r"D:\dataset\HSIMAE\Dataset\WHU-Hi-LongKou\Train100.npy")

            lrs = [1e-3, 5e-4, 1e-4, 5e-5]
            best_score = []

            for lr in lrs:
                val_results = []

                for i in range(3):
                    seed_everything(seeds[i])
                    model = get_model(model_, dataset)

                    _, [oa, aa, kappa], train_loss, val_loss = training(train_set, train_gt, model, save_path_1, model_name,
                                                                        lr=lr, wd=wd, bs=bs, epochs=epochs, early_stopping=True)
                    val_results.append([oa, aa, kappa])

                val_results = np.array(val_results)
                val_mean = np.mean(val_results, axis=0)
                val_std = np.std(val_results, axis=0)

                if best_score:
                    if best_score[0].mean() < val_mean.mean():
                        best_score = [val_mean, val_std, lr]
                    else:
                        best_score = best_score
                else:
                    best_score = [val_mean, val_std, lr]

            lr = best_score[2]
            test_results = []
            test_results_per_class = []

            for i in range(5):
                seed_everything(seeds[i])
                model = get_model(model_, dataset)

                _, [oa, aa, kappa], train_loss, val_loss = training(train_set, train_gt, model, save_path_1, model_name,
                                                                    lr=lr, wd=wd, bs=bs, epochs=epochs, early_stopping=True)

                oa, aa, kappa, ca = test_model(test_set, test_gt, all_gt, model, os.path.join(save_path_1, model_name), save_path)

                test_results.append([oa, aa, kappa])
                test_results_per_class.append(ca)

            test_results = np.array(test_results)
            test_mean = np.mean(test_results, axis=0) * 100
            test_std = np.std(test_results, axis=0) * 100

            class_accuracy_mean = np.mean(test_results_per_class, axis=0) * 100
            class_accuracy_std = np.std(test_results_per_class, axis=0) * 100

            results[k].append([lr, class_accuracy_mean, test_mean, test_std])

    for i, result in enumerate(results):
        print('current dataset: ', datasets[i])
        for j, r in enumerate(result):
            print('parameter is: ')
            print(label_nums[j])

            print('best learning rate: ')
            print(r[0])

            print('class_accuracy:')
            for ca in class_accuracy_mean:
                print(np.around(ca, 2))

            print('test oa, aa, kappa:')
            for tm in r[2]:
                print(np.around(tm, 2))
            for ts in r[3]:
                print(np.around(ts, 2))