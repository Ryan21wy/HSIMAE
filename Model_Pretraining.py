import os
import random
from tqdm import tqdm
import numpy as np
from scipy import ndimage

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.optim import AdamW
from timm.scheduler import CosineLRScheduler

from Models import HSIMAE
from Utils.Preprocessing import get_data_cut_file

import warnings
warnings.filterwarnings('ignore')


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    torch.cuda.manual_seed_all(seed)  # All GPU (Optional)
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization


def stable(dataloader, seed):
    seed_everything(seed)
    return dataloader


class HSIdataset4PT(data.Dataset):
    def __init__(self, data_cubes, train=False, device='cuda:0'):
        self.data_cubes = data_cubes[0]
        self.cut_info = data_cubes[1]
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

    def random_spectral_filp(self, data, r=0.5):
        if random.random() < r:
            return np.flip(data, 2)
        else:
            return data

    def random_spectral_crop_and_resize(self, data, range=(0.5, 1), r=0.5):
        if random.random() < r:
            s = np.random.uniform(range[0], range[1])
            h, w, len_band = data.shape
            new_len_band = int(len_band * s)
            if new_len_band < len_band:
                start_ = np.random.randint(0, len_band - new_len_band)
                data = data[:, :, start_:start_ + new_len_band]
                data = ndimage.zoom(data, (h, w, len_band) / np.array(data.shape))
            elif new_len_band > len_band:
                start_ = np.random.randint(0, new_len_band - len_band)
                data = ndimage.zoom(data, (h, w, new_len_band) / np.array(data.shape))
                data = data[:, :, start_:start_ + len_band]
            else:
                data = data
        return data

    def __getitem__(self, index):
        c, h, w, num, max_, min_ = self.cut_info[index]
        data_cube = self.data_cubes[num]
        data = data_cube[h: (h + 9), w: (w + 9), :]
        data = (data - min_) / (max_ - min_)

        if data.shape[-1] % 16 != 0:
            cut = np.random.randint(data.shape[-1] - 176)
            data = data[:, :, cut: cut + 176]

        if self.train:
            data = self.random_horizontal_filp(data)
            data = self.random_vertical_filp(data)
            # data = self.random_spectral_crop_and_resize(data, range=(0.5, 2))
        data = torch.tensor(data.copy(), dtype=torch.float32)
        data = data.unsqueeze(0).permute(0, 3, 1, 2)
        return data

    def __len__(self):
        return len(self.cut_info)


def mask_pretraining(data_cubes, save_path, model_name, img_size=9, bands=176,
                     mask_ratio=0.90, lr=2.5e-4, wd=5e-2, bs=512, epoch=100,
                     depth=12, dim=64, dec_dim=48, dec_depth=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = HSIdataset4PT(data_cubes, train=True)
    val_dataset = HSIdataset4PT([d[:100] for d in data_cubes])

    del data_cubes

    print('dataset load finished')
    print('训练集大小：' + str(len(train_dataset)))
    print('网络参数：', [dim, depth, dec_dim, dec_depth])

    model = HSIMAE(img_size=img_size, patch_size=3, in_chans=1, bands=bands, b_patch_size=16,
                   embed_dim=dim, depth=depth, num_heads=dim // 16,
                   decoder_embed_dim=dec_dim, decoder_depth=dec_depth, decoder_num_heads=dec_dim // 8,
                   norm_pix_loss=True, trunc_init=True, sep_pos_embed=True, use_learnable_pos_emb=True,
                   cls_embed=False).to(device)

    if not os.path.exists(save_path):
        print(save_path)
        os.makedirs(save_path)

    epoch_total = epoch
    base_lr = lr
    lr = base_lr * bs / 256

    train_dataload = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)
    val_dataload = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)
    print('batch load finished')
    print('训练轮次：' + str(len(train_dataload)))

    no_decay = ['bias', 'norm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': wd},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, weight_decay=wd)
    iters = epoch_total * len(train_dataload)
    scheduler = CosineLRScheduler(optimizer, t_initial=iters, lr_min=1e-5, warmup_t=int(np.ceil(iters * 0.05)),
                                  warmup_lr_init=1e-6)

    epoch_loss_list = []
    val_loss_list = []
    iter_num = 0
    for epoch in tqdm(range(epoch_total)):
        train_loss = 0
        model.train()
        for x in tqdm(stable(train_dataload, 42 + epoch)):
            inputs = x.to(device)
            loss, outputs, _ = model(inputs, mask_ratio=mask_ratio)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step(iter_num)
            iter_num += 1
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for j, x in enumerate(stable(val_dataload, 42 + epoch)):
                inputs = x.to(device)
                loss, outputs, _ = model(inputs, mask_ratio=mask_ratio)
                val_loss += loss.item()

        tloss = train_loss / len(train_dataload)
        epoch_loss_list.append(tloss)

        vloss = val_loss / len(val_dataload)
        val_loss_list.append(vloss)

    if (epoch + 1) == epoch_total:
        torch.save(model.state_dict(), os.path.join(save_path, model_name))

    history = np.array([epoch_loss_list, val_loss_list])
    np.save(save_path + '/loss.npy', history)


def get_file_list(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if 'max_min' not in file:
                file_list.append(os.path.join(root, file))
    return file_list


if __name__ == "__main__":
    seed_everything(42)

    data_paths = []
    data_dir = r'E:\hyspecnet-11k\numpy_files'
    file_list = get_file_list(data_dir)

    indices = np.arange(len(file_list))
    shuffled_indices = np.random.permutation(indices)
    use_indices = shuffled_indices[:int(0.001 * len(shuffled_indices))]
    for i, ind in enumerate(use_indices):
        data_paths.append(file_list[ind])

    img_size = 9
    data_cubes = get_data_cut_file(data_paths, patch_size=img_size, norm=False, MSPCA=True)

    bands = 64
    mask_ratio = 0.8
    lr = 2.5e-4
    wd = 5e-2
    batch_size = 1024
    epoch = 10
    paras = [12, 144]  # [12, 144] for HSIMAE-Base, [12, 256] for Large, [12 ,512] for Huge
    dec_paras = [1, 72]

    save_path = r'D:\dataset\HSIMAE\test'
    model_name = 'HSIMAE_B.pkl'

    seed_everything(42)
    mask_pretraining(data_cubes,
                     save_path,
                     model_name,
                     img_size=img_size,
                     bands=bands,
                     bs=batch_size,
                     mask_ratio=mask_ratio,
                     lr=lr,
                     wd=wd,
                     epoch=epoch,
                     depth=paras[0],
                     dim=paras[1],
                     dec_depth=dec_paras[0],
                     dec_dim=dec_paras[1],
                     )