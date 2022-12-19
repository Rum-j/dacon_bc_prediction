import os
import skimage.io
import numpy as np 
import pandas as pd 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
import cv2
from efficientnet_pytorch import model as enet

  
tile_size = 256
image_size = 256
n_tiles = 36
batch_size = 8
num_workers = 4
model_dir = './model/weight'
device = torch.device('cuda')

def feature_extraction(df, image_folder):   

    class enetv2(nn.Module):
        def __init__(self, backbone, out_dim):
            super(enetv2, self).__init__()
            self.enet = enet.EfficientNet.from_name(backbone)
            self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
            self.enet._fc = nn.Identity()

        def extract(self, x):
            return self.enet(x)

        def forward(self, x):
            x = self.extract(x)
            x = self.myfc(x)
            return x
        
        
    def load_models(model_files):
        models = []
        for model_f in model_files:
            model_f = os.path.join(model_dir, model_f)
            backbone = 'efficientnet-b0'
            model = enetv2(backbone, out_dim=5)
            model.load_state_dict(torch.load(model_f, map_location=lambda storage, loc: storage), strict=True)
            model.eval()
            model.to(device)
            models.append(model)
            
        return models


    model_files = [
        'cls_effnet_b0_Rand36r36tiles256_big_bce_lr0.3_augx2_30epo_model_fold0.pth'
    ]

    models = load_models(model_files)


    def get_tiles(img, mode=0):
            result = []
            h, w, c = img.shape
            pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
            pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

            img2 = np.pad(img,[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w//2], [0,0]], constant_values=255)
            img3 = img2.reshape(
                img2.shape[0] // tile_size,
                tile_size,
                img2.shape[1] // tile_size,
                tile_size,
                3
            )

            img3 = img3.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size,3)
            n_tiles_with_info = (img3.reshape(img3.shape[0],-1).sum(1) < tile_size ** 2 * 3 * 255).sum()
            if len(img) < n_tiles:
                img3 = np.pad(img3,[[0,N-len(img3)],[0,0],[0,0],[0,0]], constant_values=255)
            idxs = np.argsort(img3.reshape(img3.shape[0],-1).sum(-1))[:n_tiles]
            img3 = img3[idxs]
            for i in range(len(img3)):
                result.append({'img':img3[i], 'idx':i})
            return result, n_tiles_with_info >= n_tiles


    class Dataset(Dataset):
        def __init__(self,
                    df,
                    image_size,
                    n_tiles=n_tiles,
                    tile_mode=0,
                    rand=False,
                    sub_imgs=False
                    ):

            self.df = df.reset_index(drop=True)
            self.image_size = image_size
            self.n_tiles = n_tiles
            self.tile_mode = tile_mode
            self.rand = rand
            self.sub_imgs = sub_imgs

        def __len__(self):
            return self.df.shape[0]

        def __getitem__(self, index):
            row = self.df.iloc[index]
            img_id = row.ID
            
            png_file = os.path.join(image_folder, f'{img_id}.png')

            image = cv2.imread(png_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            tiles, OK = get_tiles(image, self.tile_mode)

            if self.rand:
                idxes = np.random.choice(list(range(self.n_tiles)), self.n_tiles, replace=False)
            else:
                idxes = list(range(self.n_tiles))
            idxes = np.asarray(idxes) + self.n_tiles if self.sub_imgs else idxes

            n_row_tiles = int(np.sqrt(self.n_tiles))
            images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))
            for h in range(n_row_tiles):
                for w in range(n_row_tiles):
                    i = h * n_row_tiles + w
        
                    if len(tiles) > idxes[i]:
                        this_img = tiles[idxes[i]]['img']
                    else:
                        this_img = np.ones((self.image_size, self.image_size, 3)).astype(np.uint8) * 255
                    this_img = 255 - this_img
                    h1 = h * image_size
                    w1 = w * image_size
                    images[h1:h1+image_size, w1:w1+image_size] = this_img

    
            images = images.astype(np.float32)
            images /= 255
            images = images.transpose(2, 0, 1)

            return torch.tensor(images)


    dataset1 = Dataset(df, image_size, n_tiles, 0)  # mode == 0
    loader1 = DataLoader(dataset1, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    dataset2 = Dataset(df, image_size, n_tiles, 2)  # mode == 2
    loader2 = DataLoader(dataset2, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    dataset3 = Dataset(df, image_size, n_tiles, 4)  # mode == 4
    loader3 = DataLoader(dataset3, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    dataset4 = Dataset(df, image_size, n_tiles, 6)  # mode == 6
    loader4 = DataLoader(dataset4, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    dataset5 = Dataset(df, image_size, n_tiles, 8)  # mode == 8
    loader5 = DataLoader(dataset5, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    dataset6 = Dataset(df, image_size, n_tiles, 10)  # mode == 10
    loader6 = DataLoader(dataset6, batch_size=batch_size, num_workers=num_workers, shuffle=False)



    LOGITS1,LOGITS2,LOGITS3,LOGITS4,LOGITS5,LOGITS6 = [],[],[],[],[],[]

    with torch.no_grad():
        for data in tqdm(loader1):
            data = data.to(device)
            logits = models[0](data)
            LOGITS1.append(logits)

        for data in tqdm(loader2):
            data = data.to(device)
            logits = models[0](data)
            LOGITS2.append(logits)
            
        for data in tqdm(loader3):
            data = data.to(device)
            logits = models[0](data)
            LOGITS3.append(logits)
            
        for data in tqdm(loader4):
            data = data.to(device)
            logits = models[0](data)
            LOGITS4.append(logits)
            
        for data in tqdm(loader5):
            data = data.to(device)
            logits = models[0](data)
            LOGITS5.append(logits)
            
        for data in tqdm(loader6):
            data = data.to(device)
            logits = models[0](data)
            LOGITS6.append(logits)

    LOGITS = (torch.cat(LOGITS1).sigmoid().cpu() + torch.cat(LOGITS2).sigmoid().cpu()
            + torch.cat(LOGITS3).sigmoid().cpu() + torch.cat(LOGITS4).sigmoid().cpu()
            + torch.cat(LOGITS5).sigmoid().cpu() + torch.cat(LOGITS6).sigmoid().cpu()) / 6

    for i in range(1,6):
        df[f'img_feature{i:03d}'] = LOGITS.numpy()[:,i-1]

    return df