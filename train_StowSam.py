# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""
# %% Import packages

import os
join = os.path.join
import argparse
import glob
import shutil
import h5py
import random
from tqdm import tqdm
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from skimage import transform

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize

import monai
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# %% box/mask visualization
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--h5_path",
    type=str,
    default="dataset/StowSam/input/train/",
    help="path to training h5 files; two subfolders: gts and imgs",
)
parser.add_argument(
    "-task_name",
    type=str,
    default="StowSAM-ViT-B"
)
parser.add_argument(
    "-model_type",
    type=str,
    default="vit_b"
)
parser.add_argument(
    "-checkpoint",
    type=str,
    default="work_dir/SAM/sam_vit_b_01ec64.pth"
)
parser.add_argument(
    "--load_pretrain",
    type=bool,
    default=True,
    help="use wandb to monitor training"
)
parser.add_argument(
    "-pretrain_model_path",
    type=str,
    default=""
)
parser.add_argument(
    "-work_dir",
    type=str,
    default="./work_dir"
)
# Train
parser.add_argument(
    "-num_epochs",
    type=int,
    default=10
)
parser.add_argument(
    "-batch_size",
    type=int,
    default=2
)
parser.add_argument(
    "-num_workers",
    type=int,
    default=0
)
# Optimizer parameters
parser.add_argument(
    "-weight_decay",
    type=float,
    default=0.01,
    help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr",
    type=float,
    default=0.0001,
    metavar="LR",
    help="learning rate (absolute lr)"
)
parser.add_argument(
    "-use_wandb",
    type=bool,
    default=False,
    help="use wandb to monitor training"
)
parser.add_argument(
    "-use_amp",
    action="store_true",
    default=False,
    help="use amp"
)
parser.add_argument(
    "--resume",
    type=str,
    default="",
    help="Resuming training from checkpoint"
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0"
)
args = parser.parse_args()

if args.use_wandb:
    import wandb

    wandb.login()
    wandb.init(
        project=args.task_name,
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )

## Load the dataset
train_path = '/home/sebastiangabriel/Dev/SAM/StowSAM/dataset/bin_syn/train_shard_000000.h5'
model_type ='vit_b'
checkpoint = 'work_dir/SAM/sam_vit_b_01ec64.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 0.0001
weight_decay = 0.01

# train_dataset.keys()
# # Load the training data
# for key in train_dataset.keys(): 
#     print(key) 

# train_set_x = train_dataset['data']
# print(train_set_x.shape)
# print(train_set_x[0][0].shape)
# image = train_set_x[0][0]
# image = image[:, :,:3]
# print(image.shape)

# %% h5 Dataset Class
class h5Dataset(Dataset):
    def __init__(self, data_set_path, model):
        self.data_root = data_set_path
        self.model = model
        self.dataset_dict = self.load_dict(self.data_root)
      
    def __len__(self):
        return len(self.dataset_dict["data"])
    
    def load_dict(self, file_path):
        print("loading data from", file_path)
        h5f = h5py.File(file_path, "r")
        imgs= h5f["data"]
        gts = h5f["mask"]
        depth = h5f["depth"]
        metadata = h5f["metadata"]

        dataset_dict = {
            "data": imgs,
            "mask": gts,
            "depth": depth,
            "metadata": metadata
        }

        return dataset_dict

    def __getitem__(self, index):
        image = self.dataset_dict["data"][index][0]
        mask = self.dataset_dict["mask"][index][0]
        # depth = self.dataset_dict["depth"][index][0]
        # metadata = self.dataset_dict["metadata"][index][0]

        # Image preprocessing
        # Remove any alpha channel if present.
        if image.shape[-1] > 3 and len(image.shape) == 3:
            image = image[:, :, :3]
        # If image is grayscale, then repeat the last channel to convert to rgb
        if len(image.shape) == 2:
            image = np.repeat(image[:, :, None], 3, axis=-1)

        transform = ResizeLongestSide(self.model.image_encoder.img_size)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[:, :, :]
  
        input_image = self.model.preprocess(transformed_image)
        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])

        # Mask preprocessing
        label_ids = np.unique(mask)[1:]
        gt2D = np.uint8(mask == random.choice(label_ids.tolist()))  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"

        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        gt2D = transform.apply_image(gt2D)

        # y_indices, x_indices = np.where(gt2D > 0)
        # x_min, x_max = np.min(x_indices), np.max(x_indices)
        # y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        
        #bbOX preprocessing
        bboxes = np.array([x_min, y_min, x_max, y_max])
        bboxes = transform.apply_boxes(bboxes, original_image_size)
        box_torch = torch.as_tensor(bboxes, dtype=torch.float)
        box_torch = box_torch[ :]

        return (
            input_image,
            gt2D,
            box_torch
            #torch.tensor(gt2D[None, :, :]).long(),
            #torch.tensor(bboxes).float(),
        )


#sanity check

# %% 
sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
#sam_model.to(device)
sam_model.train()

tr_dataset = h5Dataset(train_path,sam_model)
tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
for step, (images, gts, bboxes) in enumerate(tr_dataloader):
    print(images.shape, gts.shape, bboxes.shape)
    # show the example
    _, axs = plt.subplots(1, 2)

    idx = random.randint(0, 7)
    print(idx)
    #image = image / 255.0
    image = images[idx]
    print(image)
    if image.max() > 1 or image.min() < 0:
    # Normalize the image data to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
    axs[0].imshow(image.cpu().permute(1,2,0).numpy())
    show_mask(gts[idx].cpu().numpy(), axs[0])
    show_box(bboxes[idx][0].numpy(), axs[0])
    axs[0].axis("off")

    # set title
    #axs[0].set_title(names_temp[idx])
    idx = random.randint(0, 7)
    print(idx)
    image = images[idx]
    if image.max() > 1 or image.min() < 0:
    # Normalize the image data to [0, 1]
        image = (image - image.min()) / (image.max() - image.min())
    axs[1].imshow(image.cpu().permute(1,2,0).numpy())
    show_mask(gts[idx].cpu().numpy(), axs[1])
    show_box(bboxes[idx][0].numpy(), axs[1])
    axs[1].axis("off")
    # set title
   #axs[1].set_title(names_temp[idx])
    plt.show()
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)
    plt.close()
    break

# %% StowSAM Class
class StowSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder and image encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        #image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            image_embedding = self.image_encoder(image)  # (B, 256, 64, 64) Optional no gradient for image encoder
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

# %% main()
def main():
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    
    stowsam_model = StowSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    stowsam_model.train()
    print( "Number of total parameters: ",sum(p.numel() for p in stowsam_model.parameters()),)  # 93735472
    print("Number of trainable parameters: ",sum(p.numel() for p in stowsam_model.parameters() if p.requires_grad),)  # 937

    img_mask_encdec_params = list(stowsam_model.image_encoder.parameters()) + list(
        stowsam_model.mask_decoder.parameters()
    )
    optimizer = torch.optim.AdamW(img_mask_encdec_params, lr=lr, weight_decay=weight_decay)
    
    print("Number of image encoder and mask decoder parameters: ",sum(p.numel() for p in img_mask_encdec_params if p.requires_grad),)  # 93729252

    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # train
    num_epochs = 2
    iter_num = 0
    losses = []
    best_loss = 1e10
    # train_dataset = NpyDataset(args.tr_npy_path)
    train_dataset = h5Dataset(train_path, sam_model)

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    for epoch in range(num_epochs):
        epoch_loss = 0

        for step, (images, gt_masks, bboxes) in enumerate(tqdm(train_dataloader)):
            images = images.to(device)
            gt_masks = gt_masks.to(device)
            bboxes = bboxes.to(device)
            print(images.shape, gt_masks.shape, bboxes.shape)

            optimizer.zero_grad()
            pred_masks = stowsam_model(images, bboxes)

            loss = seg_loss(pred_masks, gt_masks) + ce_loss(pred_masks, gt_masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            iter_num += 1
            losses.append(loss.item())
        
        epoch_loss /= len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")



# %% test
if __name__ == "__main__":
    main()
    