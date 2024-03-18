import os
import torch
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm, trange
from skimage import transform, io, segmentation
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from typing import List

#Clear cuda cache
# torch.cuda.empty_cache()

#Set up parser
parser = argparse.ArgumentParser(description="preprocess RGB images")
# raw input (images, masks)
# parser.add_argument("--img_folder", type=str, default="data/StowSam/raw_input/train/images", help="path to a folder of images")
# parser.add_argument("--mask_folder", type=str, default="data/StowSam/raw_input/train/masks", help="path to a folder of masks (ground truth)")
# parser.add_argument("--h5_file", type=str, default="dataset/bin_syn/train_shard_000000_copy", help="path to a h5 file of rgb_images, masks, depths_images, metadatas")
parser.add_argument("--h5_file", type=str, default="dataset/bin_syn/train_shard_000000", help="path to a h5 file of rgb_images, masks, depths_images, metadatas")

# model input
parser.add_argument("--data_folder", type=str, default="dataset/StowSam/input/train", help="path to save npz files (input for training)")
parser.add_argument("--data_name", type=str, default="stow", help="dataset name; used to name the final npz file, e.g., stow.npz")
# image parameters
parser.add_argument("--img_size", type=int, default=256, help="image size")
parser.add_argument("--img_format", type=str, default='png', help="image format")
# SAM model parameters
parser.add_argument("--model_type", type=str, default="vit_b", help="model type")
parser.add_argument("--checkpoint", type=str, default="work_dir/MedSAM/sam_vit_b_01ec64.pth", help="checkpoint")
parser.add_argument("--device", type=str, default="cuda:0", help="device")
# misc
parser.add_argument("--seed", type=int, default=2023, help="random seed")
# parse the arguments
args = parser.parse_known_args()[0]
# print(f"args: {args}")

join = os.path.join

# Setup Model
sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)


def process_h5file(filename: str, num_samples: int = 1000, chunk_id: int = 0):
    with h5py.File(f"{filename}.h5", 'r') as h5_file:
         min_rand_id = chunk_id * num_samples
         for id in trange(min_rand_id, min_rand_id + num_samples):  
            for i in range(2):
                # ground truth (mask) processing
                gt_data = h5_file[f'frame{i}_mask'][id]
                print(f"gt_data.shape: {gt_data.shape}")
                if len(gt_data.shape) == 3:
                    gt_data = gt_data[:, :, 0]
                assert len(gt_data.shape) == 2, "ground truth should be 2D"

                gt_data_unique = np.unique(gt_data)[0:-2]
                success = False
                while not success:
                    # resize/filter ground truth image
                    rand_unique_id = np.random.choice(gt_data_unique)
                    gt_data_tmp = transform.resize(
                        gt_data == rand_unique_id,
                        (args.img_size, args.img_size),
                        order=0,
                        preserve_range=True,
                        mode="constant",
                    )
                    gt_data_tmp = np.uint8(gt_data_tmp)
                    if np.sum(gt_data_tmp) > 100:  # exclude tiny objects
                        gt_data = gt_data_tmp
                        success = True

                assert np.sum(gt_data) > 100, "ground truth should have more than 100 pixels"
                assert (np.max(gt_data) == 1 and np.unique(gt_data).shape[0] == 2), "ground truth should be binary"
                gts.append(gt_data)

                # image processing
                image_data = h5_file[f'frame{i}_data'][id]
                # Remove any alpha channel if present.
                if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
                    image_data = image_data[:, :, :3]
                # If image is grayscale, then repeat the last channel to convert to RGB
                if len(image_data.shape) == 2:
                    image_data = np.repeat(image_data[:, :, None], 3, axis=-1)

                # nii preprocess start
                lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
                image_data_pre = np.clip(image_data, lower_bound, upper_bound)
                # min-max normalize and scale
                image_data_pre = ((image_data_pre - np.min(image_data_pre)) / (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0)
                image_data_pre[image_data == 0] = 0
                print(image_data_pre.shape)
                image_data_pre = transform.resize(
                    image_data_pre,
                    (args.img_size, args.img_size),
                    order=3,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=True,
                )
                image_data_pre = np.uint8(image_data_pre)
                imgs.append(image_data_pre)

                # resize image to 3*1024*1024
                sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                resize_img = sam_transform.apply_image(image_data_pre)
                resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(args.device)
                input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])  # (1, 3, 1024, 1024)
                assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), "input image should be resized to 1024*1024"
                # pre-compute the image embedding
                with torch.no_grad():
                    embedding = sam_model.image_encoder(input_image)
                    img_embeddings.append(embedding.cpu().numpy()[0])


def process_newh5file(filename: str, num_samples: int = 1000, chunk_id: int = 0):
    with h5py.File(f"{filename}.h5", 'r') as h5_file:
         min_rand_id = chunk_id * num_samples

        #  gt_data = h5_file[f'mask'][0]
        #  # print(gt_data.shape)
        #  print(gt_data)
        #  print(gt_data.shape)
        #  gt_data_unique = np.unique(gt_data)[0:-2]
        #  print(gt_data_unique)

         for id in trange(min_rand_id, min_rand_id + num_samples):
            for i in range(2):
                # print(id)
                # ground truth (mask) processing
                gt_data = h5_file[f'mask'][id][i]
                # print(gt_data.shape)
                
                if len(gt_data.shape) == 3:
                    gt_data = gt_data[:, :, 0]            
                assert len(gt_data.shape) == 2, "ground truth should be 2D"
                # print(gt_data.shape)
                gt_data_unique = np.unique(gt_data)[0:-2]
                # print(gt_data_unique)
                success = False
                # print(f'test{id}')
                while not success:
                    # resize/filter ground truth image
                    rand_unique_id = np.random.choice(gt_data_unique)
                    
                    gt_data_tmp = transform.resize(
                        gt_data == rand_unique_id,
                        (args.img_size, args.img_size),
                        order=0,
                        preserve_range=True,
                        mode="constant",
                    )
                    gt_data_tmp = np.uint8(gt_data_tmp)
                    if np.sum(gt_data_tmp) > 100:  # exclude tiny objects
                        gt_data = gt_data_tmp
                        success = True
                # print(success)
                assert np.sum(gt_data) > 100, "ground truth should have more than 100 pixels"
                assert (np.max(gt_data) == 1 and np.unique(gt_data).shape[0] == 2), "ground truth should be binary"
                gts.append(gt_data)

                # image processing
                image_data = h5_file[f'data'][id][i]
                # Remove any alpha channel if present.
                if image_data.shape[-1] > 3 and len(image_data.shape) == 3:
                    image_data = image_data[:, :, :3]
                # If image is grayscale, then repeat the last channel to convert to RGB
                if len(image_data.shape) == 2:
                    image_data = np.repeat(image_data[:, :, None], 3, axis=-1)

                # nii preprocess start
                lower_bound, upper_bound = np.percentile(image_data, 0.5), np.percentile(image_data, 99.5)
                image_data_pre = np.clip(image_data, lower_bound, upper_bound)
                # min-max normalize and scale
                image_data_pre = ((image_data_pre - np.min(image_data_pre)) / (np.max(image_data_pre) - np.min(image_data_pre)) * 255.0)
                image_data_pre[image_data == 0] = 0
                # print(image_data_pre.shape)
                image_data_pre = transform.resize(
                    image_data_pre,
                    (args.img_size, args.img_size),
                    order=3,
                    preserve_range=True,
                    mode="constant",
                    anti_aliasing=True,
                )
                image_data_pre = np.uint8(image_data_pre)
                #image_data_pre = image_data_pre/255.0
                imgs.append(image_data_pre)

                # resize image to 3*1024*1024
                sam_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
                resize_img = sam_transform.apply_image(image_data_pre)
                resize_img_tensor = torch.as_tensor(resize_img.transpose(2, 0, 1)).to(args.device)
                input_image = sam_model.preprocess(resize_img_tensor[None, :, :, :])  # (1, 3, 1024, 1024)
                assert input_image.shape == (1, 3, sam_model.image_encoder.img_size, sam_model.image_encoder.img_size), "input image should be resized to 1024*1024"
                # pre-compute the image embedding
                with torch.no_grad():
                    embedding = sam_model.image_encoder(input_image)
                    img_embeddings.append(embedding.cpu().numpy()[0])


os.makedirs(args.data_folder, exist_ok=True)
for dataset_id in range(1):
    print(f"dataset_id: {dataset_id}")
    imgs: List[np.ndarray] = []
    gts: List[np.ndarray] = []
    img_embeddings: List[np.ndarray] = []

    # process_h5file(args.h5_file, num_samples=8900, chunk_id=dataset_id)
    process_newh5file(args.h5_file, num_samples=100, chunk_id=dataset_id)


    # stack the list to array
    np_imgs = np.stack(imgs, axis=0)  # (n, 256, 256, 3)
    np_gts = np.stack(gts, axis=0)  # (n, 256, 256)
    np_img_embeddings = np.stack(img_embeddings, axis=0)  # (n, 1, 256, 64, 64)

    #preprocess to save as npy
    # imgs_dir = 'dataset/StowSam/input/train/imgs'
    # gts_dir = 'dataset/StowSam/input/train/gts'

    # # Save each image in a separate .npy file
    # for i, img in enumerate(np_imgs):
    #     np.save(os.path.join(imgs_dir, f'gt_{i:02d}.npy'), img)

    # # Save each ground truth in a separate .npy file
    # for i, gt in enumerate(np_gts):
    #     np.save(os.path.join(gts_dir, f'gt_{i:02d}.npy'), gt)
    # # end preprocess to save as npy

    np.savez_compressed(
        join(args.data_folder, f"{args.data_name}_chunk{dataset_id:02d}.npz"),
        imgs=np_imgs,
        gts=np_gts,
        img_embeddings=np_img_embeddings,
    )
    
    imgs.clear()
    gts.clear()
    img_embeddings.clear()