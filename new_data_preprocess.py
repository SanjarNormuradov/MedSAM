import numpy as np
import os
join = os.path.join 
from skimage import io, transform
from tqdm import tqdm,trange
import h5py

#Clear cuda cache
# torch.cuda.empty_cache()

# #Set up parser
# parser = argparse.ArgumentParser(description="preprocess RGB images")
# # raw input (images, masks)
# # parser.add_argument("--img_folder", type=str, default="data/StowSam/raw_input/train/images", help="path to a folder of images")
# # parser.add_argument("--mask_folder", type=str, default="data/StowSam/raw_input/train/masks", help="path to a folder of masks (ground truth)")
# # parser.add_argument("--h5_file", type=str, default="dataset/bin_syn/train_shard_000000_copy", help="path to a h5 file of rgb_images, masks, depths_images, metadatas")
# parser.add_argument("--h5_file", type=str, default="dataset/bin_syn/train_shard_000000", help="path to a h5 file of rgb_images, masks, depths_images, metadatas")

# # model input
# parser.add_argument("--data_folder", type=str, default="dataset/StowSam/input/train", help="path to save npz files (input for training)")
# parser.add_argument("--data_name", type=str, default="stow", help="dataset name; used to name the final npz file, e.g., stow.npz")
# # image parameters
# parser.add_argument("--img_size", type=int, default=256, help="image size")
# parser.add_argument("--img_format", type=str, default='png', help="image format")
# # SAM model parameters
# parser.add_argument("--model_type", type=str, default="vit_b", help="model type")
# parser.add_argument("--checkpoint", type=str, default="work_dir/MedSAM/sam_vit_b_01ec64.pth", help="checkpoint")
# parser.add_argument("--device", type=str, default="cuda:0", help="device")
# # misc
# parser.add_argument("--seed", type=int, default=2023, help="random seed")
# # parse the arguments
# args = parser.parse_known_args()[0]


# convert 2D data to npy files, including images and corresponding masks
modality = 'dd' # e.g., 'Dermoscopy 
anatomy = 'dd'  # e.g., 'SkinCancer'
img_name_suffix = '.png' 
gt_name_suffix = '.png' 

prefix = modality + '_' + anatomy + '_'
save_suffix = '.npy' 
image_size = 1024
img_path = 'path to /images' # path to the images
gt_path = 'path to/labels' # path to the corresponding annotations
npy_path = 'dataset/StowSam/input/train'
os.makedirs(join(npy_path, "gts"), exist_ok=True)
os.makedirs(join(npy_path, "imgs"), exist_ok=True)
os.makedirs(join(npy_path, "npz"), exist_ok=True)
#names = sorted(os.listdir(gt_path))
#print(f'ori \# files {len(names)=}')

#NEW
filename = 'dataset/bin_syn/train_shard_000000' # path to the h5 file
npz_save_path = 'dataset/StowSam/input/train/npz'
save_name = 'stow_'


# set label ids that are excluded
remove_label_ids = [] 
tumor_id = None # only set this when there are multiple tumors in one image; convert semantic masks to instance masks
label_id_offset = 0
do_intensity_cutoff = False # True for grey images
#%% save preprocessed images and masks as npz files
#prep for new h5 file
imgs_mod = []
gts_mod = []
split = 5

def processh5file(h5_file, chunk_id, num_samples):
    with h5py.File(f"{filename}.h5", 'r') as h5_file:
        min_rand_id = chunk_id *num_samples
        for id in trange(min_rand_id,min_rand_id+num_samples):
            for i in range(1):
                #process gt data
                gt_data = h5_file[f'mask'][id][i]
                #assess shape of gt data
                if len(gt_data.shape) == 3:
                    gt_data = gt_data[:, :, 0]            
                assert len(gt_data.shape) == 2, "ground truth should be 2D"

                gt_data_ori = np.uint8(gt_data)

                #image_name = name.split(gt_name_suffix)[0] + img_name_suffix
                #gt_name = name
                #npy_save_name = prefix + gt_name.split(gt_name_suffix)[0]+save_suffix
                #gt_data_ori = np.uint8(io.imread(join(gt_path, gt_name)))
                # remove label ids
                #for remove_label_id in remove_label_ids:
                #    gt_data_ori[gt_data_ori==remove_label_id] = 0
                # label tumor masks as instances and remove from gt_data_ori
                #if tumor_id is not None:
                #    tumor_bw = np.uint8(gt_data_ori==tumor_id)
                #    gt_data_ori[tumor_bw>0] = 0
                    # label tumor masks as instances
                #    tumor_inst, tumor_n = cc3d.connected_components(tumor_bw, connectivity=26, return_N=True)
                    # put the tumor instances back to gt_data_ori
                #    gt_data_ori[tumor_inst>0] = tumor_inst[tumor_inst>0] + label_id_offset + 1
                
                #process image data
                image_data = h5_file[f'data'][id][i]

                # crop the ground truth with non-zero slices
                #image_data = io.imread(join(img_path, image_name))
                if np.max(image_data) > 255.0:
                    image_data = np.uint8((image_data-image_data.min()) / (np.max(image_data)-np.min(image_data))*255.0)
                if len(image_data.shape) == 2:
                    image_data = np.repeat(np.expand_dims(image_data, -1), 3, -1)
                assert len(image_data.shape) == 3, 'image data is not three channels: img shape:' + str(image_data.shape)
                # convert three channel to one channel
                if image_data.shape[-1] > 3:
                    image_data = image_data[:,:,:3]
                # image preprocess start
                if do_intensity_cutoff:
                    lower_bound, upper_bound = np.percentile(image_data[image_data>0], 0.5), np.percentile(image_data[image_data>0], 99.5)
                    image_data_pre = np.clip(image_data, lower_bound, upper_bound)
                    image_data_pre = (image_data_pre - np.min(image_data_pre))/(np.max(image_data_pre)-np.min(image_data_pre))*255.0
                    image_data_pre[image_data==0] = 0
                    image_data_pre = np.uint8(image_data_pre)
                else:
                    # print('no intensity cutoff')
                    image_data_pre = image_data.copy()
                
                #np.savez_compressed(join(npz_save_path, save_name+f'{id}'+f'{i}'+'.npz'), imgs=image_data_pre, gts=gt_data_ori)   
                resize_img = transform.resize(image_data_pre, (image_size, image_size), order=3, mode='constant', preserve_range=True, anti_aliasing=True)
                resize_img01 = resize_img/255.0
                resize_gt = transform.resize(gt_data_ori, (image_size, image_size), order=0, mode='constant', preserve_range=True, anti_aliasing=False)
                # save resize img and gt as npy
                #npy_save_name = save_name + f'{id}'+f'{i}'
                #np.save(join(npy_path, "imgs", npy_save_name), resize_img01)
                #np.save(join(npy_path, "gts", npy_save_name), resize_gt.astype(np.uint8))
                imgs_mod.append(resize_img01)
                gts_mod.append(resize_gt)

    # # save img_mod and gt_mod as h5 file
    new_file_h5 = f'stow_{chunk_id}'
    hf = h5py.File(join(npy_path, new_file_h5 + '.h5'), 'w')
    hf.create_dataset('imgs', data=imgs_mod)
    hf.create_dataset('gts', data=gts_mod)


for dataset_id in range(split):
    processh5file(filename, dataset_id, 1800)
    print(f'finished {dataset_id=}')
    imgs_mod.clear()
    gts_mod.clear()