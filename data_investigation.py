import h5py
import numpy as np
import matplotlib.pyplot as plt

# train_dataset = h5py.File('dataset/bin_syn/train_shard_000000.h5', "r")

# train_dataset.keys()
# # Load the training data
# for key in train_dataset.keys(): 
#     print(key) 

# child_keys = [k for k in train_dataset.keys()]
# print(child_keys)

# train_set_x = train_dataset['data'][345]
# print(train_set_x.shape)
# # if len(train_set_x.shape) == 3:
# #     gt_data = train_set_x[:, :, 0]
# # assert len(train_set_x.shape) == 2, "ground truth should be 2D"

# gt_data_unique = np.unique(train_set_x)[0:-2]
# #print(train_set_x[0])
# print(gt_data_unique)
# # if train_set_x.shape[-1] > 1 and len(train_set_x.shape) == 3:  # Ensure there are at least two channels
# #     is_same = np.array_equal(train_set_x[:, :, 0], train_set_x[:, :, 1])
# #     print(f"Channel 1 and 2 are the same: {is_same}")
# plt.imshow(train_set_x[0][:, :,2])
# plt.show()


#Investigate the .npz file
# Load the .npz file
data = np.load('dataset/StowSam/input/train/stow_chunk00.npz')

# Print the keys in the file
print(data.files)

# Access a specific array
array1 = data['array1']

# Print the shape and dtype of the array
# print(array1.shape)
# print(array1.dtype)

# # Print the unique values in the array
# print(np.unique(array1))

# # If the array is an image, you can display it
# import matplotlib.pyplot as plt
# plt.imshow(array1, cmap='gray')
# plt.show()