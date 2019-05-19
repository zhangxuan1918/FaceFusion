import glob
import os

from sandbox.check_300W_LP_data.load_data import load_data

# ================== AFW_1051618982_1_0 ===================
# image size: (450, 450)
# image mode: RGB
# image length: 202500
# roi: shape=(1, 4)
# Shape_Para: shape=(199, 1)
# Pose_Para: shape=(1, 7)
# Exp_Para: shape=(29, 1)
# Color_Para: shape=(1, 7)
# Illum_Para: shape=(1, 10)
# pt2d: shape=(2, 68)
# Tex_Para: shape=(199, 1)
# 5207 of images
# 5207 of mat

data_folder = 'H:/300W-LP/300W_LP'
image_name = 'AFW_1051618982_1_0'
dataset_name = 'AFW'
load_data(
    data_folder=data_folder,
    image_name=image_name,
    dataset_name=dataset_name
)

image_glob_pattern = os.path.join(data_folder, dataset_name, '*.jpg')
image_filenames = glob.glob(image_glob_pattern)
print('%d of images' % len(image_filenames))
mat_glob_pattern = os.path.join(data_folder, dataset_name, '*.mat')
mat_filenames = glob.glob(mat_glob_pattern)
print('%d of mat' % len(mat_filenames))


