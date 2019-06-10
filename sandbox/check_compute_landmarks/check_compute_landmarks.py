import numpy as np

from project_code.morphable_model.model.morphable_model import MorphableModel
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt

from project_code.training.opt import compute_landmarks

bfm_path = 'G:\PycharmProjects\FaceFusion\project_code\data\\3dmm\BFM\BFM.mat'
bfm = MorphableModel(bfm_path)


def load_image_mat(pic_name):
    mat_filename = 'G:\PycharmProjects\FaceFusion\project_code\data\\3dmm\\300W_LP_samples\{0}.mat'.format(pic_name)
    mat_data = sio.loadmat(mat_filename)
    image_filename = 'G:\PycharmProjects\FaceFusion\project_code\data\\3dmm\\300W_LP_samples\{0}.jpg'.format(pic_name)

    with open(image_filename, 'rb') as file:
        img = Image.open(file)
        print('image size: {0}'.format(img.size))
        img_np = np.asarray(img, dtype="int32")

    poses_param = np.expand_dims(mat_data['Pose_Para'], 0)
    shapes_param = np.expand_dims(mat_data['Shape_Para'], 0)
    exps_param = np.expand_dims(mat_data['Exp_Para'], 0)
    lms = np.expand_dims(mat_data['pt2d'], 0)
    return img_np, poses_param, shapes_param, exps_param, lms

img1, poses_param1, shapes_param1, exps_param1, landmarks1 = load_image_mat('IBUG_image_008_1_0')
img2, poses_param2, shapes_param2, exps_param2, landmarks2 = load_image_mat('IBUG_image_007_1')

poses_param = np.concatenate([poses_param1, poses_param2], 0)
shapes_param = np.concatenate([shapes_param1, shapes_param2], 0)
exps_param = np.concatenate([exps_param1, exps_param2], 0)
landmarks_param = np.concatenate([landmarks1, landmarks2], 0)

tf_landmarks = compute_landmarks(
    poses_param=poses_param,
    shapes_param=shapes_param,
    exps_param=exps_param,
    bfm=bfm,
    output_size=450)
landmarks = np.array(tf_landmarks)


def plot_landmarks(ax, image, lms):
    ax.imshow(image)
    ax.plot(lms[0, 0:17], lms[1, 0:17], marker='o', markersize=2, linestyle='-', color='w', lw=2)
    ax.plot(lms[0, 17:22], lms[1, 17:22], marker='o', markersize=2, linestyle='-', color='w', lw=2)
    ax.plot(lms[0, 22:27], lms[1, 22:27], marker='o', markersize=2, linestyle='-', color='w', lw=2)
    ax.plot(lms[0, 27:31], lms[1, 27:31], marker='o', markersize=2, linestyle='-', color='w', lw=2)
    ax.plot(lms[0, 31:36], lms[1, 31:36], marker='o', markersize=2, linestyle='-', color='w', lw=2)
    ax.plot(lms[0, 36:42], lms[1, 36:42], marker='o', markersize=2, linestyle='-', color='w', lw=2)
    ax.plot(lms[0, 42:48], lms[1, 42:48], marker='o', markersize=2, linestyle='-', color='w', lw=2)
    ax.plot(lms[0, 48:60], lms[1, 48:60], marker='o', markersize=2, linestyle='-', color='w', lw=2)
    ax.plot(lms[0, 60:68], lms[1, 60:68], marker='o', markersize=2, linestyle='-', color='w', lw=2)

fig = plt.figure()
ax = fig.add_subplot(2, 2, 1)
plot_landmarks(ax, img1, landmarks_param[0, :, :])
ax = fig.add_subplot(2, 2, 2)
plot_landmarks(ax, img1, landmarks[0, :, :])

ax = fig.add_subplot(2, 2, 3)
plot_landmarks(ax, img2, landmarks_param[1, :, :])
ax = fig.add_subplot(2, 2, 4)
plot_landmarks(ax, img2, landmarks[1, :, :])
plt.savefig('test.jpg')
