from PIL import Image
import scipy.io as sio


def load_data(image_name, dataset_name, data_folder='H:/300W-LP/300W_LP'):
    print('================== {0} ==================='.format(image_name))
    image_path = '{folder}/{dataset_name}/{image_name}.jpg'.format(
        folder=data_folder,
        dataset_name=dataset_name,
        image_name=image_name
    )
    with open(image_path, 'rb') as file:
        img = Image.open(file)
        print('image size: {0}'.format(img.size))
        print('image mode: {0}'.format(img.mode))
        print('image length: {0}'.format(len(img.getdata())))

    mat_path = '{folder}/{dataset_name}/{image_name}.mat'.format(
        folder=data_folder,
        dataset_name=dataset_name,
        image_name=image_name
    )
    mat_contents = sio.loadmat(mat_path)
    keys = ['roi', 'Shape_Para', 'Pose_Para', 'Exp_Para', 'Color_Para', 'Illum_Para', 'pt2d', 'Tex_Para']

    for key in keys:
        print('{k}: shape={s}'.format(k=key, s=mat_contents[key].shape))