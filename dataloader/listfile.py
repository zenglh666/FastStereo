import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def list_sub_dir(path, disppath):
    left_img=[]
    right_img=[]
    left_disp = []
    subdir  = os.listdir(path)
    for dd in subdir:
        fl = os.path.join(path, dd, 'left')
        fr = os.path.join(path, dd, 'right')
        fd = os.path.join(disppath, dd, 'left')
        for im in os.listdir(fl):
            flf = os.path.join(fl, im)
            frf = os.path.join(fr, im)
            fdf = os.path.join(fd, im.split(".")[0]+'.pfm')
            if is_image_file(flf):
                assert os.path.exists(frf), "left right file incompatible"
                assert os.path.exists(fdf), "left disp file incompatible"
                left_img.append(flf)
                right_img.append(frf)
                left_disp.append(fdf)
    return left_img, right_img, left_disp

def list_flow_file(filepath):
    train_left_img=[]
    train_right_img=[]
    train_left_disp = []
    test_left_img=[]
    test_right_img=[]
    test_left_disp = []

    monkaa_path = os.path.join(filepath, 'monkaa_frames_cleanpass')
    monkaa_disp = os.path.join(filepath, 'monkaa_disparity')
    monkaa_list = list_sub_dir(monkaa_path, monkaa_disp)
    train_left_img.extend(monkaa_list[0])
    train_right_img.extend(monkaa_list[1])
    train_left_disp.extend(monkaa_list[2])

    flying_path = os.path.join(filepath, 'frames_cleanpass', 'TRAIN')
    flying_disp = os.path.join(filepath, 'frames_disparity', 'TRAIN')
    subdir = ['A','B','C']

    for ss in subdir:
        flying_dir_sub = os.path.join(flying_path, ss)
        flying_disp_dir_sub = os.path.join(flying_disp, ss)
        flying_list = list_sub_dir(flying_dir_sub, flying_disp_dir_sub)
        train_left_img.extend(flying_list[0])
        train_right_img.extend(flying_list[1])
        train_left_disp.extend(flying_list[2])

    flying_path = os.path.join(filepath, 'frames_cleanpass', 'TEST')
    flying_disp = os.path.join(filepath, 'frames_disparity', 'TEST')
    subdir = ['A','B','C']

    for ss in subdir:
        flying_dir_sub = os.path.join(flying_path, ss)
        flying_disp_dir_sub = os.path.join(flying_disp, ss)
        flying_list = list_sub_dir(flying_dir_sub, flying_disp_dir_sub)
        test_left_img.extend(flying_list[0])
        test_right_img.extend(flying_list[1])
        test_left_disp.extend(flying_list[2])

    driving_path = os.path.join(filepath, 'driving_frames_cleanpass')
    driving_disp = os.path.join(filepath, 'driving_disparity')

    subdir1 = ['35mm_focallength','15mm_focallength']
    subdir2 = ['scene_backwards','scene_forwards']

    for i in subdir1:
        for j in subdir2:
            driving_dir_sub = os.path.join(driving_path, i, j)
            driving_disp_dir_sub = os.path.join(driving_disp, i, j)
            driving_list = list_sub_dir(driving_dir_sub, driving_disp_dir_sub)
            train_left_img.extend(driving_list[0])
            train_right_img.extend(driving_list[1])
            train_left_disp.extend(driving_list[2])


    return train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp


def list_kitti_file(filepath, date):
    if date == "2015":
        left_fold  = 'image_2'
        right_fold = 'image_3'
        disp_L = 'disp_occ_0'
        val_set = [1,3,6,20,26,35,38,41,43,44,49,60,67,70,81,84,89,97,109,119,122,123,
            129,130,132,134,141,144,152,158,159,165,171,174,179,182, 184,186,187,196]
    elif date == "2012":
        left_fold  = 'colored_0'
        right_fold = 'colored_1'
        disp_L   = 'disp_occ'
        val_set = [1,3,6,20,26,35,38,41,43,44,49,60,67,70,81,84,89,97,109,119,122,123,
            129,130,132,134,141,144,152,158,159,165,171,174,179,182, 184,186,187,193]

    image = [img for img in os.listdir(os.path.join(filepath, left_fold)) if img.find('_10') > -1]
    image = np.array(sorted(image))
    val_bool = np.zeros_like(image, dtype=np.bool)
    val_bool[val_set] = True
    train = image[~val_bool]
    val   = image[val_bool]

    left_train  = [os.path.join(filepath, left_fold, img) for img in train]
    right_train = [os.path.join(filepath, right_fold, img) for img in train]
    disp_train_L = [os.path.join(filepath, disp_L, img) for img in train]

    left_val  = [os.path.join(filepath, left_fold, img) for img in val]
    right_val = [os.path.join(filepath, right_fold, img) for img in val]
    disp_val_L = [os.path.join(filepath, disp_L, img) for img in val]

    return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L

def list_kitti_test_file(filepath, date):
    if date == "2015":
        left_fold  = 'image_2'
        right_fold = 'image_3'
    elif date == "2012":
        left_fold  = 'colored_0'
        right_fold = 'colored_1'

    image = [img for img in os.listdir(os.path.join(filepath, left_fold)) if img.find('_10') > -1]

    left  = [os.path.join(filepath, left_fold, img) for img in image]
    right = [os.path.join(filepath, right_fold, img) for img in image]

    return left, right

def list_middlebury_file(filepath):
    train_path = os.path.join(filepath, 'MiddEval3/trainingF')
    extra_path = os.path.join(filepath, 'extra')
    extra_path2 = os.path.join(filepath, 'extra2')

    train = [path for path in os.listdir(train_path) if path.find('.') <= -1]
    extra = [path for path in os.listdir(extra_path) if path.find('.') <= -1]
    extra2 = [path for path in os.listdir(extra_path2) if path.find('.') <= -1]

    left_train  = [os.path.join(train_path, path, 'im0.png') for path in train]
    right_train = [os.path.join(train_path, path, 'im1.png') for path in train]
    disp_train_L = [os.path.join(train_path, path, 'disp0GT.pfm') for path in train]

    left_val, right_val, disp_val_L = [], [], []
    for right_img in ['im1.png', 'im1E.png', 'im1L.png']:
        left_val.extend([os.path.join(extra_path, path, 'im0.png') for path in extra])
        right_val.extend([os.path.join(extra_path, path, right_img) for path in extra])
        disp_val_L.extend([os.path.join(extra_path, path, 'disp0.pfm') for path in extra])

    for illu, expo in [('Illum1', 'Exp1'), ('Illum1', 'Exp2'), ('Illum3', 'Exp1')]:
        left_val.extend([os.path.join(extra_path2, path, 'Illum1', 'Exp1', 'view1.png') for path in extra2])
        right_val.extend([os.path.join(extra_path2, path, illu, expo, 'view5.png') for path in extra2])
        disp_val_L.extend([os.path.join(extra_path2, path, 'disp1.png') for path in extra2])

    return left_val, right_val, disp_val_L, left_train, right_train, disp_train_L

def list_middlebury_test_file(filepath):
    train_path = os.path.join(filepath, 'MiddEval3/trainingF')
    val_path = os.path.join(filepath, 'MiddEval3/testF')

    train = [path for path in os.listdir(train_path) if path.find('.') <= -1]
    val = [path for path in os.listdir(val_path) if path.find('.') <= -1]

    left_train  = [os.path.join(train_path, path, 'im0.png') for path in train]
    right_train = [os.path.join(train_path, path, 'im1.png') for path in train]

    left_val  = [os.path.join(val_path, path, 'im0.png') for path in val]
    right_val = [os.path.join(val_path, path, 'im1.png') for path in val]

    left_train.extend(left_val)
    right_train.extend(right_val)

    return left_train, right_train