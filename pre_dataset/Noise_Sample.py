<<<<<<< HEAD
from PIL import Image
import numpy as np
import os
import argparse



parser = argparse.ArgumentParser(description='create a dataset')
parser.add_argument('--dataset', default='F:/loading/SIRST-5K-main/SIRST-5K-main/codes/Noise_Sampling/Demos/dataset3', type=str,
                    help='dataset path')
parser.add_argument('--ns_dir', default='F:/loading/SIRST-5K-main/SIRST-5K-main/codes/Noise_Sampling/Demos/Noise/N3', type=str,
                    help='noise sequences set save dir path')
opt = parser.parse_args()
# parser = argparse.ArgumentParser(description='create a dataset')
# parser.add_argument('--dataset', default='F:/loading/SIRST-5K-main/SIRST-5K-main/codes/train_images', type=str,
#                     help='dataset path')
# parser.add_argument('--ns_dir', default='F:/loading/SIRST-5K-main/SIRST-5K-main/codes/Noise', type=str,
#                     help='noise sequences set save dir path')
# opt = parser.parse_args()


def noise_window(imgSequence, sp, max_var, min_mean, max_sq_var, max_sq_mean):
    # 将输入RGB图像转化为灰度图像，并存储在img_L和img_rgb列表中。
    imgs_L = []
    imgs_rgb = []
    for img_rgb in imgSequence:
        img_L = img_rgb.convert('L')

        img_rgb = np.array(img_rgb)
        img_L = np.array(img_L)

        imgs_L.append(img_L)
        imgs_rgb.append(img_rgb)

    w, h = img_L.shape
    collect_patchs = []
    collect_vars = []

    for i in range(0, w - sp, sp):
        for j in range(0, h - sp, sp):
            vars = []
            means = []
            for img_l in imgs_L:
                patch = img_l[i:i + sp, j:j + sp]
                var_global = np.var(patch)  # window var
                mean_global = np.mean(patch)  # window mean
                
                if var_global >= max_var or mean_global <= min_mean:
                    print(f"Patch location: ({i}, {j}), var_global: {var_global}, mean_global: {mean_global}")
                    print("Skipping patch due to var or mean condition")
                    break
                vars.append(var_global)
                means.append(mean_global)
                if len(vars) != len(imgs_L):
                    continue
                if np.var(vars) <= max_sq_var and np.var(means) <= max_sq_mean:  # the var and mean of window sequence
                    imgs_patch = []
                    for img_rgb in imgs_rgb:
                        imgs_patch.append(img_rgb[i:i + sp, j:j + sp, :])
                        print(i,j)
                    collect_patchs.append(imgs_patch)
                    collect_vars.append(vars)

    return collect_patchs, collect_vars


# sp = 32
# max_var = 200              #50/500
# min_mean =20 #0/200
# max_sq_var =200 #50
# max_sq_mean = 200   #50

# max_var = 500              #50
# min_mean =100 #0
# max_sq_var =500 #50
# max_sq_mean =500   #50


# max_var = 200
# min_mean = 20
# max_sq_var = 200
# max_sq_mean = 200

# max_var = 1000000  # 临时大幅放宽
# min_mean = 127      # 允许所有均值
# max_sq_var = 1000000
# max_sq_mean = 1000000
sp = 32
max_var = 50
min_mean = 0
max_sq_var = 50
max_sq_mean = 50


if not os.path.exists(opt.ns_dir):
    os.mkdir(opt.ns_dir)
subNames = os.listdir(opt.dataset)

cnt = 0
for subName in subNames:
    imgNames = os.listdir(os.path.join(opt.dataset, subName))
    imgNames.sort()
    imgSequence = []
    for imgName in imgNames:
        img_name = os.path.splitext(os.path.basename(imgName))[0]
        img_dir = os.path.join(opt.dataset, subName, imgName)
        img = Image.open(img_dir).convert('RGB')
        # print(f"Loaded image: {img_name}, Size: {img.size}")
        imgSequence.append(img)
    patchs, var_sqs = noise_window(imgSequence, sp, max_var, min_mean, max_sq_var, max_sq_mean)
    for folder_idx, patch in enumerate(patchs):
        folderPath = os.path.join(opt.ns_dir, subName)
        if not os.path.exists(folderPath):
            os.mkdir(folderPath)
        for idx, img in enumerate(patch):
            img_name = '{:08d}'.format(idx)
            save_path = os.path.join(folderPath,
                                     '{}_{}.png'.format(subName,  img_name))  #
            cnt += 1
            # print('collect:', cnt, save_path)
            # print(var_sqs)
            Image.fromarray(img).save(save_path)
=======
<<<<<<< HEAD
from PIL import Image
import numpy as np
import os
import argparse



parser = argparse.ArgumentParser(description='create a dataset')
parser.add_argument('--dataset', default='F:/loading/SIRST-5K-main/SIRST-5K-main/codes/Noise_Sampling/Demos/dataset3', type=str,
                    help='dataset path')
parser.add_argument('--ns_dir', default='F:/loading/SIRST-5K-main/SIRST-5K-main/codes/Noise_Sampling/Demos/Noise/N3', type=str,
                    help='noise sequences set save dir path')
opt = parser.parse_args()
# parser = argparse.ArgumentParser(description='create a dataset')
# parser.add_argument('--dataset', default='F:/loading/SIRST-5K-main/SIRST-5K-main/codes/train_images', type=str,
#                     help='dataset path')
# parser.add_argument('--ns_dir', default='F:/loading/SIRST-5K-main/SIRST-5K-main/codes/Noise', type=str,
#                     help='noise sequences set save dir path')
# opt = parser.parse_args()


def noise_window(imgSequence, sp, max_var, min_mean, max_sq_var, max_sq_mean):
    # 将输入RGB图像转化为灰度图像，并存储在img_L和img_rgb列表中。
    imgs_L = []
    imgs_rgb = []
    for img_rgb in imgSequence:
        img_L = img_rgb.convert('L')

        img_rgb = np.array(img_rgb)
        img_L = np.array(img_L)

        imgs_L.append(img_L)
        imgs_rgb.append(img_rgb)

    w, h = img_L.shape
    collect_patchs = []
    collect_vars = []

    for i in range(0, w - sp, sp):
        for j in range(0, h - sp, sp):
            vars = []
            means = []
            for img_l in imgs_L:
                patch = img_l[i:i + sp, j:j + sp]
                var_global = np.var(patch)  # window var
                mean_global = np.mean(patch)  # window mean
                
                if var_global >= max_var or mean_global <= min_mean:
                    print(f"Patch location: ({i}, {j}), var_global: {var_global}, mean_global: {mean_global}")
                    print("Skipping patch due to var or mean condition")
                    break
                vars.append(var_global)
                means.append(mean_global)
                if len(vars) != len(imgs_L):
                    continue
                if np.var(vars) <= max_sq_var and np.var(means) <= max_sq_mean:  # the var and mean of window sequence
                    imgs_patch = []
                    for img_rgb in imgs_rgb:
                        imgs_patch.append(img_rgb[i:i + sp, j:j + sp, :])
                        print(i,j)
                    collect_patchs.append(imgs_patch)
                    collect_vars.append(vars)

    return collect_patchs, collect_vars


# sp = 32
# max_var = 200              #50/500
# min_mean =20 #0/200
# max_sq_var =200 #50
# max_sq_mean = 200   #50

# max_var = 500              #50
# min_mean =100 #0
# max_sq_var =500 #50
# max_sq_mean =500   #50


# max_var = 200
# min_mean = 20
# max_sq_var = 200
# max_sq_mean = 200

# max_var = 1000000  # 临时大幅放宽
# min_mean = 127      # 允许所有均值
# max_sq_var = 1000000
# max_sq_mean = 1000000
sp = 32
max_var = 50
min_mean = 0
max_sq_var = 50
max_sq_mean = 50


if not os.path.exists(opt.ns_dir):
    os.mkdir(opt.ns_dir)
subNames = os.listdir(opt.dataset)

cnt = 0
for subName in subNames:
    imgNames = os.listdir(os.path.join(opt.dataset, subName))
    imgNames.sort()
    imgSequence = []
    for imgName in imgNames:
        img_name = os.path.splitext(os.path.basename(imgName))[0]
        img_dir = os.path.join(opt.dataset, subName, imgName)
        img = Image.open(img_dir).convert('RGB')
        # print(f"Loaded image: {img_name}, Size: {img.size}")
        imgSequence.append(img)
    patchs, var_sqs = noise_window(imgSequence, sp, max_var, min_mean, max_sq_var, max_sq_mean)
    for folder_idx, patch in enumerate(patchs):
        folderPath = os.path.join(opt.ns_dir, subName)
        if not os.path.exists(folderPath):
            os.mkdir(folderPath)
        for idx, img in enumerate(patch):
            img_name = '{:08d}'.format(idx)
            save_path = os.path.join(folderPath,
                                     '{}_{}.png'.format(subName,  img_name))  #
            cnt += 1
            # print('collect:', cnt, save_path)
            # print(var_sqs)
            Image.fromarray(img).save(save_path)
=======
from PIL import Image
import numpy as np
import os
import argparse



parser = argparse.ArgumentParser(description='create a dataset')
parser.add_argument('--dataset', default='F:/loading/SIRST-5K-main/SIRST-5K-main/codes/Noise_Sampling/Demos/dataset3', type=str,
                    help='dataset path')
parser.add_argument('--ns_dir', default='F:/loading/SIRST-5K-main/SIRST-5K-main/codes/Noise_Sampling/Demos/Noise/N3', type=str,
                    help='noise sequences set save dir path')
opt = parser.parse_args()
# parser = argparse.ArgumentParser(description='create a dataset')
# parser.add_argument('--dataset', default='F:/loading/SIRST-5K-main/SIRST-5K-main/codes/train_images', type=str,
#                     help='dataset path')
# parser.add_argument('--ns_dir', default='F:/loading/SIRST-5K-main/SIRST-5K-main/codes/Noise', type=str,
#                     help='noise sequences set save dir path')
# opt = parser.parse_args()


def noise_window(imgSequence, sp, max_var, min_mean, max_sq_var, max_sq_mean):
    # 将输入RGB图像转化为灰度图像，并存储在img_L和img_rgb列表中。
    imgs_L = []
    imgs_rgb = []
    for img_rgb in imgSequence:
        img_L = img_rgb.convert('L')

        img_rgb = np.array(img_rgb)
        img_L = np.array(img_L)

        imgs_L.append(img_L)
        imgs_rgb.append(img_rgb)

    w, h = img_L.shape
    collect_patchs = []
    collect_vars = []

    for i in range(0, w - sp, sp):
        for j in range(0, h - sp, sp):
            vars = []
            means = []
            for img_l in imgs_L:
                patch = img_l[i:i + sp, j:j + sp]
                var_global = np.var(patch)  # window var
                mean_global = np.mean(patch)  # window mean
                
                if var_global >= max_var or mean_global <= min_mean:
                    print(f"Patch location: ({i}, {j}), var_global: {var_global}, mean_global: {mean_global}")
                    print("Skipping patch due to var or mean condition")
                    break
                vars.append(var_global)
                means.append(mean_global)
                if len(vars) != len(imgs_L):
                    continue
                if np.var(vars) <= max_sq_var and np.var(means) <= max_sq_mean:  # the var and mean of window sequence
                    imgs_patch = []
                    for img_rgb in imgs_rgb:
                        imgs_patch.append(img_rgb[i:i + sp, j:j + sp, :])
                        print(i,j)
                    collect_patchs.append(imgs_patch)
                    collect_vars.append(vars)

    return collect_patchs, collect_vars


# sp = 32
# max_var = 200              #50/500
# min_mean =20 #0/200
# max_sq_var =200 #50
# max_sq_mean = 200   #50

# max_var = 500              #50
# min_mean =100 #0
# max_sq_var =500 #50
# max_sq_mean =500   #50


# max_var = 200
# min_mean = 20
# max_sq_var = 200
# max_sq_mean = 200

# max_var = 1000000  # 临时大幅放宽
# min_mean = 127      # 允许所有均值
# max_sq_var = 1000000
# max_sq_mean = 1000000
sp = 32
max_var = 50
min_mean = 0
max_sq_var = 50
max_sq_mean = 50


if not os.path.exists(opt.ns_dir):
    os.mkdir(opt.ns_dir)
subNames = os.listdir(opt.dataset)

cnt = 0
for subName in subNames:
    imgNames = os.listdir(os.path.join(opt.dataset, subName))
    imgNames.sort()
    imgSequence = []
    for imgName in imgNames:
        img_name = os.path.splitext(os.path.basename(imgName))[0]
        img_dir = os.path.join(opt.dataset, subName, imgName)
        img = Image.open(img_dir).convert('RGB')
        # print(f"Loaded image: {img_name}, Size: {img.size}")
        imgSequence.append(img)
    patchs, var_sqs = noise_window(imgSequence, sp, max_var, min_mean, max_sq_var, max_sq_mean)
    for folder_idx, patch in enumerate(patchs):
        folderPath = os.path.join(opt.ns_dir, subName)
        if not os.path.exists(folderPath):
            os.mkdir(folderPath)
        for idx, img in enumerate(patch):
            img_name = '{:08d}'.format(idx)
            save_path = os.path.join(folderPath,
                                     '{}_{}.png'.format(subName,  img_name))  #
            cnt += 1
            # print('collect:', cnt, save_path)
            # print(var_sqs)
            Image.fromarray(img).save(save_path)
>>>>>>> origin/feat/update
>>>>>>> 21a2898 (update)
