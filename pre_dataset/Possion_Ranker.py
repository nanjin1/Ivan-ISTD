<<<<<<< HEAD
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor

def calculate_ssim(img1, img2, data_range=255, multichannel=True):
    # 计算图像区域的最小边长
    min_dim = min(img1.shape[0], img1.shape[1])
    # 默认使用 win_size=7，但如果图像太小则调整为不超过 min_dim 的最大奇数（最小保证为3）
    win_size = 7
    if min_dim < 7:
        win_size = min_dim if (min_dim % 2 == 1) else min_dim - 1
        win_size = max(win_size, 3)
    ssim_value = ssim(img1, img2, data_range=data_range, win_size=win_size,
                      channel_axis=2, multichannel=multichannel)
    return ssim_value

def find_best_match(img1, img2):
    img1 = cv2.imread(img1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2, cv2.IMREAD_COLOR)

    # 使用快速模板匹配算法 (cv2.TM_CCOEFF_NORMED)
    result = cv2.matchTemplate(img2, img1, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    best_match = max_loc

    # 获取img1的宽度和高度
    h, w, _ = img1.shape

    # 定义要计算SSIM的边缘区域
    edge_width = 30
    edge_regions = [
        img1[:, :edge_width, :],  # 左边缘
        img1[:, -edge_width:, :],  # 右边缘
        img1[:edge_width, :, :],    # 上边缘
        img1[-edge_width:, :, :]    # 下边缘
    ]

    # 计算每个边缘区域的SSIM并取平均
    ssims = []
    for region in edge_regions:
        # 提取img2中对应区域
        region2 = img2[max_loc[1]:max_loc[1] + region.shape[0], max_loc[0]:max_loc[0] + region.shape[1], :]
        ssims.append(calculate_ssim(region, region2))
    best_ssim = np.mean(ssims)

    return best_match, best_ssim

def paste_image(img1_path, img2_path, x, y, output_filename):
    img2 = cv2.imread(img2_path)
    img1 = cv2.imread(img1_path)

    mask = np.zeros(img1.shape[:2], dtype=np.uint8)
    mask[:img1.shape[0], :img1.shape[1]] = 255

    center = (x + img1.shape[1] // 2, y + img1.shape[0] // 2)
    output = cv2.seamlessClone(img1, img2, mask, center, cv2.NORMAL_CLONE)
    cv2.imwrite(output_filename, output)

def paste_masks(img1_path, img2_path, x, y, output_filename):
    img2 = cv2.imread(img2_path)
    img1 = cv2.imread(img1_path)

    img2[y:y + img1.shape[0], x:x + img1.shape[1]] = img1
    cv2.imwrite(output_filename, img2)

def process_image_pair(img1_path, img2_path, save_path_img, save_path_mask, mask1_dir, mask2_dir, ssim_threshold=0.7):
    best_match, best_ssim = find_best_match(img1_path, img2_path)

    if best_ssim > ssim_threshold:  # 仅当 SSIM 大于阈值时才执行粘贴
        img1_name = os.path.basename(img1_path).split('.')[0]
        img2_name = os.path.basename(img2_path).split('.')[0]
        paste_image(img1_path, img2_path, best_match[0], best_match[1],
                    os.path.join(save_path_img, f"{img1_name}_{img2_name}.png"))
        paste_masks(os.path.join(mask1_dir, os.path.basename(img1_path)),
                    os.path.join(mask2_dir, os.path.basename(img2_path)),
                    best_match[0], best_match[1],
                    os.path.join(save_path_mask, f"{img1_name}_{img2_name}.png"))
        print(f"Pasted {img2_path} with ssim {best_ssim}")

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 配置各个文件夹路径
img1_dir = 'F:/dark/real/candidate_images'
img2_dir = 'F:/dark/real/bgnew'
mask1_dir = 'F:/dark/real/candidate_masks'
mask2_dir = 'F:/dark/real/bgnewmask'
save_path_img = 'F:/dark/real/images'
save_path_mask = 'F:/dark/real/mask'

create_folder(save_path_img)
create_folder(save_path_mask)
images1 = [os.path.join(img1_dir, f) for f in os.listdir(img1_dir) if f.endswith('.png')]
images2 = [os.path.join(img2_dir, f) for f in os.listdir(img2_dir) if f.endswith('.png')]

# SIRST图片循环
with ThreadPoolExecutor() as executor:
    for img2_path in images2:
        print('SIRST is image path :', img2_path)
        ssim_results = list(executor.map(lambda img1_path: (img1_path, *find_best_match(img1_path, img2_path)), images1))

        # 按 ssim 降序排序
        ssim_results.sort(key=lambda x: x[2], reverse=True)

        # 只对前 top_k 个高分的图片执行粘贴操作
        top_k = 1
        count = 0  # 用于计数已粘贴的图像数量
        for i in range(len(ssim_results)):
            if count >= top_k:
                break  # 如果已经粘贴了 top_k 张图片，就跳出循环
            img1_path, best_match, ssim_value = ssim_results[i]
            if 0.70 < ssim_value < 1:
                executor.submit(process_image_pair, img1_path, img2_path,
                                save_path_img, save_path_mask, mask1_dir, mask2_dir)
                count += 1
=======
<<<<<<< HEAD
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor

def calculate_ssim(img1, img2, data_range=255, multichannel=True):
    # 计算图像区域的最小边长
    min_dim = min(img1.shape[0], img1.shape[1])
    # 默认使用 win_size=7，但如果图像太小则调整为不超过 min_dim 的最大奇数（最小保证为3）
    win_size = 7
    if min_dim < 7:
        win_size = min_dim if (min_dim % 2 == 1) else min_dim - 1
        win_size = max(win_size, 3)
    ssim_value = ssim(img1, img2, data_range=data_range, win_size=win_size,
                      channel_axis=2, multichannel=multichannel)
    return ssim_value

def find_best_match(img1, img2):
    img1 = cv2.imread(img1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2, cv2.IMREAD_COLOR)

    # 使用快速模板匹配算法 (cv2.TM_CCOEFF_NORMED)
    result = cv2.matchTemplate(img2, img1, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    best_match = max_loc

    # 获取img1的宽度和高度
    h, w, _ = img1.shape

    # 定义要计算SSIM的边缘区域
    edge_width = 30
    edge_regions = [
        img1[:, :edge_width, :],  # 左边缘
        img1[:, -edge_width:, :],  # 右边缘
        img1[:edge_width, :, :],    # 上边缘
        img1[-edge_width:, :, :]    # 下边缘
    ]

    # 计算每个边缘区域的SSIM并取平均
    ssims = []
    for region in edge_regions:
        # 提取img2中对应区域
        region2 = img2[max_loc[1]:max_loc[1] + region.shape[0], max_loc[0]:max_loc[0] + region.shape[1], :]
        ssims.append(calculate_ssim(region, region2))
    best_ssim = np.mean(ssims)

    return best_match, best_ssim

def paste_image(img1_path, img2_path, x, y, output_filename):
    img2 = cv2.imread(img2_path)
    img1 = cv2.imread(img1_path)

    mask = np.zeros(img1.shape[:2], dtype=np.uint8)
    mask[:img1.shape[0], :img1.shape[1]] = 255

    center = (x + img1.shape[1] // 2, y + img1.shape[0] // 2)
    output = cv2.seamlessClone(img1, img2, mask, center, cv2.NORMAL_CLONE)
    cv2.imwrite(output_filename, output)

def paste_masks(img1_path, img2_path, x, y, output_filename):
    img2 = cv2.imread(img2_path)
    img1 = cv2.imread(img1_path)

    img2[y:y + img1.shape[0], x:x + img1.shape[1]] = img1
    cv2.imwrite(output_filename, img2)

def process_image_pair(img1_path, img2_path, save_path_img, save_path_mask, mask1_dir, mask2_dir, ssim_threshold=0.7):
    best_match, best_ssim = find_best_match(img1_path, img2_path)

    if best_ssim > ssim_threshold:  # 仅当 SSIM 大于阈值时才执行粘贴
        img1_name = os.path.basename(img1_path).split('.')[0]
        img2_name = os.path.basename(img2_path).split('.')[0]
        paste_image(img1_path, img2_path, best_match[0], best_match[1],
                    os.path.join(save_path_img, f"{img1_name}_{img2_name}.png"))
        paste_masks(os.path.join(mask1_dir, os.path.basename(img1_path)),
                    os.path.join(mask2_dir, os.path.basename(img2_path)),
                    best_match[0], best_match[1],
                    os.path.join(save_path_mask, f"{img1_name}_{img2_name}.png"))
        print(f"Pasted {img2_path} with ssim {best_ssim}")

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 配置各个文件夹路径
img1_dir = 'F:/dark/real/candidate_images'
img2_dir = 'F:/dark/real/bgnew'
mask1_dir = 'F:/dark/real/candidate_masks'
mask2_dir = 'F:/dark/real/bgnewmask'
save_path_img = 'F:/dark/real/images'
save_path_mask = 'F:/dark/real/mask'

create_folder(save_path_img)
create_folder(save_path_mask)
images1 = [os.path.join(img1_dir, f) for f in os.listdir(img1_dir) if f.endswith('.png')]
images2 = [os.path.join(img2_dir, f) for f in os.listdir(img2_dir) if f.endswith('.png')]

# SIRST图片循环
with ThreadPoolExecutor() as executor:
    for img2_path in images2:
        print('SIRST is image path :', img2_path)
        ssim_results = list(executor.map(lambda img1_path: (img1_path, *find_best_match(img1_path, img2_path)), images1))

        # 按 ssim 降序排序
        ssim_results.sort(key=lambda x: x[2], reverse=True)

        # 只对前 top_k 个高分的图片执行粘贴操作
        top_k = 1
        count = 0  # 用于计数已粘贴的图像数量
        for i in range(len(ssim_results)):
            if count >= top_k:
                break  # 如果已经粘贴了 top_k 张图片，就跳出循环
            img1_path, best_match, ssim_value = ssim_results[i]
            if 0.70 < ssim_value < 1:
                executor.submit(process_image_pair, img1_path, img2_path,
                                save_path_img, save_path_mask, mask1_dir, mask2_dir)
                count += 1
=======
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor

def calculate_ssim(img1, img2, data_range=255, multichannel=True):
    # 计算图像区域的最小边长
    min_dim = min(img1.shape[0], img1.shape[1])
    # 默认使用 win_size=7，但如果图像太小则调整为不超过 min_dim 的最大奇数（最小保证为3）
    win_size = 7
    if min_dim < 7:
        win_size = min_dim if (min_dim % 2 == 1) else min_dim - 1
        win_size = max(win_size, 3)
    ssim_value = ssim(img1, img2, data_range=data_range, win_size=win_size,
                      channel_axis=2, multichannel=multichannel)
    return ssim_value

def find_best_match(img1, img2):
    img1 = cv2.imread(img1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(img2, cv2.IMREAD_COLOR)

    # 使用快速模板匹配算法 (cv2.TM_CCOEFF_NORMED)
    result = cv2.matchTemplate(img2, img1, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    best_match = max_loc

    # 获取img1的宽度和高度
    h, w, _ = img1.shape

    # 定义要计算SSIM的边缘区域
    edge_width = 30
    edge_regions = [
        img1[:, :edge_width, :],  # 左边缘
        img1[:, -edge_width:, :],  # 右边缘
        img1[:edge_width, :, :],    # 上边缘
        img1[-edge_width:, :, :]    # 下边缘
    ]

    # 计算每个边缘区域的SSIM并取平均
    ssims = []
    for region in edge_regions:
        # 提取img2中对应区域
        region2 = img2[max_loc[1]:max_loc[1] + region.shape[0], max_loc[0]:max_loc[0] + region.shape[1], :]
        ssims.append(calculate_ssim(region, region2))
    best_ssim = np.mean(ssims)

    return best_match, best_ssim

def paste_image(img1_path, img2_path, x, y, output_filename):
    img2 = cv2.imread(img2_path)
    img1 = cv2.imread(img1_path)

    mask = np.zeros(img1.shape[:2], dtype=np.uint8)
    mask[:img1.shape[0], :img1.shape[1]] = 255

    center = (x + img1.shape[1] // 2, y + img1.shape[0] // 2)
    output = cv2.seamlessClone(img1, img2, mask, center, cv2.NORMAL_CLONE)
    cv2.imwrite(output_filename, output)

def paste_masks(img1_path, img2_path, x, y, output_filename):
    img2 = cv2.imread(img2_path)
    img1 = cv2.imread(img1_path)

    img2[y:y + img1.shape[0], x:x + img1.shape[1]] = img1
    cv2.imwrite(output_filename, img2)

def process_image_pair(img1_path, img2_path, save_path_img, save_path_mask, mask1_dir, mask2_dir, ssim_threshold=0.7):
    best_match, best_ssim = find_best_match(img1_path, img2_path)

    if best_ssim > ssim_threshold:  # 仅当 SSIM 大于阈值时才执行粘贴
        img1_name = os.path.basename(img1_path).split('.')[0]
        img2_name = os.path.basename(img2_path).split('.')[0]
        paste_image(img1_path, img2_path, best_match[0], best_match[1],
                    os.path.join(save_path_img, f"{img1_name}_{img2_name}.png"))
        paste_masks(os.path.join(mask1_dir, os.path.basename(img1_path)),
                    os.path.join(mask2_dir, os.path.basename(img2_path)),
                    best_match[0], best_match[1],
                    os.path.join(save_path_mask, f"{img1_name}_{img2_name}.png"))
        print(f"Pasted {img2_path} with ssim {best_ssim}")

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 配置各个文件夹路径
img1_dir = 'F:/dark/real/candidate_images'
img2_dir = 'F:/dark/real/bgnew'
mask1_dir = 'F:/dark/real/candidate_masks'
mask2_dir = 'F:/dark/real/bgnewmask'
save_path_img = 'F:/dark/real/images'
save_path_mask = 'F:/dark/real/mask'

create_folder(save_path_img)
create_folder(save_path_mask)
images1 = [os.path.join(img1_dir, f) for f in os.listdir(img1_dir) if f.endswith('.png')]
images2 = [os.path.join(img2_dir, f) for f in os.listdir(img2_dir) if f.endswith('.png')]

# SIRST图片循环
with ThreadPoolExecutor() as executor:
    for img2_path in images2:
        print('SIRST is image path :', img2_path)
        ssim_results = list(executor.map(lambda img1_path: (img1_path, *find_best_match(img1_path, img2_path)), images1))

        # 按 ssim 降序排序
        ssim_results.sort(key=lambda x: x[2], reverse=True)

        # 只对前 top_k 个高分的图片执行粘贴操作
        top_k = 1
        count = 0  # 用于计数已粘贴的图像数量
        for i in range(len(ssim_results)):
            if count >= top_k:
                break  # 如果已经粘贴了 top_k 张图片，就跳出循环
            img1_path, best_match, ssim_value = ssim_results[i]
            if 0.70 < ssim_value < 1:
                executor.submit(process_image_pair, img1_path, img2_path,
                                save_path_img, save_path_mask, mask1_dir, mask2_dir)
                count += 1
>>>>>>> origin/feat/update
>>>>>>> 21a2898 (update)
