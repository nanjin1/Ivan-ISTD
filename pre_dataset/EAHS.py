<<<<<<< HEAD
import os
import numpy as np
from PIL import Image
import cv2

def load_and_threshold_image(image_path, threshold=0.5):
    """
    加载灰度图并将其转换为二值图
    """
    image = Image.open(image_path)
    image = np.array(image)

    # 确保是灰度图
    if len(image.shape) == 2:
        binary_image = (image > threshold * 255).astype(np.uint8) * 255
    else:
        raise ValueError("Input image is not a grayscale image.")

    return binary_image

def load_masks_from_folder(gt_masks_folder, pred_masks_folder, threshold=0.5):
    """
    从文件夹中加载真实 mask 和预测 mask  
    假设两个文件夹中的文件名一一对应（通过排序实现对应）
    """
    gt_mask_files = sorted(os.listdir(gt_masks_folder))
    pred_mask_files = sorted(os.listdir(pred_masks_folder))

    # 确保文件数量相同
    assert len(gt_mask_files) == len(pred_mask_files), "真实 mask 和预测 mask 的数量不一致！"

    gt_masks = []
    pred_masks = []

    for gt_file, pred_file in zip(gt_mask_files, pred_mask_files):
        gt_mask_path = os.path.join(gt_masks_folder, gt_file)
        pred_mask_path = os.path.join(pred_masks_folder, pred_file)

        gt_mask = load_and_threshold_image(gt_mask_path, threshold)
        pred_mask = load_and_threshold_image(pred_mask_path, threshold)

        gt_masks.append(gt_mask)
        pred_masks.append(pred_mask)

    return gt_masks, pred_masks

def get_bounding_box(mask, center):
    """
    根据给定中心点，从 mask 中查找包含该中心点的连通区域，
    返回该区域的外接矩形 (x, y, w, h)；若未找到则返回 None。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.pointPolygonTest(cnt, center, False) >= 0:
            return cv2.boundingRect(cnt)
    return None

def compute_edge_weight(M_gt):
    """
    计算边缘强化权重
    """
    grad_x = cv2.Sobel(M_gt, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(M_gt, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    max_grad = np.max(grad_magnitude)
    if max_grad == 0:
        max_grad = 1
    edge_weight = 1 + grad_magnitude / max_grad

    return edge_weight

def compute_difference_map(M_gt, M_pred, edge_weight):
    """
    计算像素级差异图 D
    """
    diff_map = np.abs(M_gt - M_pred) * edge_weight
    return diff_map

def compute_iou(Oi, gt_bbox):
    """
    计算交并比（IoU）  
    Oi 和 gt_bbox 格式均为 [u_min, v_min, u_max, v_max]
    """
    xA = max(Oi[0], gt_bbox[0])
    yA = max(Oi[1], gt_bbox[1])
    xB = min(Oi[2], gt_bbox[2])
    yB = min(Oi[3], gt_bbox[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (Oi[2] - Oi[0]) * (Oi[3] - Oi[1])
    boxBArea = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    union = boxAArea + boxBArea - interArea

    return interArea / union if union > 0 else 0

def compute_difficulty_score(Oi, diff_map, edge_weight, sigma=0.15):
    """
    计算候选区域 Oi 的困难度 H(Oi)  
    Oi 格式为 [u_min, v_min, u_max, v_max]
    """
    u_min, v_min, u_max, v_max = Oi
    region_diff = diff_map[v_min:v_max, u_min:u_max]
    region_edge_weight = edge_weight[v_min:v_max, u_min:u_max]
    
    area = (v_max - v_min) * (u_max - u_min)
    if area == 0:
        return 0
    weighted_diff = np.sum(region_diff * region_edge_weight)
    
    # 示例中假设真实框为固定值（实际情况中应使用对应真实标注）
    ground_truth_bbox = [0, 0, 100, 100]
    IoU = compute_iou(Oi, ground_truth_bbox)

    difficulty_score = (weighted_diff / area) * np.exp(-IoU / sigma)
    return difficulty_score

def select_top_k_samples(difficulty_scores, q=80):
    """
    按照困难度选择 Top-K 样本，q 表示百分比阈值  
    返回选中候选框在 difficulty_scores 列表中的索引列表
    """
    if len(difficulty_scores) == 0:
        return []
    threshold = np.percentile(difficulty_scores, 100 - q)
    selected_samples = [i for i, score in enumerate(difficulty_scores) if score > threshold]
    return selected_samples

def process_candidates(gt_masks, pred_masks, centers_file, sigma=0.15, q=80):
    """
    根据 txt 文件中每幅图像的中心点生成候选目标框，并计算困难度  
    返回一个列表，每个元素为 (img_index, candidate_bbox, difficulty_score)
    """
    difficulty_info = []  # 存储 (图像索引, 候选框, 分数)
    all_scores = []

    with open(centers_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # 确保当前图像存在对应的 mask
        if i >= len(gt_masks) or i >= len(pred_masks):
            break

        # 解析中心点，例如 "(513,330)|(320,328)|(208,328)"
        centers = []
        for coord in line.strip().split('|'):
            coord = coord.strip()
            if coord:
                try:
                    coord = coord.strip("()")
                    x, y = map(int, coord.split(','))
                    centers.append((x, y))
                except Exception as e:
                    print(f"解析中心点错误：{coord}, 错误信息: {e}")
        
        # 对当前图像的每个中心点生成候选区域
        for center in centers:
            bbox = get_bounding_box(gt_masks[i], center)
            if bbox is None:
                continue
            # get_bounding_box 返回 (x, y, w, h)，转换为 [u_min, v_min, u_max, v_max]
            u_min, v_min, w, h = bbox
            candidate_bbox = [u_min, v_min, u_min + w, v_min + h]

            # 计算当前图像的边缘权重与差异图
            edge_weight = compute_edge_weight(gt_masks[i])
            diff_map = compute_difference_map(gt_masks[i], pred_masks[i], edge_weight)
            score = compute_difficulty_score(candidate_bbox, diff_map, edge_weight, sigma)

            difficulty_info.append((i, candidate_bbox, score))
            all_scores.append(score)

    # 根据所有候选框的分数选择 Top-K（例如选择困难度高于 80% 分位数的候选框）
    if len(all_scores) == 0:
        print("没有候选目标！")
        return []

    threshold = np.percentile(all_scores, 100 - q)
    selected_candidates = [info for info in difficulty_info if info[2] > threshold]

    return selected_candidates

def save_selected_candidates(selected_candidates, images_folder, gt_masks_folder,
                             output_candidate_image_folder, output_candidate_mask_folder):
    """
    对于选中的候选目标，从原始图像和真实 mask 中裁剪候选区域，
    固定候选区域为 40x40 大小，且如果其中白色区域超过 90% 则跳过该候选目标。
    """
    os.makedirs(output_candidate_image_folder, exist_ok=True)
    os.makedirs(output_candidate_mask_folder, exist_ok=True)

    image_files = sorted(os.listdir(images_folder))
    mask_files = sorted(os.listdir(gt_masks_folder))

    def crop_patch_with_padding(image, x_min, y_min, x_max, y_max, patch_size=(40, 40)):
        """
        从 image 中裁剪 [x_min, y_min, x_max, y_max] 区域，
        如果超出边界则用 0 填充，最终输出固定 patch_size 大小的图像。
        """
        h, w = image.shape[:2]
        if len(image.shape) == 3:
            patch = np.zeros((patch_size[1], patch_size[0], image.shape[2]), dtype=image.dtype)
        else:
            patch = np.zeros((patch_size[1], patch_size[0]), dtype=image.dtype)
        # 计算图像中有效区域的坐标
        x_min_img = max(x_min, 0)
        y_min_img = max(y_min, 0)
        x_max_img = min(x_max, w)
        y_max_img = min(y_max, h)
        # 在 patch 中对应的放置位置
        x_offset = x_min_img - x_min if x_min < 0 else 0
        y_offset = y_min_img - y_min if y_min < 0 else 0
        patch[y_offset:y_offset+(y_max_img - y_min_img), x_offset:x_offset+(x_max_img - x_min_img)] = \
            image[y_min_img:y_max_img, x_min_img:x_max_img]
        return patch

    for idx, (img_index, candidate_bbox, score) in enumerate(selected_candidates):
        # 读取对应图像和 mask
        image_path = os.path.join(images_folder, image_files[img_index])
        mask_path = os.path.join(gt_masks_folder, mask_files[img_index])
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 根据候选框计算中心点，然后生成固定尺寸的 40x40 框
        u_min, v_min, u_max, v_max = candidate_bbox
        center_x = (u_min + u_max) // 2
        center_y = (v_min + v_max) // 2
        new_u_min = center_x - 20
        new_v_min = center_y - 20
        new_u_max = center_x + 20
        new_v_max = center_y + 20

        candidate_img = crop_patch_with_padding(image, new_u_min, new_v_min, new_u_max, new_v_max, patch_size=(40, 40))
        candidate_mask = crop_patch_with_padding(mask, new_u_min, new_v_min, new_u_max, new_v_max, patch_size=(40, 40))

        # 检查 candidate_mask 中白色像素的比例
        white_ratio = np.sum(candidate_mask == 255) / (40 * 40)
        if white_ratio > 0.9:
            print(f"跳过候选目标：图像 {image_files[img_index]}, 白色区域比例 {white_ratio:.2f} 超过90%")
            continue

        base_img_name = os.path.splitext(image_files[img_index])[0]
        candidate_img_name = f"{base_img_name}_cand_{idx:02d}.png"
        candidate_mask_name = f"{base_img_name}_cand_{idx:02d}.png"

        cv2.imwrite(os.path.join(output_candidate_image_folder, candidate_img_name), candidate_img)
        cv2.imwrite(os.path.join(output_candidate_mask_folder, candidate_mask_name), candidate_mask)
        print(f"保存候选目标：图像 {candidate_img_name}, 分数: {score:.4f}")


if __name__ == "__main__":
    # ----------------------------
    # 相关文件夹路径（请根据实际情况修改）
    # ----------------------------
    # 原始图像文件夹（用于裁剪候选目标）
    images_folder = "F:/dark/test/test_images_103_re"
    # 真实 mask 文件夹
    gt_masks_folder = "F:/dark/test/test_masks_103_re"
    # 预测 mask 文件夹
    pred_masks_folder = "F:/dark/temp/otherlogmo/SCTrans"
    # 包含每幅图像目标中心点的 txt 文件，每行格式：(x1,y1)|(x2,y2)|...
    centers_file = "F:/dark/test/test_masks_103_re.txt"
    # 输出候选目标图像保存文件夹
    output_candidate_image_folder = "F:/dark/candidate_images"
    # 输出候选目标 mask 保存文件夹
    output_candidate_mask_folder = "F:/dark/candidate_masks"

    

    # ----------------------------
    # 载入真实 mask 和预测 mask
    # ----------------------------
    gt_masks, pred_masks = load_masks_from_folder(gt_masks_folder, pred_masks_folder)

    # ----------------------------
    # 根据中心点生成候选目标并计算困难度，选择困难度较高的候选框
    # ----------------------------
    selected_candidates = process_candidates(gt_masks, pred_masks, centers_file, sigma=0.15, q=20)

    print("选出的候选目标（图像索引, BBox, 困难度分数）：")
    for info in selected_candidates:
        print(info)

    # ----------------------------
    # 将选出的候选目标区域（图像和 mask）裁剪后保存到指定输出文件夹中
    # ----------------------------
    save_selected_candidates(selected_candidates, images_folder, gt_masks_folder,
                             output_candidate_image_folder, output_candidate_mask_folder)
=======
<<<<<<< HEAD
import os
import numpy as np
from PIL import Image
import cv2

def load_and_threshold_image(image_path, threshold=0.5):
    """
    加载灰度图并将其转换为二值图
    """
    image = Image.open(image_path)
    image = np.array(image)

    # 确保是灰度图
    if len(image.shape) == 2:
        binary_image = (image > threshold * 255).astype(np.uint8) * 255
    else:
        raise ValueError("Input image is not a grayscale image.")

    return binary_image

def load_masks_from_folder(gt_masks_folder, pred_masks_folder, threshold=0.5):
    """
    从文件夹中加载真实 mask 和预测 mask  
    假设两个文件夹中的文件名一一对应（通过排序实现对应）
    """
    gt_mask_files = sorted(os.listdir(gt_masks_folder))
    pred_mask_files = sorted(os.listdir(pred_masks_folder))

    # 确保文件数量相同
    assert len(gt_mask_files) == len(pred_mask_files), "真实 mask 和预测 mask 的数量不一致！"

    gt_masks = []
    pred_masks = []

    for gt_file, pred_file in zip(gt_mask_files, pred_mask_files):
        gt_mask_path = os.path.join(gt_masks_folder, gt_file)
        pred_mask_path = os.path.join(pred_masks_folder, pred_file)

        gt_mask = load_and_threshold_image(gt_mask_path, threshold)
        pred_mask = load_and_threshold_image(pred_mask_path, threshold)

        gt_masks.append(gt_mask)
        pred_masks.append(pred_mask)

    return gt_masks, pred_masks

def get_bounding_box(mask, center):
    """
    根据给定中心点，从 mask 中查找包含该中心点的连通区域，
    返回该区域的外接矩形 (x, y, w, h)；若未找到则返回 None。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.pointPolygonTest(cnt, center, False) >= 0:
            return cv2.boundingRect(cnt)
    return None

def compute_edge_weight(M_gt):
    """
    计算边缘强化权重
    """
    grad_x = cv2.Sobel(M_gt, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(M_gt, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    max_grad = np.max(grad_magnitude)
    if max_grad == 0:
        max_grad = 1
    edge_weight = 1 + grad_magnitude / max_grad

    return edge_weight

def compute_difference_map(M_gt, M_pred, edge_weight):
    """
    计算像素级差异图 D
    """
    diff_map = np.abs(M_gt - M_pred) * edge_weight
    return diff_map

def compute_iou(Oi, gt_bbox):
    """
    计算交并比（IoU）  
    Oi 和 gt_bbox 格式均为 [u_min, v_min, u_max, v_max]
    """
    xA = max(Oi[0], gt_bbox[0])
    yA = max(Oi[1], gt_bbox[1])
    xB = min(Oi[2], gt_bbox[2])
    yB = min(Oi[3], gt_bbox[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (Oi[2] - Oi[0]) * (Oi[3] - Oi[1])
    boxBArea = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    union = boxAArea + boxBArea - interArea

    return interArea / union if union > 0 else 0

def compute_difficulty_score(Oi, diff_map, edge_weight, sigma=0.15):
    """
    计算候选区域 Oi 的困难度 H(Oi)  
    Oi 格式为 [u_min, v_min, u_max, v_max]
    """
    u_min, v_min, u_max, v_max = Oi
    region_diff = diff_map[v_min:v_max, u_min:u_max]
    region_edge_weight = edge_weight[v_min:v_max, u_min:u_max]
    
    area = (v_max - v_min) * (u_max - u_min)
    if area == 0:
        return 0
    weighted_diff = np.sum(region_diff * region_edge_weight)
    
    # 示例中假设真实框为固定值（实际情况中应使用对应真实标注）
    ground_truth_bbox = [0, 0, 100, 100]
    IoU = compute_iou(Oi, ground_truth_bbox)

    difficulty_score = (weighted_diff / area) * np.exp(-IoU / sigma)
    return difficulty_score

def select_top_k_samples(difficulty_scores, q=80):
    """
    按照困难度选择 Top-K 样本，q 表示百分比阈值  
    返回选中候选框在 difficulty_scores 列表中的索引列表
    """
    if len(difficulty_scores) == 0:
        return []
    threshold = np.percentile(difficulty_scores, 100 - q)
    selected_samples = [i for i, score in enumerate(difficulty_scores) if score > threshold]
    return selected_samples

def process_candidates(gt_masks, pred_masks, centers_file, sigma=0.15, q=80):
    """
    根据 txt 文件中每幅图像的中心点生成候选目标框，并计算困难度  
    返回一个列表，每个元素为 (img_index, candidate_bbox, difficulty_score)
    """
    difficulty_info = []  # 存储 (图像索引, 候选框, 分数)
    all_scores = []

    with open(centers_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # 确保当前图像存在对应的 mask
        if i >= len(gt_masks) or i >= len(pred_masks):
            break

        # 解析中心点，例如 "(513,330)|(320,328)|(208,328)"
        centers = []
        for coord in line.strip().split('|'):
            coord = coord.strip()
            if coord:
                try:
                    coord = coord.strip("()")
                    x, y = map(int, coord.split(','))
                    centers.append((x, y))
                except Exception as e:
                    print(f"解析中心点错误：{coord}, 错误信息: {e}")
        
        # 对当前图像的每个中心点生成候选区域
        for center in centers:
            bbox = get_bounding_box(gt_masks[i], center)
            if bbox is None:
                continue
            # get_bounding_box 返回 (x, y, w, h)，转换为 [u_min, v_min, u_max, v_max]
            u_min, v_min, w, h = bbox
            candidate_bbox = [u_min, v_min, u_min + w, v_min + h]

            # 计算当前图像的边缘权重与差异图
            edge_weight = compute_edge_weight(gt_masks[i])
            diff_map = compute_difference_map(gt_masks[i], pred_masks[i], edge_weight)
            score = compute_difficulty_score(candidate_bbox, diff_map, edge_weight, sigma)

            difficulty_info.append((i, candidate_bbox, score))
            all_scores.append(score)

    # 根据所有候选框的分数选择 Top-K（例如选择困难度高于 80% 分位数的候选框）
    if len(all_scores) == 0:
        print("没有候选目标！")
        return []

    threshold = np.percentile(all_scores, 100 - q)
    selected_candidates = [info for info in difficulty_info if info[2] > threshold]

    return selected_candidates

def save_selected_candidates(selected_candidates, images_folder, gt_masks_folder,
                             output_candidate_image_folder, output_candidate_mask_folder):
    """
    对于选中的候选目标，从原始图像和真实 mask 中裁剪候选区域，
    固定候选区域为 40x40 大小，且如果其中白色区域超过 90% 则跳过该候选目标。
    """
    os.makedirs(output_candidate_image_folder, exist_ok=True)
    os.makedirs(output_candidate_mask_folder, exist_ok=True)

    image_files = sorted(os.listdir(images_folder))
    mask_files = sorted(os.listdir(gt_masks_folder))

    def crop_patch_with_padding(image, x_min, y_min, x_max, y_max, patch_size=(40, 40)):
        """
        从 image 中裁剪 [x_min, y_min, x_max, y_max] 区域，
        如果超出边界则用 0 填充，最终输出固定 patch_size 大小的图像。
        """
        h, w = image.shape[:2]
        if len(image.shape) == 3:
            patch = np.zeros((patch_size[1], patch_size[0], image.shape[2]), dtype=image.dtype)
        else:
            patch = np.zeros((patch_size[1], patch_size[0]), dtype=image.dtype)
        # 计算图像中有效区域的坐标
        x_min_img = max(x_min, 0)
        y_min_img = max(y_min, 0)
        x_max_img = min(x_max, w)
        y_max_img = min(y_max, h)
        # 在 patch 中对应的放置位置
        x_offset = x_min_img - x_min if x_min < 0 else 0
        y_offset = y_min_img - y_min if y_min < 0 else 0
        patch[y_offset:y_offset+(y_max_img - y_min_img), x_offset:x_offset+(x_max_img - x_min_img)] = \
            image[y_min_img:y_max_img, x_min_img:x_max_img]
        return patch

    for idx, (img_index, candidate_bbox, score) in enumerate(selected_candidates):
        # 读取对应图像和 mask
        image_path = os.path.join(images_folder, image_files[img_index])
        mask_path = os.path.join(gt_masks_folder, mask_files[img_index])
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 根据候选框计算中心点，然后生成固定尺寸的 40x40 框
        u_min, v_min, u_max, v_max = candidate_bbox
        center_x = (u_min + u_max) // 2
        center_y = (v_min + v_max) // 2
        new_u_min = center_x - 20
        new_v_min = center_y - 20
        new_u_max = center_x + 20
        new_v_max = center_y + 20

        candidate_img = crop_patch_with_padding(image, new_u_min, new_v_min, new_u_max, new_v_max, patch_size=(40, 40))
        candidate_mask = crop_patch_with_padding(mask, new_u_min, new_v_min, new_u_max, new_v_max, patch_size=(40, 40))

        # 检查 candidate_mask 中白色像素的比例
        white_ratio = np.sum(candidate_mask == 255) / (40 * 40)
        if white_ratio > 0.9:
            print(f"跳过候选目标：图像 {image_files[img_index]}, 白色区域比例 {white_ratio:.2f} 超过90%")
            continue

        base_img_name = os.path.splitext(image_files[img_index])[0]
        candidate_img_name = f"{base_img_name}_cand_{idx:02d}.png"
        candidate_mask_name = f"{base_img_name}_cand_{idx:02d}.png"

        cv2.imwrite(os.path.join(output_candidate_image_folder, candidate_img_name), candidate_img)
        cv2.imwrite(os.path.join(output_candidate_mask_folder, candidate_mask_name), candidate_mask)
        print(f"保存候选目标：图像 {candidate_img_name}, 分数: {score:.4f}")


if __name__ == "__main__":
    # ----------------------------
    # 相关文件夹路径（请根据实际情况修改）
    # ----------------------------
    # 原始图像文件夹（用于裁剪候选目标）
    images_folder = "F:/dark/test/test_images_103_re"
    # 真实 mask 文件夹
    gt_masks_folder = "F:/dark/test/test_masks_103_re"
    # 预测 mask 文件夹
    pred_masks_folder = "F:/dark/temp/otherlogmo/SCTrans"
    # 包含每幅图像目标中心点的 txt 文件，每行格式：(x1,y1)|(x2,y2)|...
    centers_file = "F:/dark/test/test_masks_103_re.txt"
    # 输出候选目标图像保存文件夹
    output_candidate_image_folder = "F:/dark/candidate_images"
    # 输出候选目标 mask 保存文件夹
    output_candidate_mask_folder = "F:/dark/candidate_masks"

    

    # ----------------------------
    # 载入真实 mask 和预测 mask
    # ----------------------------
    gt_masks, pred_masks = load_masks_from_folder(gt_masks_folder, pred_masks_folder)

    # ----------------------------
    # 根据中心点生成候选目标并计算困难度，选择困难度较高的候选框
    # ----------------------------
    selected_candidates = process_candidates(gt_masks, pred_masks, centers_file, sigma=0.15, q=20)

    print("选出的候选目标（图像索引, BBox, 困难度分数）：")
    for info in selected_candidates:
        print(info)

    # ----------------------------
    # 将选出的候选目标区域（图像和 mask）裁剪后保存到指定输出文件夹中
    # ----------------------------
    save_selected_candidates(selected_candidates, images_folder, gt_masks_folder,
                             output_candidate_image_folder, output_candidate_mask_folder)
=======
import os
import numpy as np
from PIL import Image
import cv2

def load_and_threshold_image(image_path, threshold=0.5):
    """
    加载灰度图并将其转换为二值图
    """
    image = Image.open(image_path)
    image = np.array(image)

    # 确保是灰度图
    if len(image.shape) == 2:
        binary_image = (image > threshold * 255).astype(np.uint8) * 255
    else:
        raise ValueError("Input image is not a grayscale image.")

    return binary_image

def load_masks_from_folder(gt_masks_folder, pred_masks_folder, threshold=0.5):
    """
    从文件夹中加载真实 mask 和预测 mask  
    假设两个文件夹中的文件名一一对应（通过排序实现对应）
    """
    gt_mask_files = sorted(os.listdir(gt_masks_folder))
    pred_mask_files = sorted(os.listdir(pred_masks_folder))

    # 确保文件数量相同
    assert len(gt_mask_files) == len(pred_mask_files), "真实 mask 和预测 mask 的数量不一致！"

    gt_masks = []
    pred_masks = []

    for gt_file, pred_file in zip(gt_mask_files, pred_mask_files):
        gt_mask_path = os.path.join(gt_masks_folder, gt_file)
        pred_mask_path = os.path.join(pred_masks_folder, pred_file)

        gt_mask = load_and_threshold_image(gt_mask_path, threshold)
        pred_mask = load_and_threshold_image(pred_mask_path, threshold)

        gt_masks.append(gt_mask)
        pred_masks.append(pred_mask)

    return gt_masks, pred_masks

def get_bounding_box(mask, center):
    """
    根据给定中心点，从 mask 中查找包含该中心点的连通区域，
    返回该区域的外接矩形 (x, y, w, h)；若未找到则返回 None。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.pointPolygonTest(cnt, center, False) >= 0:
            return cv2.boundingRect(cnt)
    return None

def compute_edge_weight(M_gt):
    """
    计算边缘强化权重
    """
    grad_x = cv2.Sobel(M_gt, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(M_gt, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    max_grad = np.max(grad_magnitude)
    if max_grad == 0:
        max_grad = 1
    edge_weight = 1 + grad_magnitude / max_grad

    return edge_weight

def compute_difference_map(M_gt, M_pred, edge_weight):
    """
    计算像素级差异图 D
    """
    diff_map = np.abs(M_gt - M_pred) * edge_weight
    return diff_map

def compute_iou(Oi, gt_bbox):
    """
    计算交并比（IoU）  
    Oi 和 gt_bbox 格式均为 [u_min, v_min, u_max, v_max]
    """
    xA = max(Oi[0], gt_bbox[0])
    yA = max(Oi[1], gt_bbox[1])
    xB = min(Oi[2], gt_bbox[2])
    yB = min(Oi[3], gt_bbox[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (Oi[2] - Oi[0]) * (Oi[3] - Oi[1])
    boxBArea = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
    union = boxAArea + boxBArea - interArea

    return interArea / union if union > 0 else 0

def compute_difficulty_score(Oi, diff_map, edge_weight, sigma=0.15):
    """
    计算候选区域 Oi 的困难度 H(Oi)  
    Oi 格式为 [u_min, v_min, u_max, v_max]
    """
    u_min, v_min, u_max, v_max = Oi
    region_diff = diff_map[v_min:v_max, u_min:u_max]
    region_edge_weight = edge_weight[v_min:v_max, u_min:u_max]
    
    area = (v_max - v_min) * (u_max - u_min)
    if area == 0:
        return 0
    weighted_diff = np.sum(region_diff * region_edge_weight)
    
    # 示例中假设真实框为固定值（实际情况中应使用对应真实标注）
    ground_truth_bbox = [0, 0, 100, 100]
    IoU = compute_iou(Oi, ground_truth_bbox)

    difficulty_score = (weighted_diff / area) * np.exp(-IoU / sigma)
    return difficulty_score

def select_top_k_samples(difficulty_scores, q=80):
    """
    按照困难度选择 Top-K 样本，q 表示百分比阈值  
    返回选中候选框在 difficulty_scores 列表中的索引列表
    """
    if len(difficulty_scores) == 0:
        return []
    threshold = np.percentile(difficulty_scores, 100 - q)
    selected_samples = [i for i, score in enumerate(difficulty_scores) if score > threshold]
    return selected_samples

def process_candidates(gt_masks, pred_masks, centers_file, sigma=0.15, q=80):
    """
    根据 txt 文件中每幅图像的中心点生成候选目标框，并计算困难度  
    返回一个列表，每个元素为 (img_index, candidate_bbox, difficulty_score)
    """
    difficulty_info = []  # 存储 (图像索引, 候选框, 分数)
    all_scores = []

    with open(centers_file, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # 确保当前图像存在对应的 mask
        if i >= len(gt_masks) or i >= len(pred_masks):
            break

        # 解析中心点，例如 "(513,330)|(320,328)|(208,328)"
        centers = []
        for coord in line.strip().split('|'):
            coord = coord.strip()
            if coord:
                try:
                    coord = coord.strip("()")
                    x, y = map(int, coord.split(','))
                    centers.append((x, y))
                except Exception as e:
                    print(f"解析中心点错误：{coord}, 错误信息: {e}")
        
        # 对当前图像的每个中心点生成候选区域
        for center in centers:
            bbox = get_bounding_box(gt_masks[i], center)
            if bbox is None:
                continue
            # get_bounding_box 返回 (x, y, w, h)，转换为 [u_min, v_min, u_max, v_max]
            u_min, v_min, w, h = bbox
            candidate_bbox = [u_min, v_min, u_min + w, v_min + h]

            # 计算当前图像的边缘权重与差异图
            edge_weight = compute_edge_weight(gt_masks[i])
            diff_map = compute_difference_map(gt_masks[i], pred_masks[i], edge_weight)
            score = compute_difficulty_score(candidate_bbox, diff_map, edge_weight, sigma)

            difficulty_info.append((i, candidate_bbox, score))
            all_scores.append(score)

    # 根据所有候选框的分数选择 Top-K（例如选择困难度高于 80% 分位数的候选框）
    if len(all_scores) == 0:
        print("没有候选目标！")
        return []

    threshold = np.percentile(all_scores, 100 - q)
    selected_candidates = [info for info in difficulty_info if info[2] > threshold]

    return selected_candidates

def save_selected_candidates(selected_candidates, images_folder, gt_masks_folder,
                             output_candidate_image_folder, output_candidate_mask_folder):
    """
    对于选中的候选目标，从原始图像和真实 mask 中裁剪候选区域，
    固定候选区域为 40x40 大小，且如果其中白色区域超过 90% 则跳过该候选目标。
    """
    os.makedirs(output_candidate_image_folder, exist_ok=True)
    os.makedirs(output_candidate_mask_folder, exist_ok=True)

    image_files = sorted(os.listdir(images_folder))
    mask_files = sorted(os.listdir(gt_masks_folder))

    def crop_patch_with_padding(image, x_min, y_min, x_max, y_max, patch_size=(40, 40)):
        """
        从 image 中裁剪 [x_min, y_min, x_max, y_max] 区域，
        如果超出边界则用 0 填充，最终输出固定 patch_size 大小的图像。
        """
        h, w = image.shape[:2]
        if len(image.shape) == 3:
            patch = np.zeros((patch_size[1], patch_size[0], image.shape[2]), dtype=image.dtype)
        else:
            patch = np.zeros((patch_size[1], patch_size[0]), dtype=image.dtype)
        # 计算图像中有效区域的坐标
        x_min_img = max(x_min, 0)
        y_min_img = max(y_min, 0)
        x_max_img = min(x_max, w)
        y_max_img = min(y_max, h)
        # 在 patch 中对应的放置位置
        x_offset = x_min_img - x_min if x_min < 0 else 0
        y_offset = y_min_img - y_min if y_min < 0 else 0
        patch[y_offset:y_offset+(y_max_img - y_min_img), x_offset:x_offset+(x_max_img - x_min_img)] = \
            image[y_min_img:y_max_img, x_min_img:x_max_img]
        return patch

    for idx, (img_index, candidate_bbox, score) in enumerate(selected_candidates):
        # 读取对应图像和 mask
        image_path = os.path.join(images_folder, image_files[img_index])
        mask_path = os.path.join(gt_masks_folder, mask_files[img_index])
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 根据候选框计算中心点，然后生成固定尺寸的 40x40 框
        u_min, v_min, u_max, v_max = candidate_bbox
        center_x = (u_min + u_max) // 2
        center_y = (v_min + v_max) // 2
        new_u_min = center_x - 20
        new_v_min = center_y - 20
        new_u_max = center_x + 20
        new_v_max = center_y + 20

        candidate_img = crop_patch_with_padding(image, new_u_min, new_v_min, new_u_max, new_v_max, patch_size=(40, 40))
        candidate_mask = crop_patch_with_padding(mask, new_u_min, new_v_min, new_u_max, new_v_max, patch_size=(40, 40))

        # 检查 candidate_mask 中白色像素的比例
        white_ratio = np.sum(candidate_mask == 255) / (40 * 40)
        if white_ratio > 0.9:
            print(f"跳过候选目标：图像 {image_files[img_index]}, 白色区域比例 {white_ratio:.2f} 超过90%")
            continue

        base_img_name = os.path.splitext(image_files[img_index])[0]
        candidate_img_name = f"{base_img_name}_cand_{idx:02d}.png"
        candidate_mask_name = f"{base_img_name}_cand_{idx:02d}.png"

        cv2.imwrite(os.path.join(output_candidate_image_folder, candidate_img_name), candidate_img)
        cv2.imwrite(os.path.join(output_candidate_mask_folder, candidate_mask_name), candidate_mask)
        print(f"保存候选目标：图像 {candidate_img_name}, 分数: {score:.4f}")


if __name__ == "__main__":
    # ----------------------------
    # 相关文件夹路径（请根据实际情况修改）
    # ----------------------------
    # 原始图像文件夹（用于裁剪候选目标）
    images_folder = "F:/dark/test/test_images_103_re"
    # 真实 mask 文件夹
    gt_masks_folder = "F:/dark/test/test_masks_103_re"
    # 预测 mask 文件夹
    pred_masks_folder = "F:/dark/temp/otherlogmo/SCTrans"
    # 包含每幅图像目标中心点的 txt 文件，每行格式：(x1,y1)|(x2,y2)|...
    centers_file = "F:/dark/test/test_masks_103_re.txt"
    # 输出候选目标图像保存文件夹
    output_candidate_image_folder = "F:/dark/candidate_images"
    # 输出候选目标 mask 保存文件夹
    output_candidate_mask_folder = "F:/dark/candidate_masks"

    

    # ----------------------------
    # 载入真实 mask 和预测 mask
    # ----------------------------
    gt_masks, pred_masks = load_masks_from_folder(gt_masks_folder, pred_masks_folder)

    # ----------------------------
    # 根据中心点生成候选目标并计算困难度，选择困难度较高的候选框
    # ----------------------------
    selected_candidates = process_candidates(gt_masks, pred_masks, centers_file, sigma=0.15, q=20)

    print("选出的候选目标（图像索引, BBox, 困难度分数）：")
    for info in selected_candidates:
        print(info)

    # ----------------------------
    # 将选出的候选目标区域（图像和 mask）裁剪后保存到指定输出文件夹中
    # ----------------------------
    save_selected_candidates(selected_candidates, images_folder, gt_masks_folder,
                             output_candidate_image_folder, output_candidate_mask_folder)
>>>>>>> origin/feat/update
>>>>>>> 21a2898 (update)
