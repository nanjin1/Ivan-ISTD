<<<<<<< HEAD
import cv2
import numpy as np
import os
import pywt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt

# 方法2：设置全局字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def multi_band_filter(gray_img, wavelet='db4', level=3, filter_ratio=0.1):
    """
    多频带联合滤波
    Args:
        gray_img: 灰度图
        wavelet: 小波基类型，可选'db4'/'haar'
        level: 小波分解层数
        filter_ratio: 高频子带傅里叶滤波保留比例
    Returns:
        filtered: 滤波后的图像
    """
    # 小波分解
    coeffs = pywt.wavedec2(gray_img, wavelet, level=level)
    
    # 将tuple转换为list，以便修改
    coeffs_list = list(coeffs)
    
    # 对每个高频子带进行傅里叶滤波
    for i in range(1, len(coeffs_list)):
        # 每个层级的高频子带也需要转换为list
        detail_coeffs = list(coeffs_list[i])
        
        for j in range(len(detail_coeffs)):
            sub_band = detail_coeffs[j]
            rows, cols = sub_band.shape
            # 傅里叶低通滤波
            f = np.fft.fft2(sub_band)
            fshift = np.fft.fftshift(f)
            crow, ccol = rows//2, cols//2
            mask = np.zeros_like(fshift)
            mask[
                int(crow - filter_ratio*rows):int(crow + filter_ratio*rows),
                int(ccol - filter_ratio*cols):int(ccol + filter_ratio*cols)
            ] = 1
            fshift_filtered = fshift * mask
            sub_band_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))
            detail_coeffs[j] = sub_band_filtered
        
        # 更新修改后的子带回到coeffs_list
        coeffs_list[i] = tuple(detail_coeffs)
    
    # 小波重构并归一化
    filtered = pywt.waverec2(tuple(coeffs_list), wavelet)
    return cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def calculate_block_features(filtered_gray, original_gray, grid_size=(8,6)):
    """计算每个区块的特征和小目标概率"""
    H, W = filtered_gray.shape
    block_h = H // grid_size[0]
    block_w = W // grid_size[1]
    blocks = []
    
    # 调整Canny阈值适应滤波后特征
    edges = cv2.Canny(filtered_gray, 30, 100)
    
    # 额外计算Laplacian算子响应 (增强对小目标的检测能力)
    laplacian = cv2.Laplacian(filtered_gray, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian)
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            y = i * block_h
            x = j * block_w
            h = block_h if i != grid_size[0]-1 else H - y
            w = block_w if j != grid_size[1]-1 else W - x
            
            # 使用滤波后图像计算特征
            block_filtered = filtered_gray[y:y+h, x:x+w]
            edge_block = edges[y:y+h, x:x+w]
            laplacian_block = laplacian_abs[y:y+h, x:x+w]
            original_block = original_gray[y:y+h, x:x+w]
            
            edge_density = np.mean(edge_block) / 255.0
            variance = np.var(block_filtered)
            laplacian_response = np.mean(laplacian_block)
            
            # 局部信噪比估计 (对小目标检测有帮助)
            local_snr = 0
            if np.std(block_filtered) > 0:
                local_snr = np.mean(block_filtered) / np.std(block_filtered)
            
            # 计算原始块与滤波后块的相关性
            corr = cv2.matchTemplate(
                cv2.normalize(original_block, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F),
                cv2.normalize(block_filtered, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F),
                cv2.TM_CCORR_NORMED
            )[0][0]
            
            blocks.append({
                'edge_density': edge_density,
                'variance': variance,
                'coords': (y, x, h, w),
                'corr': corr,
                'laplacian_response': laplacian_response,
                'local_snr': local_snr,
                'grid_pos': (i, j)  # 添加网格位置信息
            })
    
    # 计算小目标概率
    norm_blocks = normalize_features(blocks)
    for block in norm_blocks:
        # 改进的小目标概率计算公式：
        # 综合考虑边缘密度、适中的方差、拉普拉斯响应、局部信噪比和相关性
        block['target_prob'] = (
            # 0.3 * block['norm_edge_density'] + 
            # 0.15 * (1 - abs(block['norm_variance'] - 0.5)) + 
            # 0.25 * block['norm_laplacian'] +
            # 0.15 * (1 - block['norm_snr']) +  # 较低的SNR可能表示有小目标
            # 0.15 * (1 - block['norm_corr'])   # 低相关性表示原图和滤波后差异大
            0.50 * edge_density + 
            0.50 * block['norm_laplacian']
        )
    
    return norm_blocks

def normalize_features(blocks):
    """对特征进行归一化"""
    edge_densities = [b['edge_density'] for b in blocks]
    variances = [b['variance'] for b in blocks]
    corrs = [b['corr'] for b in blocks]
    laplacians = [b['laplacian_response'] for b in blocks]
    snrs = [b['local_snr'] for b in blocks]
    
    min_edge = min(edge_densities)
    max_edge = max(edge_densities) if max(edge_densities) > min_edge else min_edge + 1e-6
    
    min_var = min(variances)
    max_var = max(variances) if max(variances) > min_var else min_var + 1e-6
    
    min_corr = min(corrs)
    max_corr = max(corrs) if max(corrs) > min_corr else min_corr + 1e-6
    
    min_lap = min(laplacians)
    max_lap = max(laplacians) if max(laplacians) > min_lap else min_lap + 1e-6
    
    min_snr = min(snrs)
    max_snr = max(snrs) if max(snrs) > min_snr else min_snr + 1e-6
    
    for block in blocks:
        block['norm_edge_density'] = (block['edge_density'] - min_edge) / (max_edge - min_edge)
        block['norm_variance'] = (block['variance'] - min_var) / (max_var - min_var)
        block['norm_corr'] = (block['corr'] - min_corr) / (max_corr - min_corr)
        block['norm_laplacian'] = (block['laplacian_response'] - min_lap) / (max_lap - min_lap)
        block['norm_snr'] = (block['local_snr'] - min_snr) / (max_snr - min_snr)
    
    return blocks

def find_low_prob_blocks(blocks, grid_size, prob_threshold=0.3):
    """找出全部概率低于阈值的3x3区域（恰好9个格子）"""
    # 创建网格概率映射
    grid_map = {}
    grid_positions = []
    for block in blocks:
        i, j = block['grid_pos']
        grid_map[(i, j)] = block
        grid_positions.append((i, j))
    
    # 找出概率低于阈值的区块
    low_prob_blocks = [b for b in blocks if b['target_prob'] < prob_threshold]
    
    # 如果没有满足条件的区块，返回空列表
    if not low_prob_blocks:
        return []
    
    # 获取所有可能的3x3区域中心点
    center_positions = []
    for i in range(1, grid_size[0]-1):  # 从1开始到n-2，确保中心点周围有足够的格子
        for j in range(1, grid_size[1]-1):  # 从1开始到n-2，确保中心点周围有足够的格子
            center_positions.append((i, j))
    
    # 按照概率从低到高排序中心位置
    center_positions.sort(key=lambda pos: grid_map[(pos[0], pos[1])]['target_prob'] if (pos[0], pos[1]) in grid_map else float('inf'))
    
    # 尝试找到一个3x3区域，其中所有区块概率都低于阈值
    best_area = None
    best_area_score = float('inf')  # 分数越低越好
    
    # 检查每个可能的中心点
    for center_i, center_j in center_positions:
        # 检查以这个中心的3x3区域
        area_blocks = []
        all_below_threshold = True
        area_score = 0  # 区域总分数
        all_positions_exist = True  # 确保所有9个位置都存在
        
        # 检查当前中心的3x3区域
        for i in range(center_i-1, center_i+2):
            for j in range(center_j-1, center_j+2):
                if (i, j) not in grid_map:
                    all_positions_exist = False
                    break
                    
                block = grid_map[(i, j)]
                if block['target_prob'] >= prob_threshold:
                    all_below_threshold = False
                    break
                    
                area_blocks.append(block)
                area_score += block['target_prob']  # 累积区域分数
            
            if not all_below_threshold or not all_positions_exist:
                break
        
        # 如果找到一个全部低于阈值的3x3区域（9个区块），检查它是否优于当前最佳区域
        if all_below_threshold and all_positions_exist and len(area_blocks) == 9:
            if area_score < best_area_score:
                best_area = area_blocks
                best_area_score = area_score
    
    # 如果找到满足条件的3x3区域，返回它
    if best_area:
        return best_area
    
    # 如果找不到全部满足条件的3x3区域，尝试找到一个中心点概率最低且周围有尽可能多区块低于阈值的区域
    best_fallback_center = None
    max_low_blocks = -1
    
    for center_block in low_prob_blocks:
        center_i, center_j = center_block['grid_pos']
        
        # 只考虑能形成完整3x3区域的中心点
        if center_i == 0 or center_i >= grid_size[0]-1 or center_j == 0 or center_j >= grid_size[1]-1:
            continue
            
        # 计算周围低于阈值的区块数量
        low_count = 0
        for i in range(center_i-1, center_i+2):
            for j in range(center_j-1, center_j+2):
                if (i, j) in grid_map and grid_map[(i, j)]['target_prob'] < prob_threshold:
                    low_count += 1
        
        if low_count > max_low_blocks:
            max_low_blocks = low_count
            best_fallback_center = (center_i, center_j)
    
    # 如果找到了备选中心点，返回其3x3区域
    if best_fallback_center:
        center_i, center_j = best_fallback_center
        area_blocks = []
        for i in range(center_i-1, center_i+2):
            for j in range(center_j-1, center_j+2):
                if (i, j) in grid_map:
                    area_blocks.append(grid_map[(i, j)])
        
        print(f"警告: 没有找到全部区块都低于阈值的3x3区域，使用备选区域，其中有 {max_low_blocks}/9 个区块低于阈值")
        return area_blocks
    
    # 如果连备选中心点都没有，返回概率最低的区块周围的3x3区域
    if low_prob_blocks:
        best_block = low_prob_blocks[0]
        best_i, best_j = best_block['grid_pos']
        
        # 如果最低概率区块不能形成完整3x3区域，选择一个可以的区块
        if best_i == 0 or best_i >= grid_size[0]-1 or best_j == 0 or best_j >= grid_size[1]-1:
            for block in low_prob_blocks:
                i, j = block['grid_pos']
                if i > 0 and i < grid_size[0]-1 and j > 0 and j < grid_size[1]-1:
                    best_i, best_j = i, j
                    break
            else:
                # 如果没有合适的区块可以形成3x3区域，使用网格中心
                best_i, best_j = grid_size[0]//2, grid_size[1]//2
        
        # 收集3x3区域的所有区块
        area_blocks = []
        for i in range(best_i-1, best_i+2):
            for j in range(best_j-1, best_j+2):
                if (i, j) in grid_map:
                    area_blocks.append(grid_map[(i, j)])
        
        print(f"警告: 没有找到合适的3x3低概率区域，使用最佳备选区域")
        return area_blocks
    
    return []

def crop_and_save_area(original_img, area_blocks, output_path):
    """裁剪并保存区域"""
    # 计算区域的边界
    min_y = min(block['coords'][0] for block in area_blocks)
    min_x = min(block['coords'][1] for block in area_blocks)
    max_y = max(block['coords'][0] + block['coords'][2] for block in area_blocks)
    max_x = max(block['coords'][1] + block['coords'][3] for block in area_blocks)
    
    # 裁剪图像
    cropped_img = original_img[min_y:max_y, min_x:max_x]
    
    # 检查输出目录是否存在
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存裁剪结果
    cv2.imwrite(output_path, cropped_img)
    
    # 返回裁剪区域坐标
    return (min_y, min_x, max_y - min_y, max_x - min_x)

def create_custom_colormap():
    """创建一个自定义的颜色映射，在较高概率区间颜色更丰富"""
    # 定义颜色断点 - 主要在中高概率区间细分更多颜色
    colors = [
        (0.0, (0.0, 0.0, 0.5)),      # 深蓝色 - 最低概率
        (0.2, (0.0, 0.0, 1.0)),      # 蓝色
        (0.3, (0.0, 0.5, 1.0)),      # 浅蓝色
        (0.4, (0.0, 0.7, 0.7)),      # 青色
        (0.5, (0.0, 0.8, 0.0)),      # 绿色
        (0.6, (0.5, 0.8, 0.0)),      # 黄绿色
        (0.7, (0.8, 0.8, 0.0)),      # 黄色
        (0.75, (0.9, 0.7, 0.0)),     # 橙黄色
        (0.8, (1.0, 0.6, 0.0)),      # 橙色
        (0.85, (1.0, 0.4, 0.0)),     # 深橙色
        (0.9, (1.0, 0.2, 0.0)),      # 橙红色
        (0.95, (1.0, 0.0, 0.0)),     # 红色
        (1.0, (0.8, 0.0, 0.0))       # 深红色 - 最高概率
    ]
    
    # 创建自定义颜色映射
    cmap_name = 'custom_probability_map'
    return mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)





def analyze_visualize_and_crop(image_path, vis_output_path, crop_output_path, grid_size=(8,6), 
                              wavelet='db4', level=3, filter_ratio=0.1, 
                              prob_threshold=0.3):
    """分析图像，可视化结果，并裁剪低概率区域"""
    # 读取图像
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"无法读取图像: {image_path}")
        return None, None
    
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # 直接进行滤波，跳过CLAHE增强步骤
    filtered_gray = multi_band_filter(original_gray, wavelet, level, filter_ratio)
    
    # 计算区块特征和小目标概率
    blocks = calculate_block_features(filtered_gray, original_gray, grid_size)
    
    # 找出低概率区域
    area_blocks = find_low_prob_blocks(blocks, grid_size, prob_threshold)
    
    # 如果没有找到符合条件的区块，返回None
    if not area_blocks:
        print(f"图像 {os.path.basename(image_path)} 中没有找到适合的低概率区域")
        return None, None
    
    # 检查是否有区块高于阈值
    high_prob_blocks = [b for b in area_blocks if b['target_prob'] >= prob_threshold]
    if high_prob_blocks:
        print(f"警告: 选中区域包含 {len(high_prob_blocks)} 个概率高于阈值的区块")
    
    # 确保正好有9个区块
    if len(area_blocks) != 9:
        print(f"警告: 选中区域包含 {len(area_blocks)} 个区块，不是预期的9个")
        # 如果区块数量不足9个，我们跳过这个图像
        if len(area_blocks) < 9:
            print(f"跳过图像 {os.path.basename(image_path)}: 无法形成完整的3x3区域")
            return None, None
    
    # 裁剪并保存低概率区域
    crop_coords = crop_and_save_area(original_img, area_blocks, crop_output_path)
    
    # 创建可视化图
    plt.figure(figsize=(16, 10))
    
    # 原图
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('原图')
    plt.axis('off')
    
    # 标注图
    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), alpha=0.7)  # 原图作为底图
    
    # 定义颜色映射 - 使用自定义颜色映射
    cmap = create_custom_colormap()
    
    # 使用BoundaryNorm来定义非均匀的颜色边界，在中高概率区域分配更多颜色
    boundaries = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    
    # 绘制所有区块和信息
    for block in blocks:
        y, x, h, w = block['coords']
        prob = block['target_prob']
        color = cmap(norm(prob))  # 使用归一化后的颜色
        
        # 根据概率调整填充透明度
        fill_alpha = min(0.25, prob * 0.3)  # 略微增加透明度以更好地显示颜色
        rect = Rectangle((x, y), w, h, linewidth=1, 
                         edgecolor=color, 
                         facecolor=color, 
                         alpha=fill_alpha)
        plt.gca().add_patch(rect)

        rect = Rectangle((x, y), w, h, linewidth=1, 
                         edgecolor=color, 
                         facecolor=color, 
                         alpha=fill_alpha)
        plt.gca().add_patch(rect)

    # 绘制裁剪区域（用红色粗边框标出）
    if crop_coords:
        y, x, h, w = crop_coords
        rect = Rectangle((x, y), w, h, linewidth=2, 
                         edgecolor='red', 
                         facecolor='none')
        plt.gca().add_patch(rect)
        

    
    plt.title('区块分析热力图与裁剪区域(红框)')
    plt.axis('off')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('小目标概率')
    
    # 保存可视化图像
    plt.tight_layout()
    plt.savefig(vis_output_path, dpi=300)
    plt.close()
    
    return blocks, crop_coords

def process_folder(input_dir, vis_output_dir, crop_output_dir, grid_size=(8,6), 
                  wavelet='db4', level=3, filter_ratio=0.1,
                  prob_threshold=0.3):
    """处理文件夹中的所有图像"""
    # 创建输出目录
    if not os.path.exists(vis_output_dir):
        os.makedirs(vis_output_dir)
    if not os.path.exists(crop_output_dir):
        os.makedirs(crop_output_dir)
    
    # 统计处理结果
    total_images = 0
    processed_images = 0
    
    # 处理每个图像
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        total_images += 1
        
        # 设置输入和输出路径
        image_path = os.path.join(input_dir, filename)
        vis_path = os.path.join(vis_output_dir, f"vis_{filename}")
        crop_path = os.path.join(crop_output_dir, f"crop_{filename}")
        
        print(f"处理: {filename}")
        
        # 分析图像，生成可视化结果，并裁剪低概率区域
        blocks, crop_coords = analyze_visualize_and_crop(
            image_path,
            vis_path,
            crop_path,
            grid_size=grid_size,
            wavelet=wavelet,
            level=level,
            filter_ratio=filter_ratio,
            prob_threshold=prob_threshold
        )
        
        if crop_coords:
            processed_images += 1
            print(f"  已裁剪区域: ({crop_coords[1]},{crop_coords[0]}), 尺寸: {crop_coords[3]}x{crop_coords[2]}")
    
    print(f"\n处理完成! 共处理 {total_images} 张图像，其中 {processed_images} 张生成了裁剪结果")
    print(f"可视化结果保存至: {vis_output_dir}")
    print(f"裁剪结果保存至: {crop_output_dir}")

# 主函数
if __name__ == "__main__":
    # 设置输入和输出目录
    input_dir = "F:/dark/test/test_images_103_re"  # 输入图像目录
    vis_output_dir = "F:/dark/test/visualization_resultnocha"  # 可视化结果保存目录
    crop_output_dir = "F:/dark/test/cropped_results5nocha"  # 裁剪结果保存目录
    
    print("开始处理图像...")
    print(f"输入目录: {input_dir}")
    print(f"可视化输出目录: {vis_output_dir}")
    print(f"裁剪输出目录: {crop_output_dir}")
    print("-----------------------------------")
    
    # 处理参数 - 移除了clip_limit参数
    process_folder(
        input_dir,
        vis_output_dir,
        crop_output_dir,
        grid_size=(12, 15),   # 网格大小
        wavelet='db4',       # 小波类型
        level=5,             # 小波分解层数
        filter_ratio=0.08,   # 傅里叶滤波保留比例
        prob_threshold=0.4   # 小目标概率阈值
=======
<<<<<<< HEAD
import cv2
import numpy as np
import os
import pywt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt

# 方法2：设置全局字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def multi_band_filter(gray_img, wavelet='db4', level=3, filter_ratio=0.1):
    """
    多频带联合滤波
    Args:
        gray_img: 灰度图
        wavelet: 小波基类型，可选'db4'/'haar'
        level: 小波分解层数
        filter_ratio: 高频子带傅里叶滤波保留比例
    Returns:
        filtered: 滤波后的图像
    """
    # 小波分解
    coeffs = pywt.wavedec2(gray_img, wavelet, level=level)
    
    # 将tuple转换为list，以便修改
    coeffs_list = list(coeffs)
    
    # 对每个高频子带进行傅里叶滤波
    for i in range(1, len(coeffs_list)):
        # 每个层级的高频子带也需要转换为list
        detail_coeffs = list(coeffs_list[i])
        
        for j in range(len(detail_coeffs)):
            sub_band = detail_coeffs[j]
            rows, cols = sub_band.shape
            # 傅里叶低通滤波
            f = np.fft.fft2(sub_band)
            fshift = np.fft.fftshift(f)
            crow, ccol = rows//2, cols//2
            mask = np.zeros_like(fshift)
            mask[
                int(crow - filter_ratio*rows):int(crow + filter_ratio*rows),
                int(ccol - filter_ratio*cols):int(ccol + filter_ratio*cols)
            ] = 1
            fshift_filtered = fshift * mask
            sub_band_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))
            detail_coeffs[j] = sub_band_filtered
        
        # 更新修改后的子带回到coeffs_list
        coeffs_list[i] = tuple(detail_coeffs)
    
    # 小波重构并归一化
    filtered = pywt.waverec2(tuple(coeffs_list), wavelet)
    return cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def calculate_block_features(filtered_gray, original_gray, grid_size=(8,6)):
    """计算每个区块的特征和小目标概率"""
    H, W = filtered_gray.shape
    block_h = H // grid_size[0]
    block_w = W // grid_size[1]
    blocks = []
    
    # 调整Canny阈值适应滤波后特征
    edges = cv2.Canny(filtered_gray, 30, 100)
    
    # 额外计算Laplacian算子响应 (增强对小目标的检测能力)
    laplacian = cv2.Laplacian(filtered_gray, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian)
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            y = i * block_h
            x = j * block_w
            h = block_h if i != grid_size[0]-1 else H - y
            w = block_w if j != grid_size[1]-1 else W - x
            
            # 使用滤波后图像计算特征
            block_filtered = filtered_gray[y:y+h, x:x+w]
            edge_block = edges[y:y+h, x:x+w]
            laplacian_block = laplacian_abs[y:y+h, x:x+w]
            original_block = original_gray[y:y+h, x:x+w]
            
            edge_density = np.mean(edge_block) / 255.0
            variance = np.var(block_filtered)
            laplacian_response = np.mean(laplacian_block)
            
            # 局部信噪比估计 (对小目标检测有帮助)
            local_snr = 0
            if np.std(block_filtered) > 0:
                local_snr = np.mean(block_filtered) / np.std(block_filtered)
            
            # 计算原始块与滤波后块的相关性
            corr = cv2.matchTemplate(
                cv2.normalize(original_block, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F),
                cv2.normalize(block_filtered, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F),
                cv2.TM_CCORR_NORMED
            )[0][0]
            
            blocks.append({
                'edge_density': edge_density,
                'variance': variance,
                'coords': (y, x, h, w),
                'corr': corr,
                'laplacian_response': laplacian_response,
                'local_snr': local_snr,
                'grid_pos': (i, j)  # 添加网格位置信息
            })
    
    # 计算小目标概率
    norm_blocks = normalize_features(blocks)
    for block in norm_blocks:
        # 改进的小目标概率计算公式：
        # 综合考虑边缘密度、适中的方差、拉普拉斯响应、局部信噪比和相关性
        block['target_prob'] = (
            # 0.3 * block['norm_edge_density'] + 
            # 0.15 * (1 - abs(block['norm_variance'] - 0.5)) + 
            # 0.25 * block['norm_laplacian'] +
            # 0.15 * (1 - block['norm_snr']) +  # 较低的SNR可能表示有小目标
            # 0.15 * (1 - block['norm_corr'])   # 低相关性表示原图和滤波后差异大
            0.50 * edge_density + 
            0.50 * block['norm_laplacian']
        )
    
    return norm_blocks

def normalize_features(blocks):
    """对特征进行归一化"""
    edge_densities = [b['edge_density'] for b in blocks]
    variances = [b['variance'] for b in blocks]
    corrs = [b['corr'] for b in blocks]
    laplacians = [b['laplacian_response'] for b in blocks]
    snrs = [b['local_snr'] for b in blocks]
    
    min_edge = min(edge_densities)
    max_edge = max(edge_densities) if max(edge_densities) > min_edge else min_edge + 1e-6
    
    min_var = min(variances)
    max_var = max(variances) if max(variances) > min_var else min_var + 1e-6
    
    min_corr = min(corrs)
    max_corr = max(corrs) if max(corrs) > min_corr else min_corr + 1e-6
    
    min_lap = min(laplacians)
    max_lap = max(laplacians) if max(laplacians) > min_lap else min_lap + 1e-6
    
    min_snr = min(snrs)
    max_snr = max(snrs) if max(snrs) > min_snr else min_snr + 1e-6
    
    for block in blocks:
        block['norm_edge_density'] = (block['edge_density'] - min_edge) / (max_edge - min_edge)
        block['norm_variance'] = (block['variance'] - min_var) / (max_var - min_var)
        block['norm_corr'] = (block['corr'] - min_corr) / (max_corr - min_corr)
        block['norm_laplacian'] = (block['laplacian_response'] - min_lap) / (max_lap - min_lap)
        block['norm_snr'] = (block['local_snr'] - min_snr) / (max_snr - min_snr)
    
    return blocks

def find_low_prob_blocks(blocks, grid_size, prob_threshold=0.3):
    """找出全部概率低于阈值的3x3区域（恰好9个格子）"""
    # 创建网格概率映射
    grid_map = {}
    grid_positions = []
    for block in blocks:
        i, j = block['grid_pos']
        grid_map[(i, j)] = block
        grid_positions.append((i, j))
    
    # 找出概率低于阈值的区块
    low_prob_blocks = [b for b in blocks if b['target_prob'] < prob_threshold]
    
    # 如果没有满足条件的区块，返回空列表
    if not low_prob_blocks:
        return []
    
    # 获取所有可能的3x3区域中心点
    center_positions = []
    for i in range(1, grid_size[0]-1):  # 从1开始到n-2，确保中心点周围有足够的格子
        for j in range(1, grid_size[1]-1):  # 从1开始到n-2，确保中心点周围有足够的格子
            center_positions.append((i, j))
    
    # 按照概率从低到高排序中心位置
    center_positions.sort(key=lambda pos: grid_map[(pos[0], pos[1])]['target_prob'] if (pos[0], pos[1]) in grid_map else float('inf'))
    
    # 尝试找到一个3x3区域，其中所有区块概率都低于阈值
    best_area = None
    best_area_score = float('inf')  # 分数越低越好
    
    # 检查每个可能的中心点
    for center_i, center_j in center_positions:
        # 检查以这个中心的3x3区域
        area_blocks = []
        all_below_threshold = True
        area_score = 0  # 区域总分数
        all_positions_exist = True  # 确保所有9个位置都存在
        
        # 检查当前中心的3x3区域
        for i in range(center_i-1, center_i+2):
            for j in range(center_j-1, center_j+2):
                if (i, j) not in grid_map:
                    all_positions_exist = False
                    break
                    
                block = grid_map[(i, j)]
                if block['target_prob'] >= prob_threshold:
                    all_below_threshold = False
                    break
                    
                area_blocks.append(block)
                area_score += block['target_prob']  # 累积区域分数
            
            if not all_below_threshold or not all_positions_exist:
                break
        
        # 如果找到一个全部低于阈值的3x3区域（9个区块），检查它是否优于当前最佳区域
        if all_below_threshold and all_positions_exist and len(area_blocks) == 9:
            if area_score < best_area_score:
                best_area = area_blocks
                best_area_score = area_score
    
    # 如果找到满足条件的3x3区域，返回它
    if best_area:
        return best_area
    
    # 如果找不到全部满足条件的3x3区域，尝试找到一个中心点概率最低且周围有尽可能多区块低于阈值的区域
    best_fallback_center = None
    max_low_blocks = -1
    
    for center_block in low_prob_blocks:
        center_i, center_j = center_block['grid_pos']
        
        # 只考虑能形成完整3x3区域的中心点
        if center_i == 0 or center_i >= grid_size[0]-1 or center_j == 0 or center_j >= grid_size[1]-1:
            continue
            
        # 计算周围低于阈值的区块数量
        low_count = 0
        for i in range(center_i-1, center_i+2):
            for j in range(center_j-1, center_j+2):
                if (i, j) in grid_map and grid_map[(i, j)]['target_prob'] < prob_threshold:
                    low_count += 1
        
        if low_count > max_low_blocks:
            max_low_blocks = low_count
            best_fallback_center = (center_i, center_j)
    
    # 如果找到了备选中心点，返回其3x3区域
    if best_fallback_center:
        center_i, center_j = best_fallback_center
        area_blocks = []
        for i in range(center_i-1, center_i+2):
            for j in range(center_j-1, center_j+2):
                if (i, j) in grid_map:
                    area_blocks.append(grid_map[(i, j)])
        
        print(f"警告: 没有找到全部区块都低于阈值的3x3区域，使用备选区域，其中有 {max_low_blocks}/9 个区块低于阈值")
        return area_blocks
    
    # 如果连备选中心点都没有，返回概率最低的区块周围的3x3区域
    if low_prob_blocks:
        best_block = low_prob_blocks[0]
        best_i, best_j = best_block['grid_pos']
        
        # 如果最低概率区块不能形成完整3x3区域，选择一个可以的区块
        if best_i == 0 or best_i >= grid_size[0]-1 or best_j == 0 or best_j >= grid_size[1]-1:
            for block in low_prob_blocks:
                i, j = block['grid_pos']
                if i > 0 and i < grid_size[0]-1 and j > 0 and j < grid_size[1]-1:
                    best_i, best_j = i, j
                    break
            else:
                # 如果没有合适的区块可以形成3x3区域，使用网格中心
                best_i, best_j = grid_size[0]//2, grid_size[1]//2
        
        # 收集3x3区域的所有区块
        area_blocks = []
        for i in range(best_i-1, best_i+2):
            for j in range(best_j-1, best_j+2):
                if (i, j) in grid_map:
                    area_blocks.append(grid_map[(i, j)])
        
        print(f"警告: 没有找到合适的3x3低概率区域，使用最佳备选区域")
        return area_blocks
    
    return []

def crop_and_save_area(original_img, area_blocks, output_path):
    """裁剪并保存区域"""
    # 计算区域的边界
    min_y = min(block['coords'][0] for block in area_blocks)
    min_x = min(block['coords'][1] for block in area_blocks)
    max_y = max(block['coords'][0] + block['coords'][2] for block in area_blocks)
    max_x = max(block['coords'][1] + block['coords'][3] for block in area_blocks)
    
    # 裁剪图像
    cropped_img = original_img[min_y:max_y, min_x:max_x]
    
    # 检查输出目录是否存在
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存裁剪结果
    cv2.imwrite(output_path, cropped_img)
    
    # 返回裁剪区域坐标
    return (min_y, min_x, max_y - min_y, max_x - min_x)

def create_custom_colormap():
    """创建一个自定义的颜色映射，在较高概率区间颜色更丰富"""
    # 定义颜色断点 - 主要在中高概率区间细分更多颜色
    colors = [
        (0.0, (0.0, 0.0, 0.5)),      # 深蓝色 - 最低概率
        (0.2, (0.0, 0.0, 1.0)),      # 蓝色
        (0.3, (0.0, 0.5, 1.0)),      # 浅蓝色
        (0.4, (0.0, 0.7, 0.7)),      # 青色
        (0.5, (0.0, 0.8, 0.0)),      # 绿色
        (0.6, (0.5, 0.8, 0.0)),      # 黄绿色
        (0.7, (0.8, 0.8, 0.0)),      # 黄色
        (0.75, (0.9, 0.7, 0.0)),     # 橙黄色
        (0.8, (1.0, 0.6, 0.0)),      # 橙色
        (0.85, (1.0, 0.4, 0.0)),     # 深橙色
        (0.9, (1.0, 0.2, 0.0)),      # 橙红色
        (0.95, (1.0, 0.0, 0.0)),     # 红色
        (1.0, (0.8, 0.0, 0.0))       # 深红色 - 最高概率
    ]
    
    # 创建自定义颜色映射
    cmap_name = 'custom_probability_map'
    return mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)





def analyze_visualize_and_crop(image_path, vis_output_path, crop_output_path, grid_size=(8,6), 
                              wavelet='db4', level=3, filter_ratio=0.1, 
                              prob_threshold=0.3):
    """分析图像，可视化结果，并裁剪低概率区域"""
    # 读取图像
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"无法读取图像: {image_path}")
        return None, None
    
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # 直接进行滤波，跳过CLAHE增强步骤
    filtered_gray = multi_band_filter(original_gray, wavelet, level, filter_ratio)
    
    # 计算区块特征和小目标概率
    blocks = calculate_block_features(filtered_gray, original_gray, grid_size)
    
    # 找出低概率区域
    area_blocks = find_low_prob_blocks(blocks, grid_size, prob_threshold)
    
    # 如果没有找到符合条件的区块，返回None
    if not area_blocks:
        print(f"图像 {os.path.basename(image_path)} 中没有找到适合的低概率区域")
        return None, None
    
    # 检查是否有区块高于阈值
    high_prob_blocks = [b for b in area_blocks if b['target_prob'] >= prob_threshold]
    if high_prob_blocks:
        print(f"警告: 选中区域包含 {len(high_prob_blocks)} 个概率高于阈值的区块")
    
    # 确保正好有9个区块
    if len(area_blocks) != 9:
        print(f"警告: 选中区域包含 {len(area_blocks)} 个区块，不是预期的9个")
        # 如果区块数量不足9个，我们跳过这个图像
        if len(area_blocks) < 9:
            print(f"跳过图像 {os.path.basename(image_path)}: 无法形成完整的3x3区域")
            return None, None
    
    # 裁剪并保存低概率区域
    crop_coords = crop_and_save_area(original_img, area_blocks, crop_output_path)
    
    # 创建可视化图
    plt.figure(figsize=(16, 10))
    
    # 原图
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('原图')
    plt.axis('off')
    
    # 标注图
    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), alpha=0.7)  # 原图作为底图
    
    # 定义颜色映射 - 使用自定义颜色映射
    cmap = create_custom_colormap()
    
    # 使用BoundaryNorm来定义非均匀的颜色边界，在中高概率区域分配更多颜色
    boundaries = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    
    # 绘制所有区块和信息
    for block in blocks:
        y, x, h, w = block['coords']
        prob = block['target_prob']
        color = cmap(norm(prob))  # 使用归一化后的颜色
        
        # 根据概率调整填充透明度
        fill_alpha = min(0.25, prob * 0.3)  # 略微增加透明度以更好地显示颜色
        rect = Rectangle((x, y), w, h, linewidth=1, 
                         edgecolor=color, 
                         facecolor=color, 
                         alpha=fill_alpha)
        plt.gca().add_patch(rect)

        rect = Rectangle((x, y), w, h, linewidth=1, 
                         edgecolor=color, 
                         facecolor=color, 
                         alpha=fill_alpha)
        plt.gca().add_patch(rect)

    # 绘制裁剪区域（用红色粗边框标出）
    if crop_coords:
        y, x, h, w = crop_coords
        rect = Rectangle((x, y), w, h, linewidth=2, 
                         edgecolor='red', 
                         facecolor='none')
        plt.gca().add_patch(rect)
        

    
    plt.title('区块分析热力图与裁剪区域(红框)')
    plt.axis('off')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('小目标概率')
    
    # 保存可视化图像
    plt.tight_layout()
    plt.savefig(vis_output_path, dpi=300)
    plt.close()
    
    return blocks, crop_coords

def process_folder(input_dir, vis_output_dir, crop_output_dir, grid_size=(8,6), 
                  wavelet='db4', level=3, filter_ratio=0.1,
                  prob_threshold=0.3):
    """处理文件夹中的所有图像"""
    # 创建输出目录
    if not os.path.exists(vis_output_dir):
        os.makedirs(vis_output_dir)
    if not os.path.exists(crop_output_dir):
        os.makedirs(crop_output_dir)
    
    # 统计处理结果
    total_images = 0
    processed_images = 0
    
    # 处理每个图像
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        total_images += 1
        
        # 设置输入和输出路径
        image_path = os.path.join(input_dir, filename)
        vis_path = os.path.join(vis_output_dir, f"vis_{filename}")
        crop_path = os.path.join(crop_output_dir, f"crop_{filename}")
        
        print(f"处理: {filename}")
        
        # 分析图像，生成可视化结果，并裁剪低概率区域
        blocks, crop_coords = analyze_visualize_and_crop(
            image_path,
            vis_path,
            crop_path,
            grid_size=grid_size,
            wavelet=wavelet,
            level=level,
            filter_ratio=filter_ratio,
            prob_threshold=prob_threshold
        )
        
        if crop_coords:
            processed_images += 1
            print(f"  已裁剪区域: ({crop_coords[1]},{crop_coords[0]}), 尺寸: {crop_coords[3]}x{crop_coords[2]}")
    
    print(f"\n处理完成! 共处理 {total_images} 张图像，其中 {processed_images} 张生成了裁剪结果")
    print(f"可视化结果保存至: {vis_output_dir}")
    print(f"裁剪结果保存至: {crop_output_dir}")

# 主函数
if __name__ == "__main__":
    # 设置输入和输出目录
    input_dir = "F:/dark/test/test_images_103_re"  # 输入图像目录
    vis_output_dir = "F:/dark/test/visualization_resultnocha"  # 可视化结果保存目录
    crop_output_dir = "F:/dark/test/cropped_results5nocha"  # 裁剪结果保存目录
    
    print("开始处理图像...")
    print(f"输入目录: {input_dir}")
    print(f"可视化输出目录: {vis_output_dir}")
    print(f"裁剪输出目录: {crop_output_dir}")
    print("-----------------------------------")
    
    # 处理参数 - 移除了clip_limit参数
    process_folder(
        input_dir,
        vis_output_dir,
        crop_output_dir,
        grid_size=(12, 15),   # 网格大小
        wavelet='db4',       # 小波类型
        level=5,             # 小波分解层数
        filter_ratio=0.08,   # 傅里叶滤波保留比例
        prob_threshold=0.4   # 小目标概率阈值
=======
import cv2
import numpy as np
import os
import pywt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import shutil
import matplotlib as mpl
import matplotlib.pyplot as plt

# 方法2：设置全局字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def multi_band_filter(gray_img, wavelet='db4', level=3, filter_ratio=0.1):
    """
    多频带联合滤波
    Args:
        gray_img: 灰度图
        wavelet: 小波基类型，可选'db4'/'haar'
        level: 小波分解层数
        filter_ratio: 高频子带傅里叶滤波保留比例
    Returns:
        filtered: 滤波后的图像
    """
    # 小波分解
    coeffs = pywt.wavedec2(gray_img, wavelet, level=level)
    
    # 将tuple转换为list，以便修改
    coeffs_list = list(coeffs)
    
    # 对每个高频子带进行傅里叶滤波
    for i in range(1, len(coeffs_list)):
        # 每个层级的高频子带也需要转换为list
        detail_coeffs = list(coeffs_list[i])
        
        for j in range(len(detail_coeffs)):
            sub_band = detail_coeffs[j]
            rows, cols = sub_band.shape
            # 傅里叶低通滤波
            f = np.fft.fft2(sub_band)
            fshift = np.fft.fftshift(f)
            crow, ccol = rows//2, cols//2
            mask = np.zeros_like(fshift)
            mask[
                int(crow - filter_ratio*rows):int(crow + filter_ratio*rows),
                int(ccol - filter_ratio*cols):int(ccol + filter_ratio*cols)
            ] = 1
            fshift_filtered = fshift * mask
            sub_band_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_filtered)))
            detail_coeffs[j] = sub_band_filtered
        
        # 更新修改后的子带回到coeffs_list
        coeffs_list[i] = tuple(detail_coeffs)
    
    # 小波重构并归一化
    filtered = pywt.waverec2(tuple(coeffs_list), wavelet)
    return cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def calculate_block_features(filtered_gray, original_gray, grid_size=(8,6)):
    """计算每个区块的特征和小目标概率"""
    H, W = filtered_gray.shape
    block_h = H // grid_size[0]
    block_w = W // grid_size[1]
    blocks = []
    
    # 调整Canny阈值适应滤波后特征
    edges = cv2.Canny(filtered_gray, 30, 100)
    
    # 额外计算Laplacian算子响应 (增强对小目标的检测能力)
    laplacian = cv2.Laplacian(filtered_gray, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian)
    
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            y = i * block_h
            x = j * block_w
            h = block_h if i != grid_size[0]-1 else H - y
            w = block_w if j != grid_size[1]-1 else W - x
            
            # 使用滤波后图像计算特征
            block_filtered = filtered_gray[y:y+h, x:x+w]
            edge_block = edges[y:y+h, x:x+w]
            laplacian_block = laplacian_abs[y:y+h, x:x+w]
            original_block = original_gray[y:y+h, x:x+w]
            
            edge_density = np.mean(edge_block) / 255.0
            variance = np.var(block_filtered)
            laplacian_response = np.mean(laplacian_block)
            
            # 局部信噪比估计 (对小目标检测有帮助)
            local_snr = 0
            if np.std(block_filtered) > 0:
                local_snr = np.mean(block_filtered) / np.std(block_filtered)
            
            # 计算原始块与滤波后块的相关性
            corr = cv2.matchTemplate(
                cv2.normalize(original_block, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F),
                cv2.normalize(block_filtered, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F),
                cv2.TM_CCORR_NORMED
            )[0][0]
            
            blocks.append({
                'edge_density': edge_density,
                'variance': variance,
                'coords': (y, x, h, w),
                'corr': corr,
                'laplacian_response': laplacian_response,
                'local_snr': local_snr,
                'grid_pos': (i, j)  # 添加网格位置信息
            })
    
    # 计算小目标概率
    norm_blocks = normalize_features(blocks)
    for block in norm_blocks:
        # 改进的小目标概率计算公式：
        # 综合考虑边缘密度、适中的方差、拉普拉斯响应、局部信噪比和相关性
        block['target_prob'] = (
            # 0.3 * block['norm_edge_density'] + 
            # 0.15 * (1 - abs(block['norm_variance'] - 0.5)) + 
            # 0.25 * block['norm_laplacian'] +
            # 0.15 * (1 - block['norm_snr']) +  # 较低的SNR可能表示有小目标
            # 0.15 * (1 - block['norm_corr'])   # 低相关性表示原图和滤波后差异大
            0.50 * edge_density + 
            0.50 * block['norm_laplacian']
        )
    
    return norm_blocks

def normalize_features(blocks):
    """对特征进行归一化"""
    edge_densities = [b['edge_density'] for b in blocks]
    variances = [b['variance'] for b in blocks]
    corrs = [b['corr'] for b in blocks]
    laplacians = [b['laplacian_response'] for b in blocks]
    snrs = [b['local_snr'] for b in blocks]
    
    min_edge = min(edge_densities)
    max_edge = max(edge_densities) if max(edge_densities) > min_edge else min_edge + 1e-6
    
    min_var = min(variances)
    max_var = max(variances) if max(variances) > min_var else min_var + 1e-6
    
    min_corr = min(corrs)
    max_corr = max(corrs) if max(corrs) > min_corr else min_corr + 1e-6
    
    min_lap = min(laplacians)
    max_lap = max(laplacians) if max(laplacians) > min_lap else min_lap + 1e-6
    
    min_snr = min(snrs)
    max_snr = max(snrs) if max(snrs) > min_snr else min_snr + 1e-6
    
    for block in blocks:
        block['norm_edge_density'] = (block['edge_density'] - min_edge) / (max_edge - min_edge)
        block['norm_variance'] = (block['variance'] - min_var) / (max_var - min_var)
        block['norm_corr'] = (block['corr'] - min_corr) / (max_corr - min_corr)
        block['norm_laplacian'] = (block['laplacian_response'] - min_lap) / (max_lap - min_lap)
        block['norm_snr'] = (block['local_snr'] - min_snr) / (max_snr - min_snr)
    
    return blocks

def find_low_prob_blocks(blocks, grid_size, prob_threshold=0.3):
    """找出全部概率低于阈值的3x3区域（恰好9个格子）"""
    # 创建网格概率映射
    grid_map = {}
    grid_positions = []
    for block in blocks:
        i, j = block['grid_pos']
        grid_map[(i, j)] = block
        grid_positions.append((i, j))
    
    # 找出概率低于阈值的区块
    low_prob_blocks = [b for b in blocks if b['target_prob'] < prob_threshold]
    
    # 如果没有满足条件的区块，返回空列表
    if not low_prob_blocks:
        return []
    
    # 获取所有可能的3x3区域中心点
    center_positions = []
    for i in range(1, grid_size[0]-1):  # 从1开始到n-2，确保中心点周围有足够的格子
        for j in range(1, grid_size[1]-1):  # 从1开始到n-2，确保中心点周围有足够的格子
            center_positions.append((i, j))
    
    # 按照概率从低到高排序中心位置
    center_positions.sort(key=lambda pos: grid_map[(pos[0], pos[1])]['target_prob'] if (pos[0], pos[1]) in grid_map else float('inf'))
    
    # 尝试找到一个3x3区域，其中所有区块概率都低于阈值
    best_area = None
    best_area_score = float('inf')  # 分数越低越好
    
    # 检查每个可能的中心点
    for center_i, center_j in center_positions:
        # 检查以这个中心的3x3区域
        area_blocks = []
        all_below_threshold = True
        area_score = 0  # 区域总分数
        all_positions_exist = True  # 确保所有9个位置都存在
        
        # 检查当前中心的3x3区域
        for i in range(center_i-1, center_i+2):
            for j in range(center_j-1, center_j+2):
                if (i, j) not in grid_map:
                    all_positions_exist = False
                    break
                    
                block = grid_map[(i, j)]
                if block['target_prob'] >= prob_threshold:
                    all_below_threshold = False
                    break
                    
                area_blocks.append(block)
                area_score += block['target_prob']  # 累积区域分数
            
            if not all_below_threshold or not all_positions_exist:
                break
        
        # 如果找到一个全部低于阈值的3x3区域（9个区块），检查它是否优于当前最佳区域
        if all_below_threshold and all_positions_exist and len(area_blocks) == 9:
            if area_score < best_area_score:
                best_area = area_blocks
                best_area_score = area_score
    
    # 如果找到满足条件的3x3区域，返回它
    if best_area:
        return best_area
    
    # 如果找不到全部满足条件的3x3区域，尝试找到一个中心点概率最低且周围有尽可能多区块低于阈值的区域
    best_fallback_center = None
    max_low_blocks = -1
    
    for center_block in low_prob_blocks:
        center_i, center_j = center_block['grid_pos']
        
        # 只考虑能形成完整3x3区域的中心点
        if center_i == 0 or center_i >= grid_size[0]-1 or center_j == 0 or center_j >= grid_size[1]-1:
            continue
            
        # 计算周围低于阈值的区块数量
        low_count = 0
        for i in range(center_i-1, center_i+2):
            for j in range(center_j-1, center_j+2):
                if (i, j) in grid_map and grid_map[(i, j)]['target_prob'] < prob_threshold:
                    low_count += 1
        
        if low_count > max_low_blocks:
            max_low_blocks = low_count
            best_fallback_center = (center_i, center_j)
    
    # 如果找到了备选中心点，返回其3x3区域
    if best_fallback_center:
        center_i, center_j = best_fallback_center
        area_blocks = []
        for i in range(center_i-1, center_i+2):
            for j in range(center_j-1, center_j+2):
                if (i, j) in grid_map:
                    area_blocks.append(grid_map[(i, j)])
        
        print(f"警告: 没有找到全部区块都低于阈值的3x3区域，使用备选区域，其中有 {max_low_blocks}/9 个区块低于阈值")
        return area_blocks
    
    # 如果连备选中心点都没有，返回概率最低的区块周围的3x3区域
    if low_prob_blocks:
        best_block = low_prob_blocks[0]
        best_i, best_j = best_block['grid_pos']
        
        # 如果最低概率区块不能形成完整3x3区域，选择一个可以的区块
        if best_i == 0 or best_i >= grid_size[0]-1 or best_j == 0 or best_j >= grid_size[1]-1:
            for block in low_prob_blocks:
                i, j = block['grid_pos']
                if i > 0 and i < grid_size[0]-1 and j > 0 and j < grid_size[1]-1:
                    best_i, best_j = i, j
                    break
            else:
                # 如果没有合适的区块可以形成3x3区域，使用网格中心
                best_i, best_j = grid_size[0]//2, grid_size[1]//2
        
        # 收集3x3区域的所有区块
        area_blocks = []
        for i in range(best_i-1, best_i+2):
            for j in range(best_j-1, best_j+2):
                if (i, j) in grid_map:
                    area_blocks.append(grid_map[(i, j)])
        
        print(f"警告: 没有找到合适的3x3低概率区域，使用最佳备选区域")
        return area_blocks
    
    return []

def crop_and_save_area(original_img, area_blocks, output_path):
    """裁剪并保存区域"""
    # 计算区域的边界
    min_y = min(block['coords'][0] for block in area_blocks)
    min_x = min(block['coords'][1] for block in area_blocks)
    max_y = max(block['coords'][0] + block['coords'][2] for block in area_blocks)
    max_x = max(block['coords'][1] + block['coords'][3] for block in area_blocks)
    
    # 裁剪图像
    cropped_img = original_img[min_y:max_y, min_x:max_x]
    
    # 检查输出目录是否存在
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存裁剪结果
    cv2.imwrite(output_path, cropped_img)
    
    # 返回裁剪区域坐标
    return (min_y, min_x, max_y - min_y, max_x - min_x)

def create_custom_colormap():
    """创建一个自定义的颜色映射，在较高概率区间颜色更丰富"""
    # 定义颜色断点 - 主要在中高概率区间细分更多颜色
    colors = [
        (0.0, (0.0, 0.0, 0.5)),      # 深蓝色 - 最低概率
        (0.2, (0.0, 0.0, 1.0)),      # 蓝色
        (0.3, (0.0, 0.5, 1.0)),      # 浅蓝色
        (0.4, (0.0, 0.7, 0.7)),      # 青色
        (0.5, (0.0, 0.8, 0.0)),      # 绿色
        (0.6, (0.5, 0.8, 0.0)),      # 黄绿色
        (0.7, (0.8, 0.8, 0.0)),      # 黄色
        (0.75, (0.9, 0.7, 0.0)),     # 橙黄色
        (0.8, (1.0, 0.6, 0.0)),      # 橙色
        (0.85, (1.0, 0.4, 0.0)),     # 深橙色
        (0.9, (1.0, 0.2, 0.0)),      # 橙红色
        (0.95, (1.0, 0.0, 0.0)),     # 红色
        (1.0, (0.8, 0.0, 0.0))       # 深红色 - 最高概率
    ]
    
    # 创建自定义颜色映射
    cmap_name = 'custom_probability_map'
    return mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)





def analyze_visualize_and_crop(image_path, vis_output_path, crop_output_path, grid_size=(8,6), 
                              wavelet='db4', level=3, filter_ratio=0.1, 
                              prob_threshold=0.3):
    """分析图像，可视化结果，并裁剪低概率区域"""
    # 读取图像
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"无法读取图像: {image_path}")
        return None, None
    
    original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # 直接进行滤波，跳过CLAHE增强步骤
    filtered_gray = multi_band_filter(original_gray, wavelet, level, filter_ratio)
    
    # 计算区块特征和小目标概率
    blocks = calculate_block_features(filtered_gray, original_gray, grid_size)
    
    # 找出低概率区域
    area_blocks = find_low_prob_blocks(blocks, grid_size, prob_threshold)
    
    # 如果没有找到符合条件的区块，返回None
    if not area_blocks:
        print(f"图像 {os.path.basename(image_path)} 中没有找到适合的低概率区域")
        return None, None
    
    # 检查是否有区块高于阈值
    high_prob_blocks = [b for b in area_blocks if b['target_prob'] >= prob_threshold]
    if high_prob_blocks:
        print(f"警告: 选中区域包含 {len(high_prob_blocks)} 个概率高于阈值的区块")
    
    # 确保正好有9个区块
    if len(area_blocks) != 9:
        print(f"警告: 选中区域包含 {len(area_blocks)} 个区块，不是预期的9个")
        # 如果区块数量不足9个，我们跳过这个图像
        if len(area_blocks) < 9:
            print(f"跳过图像 {os.path.basename(image_path)}: 无法形成完整的3x3区域")
            return None, None
    
    # 裁剪并保存低概率区域
    crop_coords = crop_and_save_area(original_img, area_blocks, crop_output_path)
    
    # 创建可视化图
    plt.figure(figsize=(16, 10))
    
    # 原图
    plt.subplot(2, 1, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title('原图')
    plt.axis('off')
    
    # 标注图
    plt.subplot(2, 1, 2)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), alpha=0.7)  # 原图作为底图
    
    # 定义颜色映射 - 使用自定义颜色映射
    cmap = create_custom_colormap()
    
    # 使用BoundaryNorm来定义非均匀的颜色边界，在中高概率区域分配更多颜色
    boundaries = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    
    # 绘制所有区块和信息
    for block in blocks:
        y, x, h, w = block['coords']
        prob = block['target_prob']
        color = cmap(norm(prob))  # 使用归一化后的颜色
        
        # 根据概率调整填充透明度
        fill_alpha = min(0.25, prob * 0.3)  # 略微增加透明度以更好地显示颜色
        rect = Rectangle((x, y), w, h, linewidth=1, 
                         edgecolor=color, 
                         facecolor=color, 
                         alpha=fill_alpha)
        plt.gca().add_patch(rect)

        rect = Rectangle((x, y), w, h, linewidth=1, 
                         edgecolor=color, 
                         facecolor=color, 
                         alpha=fill_alpha)
        plt.gca().add_patch(rect)

    # 绘制裁剪区域（用红色粗边框标出）
    if crop_coords:
        y, x, h, w = crop_coords
        rect = Rectangle((x, y), w, h, linewidth=2, 
                         edgecolor='red', 
                         facecolor='none')
        plt.gca().add_patch(rect)
        

    
    plt.title('区块分析热力图与裁剪区域(红框)')
    plt.axis('off')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('小目标概率')
    
    # 保存可视化图像
    plt.tight_layout()
    plt.savefig(vis_output_path, dpi=300)
    plt.close()
    
    return blocks, crop_coords

def process_folder(input_dir, vis_output_dir, crop_output_dir, grid_size=(8,6), 
                  wavelet='db4', level=3, filter_ratio=0.1,
                  prob_threshold=0.3):
    """处理文件夹中的所有图像"""
    # 创建输出目录
    if not os.path.exists(vis_output_dir):
        os.makedirs(vis_output_dir)
    if not os.path.exists(crop_output_dir):
        os.makedirs(crop_output_dir)
    
    # 统计处理结果
    total_images = 0
    processed_images = 0
    
    # 处理每个图像
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        total_images += 1
        
        # 设置输入和输出路径
        image_path = os.path.join(input_dir, filename)
        vis_path = os.path.join(vis_output_dir, f"vis_{filename}")
        crop_path = os.path.join(crop_output_dir, f"crop_{filename}")
        
        print(f"处理: {filename}")
        
        # 分析图像，生成可视化结果，并裁剪低概率区域
        blocks, crop_coords = analyze_visualize_and_crop(
            image_path,
            vis_path,
            crop_path,
            grid_size=grid_size,
            wavelet=wavelet,
            level=level,
            filter_ratio=filter_ratio,
            prob_threshold=prob_threshold
        )
        
        if crop_coords:
            processed_images += 1
            print(f"  已裁剪区域: ({crop_coords[1]},{crop_coords[0]}), 尺寸: {crop_coords[3]}x{crop_coords[2]}")
    
    print(f"\n处理完成! 共处理 {total_images} 张图像，其中 {processed_images} 张生成了裁剪结果")
    print(f"可视化结果保存至: {vis_output_dir}")
    print(f"裁剪结果保存至: {crop_output_dir}")

# 主函数
if __name__ == "__main__":
    # 设置输入和输出目录
    input_dir = "F:/dark/test/test_images_103_re"  # 输入图像目录
    vis_output_dir = "F:/dark/test/visualization_resultnocha"  # 可视化结果保存目录
    crop_output_dir = "F:/dark/test/cropped_results5nocha"  # 裁剪结果保存目录
    
    print("开始处理图像...")
    print(f"输入目录: {input_dir}")
    print(f"可视化输出目录: {vis_output_dir}")
    print(f"裁剪输出目录: {crop_output_dir}")
    print("-----------------------------------")
    
    # 处理参数 - 移除了clip_limit参数
    process_folder(
        input_dir,
        vis_output_dir,
        crop_output_dir,
        grid_size=(12, 15),   # 网格大小
        wavelet='db4',       # 小波类型
        level=5,             # 小波分解层数
        filter_ratio=0.08,   # 傅里叶滤波保留比例
        prob_threshold=0.4   # 小目标概率阈值
>>>>>>> origin/feat/update
>>>>>>> 21a2898 (update)
    )