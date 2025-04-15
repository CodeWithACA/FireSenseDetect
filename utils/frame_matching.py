import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import os

# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号


def extract_frames(video_path):
    """从视频中提取所有帧"""
    frames = []
    timestamps = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return frames, timestamps, fps
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转为灰度图像以便于比较
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
        timestamps.append(frame_count / fps)  # 记录时间戳（秒）
        frame_count += 1
    cap.release()
    return frames, timestamps, fps

def compute_frame_features(frame):
    """计算帧的特征（使用多种特征提取方法）"""
    # 1. 直方图特征
    hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    # 2. 降采样（简化特征）
    resized = cv2.resize(frame, (32, 32))
    resized_flat = resized.flatten() / 255.0
    # 3. 边缘特征
    edges = cv2.Canny(frame, 100, 200)
    edge_count = np.count_nonzero(edges) / (frame.shape[0] * frame.shape[1])
    return {
        'hist': hist,
        'resized': resized_flat,
        'edge_ratio': edge_count
    }

def compute_similarity(features1, features2, hist_weight=0.3, pixel_weight=0.3, edge_weight=0.4):
    """计算两帧之间的相似度"""
    # 直方图相似度 (1 - 余弦距离)
    hist_sim = 1 - cosine(features1['hist'], features2['hist'])
    # 像素相似度 (1 - 余弦距离)
    pixel_sim = 1 - cosine(features1['resized'], features2['resized'])
    # 边缘特征相似度
    edge_sim = 1 - abs(features1['edge_ratio'] - features2['edge_ratio'])
    # 综合相似度 (可以调整权重) 
    similarity = hist_weight * hist_sim + pixel_weight * pixel_sim + edge_weight * edge_sim
    
    return similarity

def match_frames(ir_video_path, gray_video_path, similarity_threshold=0.7, output_dir=None, max_time_diff=3.0,hist_weight=0.3, pixel_weight=0.3, edge_weight=0.4):
    """匹配两个视频中的相似帧
    
    Args:
        ir_video_path: 红外视频路径
        gray_video_path: 灰度视频路径
        similarity_threshold: 相似度阈值
        output_dir: 输出目录
        max_time_diff: 最大时间差（秒）
    """
    # 提取帧
    ir_frames, ir_timestamps, ir_fps = extract_frames(ir_video_path)
    gray_frames, gray_timestamps, gray_fps = extract_frames(gray_video_path)
    print(f"红外视频: {len(ir_frames)}帧, {ir_fps}fps")
    print(f"灰度视频: {len(gray_frames)}帧, {gray_fps}fps")
    # 计算每一帧的特征
    print("计算红外视频特征...")
    ir_features = [compute_frame_features(frame) for frame in ir_frames]
    print("计算灰度视频特征...")
    gray_features = [compute_frame_features(frame) for frame in gray_frames]    
    # 确定哪个视频的帧数较少，用它作为基准进行匹配
    if len(ir_frames) <= len(gray_frames):
        # 红外帧数较少，以红外帧为基准
        base_frames = ir_frames
        base_features = ir_features
        base_timestamps = ir_timestamps
        base_fps = ir_fps
        target_frames = gray_frames
        target_features = gray_features
        target_fps = gray_fps
        is_ir_base = True
    else:
        # 灰度帧数较少，以灰度帧为基准
        base_frames = gray_frames
        base_features = gray_features
        base_timestamps = gray_timestamps
        base_fps = gray_fps
        target_frames = ir_frames
        target_features = ir_features
        target_fps = ir_fps
        is_ir_base = False
    
    # 匹配帧
    matches = []
    print("开始匹配帧...")
    
    # 记录已匹配的目标帧索引
    matched_target_indices = set()
    last_match_idx = -1
    
    # 对于基准视频的每一帧，找到最匹配的目标帧
    for i, base_feat in enumerate(base_features):
        base_time = base_timestamps[i]
        
        # 估计对应的目标视频时间点
        if last_match_idx == -1:
            expected_target_idx = int(base_time * target_fps)
        else:
            # 从上一次匹配位置开始搜索
            expected_target_idx = last_match_idx + 1
        
        # 在预期时间点附近搜索
        window = int(max_time_diff * target_fps)
        start_idx = max(0, expected_target_idx - window)
        end_idx = min(len(target_frames) - 1, expected_target_idx + window)
        
        best_match_idx = -1
        best_similarity = -1
        
        # 在时间窗口内寻找最佳匹配
        for j in range(start_idx, end_idx + 1):
            # 跳过已匹配的帧
            if j in matched_target_indices:
                continue
                
            similarity = compute_similarity(base_feat, target_features[j], hist_weight, pixel_weight, edge_weight)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = j
        
        # 如果相似度超过阈值，则认为是匹配的
        if best_similarity >= similarity_threshold and best_match_idx not in matched_target_indices:
            matched_target_indices.add(best_match_idx)
            last_match_idx = best_match_idx
            
            # 根据基准视频类型，调整匹配对的顺序
            if is_ir_base:
                matches.append((i, best_match_idx, best_similarity))
            else:
                matches.append((best_match_idx, i, best_similarity))
    
    # 按时间顺序排序匹配结果
    matches.sort(key=lambda x: x[0])
    
    print(f"找到 {len(matches)} 对匹配帧")
    
    # 可视化并保存匹配结果
    if output_dir and len(matches) > 0:
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, (ir_idx, gray_idx, sim) in enumerate(matches):
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(ir_frames[ir_idx], cmap='gray')
            plt.title(f"红外帧 #{ir_idx} (时间: {ir_timestamps[ir_idx]:.2f}s)")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(gray_frames[gray_idx], cmap='gray')
            plt.title(f"灰度帧 #{gray_idx} (时间: {gray_timestamps[gray_idx]:.2f}s)")
            plt.axis('off')
            
            plt.suptitle(f"匹配对 #{idx+1} - 相似度: {sim:.4f}")
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f"match_{idx+1}.png"))
            plt.close()
    
    return matches, ir_frames, gray_frames

if __name__ == "__main__":
    # 替换为你的视频路径
    ir_video_path = r"D:\WorkSpace\codespace\fwwb\videos\output_tr_smoked_30fps.mp4"  # 红外视频
    gray_video_path = r"D:\WorkSpace\codespace\fwwb\videos\output_rgb_smoked.mp4"  # 灰度视频
    
    # 创建输出目录
    output_dir = "d:/WorkSpace/codespace/fwwb/matched_frames"
    
    # 执行帧匹配，考虑最多3秒的时间差
    matches, ir_frames, gray_frames = match_frames(
        ir_video_path, 
        gray_video_path,
        similarity_threshold=0.7,  # 相似度阈值
        output_dir=output_dir,
        max_time_diff=3.0,  # 最大时间差（秒）
        hist_weight = 0.01,
        pixel_weight = 0.01,
        edge_weight = 0.98)
    
    # 输出匹配结果
    print("\n匹配结果:")
    for idx, (ir_idx, gray_idx, sim) in enumerate(matches):
        print(f"匹配 #{idx+1}: 红外帧 #{ir_idx} <-> 灰度帧 #{gray_idx} (相似度: {sim:.4f})")