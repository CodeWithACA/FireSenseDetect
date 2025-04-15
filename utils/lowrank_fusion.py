import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.linalg import inv, svd
import time

# 添加中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号


def latent_lrr(X, lambda_val):
    """
    潜在低秩表示(Latent Low-Rank Representation)算法
    
    参数:
    X: 输入矩阵
    lambda_val: 正则化参数
    
    返回:
    Z, L, E: 低秩表示的结果矩阵
    """
    A = X
    tol = 1e-5
    rho = 1.1
    max_mu = 1e3
    mu = 1e-6
    maxIter = 100
    
    d, n = X.shape
    m = A.shape[1]
    atx = X.T @ X
    inv_a = inv(A.T @ A + np.eye(m))
    inv_b = inv(A @ A.T + np.eye(d))
    
    J = np.zeros((m, n))
    Z = np.zeros((m, n))
    L = np.zeros((d, d))
    S = np.zeros((d, d))
    E = np.zeros((d, n))
    
    Y1 = np.zeros((d, n))
    Y2 = np.zeros((m, n))
    Y3 = np.zeros((d, d))
    
    iter_num = 0
    while iter_num < maxIter:
        iter_num += 1
        
        # 更新J
        temp_J = Z + Y2/mu
        U_J, sigma_J, V_J = svd(temp_J, full_matrices=False)
        svp_J = len(np.where(sigma_J > 1/mu)[0])
        if svp_J >= 1:
            sigma_J = sigma_J[:svp_J] - 1/mu
        else:
            svp_J = 1
            sigma_J = np.array([0])
        J = U_J[:, :svp_J] @ np.diag(sigma_J) @ V_J[:svp_J, :]
        
        # 更新S
        temp_S = L + Y3/mu
        U_S, sigma_S, V_S = svd(temp_S, full_matrices=False)
        svp_S = len(np.where(sigma_S > 1/mu)[0])
        if svp_S >= 1:
            sigma_S = sigma_S[:svp_S] - 1/mu
        else:
            svp_S = 1
            sigma_S = np.array([0])
        S = U_S[:, :svp_S] @ np.diag(sigma_S) @ V_S[:svp_S, :]
        
        # 更新Z
        Z = inv_a @ (atx - X.T @ L @ X - X.T @ E + J + (X.T @ Y1 - Y2)/mu)
        
        # 更新L
        L = ((X - X @ Z - E) @ X.T + S + (Y1 @ X.T - Y3)/mu) @ inv_b
        
        # 更新E
        xmaz = X - X @ Z - L @ X
        temp = xmaz + Y1/mu
        E = np.maximum(0, temp - lambda_val/mu) + np.minimum(0, temp + lambda_val/mu)
        
        leq1 = xmaz - E
        leq2 = Z - J
        leq3 = L - S
        max_l1 = np.max(np.abs(leq1))
        max_l2 = np.max(np.abs(leq2))
        max_l3 = np.max(np.abs(leq3))
        
        stopC1 = max(max_l1, max_l2)
        stopC = max(stopC1, max_l3)
        if stopC < tol:
            print(f'收敛于第{iter_num}次迭代, 误差:{stopC:.2e}')
            break
        # 添加进度打印
        if iter_num % 100 == 0:
            print(f'迭代进度: {iter_num}/{maxIter}, 当前误差: {stopC:.2e}')

        else:
            Y1 = Y1 + mu * leq1
            Y2 = Y2 + mu * leq2
            Y3 = Y3 + mu * leq3
            mu = min(max_mu, mu * rho)
    
    return Z, L, E

def image_fusion(image1_path, image2_path, output_path=None, show_result=False):
    """
    基于潜在低秩表示的图像融合算法
    
    参数:
    image1_path: 第一张输入图像的路径
    image2_path: 第二张输入图像的路径
    output_path: 融合图像的保存路径，默认为None（不保存）
    show_result: 是否显示结果图像，默认为False
    
    返回:
    F: 融合后的图像
    """
    # 读取图像
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    if image1 is None or image2 is None:
        raise ValueError("无法读取图像，请检查路径是否正确")
    
    # 转换为double类型（0-1范围）
    image1 = image1.astype(np.float64) / 255.0
    image2 = image2.astype(np.float64) / 255.0
    
    # 设置lambda参数
    lambda_val = 0.8
    
    print('开始执行latlrr算法...')
    start_time = time.time()
    
    # 对第一张图像应用latent_lrr
    X1 = image1
    Z1, L1, E1 = latent_lrr(X1, lambda_val)
    
    # 对第二张图像应用latent_lrr
    X2 = image2
    Z2, L2, E2 = latent_lrr(X2, lambda_val)
    
    end_time = time.time()
    print(f'latlrr算法完成，耗时: {end_time - start_time:.2f}秒')
    
    # 计算低秩部分
    I_lrr1 = X1 @ Z1
    I_saliency1 = L1 @ X1
    I_lrr1 = np.clip(I_lrr1, 0, 1)
    I_saliency1 = np.clip(I_saliency1, 0, 1)
    I_e1 = E1
    
    I_lrr2 = X2 @ Z2
    I_saliency2 = L2 @ X2
    I_lrr2 = np.clip(I_lrr2, 0, 1)
    I_saliency2 = np.clip(I_saliency2, 0, 1)
    I_e2 = E2
    
    # 融合
    # lrr部分
    F_lrr = (I_lrr1 + I_lrr2) / 2
    # 显著性部分
    F_saliency = I_saliency1 + I_saliency2
    
    # 最终融合结果
    F = F_lrr + F_saliency
    
    # 显示结果
    if show_result:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 4, 1)
        plt.imshow(image1, cmap='gray')
        plt.title('输入图像1')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(image2, cmap='gray')
        plt.title('输入图像2')
        plt.axis('off')
        
        plt.subplot(1, 4, 3)
        plt.imshow(I_saliency1, cmap='gray')
        plt.title('显著性图1')
        plt.axis('off')
        
        plt.subplot(1, 4, 4)
        plt.imshow(I_saliency2, cmap='gray')
        plt.title('显著性图2')
        plt.axis('off')
        
        plt.figure()
        plt.imshow(F, cmap='gray')
        plt.title('融合结果')
        plt.axis('off')
        
        plt.show()
    
    # 保存结果
    if output_path:
        # 将浮点数图像转换为8位无符号整数
        F_uint8 = (F * 255).astype(np.uint8)
        cv2.imwrite(output_path, F_uint8)
        print(f'融合图像已保存至: {output_path}')
    
    return F

# 示例用法
if __name__ == "__main__":
    # 设置图像路径
    image1_path = r"./demo/demo.jpg"
    image2_path = r"./demo/demo2.jpg"
    output_path = None
    
    # 执行融合
    fused_image = image_fusion(image1_path, image2_path, output_path, show_result=True)