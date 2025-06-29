import cv2
import numpy as np
import os

def convert_to_binary(image, threshold=127, max_value=255):
    """
    将图像转换为黑白二值图
    
    Args:
        image: 输入图像，可以是彩色或灰度图像
        threshold: 二值化阈值，默认127
        max_value: 二值化后的最大值，默认255
    
    Returns:
        二值化后的图像
    """
    # 如果是彩色图像，先转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 使用OTSU自适应阈值进行二值化
    _, binary = cv2.threshold(gray, threshold, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def remove_small_regions(binary_image, min_area=10):
    """
    移除小于指定面积的连通区域
    
    Args:
        binary_image: 二值化图像
        min_area: 最小连通域面积
    
    Returns:
        处理后的图像
    """
    binary = binary_image.copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (255 - binary),  # 将黑色区域转为前景
        connectivity=8    # 使用8连通
    )
    
    result = binary.copy()
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            result[labels == i] = 255
            
    return result

def morphological_operations(binary_image, operation='dilate', kernel_size=3, iterations=1):
    """
    对二值图像进行形态学操作
    
    Args:
        binary_image: 二值化图像
        operation: 操作类型，'dilate'为膨胀，'erode'为侵蚀
        kernel_size: 结构元素大小
        iterations: 操作次数
    
    Returns:
        处理后的图像
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    if operation == 'dilate':
        return cv2.dilate(binary_image, kernel, iterations=iterations)
    elif operation == 'erode':
        return cv2.erode(binary_image, kernel, iterations=iterations)
    else:
        raise ValueError("不支持的操作类型")

def custom_median_filter(image, kernel_size=3):
    """
    使用1/3位中值滤波进行图像平滑
    
    Args:
        image: 输入的二值图像
        kernel_size: 滤波窗口大小
        
    Returns:
        处理后的图像
    """
    height, width = image.shape
    output_image = np.zeros_like(image)
    half_kernel = kernel_size // 2
    
    for i in range(half_kernel, height - half_kernel):
        for j in range(half_kernel, width - half_kernel):
            window = image[i-half_kernel:i+half_kernel+1, j-half_kernel:j+half_kernel+1]
            sorted_window = np.sort(window.flatten())
            index = len(sorted_window) // 3
            output_image[i, j] = sorted_window[index]
    
    return output_image

def process_image(image_path, min_area1=2, min_area2=3, kernel_size=3, iterations=1):
    """
    完整的图像处理流程
    
    Args:
        image_path: 输入图像路径
        min_area1: 第一次去除小连通域的面积阈值
        min_area2: 第二次去除小连通域的面积阈值
        kernel_size: 形态学操作的结构元素大小
        iterations: 形态学操作的迭代次数
    
    Returns:
        处理后的图像
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("无法读取图像")
    
    # 1. 转换为二值图
    binary = convert_to_binary(image)
    
    # 2. 第一次移除小连通域
    cleaned = remove_small_regions(binary, min_area1)
    
    # 3. 形态学操作：先侵蚀后膨胀（开运算）
    eroded = morphological_operations(cleaned, operation='erode', kernel_size=kernel_size, iterations=iterations)
    dilated = morphological_operations(eroded, operation='dilate', kernel_size=kernel_size, iterations=iterations)
    
    # 4. 1/3位中值滤波去除噪点
    smoothed = custom_median_filter(dilated, kernel_size=3)
    
    # 5. 第二次移除小连通域
    final = remove_small_regions(smoothed, min_area2)
    
    return final

if __name__ == "__main__":
    image_path = "C:/Users/wangtz/Desktop/vscode/ComfyUI-Custom-node-wtz/img/111.jpg"
    result = process_image(
        image_path,
        min_area1=5,
        min_area2=21,
        kernel_size=3,
        iterations=1
    )
    # 保存最终结果
    output_path = os.path.join(os.path.dirname(image_path), "output.jpg")
    cv2.imwrite(output_path, result)

