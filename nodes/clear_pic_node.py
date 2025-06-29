import cv2
import numpy as np
from PIL import Image
import torch

# Tensor to cv2
def tensor2cv2(image):
    image_array = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    # 将PIL图像转换为OpenCV格式
    image = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)
    return image

# Convert cv2 to Tensor
def cv22tensor(image):
    image_pil_after = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return torch.from_numpy(np.array(image_pil_after).astype(np.float32) / 255.0).unsqueeze(0)

def convert_to_binary(image, threshold=127, max_value=255):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, binary = cv2.threshold(gray, threshold, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def remove_small_regions(binary_image, min_area=10):
    binary = binary_image.copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (255 - binary),
        connectivity=8
    )
    result = binary.copy()
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            result[labels == i] = 255
    return result

def morphological_operations(binary_image, operation='dilate', kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'dilate':
        return cv2.dilate(binary_image, kernel, iterations=iterations)
    elif operation == 'erode':
        return cv2.erode(binary_image, kernel, iterations=iterations)
    else:
        raise ValueError("不支持的操作类型")

def custom_median_filter(image, kernel_size=3):
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

class ClearPicNode:
    def __init__(self):
        pass

    CATEGORY = "wtz_erthor_node👾👾👾/clear_pic_node"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "min_area1": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "min_area2": ("INT", {
                    "default": 21,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
                "kernel_size": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 21,
                    "step": 2,
                    "display": "number"
                }),
                "iterations": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "display": "number"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cleaned_image",)
    FUNCTION = "process_image"

    def process_image(self, image, min_area1=2, min_area2=3, kernel_size=3, iterations=1):
        # 将tensor转换为cv2格式
        cv2_image = tensor2cv2(image)
        
        # 1. 转换为二值图
        binary = convert_to_binary(cv2_image)
        
        # 2. 第一次移除小连通域
        cleaned = remove_small_regions(binary, min_area1)
        
        # 3. 形态学操作：先侵蚀后膨胀（开运算）
        eroded = morphological_operations(cleaned, operation='erode', kernel_size=kernel_size, iterations=iterations)
        dilated = morphological_operations(eroded, operation='dilate', kernel_size=kernel_size, iterations=iterations)
        
        # 4. 1/3位中值滤波去除噪点
        smoothed = custom_median_filter(dilated, kernel_size=3)
        
        # 5. 第二次移除小连通域
        final = remove_small_regions(smoothed, min_area2)
        
        # 将结果转换回tensor格式
        # 由于结果是二值图像，需要先转换为3通道
        final_3ch = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
        result_tensor = cv22tensor(final_3ch)
        
        return (result_tensor,) 