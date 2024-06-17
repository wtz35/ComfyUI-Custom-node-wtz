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


# 0 定义一个类，一个节点就是一个类。comfyui会引入这个类作为一个节点==============================
class cut_face_node:
    def __init__(self):
        pass

    # 1 定义这个节点在ui界面中的位置=======================================================
    CATEGORY = "wtz_erthor_node👾👾👾/cut_face_node"


    #====================定义输入===========================================================
    @classmethod
    def INPUT_TYPES(s):# 固定格式，输入参数种类
        # 返回一个包含所需输入类型的字典
        return {
            "required": {
                "seg_image": ("IMAGE",),
                "ori_image": ("IMAGE",),
                # # 2 左边的输入点在这里定义=================================================
                # "左边的输入": ("STRING", {"forceInput": True}),


                # # 3 中间的参数栏在这里定义=================================================
                # "参数：整数": ("INT", {
                #     "default": 20,  # 默认
                #     "min": 1,
                #     "max": 10000,
                #     "step": 2,  # 步长
                #     "display": "number"}),  # 数值调整

            },
        }


    # 4 右边的输出点在这里定义=============================================================
    OUTPUT_NODE = True  # 表明它是一个输出节点
    # 输出的数据类型，需要大写
    RETURN_TYPES = ("IMAGE",)
    # 自定义输出名称
    RETURN_NAMES = ("face image",)


    # 5 节点的核心功能逻辑在这里定义========================================================
    FUNCTION = "cut_face" # 核心功能函数名称，将运行这个类中的这个方法


    # 定义剪切函数
    def cut_face(self, seg_image, ori_image):
        image = tensor2cv2(seg_image)
        ori_image_cv2 = tensor2cv2(ori_image)
        
        
        # 将图片按比例缩放到短边为r1
        r1 = 704
        if image.shape[0] < image.shape[1]:
            image = cv2.resize(image, (int(r1/image.shape[0]*image.shape[1]), r1))
            ori_image_cv2 = cv2.resize(ori_image_cv2, (int(r1/image.shape[0]*image.shape[1]), r1))
        else:
            image = cv2.resize(image, (r1, int(r1/image.shape[1]*image.shape[0])))
            ori_image_cv2 = cv2.resize(ori_image_cv2, (r1, int(r1/image.shape[1]*image.shape[0])))

        # 定义绿色的HSV范围
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        x1, y1, w1, h1 = 0, 0, 0, 0
        # 在绿色脸部周围画矩形框
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w*h > w1*h1:
                x1, y1, w1, h1 = x, y, w, h

        l1 = 20
        l2 = 70
        l3 = 110

        x11 = x1 - l2 if x1 - l2 > 0 else 0
        y11 = y1 - l3 if y1 - l3 > 0 else 0
        x2 = x1 + w1 + l2 if x1 + w1 + l2 < image.shape[1] else image.shape[1]
        y2 = y1 + h1 + l1 if y1 + h1 + l1 < image.shape[0] else image.shape[0]

        # 剪裁出图片中的脸部
        face = ori_image_cv2[y11:y2, x11:x2]
        face_image = cv22tensor(face)
        return (face_image,)

