import sys
sys.path.append('/root/autodl-tmp/diffuser-pipe/src/app/ComfyUI')

import torch
from mediapipe import solutions
import cv2
import numpy as np
from PIL import Image, ImageFilter
from ultralytics import YOLO
import os
import mediapipe as mp
import math
from media_face import Media_Pipe_Face_Mesh_detect
from media_utils import load_image
# import sampling as k_diffusion_sampling

min_detection_confidence = 0.1
kernel_size = 7
set_resolution = 512
def make_face_mask(ref_path):
    ref_img = Image.open(ref_path)
    tensor_image = load_image(ref_path)
    images = Media_Pipe_Face_Mesh_detect(tensor_image, max_faces=1, min_confidence=min_detection_confidence, resolution=set_resolution)
    image = images[0]
    # 保存图片
    image_array = image.squeeze(0).cpu().numpy()   # 转换为 HWC 格式并移除梯度
    # 将数组转换为 PIL 图像
    image_pil = Image.fromarray((image_array * 255).astype(np.uint8))  # 值在 [0, 1] 范围内
    # image_pil.save("/home/output/img.png")

    # 将 PIL.Image 转换为 numpy 数组
    image_np = np.array(image_pil)

    # image_cv2 = cv2.imread("/home/output/img.png")
    
    # 转换 RGB 到 BGR (PIL 使用 RGB，而 OpenCV 使用 BGR)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    # 转换为 HSV 色彩空间
    hsv = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2HSV)

    # 定义绿色的颜色范围 (可以根据图像调整颜色上下限)
    lower_green = np.array([35, 100, 100])
    upper_green = np.array([85, 255, 255])

    # 创建绿色区域的掩膜
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 找到绿色线条的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个与原图像相同大小的全黑图像
    mask_filled = np.zeros_like(image_cv2)

    # 填充轮廓内部为白色
    cv2.drawContours(mask_filled, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    # 转换为灰度图，确保是单通道
    mask_filled_gray = cv2.cvtColor(mask_filled, cv2.COLOR_BGR2GRAY)

    # 扩展区域（可选步骤，调整 kernel 大小来扩展区域）
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_filled_dilated = cv2.dilate(mask_filled_gray, kernel, iterations=1)

    # # 保存最终的白色遮罩
    # cv2.imwrite('C:/Users/wangtz/Desktop/vscode/diffuser-pipe/filled_white_mask.png', mask_filled_dilated)
    
    # 使用 PIL 将单通道的 OpenCV 图像转换为 PIL 图像
    pil_image = Image.fromarray(mask_filled_dilated)
    
    pil_image_resized = pil_image.resize(ref_img.size, Image.BILINEAR)
    return pil_image_resized

def crop_face_area(image_to_cut_path, black_white_image, output_image_path='', save=False):
    """
    在一张包含白色区域的单通道图像上找到最小的白色区域外接矩形，
    并根据该区域在另一张图片中裁剪相同区域，并保存结果。

    参数：
    black_white_image: PIL单通道图像。
    image_to_cut_path (str): 需要裁剪的目标图像的路径。
    output_image_path (str): 保存裁剪后图片的路径。
    """
    # 1. 读取单通道的PIL图像，并将其转为NumPy数组
    img_with_white_area = black_white_image  # 单通道黑白图
    img_to_cut = Image.open(image_to_cut_path)  # 另一张需要裁切的PIL图像
    width, height = img_to_cut.size
    
    # 将PIL图像转换为NumPy数组
    img_with_white_area_np = np.array(img_with_white_area)

    # 2. 找到不规则白色区域的最小外接矩形
    # 阈值分割，假设白色区域接近255
    binary = np.where(img_with_white_area_np > 240, 255, 0).astype(np.uint8)

    # 找到白色区域的非零像素的坐标
    coords = np.column_stack(np.where(binary > 0))

    # 如果没有白色区域，返回提示
    if coords.size == 0:
        print("没有找到白色区域")
        return

    # 找到白色区域的最小外接矩形
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # 计算矩形的宽和高
    w, h = x_max - x_min + 1, y_max - y_min + 1

    # 3. 在第二张图像上裁剪相同区域
    img_to_cut_np = np.array(img_to_cut)  # 将PIL图像转换为NumPy数组
    cut_image_np = img_to_cut_np[y_min:y_min+h, x_min:x_min+w]

    # 4. 将裁剪后的NumPy数组转换回PIL图像
    face_cut_image = Image.fromarray(cut_image_np)

    # 找到白色区域的外部参考矩形
    x_min_1 = max(x_min - w // 2, 0)
    y_min_1 = max(y_min - h // 2, 0)
    x_max_1 = min(x_min + w * 3 // 2, width)
    y_max_1 = min(y_min + h * 3 // 2, height)
    cut_image_np_1 = img_to_cut_np[y_min_1:y_max_1, x_min_1:x_max_1]
    face_cut_image_1 = Image.fromarray(cut_image_np_1)
    
    if save:
        # 5. 保存裁切后的PIL图像
        face_cut_image.save(output_image_path)

        print(f"裁切区域的坐标为: x={x_min}, y={y_min}, 宽度={w}, 高度={h}")
        print(f"裁切后的图片已保存为: {output_image_path}")

    return face_cut_image, face_cut_image_1, (y_min_1, y_max_1, x_min_1, x_max_1)

# sampler
def repeat_to_batch_size(tensor, batch_size, dim=0):
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    elif tensor.shape[dim] < batch_size:
        return tensor.repeat(dim * [1] + [math.ceil(batch_size / tensor.shape[dim])] + [1] * (len(tensor.shape) - 1 - dim)).narrow(dim, 0, batch_size)
    return tensor

def fix_empty_latent_channels(model, latent_image):
    latent_channels = model.get_model_object("latent_format").latent_channels #Resize the empty latent image so it has the right number of channels
    if latent_channels != latent_image.shape[1] and torch.count_nonzero(latent_image) == 0:
        latent_image = repeat_to_batch_size(latent_image, latent_channels, dim=1)
    return latent_image

def prepare_noise(latent_image, seed, noise_inds=None):
    """
    creates random noise given a latent image and a seed.
    optional arg skip can be used to skip and discard x number of noise generations for a given seed
    """
    generator = torch.manual_seed(seed)
    if noise_inds is None:
        return torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
    
    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1]+1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device="cpu")
        if i in unique_inds:
            noises.append(noise)
    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises


def prepare_callback(model, steps, x0_output_dict=None):
    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"

    # previewer = get_previewer(model.load_device, model.model.latent_format)

    # pbar = comfy.utils.ProgressBar(steps)
    def callback(step, x0, x, total_steps):
        if x0_output_dict is not None:
            x0_output_dict["x0"] = x0

        preview_bytes = None
        # if previewer:
        #     preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
        # pbar.update_absolute(step + 1, total_steps, preview_bytes)
    return callback

KSAMPLER_NAMES = ["euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
                  "ipndm", "ipndm_v", "deis"]
SCHEDULER_NAMES = ["karras"]
# SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform", "beta"]
SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)

def calculate_sigmas(model_sampling, scheduler_name, steps):
    if scheduler_name == "karras":
        sigmas = get_sigmas_karras(n=steps, sigma_min=float(model_sampling.sigma_min), sigma_max=float(model_sampling.sigma_max))
    # elif scheduler_name == "exponential":
    #     sigmas = k_diffusion_sampling.get_sigmas_exponential(n=steps, sigma_min=float(model_sampling.sigma_min), sigma_max=float(model_sampling.sigma_max))
    # elif scheduler_name == "normal":
    #     sigmas = normal_scheduler(model_sampling, steps)
    # elif scheduler_name == "simple":
    #     sigmas = simple_scheduler(model_sampling, steps)
    # elif scheduler_name == "ddim_uniform":
    #     sigmas = ddim_scheduler(model_sampling, steps)
    # elif scheduler_name == "sgm_uniform":
    #     sigmas = normal_scheduler(model_sampling, steps, sgm=True)
    # elif scheduler_name == "beta":
    #     sigmas = beta_scheduler(model_sampling, steps)
    else:
        print("error invalid scheduler {}".format(scheduler_name))
    return sigmas

class Sampler:
    def sample(self):
        pass

    def max_denoise(self, model_wrap, sigmas):
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

class KSamplerX0Inpaint:
    def __init__(self, model, sigmas):
        self.inner_model = model
        self.sigmas = sigmas
    def __call__(self, x, sigma, denoise_mask, model_options={}, seed=None):
        if denoise_mask is not None:
            if "denoise_mask_function" in model_options:
                denoise_mask = model_options["denoise_mask_function"](sigma, denoise_mask, extra_options={"model": self.inner_model, "sigmas": self.sigmas})
            latent_mask = 1. - denoise_mask
            x = x * denoise_mask + self.inner_model.inner_model.model_sampling.noise_scaling(sigma.reshape([sigma.shape[0]] + [1] * (len(self.noise.shape) - 1)), self.noise, self.latent_image) * latent_mask
        out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
        if denoise_mask is not None:
            out = out * denoise_mask + self.latent_image * latent_mask
        return out
    
class KSAMPLER(Sampler):
    def __init__(self, sampler_function, extra_options={}, inpaint_options={}):
        self.sampler_function = sampler_function
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        if self.inpaint_options.get("random", False): #TODO: Should this be the default?
            generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
            model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)
        else:
            model_k.noise = noise

        noise = model_wrap.inner_model.model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas))

        k_callback = None
        total_steps = len(sigmas) - 1
        if callback is not None:
            k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)

        samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
        return samples
    
class KSampler:
    SCHEDULERS = SCHEDULER_NAMES
    SAMPLERS = SAMPLER_NAMES
    DISCARD_PENULTIMATE_SIGMA_SAMPLERS = set(('dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2'))

    def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None, model_options={}):
        self.model = model
        self.device = device
        if scheduler not in self.SCHEDULERS:
            scheduler = self.SCHEDULERS[0]
        if sampler not in self.SAMPLERS:
            sampler = self.SAMPLERS[0]
        self.scheduler = scheduler
        self.sampler = sampler
        self.set_steps(steps, denoise)
        self.denoise = denoise
        self.model_options = model_options

    def calculate_sigmas(self, steps):
        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler in self.DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
            steps += 1
            discard_penultimate_sigma = True

        sigmas = calculate_sigmas(self.model.get_model_object("model_sampling"), self.scheduler, steps)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            if denoise <= 0.0:
                self.sigmas = torch.FloatTensor([])
            else:
                new_steps = int(steps/denoise)
                sigmas = self.calculate_sigmas(new_steps).to(self.device)
                self.sigmas = sigmas[-(steps + 1):]

    def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None, force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
        if sigmas is None:
            sigmas = self.sigmas

        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)

        sampler = sampler_object(self.sampler)

        return sample(self.model, noise, positive, negative, cfg, self.device, sampler, sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)


def ksampler(sampler_name, extra_options={}, inpaint_options={}):
    # if sampler_name == "dpm_fast":
    #     def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):
    #         if len(sigmas) <= 1:
    #             return noise

    #         sigma_min = sigmas[-1]
    #         if sigma_min == 0:
    #             sigma_min = sigmas[-2]
    #         total_steps = len(sigmas) - 1
    #         return k_diffusion_sampling.sample_dpm_fast(model, noise, sigma_min, sigmas[0], total_steps, extra_args=extra_args, callback=callback, disable=disable)
    #     sampler_function = dpm_fast_function
    # elif sampler_name == "dpm_adaptive":
    #     def dpm_adaptive_function(model, noise, sigmas, extra_args, callback, disable, **extra_options):
    #         if len(sigmas) <= 1:
    #             return noise

    #         sigma_min = sigmas[-1]
    #         if sigma_min == 0:
    #             sigma_min = sigmas[-2]
    #         return k_diffusion_sampling.sample_dpm_adaptive(model, noise, sigma_min, sigmas[0], extra_args=extra_args, callback=callback, disable=disable, **extra_options)
    #     sampler_function = dpm_adaptive_function
    # else:
    #     sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))
    sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))
    return KSAMPLER(sampler_function, extra_options, inpaint_options)

def sampler_object(name):
    # if name == "uni_pc":
    #     sampler = KSAMPLER(uni_pc.sample_unipc)
    # elif name == "uni_pc_bh2":
    #     sampler = KSAMPLER(uni_pc.sample_unipc_bh2)
    # elif name == "ddim":
    #     sampler = ksampler("euler", inpaint_options={"random": True})
    # else:
    #     sampler = ksampler(name)
    sampler = ksampler(name)
    return sampler



def sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, noise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
    sampler = KSampler(model, steps=steps, device=model.load_device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

    samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
    samples = samples.to("GPU")
    return samples

def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = fix_empty_latent_channels(model, latent_image)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = prepare_callback(model, steps)
    disable_pbar = True
    samples = sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.copy()
    out["samples"] = samples
    return (out, )

# face_model_path = os.path.join(base_path, "models/dz_facedetailer/yolo/face_yolov8m.pt")
MASK_CONTROL = ["dilate", "erode", "disabled"]
MASK_TYPE = ["face", "box"]
min_detection_confidence = 0.2
yolo_shreshold = 0.2
# class FaceDetailer:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {"required":
#                 {
#                     "model": ("MODEL",),
#                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
#                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
#                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
#                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
#                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
#                     "positive": ("CONDITIONING", ),
#                     "negative": ("CONDITIONING", ),
#                     "latent_image": ("LATENT", ),
#                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
#                     "latent_image": ("LATENT", ),
#                     "vae": ("VAE",),
#                     "mask_blur": ("INT", {"default": 0, "min": 0, "max": 100}),
#                     "mask_type": (MASK_TYPE, ),
#                     "mask_control": (MASK_CONTROL, ),
#                     "dilate_mask_value": ("INT", {"default": 3, "min": 0, "max": 100}),
#                     "erode_mask_value": ("INT", {"default": 3, "min": 0, "max": 100}),
#                 }
#                 }
        
#     def detailer(model, positive, negative, latent_image, steps=20, cfg=7, sampler_name="dpmpp_2m", scheduler="karras", denoise=0.7):

#         mask = make_mask()#(ref_path)

#         latent_mask = set_mask(latent_image, mask)
        
#         random_seed = np.random.randint(10**19, 10**20)
#         latent = common_ksampler(model, random_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_mask, denoise=denoise)

#         return (latent[0], latent[0]["noise_mask"],)

# class Detection:
#     def __init__(self):
#         pass

#     def detect_faces(self, tensor_img, batch_size, mask_type, mask_control, mask_blur, mask_dilate, mask_erode):
#         mask_imgs = []
#         for i in range(0, batch_size):
#             # print(input_tensor_img[i, :,:,:].shape)
#             # convert input latent to numpy array for yolo model
#             img = image2nparray(tensor_img[i], False)
#             # Process the face mesh or make the face box for masking
#             if mask_type == "box":
#                 final_mask = facebox_mask(img)
#             else:
#                 final_mask = facemesh_mask(img)

#             final_mask = self.mask_control(final_mask, mask_control, mask_blur, mask_dilate, mask_erode)

#             final_mask = np.array(Image.fromarray(final_mask).getchannel('A')).astype(np.float32) / 255.0
#             # Convert mask to tensor and assign the mask to the input tensor
#             final_mask = torch.from_numpy(final_mask)

#             mask_imgs.append(final_mask)

#         final_mask = torch.stack(mask_imgs)

#         return final_mask

#     def mask_control(self, numpy_img, mask_control, mask_blur, mask_dilate, mask_erode):
#         numpy_image = numpy_img.copy();
#         # Erode/Dilate mask
#         if mask_control == "dilate":
#             if mask_dilate > 0:
#                 numpy_image = self.dilate_mask(numpy_image, mask_dilate)
#         elif mask_control == "erode":
#             if mask_erode > 0:
#                 numpy_image = self.erode_mask(numpy_image, mask_erode)
#         if mask_blur > 0:
#             final_mask_image = Image.fromarray(numpy_image)
#             blurred_mask_image = final_mask_image.filter(
#                 ImageFilter.GaussianBlur(radius=mask_blur))
#             numpy_image = np.array(blurred_mask_image)

#         return numpy_image

#     def erode_mask(self, mask, dilate):
#         # I use erode function because the mask is inverted
#         # later I will fix it
#         kernel = np.ones((int(dilate), int(dilate)), np.uint8)
#         dilated_mask = cv2.dilate(mask, kernel, iterations=1)
#         return dilated_mask

#     def dilate_mask(self, mask, erode):
#         # I use dilate function because the mask is inverted like the other function
#         # later I will fix it
#         kernel = np.ones((int(erode), int(erode)), np.uint8)
#         eroded_mask = cv2.erode(mask, kernel, iterations=1)
#         return eroded_mask

# def facebox_mask(image):
#     # Create an empty image with alpha
#     mask = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)

#     # setup yolov8n face detection model
#     face_model = YOLO(face_model_path)
#     face_bbox = face_model(image, conf=yolo_shreshold)
#     boxes = face_bbox[0].boxes
#     print('1', boxes.xyxy)
#     # box = boxes[0].xyxy
#     for box in boxes.xyxy:
#         x_min, y_min, x_max, y_max = box.tolist()
#         # Calculate the center of the bounding box
#         center_x = (x_min + x_max) / 2
#         center_y = (y_min + y_max) / 2

#         # Calcule the maximum width and height
#         width = x_max - x_min
#         height = y_max - y_min
#         max_size = max(width, height)

#         # Get the new WxH for a ratio of 1:1
#         new_width = max_size
#         new_height = max_size

#         # Calculate the new coordinates
#         new_x_min = int(center_x - new_width / 2)
#         new_y_min = int(center_y - new_height / 2)
#         new_x_max = int(center_x + new_width / 2)
#         new_y_max = int(center_y + new_height / 2)

#         # print((new_x_min, new_y_min), (new_x_max, new_y_max))
#         # set the square in the face location
#         cv2.rectangle(mask, (new_x_min, new_y_min), (new_x_max, new_y_max), (0, 0, 0, 255), -1)

#     # mask[:, :, 3] = ~mask[:, :, 3]  # invert the mask

#     return mask


# def facemesh_mask(image):

#     faces_mask = []

#     # Empty image with the same shape as input
#     mask = np.zeros(
#         (image.shape[0], image.shape[1], 4), dtype=np.uint8)
    
#     # setup yolov8n face detection model
#     face_model = YOLO(face_model_path)
#     face_bbox = face_model(image, conf=yolo_shreshold)
#     boxes = face_bbox[0].boxes
#     print(boxes.xyxy)
#     # box = boxes[0].xyxy
#     for box in boxes.xyxy:
#         x_min, y_min, x_max, y_max = box.tolist()
#         # Calculate the center of the bounding box
#         center_x = (x_min + x_max) / 2
#         center_y = (y_min + y_max) / 2

#         # Calcule the maximum width and height
#         width = x_max - x_min
#         height = y_max - y_min
#         max_size = max(width, height)

#         # Get the new WxH for a ratio of 1:1
#         new_width = max_size
#         new_height = max_size

#         # Calculate the new coordinates
#         new_x_min = int(center_x - new_width / 2)
#         new_y_min = int(center_y - new_height / 2)
#         new_x_max = int(center_x + new_width / 2)
#         new_y_max = int(center_y + new_height / 2)

#         print((new_x_min, new_y_min), (new_x_max, new_y_max))
#         # set the square in the face location
#         face = image[new_y_min:new_y_max, new_x_min:new_x_max, :]
#         print(face.shape)
#         # 保存人脸图片
#         cv2.imwrite("C:/Users/wangtz/Desktop/vscode/face.jpg", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
#         mp_face_mesh = solutions.face_mesh
#         face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=min_detection_confidence)
#         results = face_mesh.process(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
#         print(results.multi_face_landmarks)
#         results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         print(results.multi_face_landmarks)
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 # List of detected face points
#                 points = []
#                 for landmark in face_landmarks.landmark:
#                     cx, cy = int(
#                         landmark.x * face.shape[1]), int(landmark.y * face.shape[0])
#                     points.append([cx, cy])

#                 face_mask = np.zeros((face.shape[0], face.shape[1], 4), dtype=np.uint8)

#                 # Obtain the countour of the face
#                 convex_hull = cv2.convexHull(np.array(points))
                
#                 # Fill the contour and store it in alpha for the mask
#                 cv2.fillConvexPoly(face_mask, convex_hull, (0, 0, 0, 255))

#                 faces_mask.append([face_mask, [new_x_min, new_x_max, new_y_min, new_y_max]])
            
#     for face_mask in faces_mask:
#         paste_numpy_images(mask, face_mask[0], face_mask[1][0], face_mask[1][1], face_mask[1][2], face_mask[1][3])

#     # print(f"{len(faces_mask)} faces detected")
#     # mask[:, :, 3] = ~mask[:, :, 3]
#     return mask


def paste_numpy_images(target_image, source_image, x_min, x_max, y_min, y_max):
    # Paste the source image into the target image at the specified coordinates
    target_image[y_min:y_max, x_min:x_max, :] = source_image

    return target_image



def image2nparray(image, BGR):
    """
    convert tensor image to numpy array

    Args:
        image (Tensor): Tensor image

    Returns:
        returns: Numpy array.

    """
    narray = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

    if BGR:
        return narray
    else:
        return narray[:, :, ::-1]


def set_mask(samples, mask):
    s = samples.copy()
    s["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
    return s
