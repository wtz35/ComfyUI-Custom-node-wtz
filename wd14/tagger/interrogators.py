from typing import List, Dict

from wd14.tagger.interrogator import Interrogator, WaifuDiffusionInterrogator, MLDanbooruInterrogator

import os

# 获取环境变量的值
home_path = os.environ.get("HOME_PATH") 

if not home_path:
    raise ValueError("HOME PATH must be provided either as an argument or via the HOME PATH environment variable.")

wd14_model_path = home_path + "/models/wd14"


interrogators: Dict[str, Interrogator] = {
    'wd14-vit.v1': WaifuDiffusionInterrogator(
        'WD14 ViT v1',
        repo_id='SmilingWolf/wd-v1-4-vit-tagger'
    ),
    'wd14-vit.v2': WaifuDiffusionInterrogator(
        'WD14 ViT v2',
        repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2',
    ),
    'wd14-convnext.v1': WaifuDiffusionInterrogator(
        'WD14 ConvNeXT v1',
        repo_id='SmilingWolf/wd-v1-4-convnext-tagger'
    ),
    'wd14-convnext.v2': WaifuDiffusionInterrogator(
        'WD14 ConvNeXT v2',
        repo_id='SmilingWolf/wd-v1-4-convnext-tagger-v2',
    ),
    'wd14-convnextv2.v1': WaifuDiffusionInterrogator(
        'WD14 ConvNeXTV2 v1',
        # the name is misleading, but it's v1
        repo_id='SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
    ),
    'wd14-swinv2-v1': WaifuDiffusionInterrogator(
        'WD14 SwinV2 v1',
        # again misleading name
        repo_id='SmilingWolf/wd-v1-4-swinv2-tagger-v2',
    ),
    'wd-v1-4-moat-tagger.v2': WaifuDiffusionInterrogator(
        'WD14 moat tagger v2',
        repo_id='SmilingWolf/wd-v1-4-moat-tagger-v2'
    ),
    'wd-v1-4-vit-tagger.v3': WaifuDiffusionInterrogator(
        'WD14 ViT v3',
        repo_id='SmilingWolf/wd-vit-tagger-v3'
    ),
    'wd-v1-4-convnext-tagger.v3': WaifuDiffusionInterrogator(
        'WD14 ConvNext v3',
        repo_id='SmilingWolf/wd-convnext-tagger-v3'
    ),
    'wd-v1-4-swinv2-tagger.v3': WaifuDiffusionInterrogator(
        'WD14 SwinV2 v3',
        repo_id='SmilingWolf/wd-swinv2-tagger-v3'
    ),
    'wd-vit-large-tagger-v3': WaifuDiffusionInterrogator(
        'WD ViT-Large Tagger v3',
        repo_id='SmilingWolf/wd-vit-large-tagger-v3'
    ),
    'wd-eva02-large-tagger-v3': WaifuDiffusionInterrogator(
        'WD EVA02-Large Tagger v3',
        model_path=wd14_model_path + '/model.onnx',
        tags_path=wd14_model_path + "/selected_tags.csv"
    ),
    'z3d-e621-convnext-toynya': WaifuDiffusionInterrogator(
        'Z3D-E621-Convnext(toynya)',
        repo_id='toynya/Z3D-E621-Convnext',
        tags_path='tags-selected.csv'
    ),
    'z3d-e621-convnext-silveroxides': WaifuDiffusionInterrogator(
        'Z3D-E621-Convnext(silveroxides)',
        repo_id='silveroxides/Z3D-E621-Convnext',
        tags_path='tags-selected.csv'
    ),
    'mld-caformer.dec-5-97527': MLDanbooruInterrogator(
        'ML-Danbooru Caformer dec-5-97527',
        repo_id='deepghs/ml-danbooru-onnx',
        model_path='ml_caformer_m36_dec-5-97527.onnx'
    ),
    'mld-tresnetd.6-30000': MLDanbooruInterrogator(
        'ML-Danbooru TResNet-D 6-30000',
        repo_id='deepghs/ml-danbooru-onnx',
        model_path='TResnet-D-FLq_ema_6-30000.onnx'
    ),
}
