import torchvision.models as models
from torch import nn    
import torch
from transformers import DPTModel, PvtV2ForImageClassification, ViTHybridModel, CvtModel, LevitConfig, LevitModel, AutoImageProcessor, AutoModel, AutoFeatureExtractor, SwinForImageClassification, MobileViTFeatureExtractor, MobileViTForImageClassification, ViTImageProcessor, ViTForImageClassification, AutoFeatureExtractor, DeiTForImageClassificationWithTeacher, BeitImageProcessor, BeitForImageClassification, SegformerModel, SegformerForImageClassification, DetrFeatureExtractor, DetrModel
from sentence_transformers import SentenceTransformer
#!UPDATE!# 
teachers_dict_torchvision = {
    "squeezenet": models.squeezenet1_0(pretrained = True),#
    "shufflenet": models.shufflenet_v2_x1_0(pretrained = True),#
    "mobilenet": models.mobilenet_v2(pretrained = True),#
    "mnasnet": models.mnasnet1_0(pretrained = True),
    "googlenet": models.googlenet(pretrained = True),#
    "resnet18": models.resnet18(pretrained = True),#
    "resnext50_32x4d": models.resnext50_32x4d(pretrained = True),
    "densenet": models.densenet161(pretrained = True),#
    "wide_resnet50_2": models.wide_resnet50_2(pretrained = True),
    }
teachers_dict_vit_all = {
    "SegFormer": SegformerForImageClassification.from_pretrained("nvidia/mit-b5"),
    "Swin": SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224", return_dict=False),
    "ViT": ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', return_dict=False),
    "DINOv2": AutoModel.from_pretrained('facebook/dinov2-base', return_dict=False),
    "BEiT": BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224', return_dict=False),
    "PVTv2": PvtV2ForImageClassification.from_pretrained("OpenGVLab/pvt_v2_b0", return_dict=False),
    "DETR": DetrModel.from_pretrained("facebook/detr-resnet-101"),
    "DINOv2_": AutoModel.from_pretrained('facebook/dinov2-base', return_dict=False),
    "PVTv2_": PvtV2ForImageClassification.from_pretrained("OpenGVLab/pvt_v2_b0", return_dict=False),
}
teachers_size = {
    "SegFormer": 512,
    "resnet18": 512,
    "squeezenet": 512,
    "densenet": 2208,
    "googlenet": 1024,
    "shufflenet": 1024,
    "mobilenet": 1280,
    "resnext50_32x4d": 2048,
    "wide_resnet50_2": 2048,
    "mnasnet": 1280,
    "DINOv2": 768,
    "DINOv2_": 768,
    "ViT": 768,
    "Swin": 1024,
    "BEiT": 768,
    "MobileViT": 640,
    "ConvNeXT": 1024,
    "PVTv2": 256,
    "PVTv2_": 256,
    "DETR": 256,
    "Snowflake/snowflake-arctic-embed-xs": 384,
    "Snowflake/snowflake-arctic-embed-s": 384,
    }

def get_embedder_size(name):
    return teachers_size[name]

    
class EmbedderFromTorchvision:
    def __new__(cls, name, *args, **kwargs):
        model = teachers_dict_torchvision[name]
        model.eval()
        if name in [
            "resnet18",
            "shufflenet",
            "resnext50_32x4d",
            "wide_resnet50_2",
            "googlenet",
        ]:
            model.fc = nn.Identity()
        elif name in ["squeezenet"]:
            model.classifier = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif hasattr(model, "classifier"):
            model.classifier = nn.Identity()
        return model


class EmbedderFromViT:
    def __new__(cls, name, *args, **kwargs):
        model = teachers_dict_vit_all[name]
        if name in ["Swin", "ViT", "BEiT", "SegFormer"]:
            # Remove the classifier layer in Swin
            model.classifier = torch.nn.Identity()
        elif name in ["PVTv2"]:
            # Remove the classifier layer in Swin
            model.classifier = torch.nn.Identity()
        return model

def get_embedder(name, no_float16 = True, flash_attn = False):
    if name in ["Swin", "ViT", "BEiT", "DINOv2", "PVTv2", "DINOv2_", "PVTv2_"]:
        return EmbedderFromViT(name.replace("_", ""))
    elif name in ["Snowflake/snowflake-arctic-embed-s"]:
        try:
            model = SentenceTransformer(
                name,
                model_kwargs={
                    "torch_dtype": torch.float16 if not no_float16 else torch.float32,
                    "attn_implementation": (
                        "flash_attention_2" if flash_attn else None
                    ),
                },
                device="cuda",
                trust_remote_code=True,
            )
        except Exception as e:
            model = SentenceTransformer(
                name,
                model_kwargs={
                    "torch_dtype": torch.float16 if not no_float16 else torch.float32,
                },
                device="cuda",
                trust_remote_code=True,
            )

        if model.tokenizer.eos_token is not None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
        return model
    return EmbedderFromTorchvision(name)