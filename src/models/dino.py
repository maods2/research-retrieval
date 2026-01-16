import timm
import torch
import torch.nn as nn

from transformers import AutoImageProcessor, AutoModel

class DINOWrapper(nn.Module):
    def __init__(self, dino_version: int, model_name: str, pretrained=True) -> None:
        super(DINOWrapper, self).__init__()

        repo_name = "facebookresearch/dino"
        if dino_version > 1:
            repo_name += f"v{dino_version}"

        # # Load pretrained DINO model from timm
        # self.backbone = timm.create_model(
        #     repo_name, pretrained=pretrained, num_classes=0
        # )  # num_classes=0 returns features

        self.processor = None
        if dino_version == 3: # TODO: Maybe make all dinos come from same source
            self.processor = AutoImageProcessor.from_pretrained(
                f"facebook/{model_name}-pretrain-lvd1689m"
            )
            self.backbone = AutoModel.from_pretrained(
                f"facebook/{model_name}-pretrain-lvd1689m",
            )
        elif dino_version == 2:
            self.processor = AutoImageProcessor.from_pretrained(
                f"facebook/{model_name}-imagenet1k-1-layer"
            )
            self.backbone = AutoModel.from_pretrained(
                f"facebook/{model_name}-imagenet1k-1-layer",
            )

        else:
            print(repo_name)
            self.backbone = timm.create_model(model_name=model_name, pretrained=True)

    def forward(self, x):
        if self.processor is not None:
            x = self.processor(x, return_tensors="pt")
            return self.backbone(**x)  # Already returns flattened embeddings
        return self.backbone(x)

class DINO(DINOWrapper):
    def __init__(
        self, model_name='vit_small_patch16_224_dino', pretrained=True
    ):
        """
        'vit_small_patch16_224_dino' → 384 dims
        'vit_base_patch16_224_dino' → 768 dims
        'vit_base_patch8_224_dino'
        """
        super(DINO, self).__init__(
            dino_version=1,
            model_name=model_name, 
            pretrained=pretrained
        )

class DINOv2(DINOWrapper):
    def __init__(self, model_name='dinov2-base', pretrained=True):
        """
        dinov2-giant: [1, 1024]
        dinov2-large: [1, 1024]
        dinov2-base: [1, 768]
        dinov2-small: [1, 384]
        """
        super(DINOv2, self).__init__(
            dino_version=2,
            model_name=model_name, 
            pretrained=pretrained
        )

class DINOv3(DINOWrapper):
    def __init__(self, model_name='dinov3_vitb16', pretrained=True):
        """
        dinov3_vits16: 
        dinov3_vits16plus: 
        dinov3_vitb16: 
        dinov3_vitl16:
        dinov3_vith16plus:
        """
        super(DINOv3, self).__init__(
            dino_version=3,
            model_name=model_name, 
            pretrained=pretrained
        )


if __name__ == '__main__':
    model = DINO()
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)
    # print(output)
    model = DINOv2()
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)
    
    model = DINOv3()
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)
