import torch
from glide_text2im.model_creation import Text2ImUNet


class ImageText2ImUNet(Text2ImUNet):
    """
    A text2im model which can perform inpainting.
    """

    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["in_channels"] = kwargs["in_channels"] + kwargs.pop("cond_channels")
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, cond_image=None, **kwargs):
        return super().forward(
            torch.cat([x, cond_image], dim=1),
            timesteps,
            **kwargs,
        )
