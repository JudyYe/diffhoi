import torch
from glide_text2im.model_creation import Text2ImUNet


class ImageText2ImUNet(Text2ImUNet):
    """
    A text2im model which can perform inpainting.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2 + 1
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, cond_image=None, **kwargs):
        return super().forward(
            torch.cat([x, cond_image], dim=1),
            timesteps,
            **kwargs,
        )
