"""Inspired by pytorch3d softmax_blending"""
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import NamedTuple, Sequence, Union
from pytorch3d.renderer.blending import BlendParams
import torch


# Example functions for blending the top K colors per pixel using the outputs
# from rasterization.
# NOTE: All blending function should return an RGBA image per batch element


def hard_rgb_blend(
    color_list, zbuf_list, mask_list,
    blend_params: BlendParams
) -> torch.Tensor:
    """
    Naive blending of top K faces to return an RGBA image
      - **RGB** - choose color of the closest point i.e. K=0
      - **A** - 1.0

    Args:
        colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
        fragments: the outputs of rasterization. From this we use
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image. This is used to
              determine the output shape.
        blend_params: BlendParams instance that contains a background_color
        field specifying the color for the background
    Returns:
        RGBA pixel_colors: (N, H, W, 4)
    """
    N, nP, _ = color_list[0].shape
    device = color_list[0].device

    # compose and sort
    zbuf = torch.stack(zbuf_list, -1)    # (N, P, K)
    colors = torch.stack(color_list, 2)  # (N, P, K, 3)
    zbuf, sort_indices_masked = torch.sort(zbuf, dim=-1)  # sort from near to far
    colors = torch.gather(colors, dim=-2, index=torch.stack([sort_indices_masked]*3, dim=-1))

    # Mask for the background.
    eps = 0.1
    is_background = torch.stack(mask_list, -1).sum(-1) <= eps    # (N, P, K)
    
    background_color_ = blend_params.background_color
    if isinstance(background_color_, torch.Tensor):
        background_color = background_color_.to(device)
    else:
        background_color = colors.new_tensor(background_color_)

    # Find out how much background_color needs to be expanded to be used for masked_scatter.
    num_background_pixels = is_background.sum()

    # Set background color.
    pixel_colors = colors[..., 0, :].masked_scatter(
        is_background[..., None],
        background_color[None, :].expand(num_background_pixels, -1),
    )  # (N, H, W, 3)

    # Concat with the alpha channel.
    alpha = (~is_background).type_as(pixel_colors)[..., None]

    return torch.cat([pixel_colors, alpha], dim=-1)  # (N, H, W, 4)


def volumetric_rgb_blend(color_list, zbuf_list, mask_list,
    blend_params: BlendParams,
    znear: Union[float, torch.Tensor] = 1.0,
    zfar: Union[float, torch.Tensor] = 100,
) -> torch.Tensor:
    N, nP, _ = color_list[0].shape
    device = color_list[0].device
    pixel_colors = torch.ones((N, nP, 4), device=device)
    background_ = blend_params.background_color
    if not isinstance(background_, torch.Tensor):
        background = torch.tensor(background_, dtype=torch.float32, device=device)
    else:
        background = background_.to(device)

    # Weight for background color
    eps = 1e-10
    
    # compose and sort
    colors = torch.stack(color_list, 2)  # (N, P, K, 3)
    zbuf = torch.stack(zbuf_list, -1)    # (N, P, K)
    mask = torch.stack(mask_list, -1)    # (N, P, K)

    zbuf, sort_indices_masked = torch.sort(zbuf, dim=-1)  # sort from near to far
    colors = torch.gather(colors, dim=-2, index=torch.stack([sort_indices_masked]*3, dim=-1))
    mask = torch.gather(mask, dim=-1, index=sort_indices_masked) 

    # # Mask for padded pixels.
    # mask = fragments.pix_to_face >= 0

    # # Sigmoid probability map based on the distance of the pixel to the face.
    # prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
    prob_map = mask
    # The cumulative product ensures that alpha will be 0.0 if at least 1
    # face fully covers the pixel as for that face, prob will be 1.0.
    # This results in a multiplication by 0.0 because of the (1.0 - prob)
    # term. Therefore 1.0 - alpha will be 1.0.
    alpha = torch.prod((1.0 - prob_map), dim=-1)    
    
    miss = torch.cumprod(1 - prob_map, dim=-1)
    miss = torch.cat([torch.ones([N, nP, 1], device=device), miss], -1)  # nP+1
    prob_map = torch.cat([prob_map, torch.ones([N, nP, 1], device=device)], -1)  # nP+1
    weights_num = prob_map * miss  # hit probability at [k]

    # Sum: weights * textures + background color
    weighted_colors = (weights_num[..., :-1, None] * colors).sum(dim=-2)
    weighted_background = weights_num[..., -1, None] * background  # (N, 3)
    pixel_colors[..., :3] = (weighted_colors + weighted_background)
    pixel_colors[..., 3] = 1.0 - alpha
    return pixel_colors


def softmax_rgb_blend(
    color_list, zbuf_list, mask_list,
    blend_params: BlendParams,
    znear: Union[float, torch.Tensor] = 1.0,
    zfar: Union[float, torch.Tensor] = 100,
) -> torch.Tensor:
    """
    RGB and alpha channel blending to return an RGBA image based on the method
    proposed in [1]
      - **RGB** - blend the colors based on the 2D distance based probability map and
        relative z distances.
      - **A** - blend based on the 2D distance based probability map.

    Args:
        color_list: list of (N, -1, 3) --> (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
        zbuf_list: list of (N, -1, ) --> (N, H, W, K, ) will become fragment.zbuf
        mask_list: list of (N, -1, ) --> (N, H, W, K, ) will become prob_map
        fragments: namedtuple with outputs of rasterization. We use properties
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image.
            - dists: FloatTensor of shape (N, H, W, K) specifying
              the 2D euclidean distance from the center of each pixel
              to each of the top K overlapping faces.
            - zbuf: FloatTensor of shape (N, H, W, K) specifying
              the interpolated depth from each pixel to to each of the
              top K overlapping faces.
        blend_params: instance of BlendParams dataclass containing properties
            - sigma: float, parameter which controls the width of the sigmoid
              function used to calculate the 2D distance based probability.
              Sigma controls the sharpness of the edges of the shape.
            - gamma: float, parameter which controls the scaling of the
              exponential function used to control the opacity of the color.
            - background_color: (3) element list/tuple/torch.Tensor specifying
              the RGB values for the background color.
        znear: float, near clipping plane in the z direction
        zfar: float, far clipping plane in the z direction

    Returns:
        RGBA pixel_colors: (N, -1, 4)

    [0] Shichen Liu et al, 'Soft Rasterizer: A Differentiable Renderer for
    Image-based 3D Reasoning'
    """

    N, nP, _ = color_list[0].shape
    device = color_list[0].device
    pixel_colors = torch.ones((N, nP, 4), device=device)
    background_ = blend_params.background_color
    if not isinstance(background_, torch.Tensor):
        background = torch.tensor(background_, dtype=torch.float32, device=device)
    else:
        background = background_.to(device)

    # Weight for background color
    eps = 1e-10
    
    # compose and sort
    colors = torch.stack(color_list, 2)  # (N, P, K, 3)
    zbuf = torch.stack(zbuf_list, -1)    # (N, P, K)
    mask = torch.stack(mask_list, -1)    # (N, P, K)

    zbuf, sort_indices_masked = torch.sort(zbuf, dim=-1)  # sort from near to far
    colors = torch.gather(colors, dim=-2, index=torch.stack([sort_indices_masked]*3, dim=-1))
    mask = torch.gather(mask, dim=-1, index=sort_indices_masked) 

    # # Mask for padded pixels.
    # mask = fragments.pix_to_face >= 0

    # # Sigmoid probability map based on the distance of the pixel to the face.
    # prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
    prob_map = mask
    # The cumulative product ensures that alpha will be 0.0 if at least 1
    # face fully covers the pixel as for that face, prob will be 1.0.
    # This results in a multiplication by 0.0 because of the (1.0 - prob)
    # term. Therefore 1.0 - alpha will be 1.0.
    alpha = torch.prod((1.0 - prob_map), dim=-1)

    # Weights for each face. Adjust the exponential by the max z to prevent
    # overflow. zbuf shape (N, H, W, K), find max over K.
    # TODO: there may still be some instability in the exponent calculation.

    # Reshape to be compatible with (N, H, W, K) values in fragments
    if torch.is_tensor(zfar):
        # pyre-fixme[16]
        zfar = zfar[:, None, None]
    if torch.is_tensor(znear):
        znear = znear[:, None, None]

    z_inv = (zfar - zbuf) / (zfar - znear) * mask
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None].clamp(min=eps)
    weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)

    # Also apply exp normalize trick for the background color weight.
    # Clamp to ensure delta is never 0.
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    delta = torch.exp((eps - z_inv_max) / blend_params.gamma).clamp(min=eps)

    # Normalize weights.
    # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
    denom = weights_num.sum(dim=-1)[..., None] + delta

    # Sum: weights * textures + background color
    weighted_colors = (weights_num[..., None] * colors).sum(dim=-2)
    weighted_background = delta * background
    pixel_colors[..., :3] = (weighted_colors + weighted_background) / denom
    pixel_colors[..., 3] = 1.0 - alpha

    return pixel_colors
