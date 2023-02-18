## glide_util.py
# Utilities for tokenizing, padding, and batching data and sampling from GLIDE.
from copy import deepcopy
import os
from typing import Tuple

import PIL
import numpy as np
import torch as th
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    # create_gaussian_diffusion,
    # create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
    get_named_beta_schedule,
)
from glide_text2im.respace import SpacedDiffusion, space_timesteps
from glide_text2im.tokenizer.bpe import get_encoder, Encoder
from glide_text2im.text2im_model import Text2ImUNet
from ..models.network import ImageText2ImUNet
from jutils import model_utils


MODEL_TYPES = ["base", "upsample", "base-inpaint", "upsample-inpaint"]

def init_best_efforts(new_shape, param, split_in_half, dim):
    K = new_shape[dim] // (param.shape[dim]) 
    m = new_shape[dim] % (param.shape[dim])
    dims = [1,] * param.ndim
    if split_in_half:
        param1, param2 = th.chunk(param, 2, dim=dim)
        half_dim = list(new_shape); half_dim[dim] //= 2
        first = init_best_efforts(half_dim, param1, False, dim)
        second = init_best_efforts(half_dim, param2, False, dim)
        new_param = th.cat([first, second], 0)
    else:
        to_dims1 = deepcopy(dims); to_dims1[dim] = K
        to_dims2 = deepcopy(dims); to_dims2[dim] = m
        new_param = param.repeat(*to_dims1) 
        if m > 0:
            param2 = th.mean(param, dim, True).repeat(*to_dims2)
            new_param = th.cat([new_param, param2], dim)
    return new_param

def create_model_and_diffusion(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    text_ctx,
    xf_width,
    xf_layers,
    xf_heads,
    xf_final_ln,
    xf_padding,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    cache_text_emb,
    inpaint,
    super_res,
    in_channels=3,
    cond_channels=0,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        xf_padding=xf_padding,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        cache_text_emb=cache_text_emb,
        inpaint=inpaint,
        super_res=super_res,
        in_channels=in_channels,
        cond_channels=cond_channels,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        noise_schedule=noise_schedule,
        timestep_respacing=timestep_respacing,
    )

    # diffusion = create_gaussian_diffusion(
    #     steps=diffusion_steps,
    #     noise_schedule=noise_schedule,
    #     timestep_respacing=timestep_respacing,
    # )
    return model, diffusion


def create_gaussian_diffusion(
    steps,
    noise_schedule,
    timestep_respacing,
):
    betas = make_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
    )


def make_beta_schedule(schedule, n_timestep, linear_start=0.0015, linear_end=0.0195, cosine_s=8e-3):
    """LDM"""
    if schedule == "linear":
        betas = (
                th.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=th.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                th.arange(n_timestep + 1, dtype=th.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = th.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = th.linspace(linear_start, linear_end, n_timestep, dtype=th.float64)
    elif schedule == "sqrt":
        betas = th.linspace(linear_start, linear_end, n_timestep, dtype=th.float64) ** 0.5
    else:
        try:
            betas = get_named_beta_schedule(schedule, n_timestep)
            betas = th.FloatTensor(betas)
        except NotImplementedError:
            pass
    return betas.numpy()


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    text_ctx,
    xf_width,
    xf_layers,
    xf_heads,
    xf_final_ln,
    xf_padding,
    resblock_updown,
    use_fp16,
    cache_text_emb,
    inpaint,
    super_res,
    in_channels,
    cond_channels,
):
    if channel_mult == "":
        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
        assert 2 ** (len(channel_mult) + 2) == image_size

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    if not inpaint:
        Model = Text2ImUNet
        model_kwargs = {}
    else:
        Model = ImageText2ImUNet
        model_kwargs = {'cond_channels': cond_channels}
    return Model(
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        tokenizer=get_encoder(),
        xf_padding=xf_padding,
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=in_channels * 2,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        cache_text_emb=cache_text_emb,
        **model_kwargs,
    )
 

def get_uncond_tokens_mask(tokenizer: Encoder):
    uncond_tokens, uncond_mask = tokenizer.padded_tokens_and_mask([], 128)
    return th.tensor(uncond_tokens), th.tensor(uncond_mask, dtype=th.bool)


def get_tokens_and_mask(
    tokenizer: Encoder, prompt: str = "", context_len: int = 128
) -> Tuple[th.tensor, th.tensor]:
    if len(prompt) == 0:
        return get_uncond_tokens_mask(tokenizer)
    else:
        tokens = tokenizer.encode(prompt)
        tokens, mask = tokenizer.padded_tokens_and_mask(tokens, context_len)
        tokens = th.tensor(tokens)  # + uncond_tokens)
        mask = th.tensor(mask, dtype=th.bool)  # + uncond_mask, dtype=th.bool)
        return tokens, mask


def load_model(
    glide_path: str = "",
    use_fp16: bool = False,
    disable_transformer: bool = False,
    freeze_transformer: bool = False,
    freeze_diffusion: bool = False,
    activation_checkpointing: bool = False,
    model_type: str = "base",
    in_channels=3,
    cond_channels=0,
    beta_schdl=None,
):
    assert model_type in MODEL_TYPES, f"Model must be one of {MODEL_TYPES}. Exiting."
    if model_type in ["base", "base-inpaint"]:
        options = model_and_diffusion_defaults()
    elif model_type in ["upsample", "upsample-inpaint"]:
        options = model_and_diffusion_defaults_upsampler()
    if "inpaint" in model_type:
        options["inpaint"] = True
    
    if beta_schdl is not None:
        options['noise_schedule'] = beta_schdl
    options["use_fp16"] = use_fp16
    if disable_transformer:
        options["xf_width"] = 0
    glide_model, glide_diffusion = create_model_and_diffusion(**options, in_channels=in_channels, cond_channels=cond_channels)
    if activation_checkpointing:
        glide_model.use_checkpoint = True

    glide_model.requires_grad_(True)
    if freeze_transformer:
        glide_model.transformer.requires_grad_(False)
        glide_model.transformer_proj.requires_grad_(False)
        glide_model.token_embedding.requires_grad_(False)
        glide_model.padding_embedding.requires_grad_(False)
        glide_model.positional_embedding.requires_grad_(False)
    if freeze_diffusion:
        glide_model.out.requires_grad_(False)
        glide_model.input_blocks.requires_grad_(False)
        glide_model.middle_block.requires_grad_(False)
        glide_model.output_blocks.requires_grad_(False)
    if glide_path and os.path.exists(glide_path):  # user provided checkpoint
        weights = th.load(glide_path, map_location="cpu")
        _, _, mismatch_keys =  model_utils.load_my_state_dict(glide_model, weights)
        if len(mismatch_keys) > 0:
            # try to fit outermost layer?
            own_state = glide_model.state_dict()
            for key in mismatch_keys:
                if key not in ['input_blocks.0.0.weight', 'out.2.weight', 'out.2.bias']:
                    continue
                dim = 1 if 'input' in key else 0
                v = init_best_efforts(
                    own_state[key].shape, weights[key].data, 'out' in key, dim)
                own_state[key].copy_(v)
                print('init w best efforts', key)
    elif glide_path is None:  # use default checkpoint from openai
        pass
    else:
        # glide_model.load_state_dict(
        #     load_checkpoint(model_type, "cpu", cache_dir=os.path.dirname(glide_path))
        # )  # always load to cpu, saves memory
        model_utils.load_my_state_dict(
            glide_model,
            load_checkpoint(model_type, "cpu", cache_dir=os.path.dirname(glide_path)))        
    if use_fp16:
        glide_model.convert_to_fp16()
        print("Converted to fp16, likely gradients will explode")
    return glide_model, glide_diffusion, options

def read_image(path: str, shape: Tuple[int, int]):
    pil_img = PIL.Image.open(path).convert('RGB')
    pil_img = pil_img.resize(shape, resample=PIL.Image.BICUBIC)
    img = np.array(pil_img)
    return th.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1

# Sample from the base model.

@th.inference_mode()
def sample(
    glide_model,
    glide_options,
    side_x=None,
    side_y=None,
    prompt="",
    val_batch=None,
    size=(3, 64, 64),
    batch_size=1,
    guidance_scale=4,
    device="cpu",
    prediction_respacing="100",
    upsample_enabled=False,
    image_to_upsample='',
    upsample_temp=0.997,
    uncond_image=False,
):
    glide_model.del_cache()
    eval_diffusion = create_gaussian_diffusion(
        steps=glide_options["diffusion_steps"],
        noise_schedule=glide_options["noise_schedule"],
        timestep_respacing=prediction_respacing,
    )
    print(glide_options['noise_schedule'])
    # Create the text tokens to feed to the model.
    # tokens = glide_model.tokenizer.encode(prompt)
    # tokens, mask = glide_model.tokenizer.padded_tokens_and_mask(
    #     tokens, glide_options["text_ctx"]
    # )
    val_batch = model_utils.to_cuda(val_batch, device)
    tokens, mask = val_batch['token'], val_batch['token_mask']
    batch_size = len(tokens)

    # Create the classifier-free guidance tokens (empty)
    full_batch_size = batch_size * 2
    uncond_tokens, uncond_mask = glide_model.tokenizer.padded_tokens_and_mask( [], glide_options["text_ctx"])

    # Pack the tokens together into model kwargs.
    model_kwargs = dict(
        tokens=th.cat([
            tokens, th.tensor([uncond_tokens] * batch_size, device=device)
        ], 0),
        mask=th.cat([
            mask, th.tensor(
            [uncond_mask] * batch_size,
            dtype=th.bool,
            device=device,
        )], 0),
        # cond_image=val_batch.get('cond_image', None)
    )
    if 'cond_image' in val_batch:
        if uncond_image:
            model_kwargs['cond_image'] = th.cat([
                th.zeros_like(val_batch['cond_image']), val_batch['cond_image'],
            ], 0)
            print('condition on image!!!')
        else:
            model_kwargs['cond_image'] = val_batch['cond_image'].repeat(2, 1, 1, 1)

    def cfg_model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = glide_model(combined, ts, **kwargs)
        C = model_out.shape[1]
        eps, rest = model_out[:, :C//2], model_out[:, C//2:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        beta = eval_diffusion.betas[
            int(
                ts.flatten()[0].item()
                / glide_options["diffusion_steps"]
                * len(eval_diffusion.betas)
            )
        ]
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        # current_prediction_pil = pred_to_pil(
        #     (x_t - eps * (beta**0.5))[:batch_size]
        # )
        # current_prediction_pil.save("current_prediction.png")
        return th.cat([eps, rest], dim=1)

    model_fn = cfg_model_fn # so we use CFG for the base model.
    if upsample_enabled:
        assert image_to_upsample != '', "You must specify a path to an image to upsample."
        low_res_samples = read_image(image_to_upsample, size=(side_x, side_y))
        model_kwargs['low_res'] = low_res_samples
        noise = th.randn([batch_size, ] + size, device=device) * upsample_temp
        model_kwargs['noise'] = noise
        model_fn = glide_model # just use the base model, no need for CFG.

    samples = eval_diffusion.plms_sample_loop(
        model_fn,
        [full_batch_size,] + size, # 3, side_y, side_x),  # only thing that's changed
        device=device,
        clip_denoised=True,
        progress=True,
        model_kwargs=model_kwargs,
        cond_fn=None,
    )[:batch_size]
    glide_model.del_cache()
    return samples, []