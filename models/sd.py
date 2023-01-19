# modified from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py
import time
import torch
import torch.nn as nn
from ddpm.utils.train_util import load_from_checkpoint
from ddpm.models.glide_base import BaseModule
from jutils import model_utils
from glide_text2im.model_creation import create_gaussian_diffusion
from glide_text2im.gaussian_diffusion import _extract_into_tensor


class SDLoss:
    def __init__(
        self, 
        ckpt_path, 
        cfg={}, 
        min_step=0.02, 
        max_step=0.98, 
        prediction_respacing=100, 
        guidance_scale=4, 
        prompt='a semantic segmentation of a hand grasping an object'
    ) -> None:
        super().__init__()
        self.min_step = min_step
        self.max_step = max_step
        self.num_step = prediction_respacing
        self.guidance_scale = guidance_scale
        self.ckpt_path = ckpt_path
        self.model: BaseModule = None
        self.alphas = None #  self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.cfg = cfg
        self.const_str = prompt
        self.reso = 64

    def to(self, device):
        self.model.to(device)

    def init_model(self, device='cuda'):
        self.model = load_from_checkpoint(self.ckpt_path)
        model_utils.freeze(self.model)
        self.unet = self.model.glide_model
        self.options = self.model.glide_options
        self.diffusion = create_gaussian_diffusion(
            steps=self.options["diffusion_steps"],
            noise_schedule=self.options["noise_schedule"],
            timestep_respacing=str(self.num_step),
        )        
        self.min_step = int(self.min_step * self.num_step)
        self.max_step = int(self.max_step * self.num_step)
        self.alphas = self.diffusion.alphas_cumprod
        
        self.to(device)
        
    def apply_sd(self, latents, weight=1):
        device = latents.device
        batch_size = len(latents)
        guidance_scale = self.guidance_scale
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [batch_size], dtype=torch.long, device=device)
        # predict the noise residual with unet, NO grad!
        _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents, device=device)
            latents_noisy = self.diffusion.q_sample(latents, t, noise)
            # latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # apply CF-guidance
            # noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            # from glide_utils
            tokens = self.unet.tokenizer.encode(self.const_str)
            tokens, mask = self.unet.tokenizer.padded_tokens_and_mask( [], self.options["text_ctx"])
            uncond_tokens, uncond_mask = self.unet.tokenizer.padded_tokens_and_mask( [], self.options["text_ctx"])
            model_kwargs = dict(
                tokens=torch.tensor([tokens] * batch_size + [uncond_tokens] * batch_size, device=device),
                mask=torch.tensor([mask] * batch_size + [uncond_mask] * batch_size,
                    dtype=torch.bool,
                    device=device,
                )
            )
            # from # GaussinDiffusion:L633 get_eps()
            tt = torch.cat([t, t], 0)
            model_output = self.unet(latent_model_input, tt, **model_kwargs)
            if isinstance(model_output, tuple):
                model_output, _ = model_output
            noise_pred = eps = model_output[:, :3]

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        # TODO: judy: replace w EDM? 
        w = 1 - _extract_into_tensor(self.alphas, t, noise_pred.shape) 
        # w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = weight * w * (noise_pred - noise)
        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        grad = torch.nan_to_num(grad)
        # print('sd.py: L100, diffusion grad', grad, torch.norm(grad, dim=-1).mean())
        # latents.retain_grad()  # just for debug
        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        latents.backward(gradient=grad, retain_graph=True)    

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings        

    def get_noisy_image(self, image, t, noise=None):
        """return I_t+1"""
        if noise is not None:
            noise = torch.randn([1, 3, self.reso, self.reso]).to(image)
        t = torch.tensor([t] * len(image), dtype=torch.long, device=image.device)
        noisy_image = self.diffusion.q_sample(image, t, noise)
        return noisy_image

    def vis_single_step(self, noisy, t, guidance_scale=None):
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        N = len(noisy)
        device = noisy.device
        t = torch.tensor([t] * len(noisy), dtype=torch.long, device=device)

        noise_pred = self.get_pred_noise(noisy, t, guidance_scale)
        start_pred = self.diffusion.eps_to_pred_xstart(noisy, noise_pred, t)
        return start_pred
        
    def vis_multi_step(self, noisy, t, guidance_scale=None, loop='plms'):
        N = len(noisy)
        device = noisy.device
        t = torch.tensor([t] * len(noisy), dtype=torch.long, device=device)
        # sample for a bunch of steps
        ## something about xxx_loop_progressive

        # Pack the tokens together into model kwargs.
        model_kwargs = self.get_model_kwargs(device, N)
        glide_model = self.unet
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
        th = torch

        def cfg_model_fn(x_t, ts, **kwargs):
            half = x_t[: len(x_t) // 2]
            combined = th.cat([half, half], dim=0)
            model_out = glide_model(combined, ts, **kwargs)
            eps, rest = model_out[:, :3], model_out[:, 3:]
            cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = th.cat([half_eps, half_eps], dim=0)
            return th.cat([eps, rest], dim=1)

        noisy_double = torch.cat([noisy, noisy], 0)
        model_fn = cfg_model_fn # so we use CFG for the base model.
        if loop == 'plms':
            img = self.plms_sample_from_middle(model_fn, noisy_double, t, model_kwargs=model_kwargs)
        elif loop == 'ddim':
            img = self.ddim_sample_from_middle(model_fn, noisy_double, t, model_kwargs=model_kwargs)
        return img[:N]

        
    def ddim_sample_from_middle(
        self,
        model,
        img, t0, 
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        shape = img.shape
        device = img.device
        assert isinstance(shape, (tuple, list))
        indices = list(range(self.diffusion.num_timesteps))[::-1]
        indices = list(range(t0))[::-1]  # internal function ignore t=0 step. 

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.diffusion.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                # yield out
                img = out["sample"]
        return out['sample']

    def plms_sample_from_middle(
        self,
        model,
        img, t0, 
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Use PLMS to sample from the model and yield intermediate samples from
        each timestep of PLMS.

        Same usage as p_sample_loop_progressive().
        """
        shape = img.shape
        device = img.device
        # say T=num_timestpe=100, indices = [98, ..., 1], then noise is Img_99
        indices = list(range(self.diffusion.num_timesteps))[::-1][1:-1] # 
        indices = list(range(t0))[::-1][1:-1]  #??

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        old_eps = []

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                if len(old_eps) < 3:
                    out = self.diffusion.prk_sample(
                        model,
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs,
                    )
                else:
                    out = self.diffusion.plms_sample(
                        model,
                        img,
                        old_eps,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs,
                    )
                    old_eps.pop(0)
                old_eps.append(out["eps"])
                img = out["sample"]
        # use out['sample']
        return out['sample']

    def get_model_kwargs(self, device, batch_size):
        tokens = self.unet.tokenizer.encode(self.const_str)
        tokens, mask = self.unet.tokenizer.padded_tokens_and_mask( [], self.options["text_ctx"])
        uncond_tokens, uncond_mask = self.unet.tokenizer.padded_tokens_and_mask( [], self.options["text_ctx"])
        model_kwargs = dict(
            tokens=torch.tensor([tokens] * batch_size + [uncond_tokens] * batch_size, device=device),
            mask=torch.tensor([mask] * batch_size + [uncond_mask] * batch_size,
                dtype=torch.bool,
                device=device,
            )
        )
        return model_kwargs

    def get_pred_noise(self, latents_noisy, t, guidance_scale):
        # with CFG
        batch_size = len(latents_noisy)
        device = latents_noisy.device

        with torch.no_grad():
            latent_model_input = torch.cat([latents_noisy] * 2)
            # apply CF-guidance
            # noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            model_kwargs = self.get_model_kwargs(device, batch_size)
            # from # GaussinDiffusion:L633 get_eps()
            tt = torch.cat([t, t], 0)
            model_output = self.unet(latent_model_input, tt, **model_kwargs)
            if isinstance(model_output, tuple):
                model_output, _ = model_output
            noise_pred = eps = model_output[:, :3]

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
        return noise_pred

def test_sd():
    N = 2
    device = 'cuda'
    image = nn.Parameter(torch.randn([N, 3, 64, 64]).to(device).requires_grad_(True))
    # image.retain_grad()
    sd = SDLoss('/home/yufeiy2/scratch/result/vhoi/ddpm/glide_SM2/checkpoints/last.ckpt')
    sd.init_model(device)
    sd.apply_sd(image)
    print(image.grad.shape)  # 1.46G, 0.547s


if __name__ == '__main__':
    test_sd()