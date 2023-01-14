# https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py
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

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
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