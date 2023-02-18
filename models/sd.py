# modified from https://github.com/ashawkey/stable-dreamfusion/blob/main/nerf/sd.py
from tqdm import tqdm
import pickle
import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from ddpm2d.utils.train_util import load_from_checkpoint
from ddpm2d.models.glide_base import BaseModule
from jutils import model_utils, image_utils
from glide_text2im.model_creation import create_gaussian_diffusion
from glide_text2im.gaussian_diffusion import _extract_into_tensor
from glide_text2im.respace import _WrappedModel

class SDLoss:
    def __init__(
        self, 
        ckpt_path, 
        cfg={}, 
        min_step=0.02, 
        max_step=0.98, 
        prediction_respacing=100, 
        guidance_scale=4, 
        prompt='a semantic segmentation of a hand grasping an object', **kwargs,
    ) -> None:
        super().__init__()
        self._warp = None
        self.min_step = min_step
        self.max_step = max_step
        self.num_step = prediction_respacing
        self.guidance_scale = guidance_scale
        self.ckpt_path = ckpt_path
        self.model: BaseModule = None
        self.in_channles = 0
        self.alphas_bar = None #  self.scheduler.alphas_cumprod.to(self.device) # for convenience
        self.cfg = cfg
        # TODO: change constant prompt
        self.const_str = prompt
        self.reso = 64
        self.cond = False

    def to(self, device):
        self.model.to(device)

    def init_model(self, device='cuda'):
        self.model = load_from_checkpoint(self.ckpt_path)
        self.model.eval()
        model_utils.freeze(self.model)
        self.unet = self.model.glide_model
        self.options = self.model.glide_options
        print(self.options['noise_schedule'])
        self.diffusion = create_gaussian_diffusion(
            steps=self.options["diffusion_steps"],
            noise_schedule=self.options["noise_schedule"],
            timestep_respacing=str(self.num_step),
        )        
        self.min_step = int(self.min_step * self.num_step)
        self.max_step = int(self.max_step * self.num_step)
        self.alphas_bar = self.diffusion.alphas_cumprod
        self.in_channles = self.model.template_size[0]
        if self.model.cfg.mode.cond:  # False or None.. 
            self.cond = True
        self.to(device)  # do this since loss is not a nn.Module?
    
    def get_weight(self, t, shape, method):
        if method == 'dream':
            w = 1 - _extract_into_tensor(self.alphas_bar, t, shape) 
        elif method == 'decay':
            w = (1 - _extract_into_tensor(self.alphas_bar, t, shape))**1.5 \
                / _extract_into_tensor(np.sqrt(self.alphas_bar), t, shape)
            # w *= sqrt(1-alpha_bar) / sqrt(alpha_bar)
        elif method == 'bell':
            w = (1 - _extract_into_tensor(self.alphas_bar, t, shape)) ** 0.5 \
                * _extract_into_tensor(np.sqrt(self.alphas_bar), t, shape)
        elif method == 'idty':
            w = 1. + torch.zeros(shape)
        return w

    def apply_sd(self, image, weight=1,
            w_mask=1, w_normal=1, w_depth=1, w_spatial=False, 
            t=None, noise=None, cond_image=None, w_schdl='dream', 
            debug=False):
        latents = image
        device = latents.device
        batch_size = len(latents)
        guidance_scale = self.guidance_scale
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if t is None:
            t = torch.randint(self.min_step, self.max_step + 1, [batch_size], dtype=torch.long, device=device)
        if not torch.is_tensor(t):
            t = torch.tensor([t] * batch_size, dtype=torch.long, device=device,)
        if noise is None:
            noise = torch.randn_like(image)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            latents_noisy = self.get_noisy_image(latents, t, noise)
            # pred noise
            model_fn = self.get_cfg_model_fn(self.unet, guidance_scale)
            noise_pred = self.get_pred_noise(model_fn, latents_noisy, t, cond_image)

        # w(t), sigma_t^2
        w = self.get_weight(t, noise_pred.shape, w_schdl)
        w = w.to(device)
        # w = 1 - _extract_into_tensor(self.alphas_bar, t, noise_pred.shape) 
        # TODO: 
        # w *= sqrt(1-alpha_bar) / sqrt(alpha_bar)
        
        grad = weight * w * (noise_pred - noise)
        grad = self.model.distribute_weight(grad, w_mask, w_normal, w_depth)  # interpolate your own image
        grad = torch.nan_to_num(grad)
        # latents.retain_grad()  # just for debug
        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        if debug:
            return grad, {'noisy': latents_noisy, }
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
        """return I_t"""
        if noise is None:
            noise = torch.randn_like(image)
        if not torch.is_tensor(t):
            t = torch.tensor([t] * len(image), dtype=torch.long, device=image.device)
        noisy_image = self.diffusion.q_sample(image, t, noise)
        return noisy_image

    def vis_single_step(self, noisy, t, guidance_scale=None, noise=None, cond_image=None):
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        N = len(noisy)
        device = noisy.device
        t = torch.tensor([t] * len(noisy), dtype=torch.long, device=device)
        model_fn = self.get_cfg_model_fn(self.unet, guidance_scale)
        model_fn = self.diffusion._wrap_model(model_fn)
        noise_pred = self.get_pred_noise(model_fn, noisy, t, cond_image)
        # noise_pred = noise
        start_pred = self.diffusion.eps_to_pred_xstart(noisy, noise_pred, t)
        return start_pred, {'noise': noise_pred}
    
    def get_cfg_model_fn(self, glide_model, guidance_scale):
        """

        :param glide_model: _description_
        :param guidance_scale: _description_
        :return: a function that takes in x_t in shape of (N*2, C*2, H, W)
        but only use the first N batch of x_t, and return 
        (2N, 2C, H, W),  where the :N, ... == N:2N, ...
        """
        th = torch
        def cfg_model_fn(x_t, ts, **kwargs):
            # with classifier-free guidance
            _3 = x_t.shape[1] // 2
            half = x_t[: len(x_t) // 2]
            combined = th.cat([half, half], dim=0)
            model_out = glide_model(combined, ts, **kwargs)

            eps, rest = model_out[:, :_3], model_out[:, _3:]
            cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)  # (N, C, H, W)
            half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
            eps = th.cat([half_eps, half_eps], dim=0)  # 2*N, C, H, W
            return th.cat([eps, rest], dim=1)  # 2N, 2C, H, W
        if self._warp is None:
            self._warp = self.diffusion._wrap_model(cfg_model_fn)
        return self._warp # self.diffusion._wrap_model(cfg_model_fn)

    def vis_multi_step(self, noisy, t, guidance_scale=None, cond_image=None, loop='plms'):
        N = len(noisy)
        device = noisy.device
        t = torch.tensor([t] * len(noisy), dtype=torch.long, device=device)
        # sample for a bunch of steps
        ## something about xxx_loop_progressive

        # Pack the tokens together into model kwargs.
        model_kwargs = self.get_model_kwargs(device, N, cond_image)
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        noisy_double = torch.cat([noisy, noisy], 0)
        model_fn = self.get_cfg_model_fn(self.unet, guidance_scale) # so we use CFG for the base model.
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

    def get_model_kwargs(self, device, batch_size, cond_image):
        uncond_image = self.model.cfg.get('uncond_image', False)
        tokens = self.unet.tokenizer.encode(self.const_str)
        # TODO change to constant string~~
        tokenizer = self.unet.tokenizer
        tokens = tokenizer.encode(self.const_str)
        tokens, mask = tokenizer.padded_tokens_and_mask(tokens, self.options["text_ctx"])

        # tokens, mask = self.unet.tokenizer.padded_tokens_and_mask( [self.const_str], self.options["text_ctx"])
        uncond_tokens, uncond_mask = self.unet.tokenizer.padded_tokens_and_mask( [], self.options["text_ctx"])
        model_kwargs = dict(
            tokens=torch.tensor([tokens] * batch_size + [uncond_tokens] * batch_size, device=device),
            mask=torch.tensor([mask] * batch_size + [uncond_mask] * batch_size,
                dtype=torch.bool,
                device=device,
            )
        )
        if cond_image is not None:
            if uncond_image:
                model_kwargs['cond_image'] = torch.cat([
                    torch.zeros_like(cond_image), cond_image,
                ], 0)
            else:
                model_kwargs['cond_image'] = cond_image.repeat(2, 1, 1, 1)
        return model_kwargs

    def get_pred_noise(self, model_fn, latents_noisy, t, cond_image):
        """
        inside the method, it 
        :param model_fn: CFG func! Wrapped! function
        :param latents_noisy:in shape of (N, C, H, W)
        :param t: in shape of (N, )
        :return: (N, C, H, W)
        """
        # with CFG
        if not isinstance(model_fn, _WrappedModel):
            print('##### Should not appear, sd.py:L329')
            model_fn = self.diffusion._wrap_model(model_fn)
        batch_size = len(latents_noisy)
        device = latents_noisy.device

        with torch.no_grad():
            latent_model_input = torch.cat([latents_noisy] * 2)
            # apply CF-guidance
            model_kwargs = self.get_model_kwargs(device, batch_size, cond_image)
            # from # GaussinDiffusion:L633 get_eps()
            tt = torch.cat([t, t], 0)
            model_output = model_fn(latent_model_input, tt, **model_kwargs)

            # model_output = self.unet(latent_model_input, tt, **model_kwargs)
            if isinstance(model_output, tuple):
                model_output, _ = model_output
            noise_pred = eps = model_output[:, :model_output.shape[1] // 2]

        # perform guidance (high scale from paper!)
        # noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        # noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)
        return noise_pred[:len(noise_pred)//2]


def test_sd():
    N = 2
    device = 'cuda'
    image = nn.Parameter(torch.randn([N, 3, 64, 64]).to(device).requires_grad_(True))
    # image.retain_grad()
    sd = SDLoss('/home/yufeiy2/scratch/result/vhoi/ddpm/glide_SM2/checkpoints/last.ckpt')
    sd.init_model(device)
    sd.apply_sd(image)
    print(image.grad.shape)  # 1.46G, 0.547s


def vis_alpha():
    from tqdm import tqdm
    device = 'cpu'
    num = 100
    vis_dir = '/home/yufeiy2/scratch/result/vis/'
    sd = SDLoss('/home/yufeiy2/scratch/result/vhoi/hand_ddpm_geom/train_seg_CondGeomGlide/checkpoints/last.ckpt')
    sd.init_model(device)

    ms = ['dream', 'decay']
    for method in ms:
        ts = list(range(sd.min_step, sd.max_step))
        ws = []
        for t in tqdm(ts):
            w = sd.get_weight(torch.LongTensor([t]), [1, ], method)
            print(w.shape)
            ws.append(w[0].cpu().detach().numpy())
        ws = np.array(ws)
        xs = np.array(ts)/sd.num_step
        plt.plot(xs, ws)
    plt.plot(xs, sd.alphas_bar[ts])
    plt.plot(xs, 1-sd.alphas_bar[ts])
    plt.plot(xs, np.sqrt((1-sd.alphas_bar[ts]) * sd.alphas_bar[ts]))
    plt.legend(ms+['alpha_bar', '1 - alpha_bar', 'alpha*(1-alpha)'])
    plt.savefig(osp.join(vis_dir, 'schedule.png'))

    return 


def get_wgrad(sd: SDLoss, ts, image, method):
    ws = []
    for t in ts:
        print(t)
        # w = sd.get_weight(torch.LongTensor([t]), [1, ], method)
        grad, extra = sd.apply_sd(**image, t=t, w_schdl=method, debug=True)
        grad_norm = grad.reshape(len(image['image']), -1).norm(2, dim=-1).mean()
        print(grad_norm)
        ws.append(grad_norm.detach().cpu().numpy())

        # noisy = sd.get_noisy_image(image['image'], t)
        # noisy = extra['noisy']
        # start_pred, _ = sd.vis_single_step(noisy, t, cond_image = image['cond_image'])
    #     # out = sd.model.decode_samples(start_pred)
    #     for k, v in out.items():
    #         image_utils.save_images(v[0:4], osp.join(save_dir, f'{k}_{t}'))
    # out = sd.model.decode_samples(image['image'])
    # for k, v in out.items():
    #     image_utils.save_images(v[0:4], osp.join(save_dir, f'{k}_inp'))
    # out = sd.model.decode_samples(image['cond_image'])
    # for k, v in out.items():
    #     image_utils.save_images(v[0:4], osp.join(save_dir, f'{k}_cond'))
    return ws


def read_input(inp_file, mode, scale=False):
    obj = pickle.load(open(inp_file, 'rb'))
    print(obj['image'].shape, obj['cond_image'].shape)

    def get_inp(image):
        if mode == 'mask':
            img = image[:, 0:1]
            if scale:
                img *= 0.5
        elif mode == 'normal':
            img = image[:, 1:4]
        elif mode == 'depth':
            img = image[:, 4:5]
            if scale:
                img[img==1] = 0
        elif mode == 'all':
            img = image
            if scale:
                img[:, 0:1] *= 0.5
                depth = img[:, 4:5]
                depth[depth==1] = 0
                img[:, 4:5] = depth
        return img
    out = {}
    out['image'] = get_inp(obj['image'])
    out['cond_image'] = get_inp(obj['cond_image'])
    return out


def vis_wgrad():
    import sys
    device = 'cuda:0'
    torch.manual_seed(123)
    N = 40
    ckpt = sys.argv[1] # 'ddpm_novel_sunday/hoi4d_CondGeomGlide_1'
    vis_dir = '/home/yufeiy2/scratch/result/vis/'
    sd = SDLoss(f'/home/yufeiy2/scratch/result/vhoi/{ckpt}/checkpoints/last.ckpt')
    sd.init_model(device)
    ms = ['idty', 'dream', 'decay', 'bell']
    # ms = ['idty']
    image = read_input('/home/yufeiy2/scratch/result/vis_ddpm/input/handuv.pkl', 'all')
    image = model_utils.to_cuda(image, device)
    ts = list(range(sd.min_step, sd.max_step, ))
    # ts = [1, 2, 95, 98,]
    xs = np.array(ts)/sd.num_step

    plt.subplot(1, 2, 2)
    # plt.close()
    for method in ms:
        ws_list = []
        for t in ts:
            w = sd.get_weight(torch.tensor([t]).long(), [1, ], method)
            ws_list.append(w[0].detach().cpu().numpy())
        ws = np.array(ws_list)
        plt.plot(xs, ws)

    plt.subplot(1, 2, 1)
    for k,v in image.items():
        image[k] = v.repeat(N, 1, 1, 1)
    for method in ms:
        ws = get_wgrad(sd, ts, image, method)
        ws = np.array(ws)
        plt.plot(xs, ws)

    plt.legend(ms)
    # plt.savefig(osp.join(vis_dir, 'wgrad.png'))

    plt.legend(ms)
    save_index = ckpt.replace('/', '_')
    plt.savefig(osp.join(vis_dir, f'wgrad_alpa_{save_index}.png'))
    return


if __name__ == '__main__':
    # test_sd()
    # vis_alpha()
    save_dir = '/home/yufeiy2/scratch/result/vis/'

    vis_wgrad()