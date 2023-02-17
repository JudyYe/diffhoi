from tqdm import tqdm
import argparse
import pickle
import imageio
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
from glide_text2im.gaussian_diffusion import get_named_beta_schedule, get_beta_schedule
from glide_text2im.respace import SpacedDiffusion, space_timesteps


from pytorch3d.renderer import ImplicitRenderer, GridRaysampler
num_steps = 100
vis_dir = '/home/yufeiy2/scratch/result/vis/'


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    """LDM"""
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        try:
            betas = get_beta_schedule(schedule, beta_start=linear_start, beta_end=linear_end, num_diffusion_timesteps=n_timestep)
        except NotImplementedError:
            betas = get_named_beta_schedule(schedule, n_timestep)
            betas = torch.FloatTensor(betas)
            pass
    return betas.numpy()

def get_diffuser(noise_schedule, steps, linear_start=0.0015, linear_end=0.0195, timestep_respacing=""):
    betas = make_beta_schedule(noise_schedule, steps, linear_start, linear_end)
    if not timestep_respacing:
        timestep_respacing = [steps]
    diffuser = SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
    )
    return diffuser

def main():
    steps = num_steps
    timestep_respacing = ""
    candidates = ['linear', 'cosine', 'sqrt_linear', 'sqrt', 'squaredcos_cap_v2']
    linear_start = 0.0015
    linear_end = 0.0195    
    plt.close()
    f, axes = plt.subplots(1, 2)
    ts = np.linspace(0, 1, steps)
    for noise_schedule in candidates:
        betas = make_beta_schedule(noise_schedule, steps, linear_start, linear_end)
        if not timestep_respacing:
            timestep_respacing = [steps]
        diffuser = SpacedDiffusion(
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
        )
        axes[0].plot(ts, diffuser.alphas_cumprod)
        axes[1].plot(ts, diffuser.betas)
        # diffuser.alphas_cumprod
        # diffuser.betas
    axes[0].set_title('alpha_bar')
    axes[1].set_title('beta')
    axes[0].legend(candidates)
    axes[1].legend(candidates)
    plt.savefig(osp.join(vis_dir, 'schedule.png'))


def forward_process():
    N = 10
    save_dir = osp.join(vis_dir, 'schedule')
    os.makedirs(save_dir, exist_ok=True)
    steps = num_steps

    ts = np.linspace(0, 1, steps)
    noise = args.noise
    mode = args.mode
    
    legend_list, kl_list = [], []

    mask = read_input(args.inp, 'normal')
    np.tile(mask, [N, 1, 1, 1])

    
    mode = 'normal'
    diffuser = get_diffuser(noise, steps,)
    mask = mask.reshape([-1])
    legend_list.append('normal ' + noise + '%f %f' % (1.5e-3, 1.95e-2), )
    kl_list.append(forward_w_noise(mask, mode, noise, diffuser, steps, save_dir))

    # se_list = [(1e-4, 2e-2), (1.5e-3, 1.95e-2), (1.5e-3, 1.95e-1), (1.5e-2, 1.95e-1)]
    se_list = [(1.5e-3, 1.95e-2), ]
    for se in se_list:
        mode = args.mode
        # mode = 'mask'
        start, end = se
        diffuser = get_diffuser(noise, steps, start, end,)
        mask = read_input(args.inp, mode)
        np.tile(mask, [N, 1, 1, 1])
        mask = mask.reshape([-1])
        kl_list.append(forward_w_noise(mask, mode, noise, diffuser, steps, save_dir))
        legend_list.append('%s %f %f' % (mode, start, end))
    plt.close()
    for kl in kl_list:
        plt.plot(ts, np.array(kl))
    plt.legend(legend_list)
    plt.title(f'{noise} ')
    plt.savefig(osp.join(save_dir, f'{args.mode}_{args.noise}kl.png'))


def forward_w_noise(mask, mode, noise, diffuser, steps, save_dir):
    image_list = []
    eps = np.random.randn(*mask.shape)
    gaussian_counts, gaussian_bins = np.histogram(eps, np.linspace(-2, 2,), density=True)
    kl_list = []
    for t in tqdm(range(steps)):
        plt.close()
        x0 = mask.reshape([-1])
        eps = np.random.randn(*x0.shape)
        x_t = diffuser.sqrt_alphas_cumprod[t] * x0 + diffuser.sqrt_one_minus_alphas_cumprod[t] * eps
        counts, bins = np.histogram(x_t, np.linspace(-2, 2), density=True)
        ax = plt.gca()
        ax.set_ylim([0, 2])

        kl = kl_div(counts, gaussian_counts, bins[1] - bins[0])
        kl_list.append(kl)
        plt.title(f't={t:3d}, kl={kl:2.4f}')
        plt.hist(bins[:-1], bins, weights=counts)
        plt.savefig(osp.join(save_dir, '%03d.png' % t))

        image_list.append(imageio.imread(osp.join(save_dir, '%03d.png' % t)))
    imageio.mimsave(osp.join(save_dir, f'{noise}_{mode}.gif'), image_list)
    return kl_list

def kl_div(hist_a, hist_b, dx=4/50):
    # sum px log (px / qx)
    hist_a = hist_a * dx
    hist_b = hist_b * dx
    kl = np.sum(hist_a * np.log((hist_a+1e-7) / hist_b))
    return kl


def read_input(inp_file, mode):
    obj = pickle.load(open(inp_file, 'rb'))
    print(obj['image'].shape)
    if mode == 'mask':
        img = obj['image'][:, 0:1]
        # img = img * 0.5 + 0.5
        if args.scale:
            img *= 0.5
    elif mode == 'normal':
        img = obj['image'][:, 1:4]
    elif mode == 'depth':
        img = obj['image'][:, 4:5]
        if args.scale:
            img[img==1] = 0
    elif mode == 'all':
        img = obj['image']
        if args.scale:
            img[:, 0:1] *= 0.5
            depth = img[:, 4:5]
            depth[depth==1] = 0
            img[:, 4:5] = depth

    img= img.cpu().detach().numpy()
    return img



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str, default="/home/yufeiy2/scratch/result/vis_ddpm/input/handuv.pkl", help='output ply file name')
    # parser.add_argument("--out", type=str, default="/home/yufeiy2/scratch/result/vis_ddpm/", help='output ply file name')
    parser.add_argument("--scale", action='store_true')
    parser.add_argument("--mode", type=str, default="normal", help='output ply file name')
    parser.add_argument("--noise", type=str, default="linear", help='output ply file name')
    args, unknown = parser.parse_known_args()

    return args

if __name__ == '__main__':
    # main()
    args = parse_args()
    forward_process()