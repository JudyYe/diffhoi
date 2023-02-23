from hydra import main
import torch
import torch.nn as nn


class PixelSampler(nn.Module):
    def __init__(self, cfg, gen_cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.gen_cfg = gen_cfg

    def forward(self, model_input, H, W, N_rays, it):
        """

        :param model_input: _description_
        :param H: _description_
        :param W: _description_
        :param N_rays: _description_
        :param it: _description_
        :returns: select_inds (LongTensor) in shape [N_rays]
        """
        device = model_input['hand_mask'].device
        select_hs = torch.randint(0, H, size=[N_rays]).to(device)
        select_ws = torch.randint(0, W, size=[N_rays]).to(device)
        select_inds = select_hs * W + select_ws
        return select_inds


class ProportionPixelSampler(PixelSampler):
    def __init__(self, cfg, gen_cfg) -> None:
        super().__init__(cfg, gen_cfg)
        self.data_init_factor = cfg.data_init_factor
        self.data_final_factor = cfg.data_final_factor
        # data_init_factor : [0.35, 0.35, 0.3]  # background, hand, object
        # data_final_factor : [0.1, 0.1, 0.80]
    
    def forward(self, model_input, H, W, N_rays, it):
        assert model_input['obj_mask'].shape[0] == 1
        obj_indices = torch.nonzero(model_input['obj_mask'].view(-1)).squeeze(1)
        hand_indices = torch.nonzero(model_input['hand_mask'].view(-1)).squeeze(1)
        background_indices = torch.nonzero(((1-model_input['obj_mask']) * (1-model_input['hand_mask'])).view(-1)).squeeze(1)

        obj_factor = self.data_init_factor[2] + (self.data_final_factor[2]-self.data_init_factor[2]) * (it/self.gen_cfg.training.num_iters)
        hand_factor = self.data_init_factor[1] + (self.data_final_factor[1]-self.data_init_factor[1]) * (it/self.gen_cfg.training.num_iters)

        num_obj = int(obj_factor * N_rays)
        num_hand = int(hand_factor * N_rays)
        num_background = N_rays - num_obj - num_hand
        
        sub_obj_indices = torch.multinomial(torch.ones_like(obj_indices).float(), num_obj, replacement=True)
        obj_indices = obj_indices[sub_obj_indices]
        sub_hand_indices = torch.multinomial(torch.ones_like(hand_indices).float(), num_hand, replacement=True)
        hand_indices = hand_indices[sub_hand_indices]
        sub_background_indices = torch.multinomial(torch.ones_like(background_indices).float(), num_background, replacement=True)
        background_indices = background_indices[sub_background_indices]

        select_inds = torch.cat([obj_indices, hand_indices, background_indices], dim=0)
        return select_inds


class ActivePixelSampler(PixelSampler):
    def __init__(self, cfg, gen_cfg) -> None:
        super().__init__(cfg, gen_cfg)


def get_pixel_sampler(name, sampler_args, args):
    if name == 'naive':
        Model = PixelSampler
    elif name == 'proportion':
        Model = ProportionPixelSampler
    elif name == 'active':
        Model = ActivePixelSampler
    else:
        raise NotImplementedError('Unknown pixel sampler: %s' % name)
    return Model(sampler_args, args)


@main(config_path='../configs', config_name='volsdf_nogt')
def test_pixel_sampler(cfg):
    vis_dir = '/home/yufeiy2/scratch/result/vis/'
    pixel_sampler = get_pixel_sampler(cfg.pixel_sampler.name, cfg.pixel_sampler, cfg)
    model_input = {
        'obj_mask': torch.zeros(1, 1, 256, 256),
        'hand_mask': torch.zeros(1, 1, 256, 256),
    }
    H = W = 256
    model_input['hand_mask'][0, 0, 200:, 200:] = 1
    model_input['obj_mask'][0, 0, :100, :100] = 1

    it = cfg.iter
    inds = pixel_sampler(model_input, H, W, 1000, it)
    print(cfg.pixel_sampler.name, inds.shape)
    canvas = torch.zeros(1, 1, H, W).view(-1)
    canvas = torch.scatter_add(canvas, 0, inds, torch.ones(inds.shape))
    
    from jutils import image_utils
    image_utils.save_images(canvas.view(1, 1, H, W), vis_dir + f'{cfg.pixel_sampler.name}_{it}.png')


if __name__ == '__main__':
    test_pixel_sampler()