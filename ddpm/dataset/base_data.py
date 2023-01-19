import time
from random import randint, choice, random

import PIL

from torch.utils.data import Dataset
from torchvision import transforms as T
from ..utils.glide_util import get_tokens_and_mask, get_uncond_tokens_mask
from ..utils.train_util import pil_image_to_norm_tensor



def random_resized_crop(image, shape, resize_ratio=1.0):
    """
    Randomly resize and crop an image to a given size.

    Args:
        image (PIL.Image): The image to be resized and cropped.
        shape (tuple): The desired output shape.
        resize_ratio (float): The ratio to resize the image.
    """
    image_transform = T.RandomResizedCrop(shape, scale=(resize_ratio, 1.0), ratio=(1.0, 1.0))
    return image_transform(image)


class TextImageDataset(Dataset):
    def __init__(
        self,
        parsed_data,
        side_x=64,
        side_y=64,
        resize_ratio=0.75,
        shuffle=False,
        tokenizer=None,
        text_ctx_len=128,
        uncond_p=0.0,
        use_captions=False,
        enable_glide_upsample=False,
        upscale_factor=4,
        use_flip=False,
        is_train=False,
        cfg={},
        data_cfg={},
    ):
        """
        :param parsed_data: a Dict of 'image': [], 'text': [], 'meta': [], 'img_func': 
        :param side_x: _description_, defaults to 64
        :param side_y: _description_, defaults to 64
        :param resize_ratio: _description_, defaults to 0.75
        :param shuffle: _description_, defaults to False
        :param tokenizer: _description_, defaults to None
        :param text_ctx_len: _description_, defaults to 128
        :param uncond_p: _description_, defaults to 0.0
        :param use_captions: _description_, defaults to False
        :param enable_glide_upsample: _description_, defaults to False
        :param upscale_factor: _description_, defaults to 4
        """
        super().__init__()
        self.parsed_data = parsed_data
        self.image_files = parsed_data['image']
        if use_captions:
            self.text_files = parsed_data['text']
        else:
            self.text_files = None
            print(f"NOT using text files. Restart with --use_captions to enable...")
            time.sleep(3)

        self.resize_ratio = resize_ratio
        self.text_ctx_len = text_ctx_len

        self.shuffle = shuffle
        self.side_x = side_x
        self.side_y = side_y
        self.tokenizer = tokenizer
        self.uncond_p = uncond_p
        self.enable_upsample = enable_glide_upsample
        self.upscale_factor = upscale_factor

    def __len__(self):
        return len(self.image_files)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def get_caption(self, ind):
        descriptions = self.text_files[ind]
        try:
            if isinstance(descriptions, list):
                description = choice(descriptions).strip()
            elif isinstance(descriptions, str):
                description = descriptions.strip()
            else:
                raise TypeError(type(descriptions))
            return get_tokens_and_mask(tokenizer=self.tokenizer, prompt=description)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load text for files {self.image_files[ind]}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

    def __getitem__(self, ind):
        image_file = self.image_files[ind]
        if self.text_files is None or random() < self.uncond_p:
            tokens, mask = get_uncond_tokens_mask(self.tokenizer)
        else:
            tokens, mask = self.get_caption(ind)

        try:
            if self.parsed_data.get('img_func', None) is not None:
                # return a tensor in scale [-1, 1], (C, H, W)
                base_tensor = self.parsed_data['img_func'](image_file, ind, self.parsed_data['meta'])
            else:
                original_pil_image = PIL.Image.open(image_file).convert("RGB")
                base_tensor = pil_image_to_norm_tensor(original_pil_image)
        except (OSError, ValueError) as e:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
        
        base_tensor = random_resized_crop(base_tensor, (self.side_x, self.side_y), resize_ratio=self.resize_ratio)
        # base_tensor = pil_image_to_norm_tensor(base_pil_image)
        # return th.tensor(tokens), th.tensor(mask, dtype=th.bool), base_tensor
        return {
            'token': tokens, 
            'token_mask': mask, 
            'image': base_tensor,
            }