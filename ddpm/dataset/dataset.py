import importlib
import torch
from typing import Iterable, List
import warnings
from torch.utils.data import ConcatDataset, DataLoader, Dataset, IterableDataset
from .base_data import TextImageDataset

class ConcatDatasetProb(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset], prob_list=None) -> None:
        super().__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)
        self.all_sizes = [len(e) for e in self.datasets]
        if prob_list is None:
            prob_list = [1.  for _ in  self.datasets]
        print('mix up dataset with prob', prob_list)
        prob_list = torch.FloatTensor(prob_list[0:len(datasets)])
        self.unnorm_prob_list = prob_list
        self.max_size = max(self.all_sizes)

    def __len__(self):
        return self.max_size
        # return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if len(self.unnorm_prob_list) > 0:
            dataset_idx = torch.multinomial(self.unnorm_prob_list, 1, )[0].item()
        else:
            dataset_idx = 0
        sample_idx = idx % self.all_sizes[dataset_idx]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


def get_data_parsed(data_cfg):
    """
    :param data_cfg: a dict with keys: target, data_dir, split, .... defined in data/xxx.yaml
    """
    mod = importlib.import_module('.' + data_cfg.target, 'ddpm.dataset')
    met = getattr(mod, 'parse_data')
    parsed_data = met(data_cfg.data_dir, data_cfg.split, data_cfg)
    return parsed_data


def build_dataloader(args, datasets, tokenizer, text_ctx_len, is_train, bs, shuffle=None):
    dataset_list = []
    print(type(datasets))
    for ds in datasets:
        data_cfg = datasets[ds]
        data_parserd = get_data_parsed(data_cfg)
        s = data_cfg.split
        DSet = TextImageDataset 
        dataset = DSet(
            data_parserd,
            side_x=args.side_x,
            side_y=args.side_y,
            resize_ratio=data_cfg.resize_ratio,
            uncond_p=args.uncond_p,
            shuffle=shuffle,
            tokenizer=tokenizer,
            text_ctx_len=text_ctx_len,
            use_captions=args.use_captions,
            enable_glide_upsample=args.enable_upsample,
            upscale_factor=args.upsample_factor,  # TODO: make this a parameter  
            use_flip=args.use_flip,
            is_train=is_train,
            cfg=args,
            data_cfg=data_cfg,
        )    
        print(dataset_list)
        dataset_list.append(dataset)
    if is_train:
        dataset = ConcatDatasetProb(dataset_list, args.get('train_prob', None))
    else:
        dataset = ConcatDataset(dataset_list)
 
    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=10,
        pin_memory=True,
    )

    return dataloader
