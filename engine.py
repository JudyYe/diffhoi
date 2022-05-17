from importlib import import_module
import random
from typing import Any

from omegaconf import OmegaConf
from utils import io_util
import warnings
import submitit
import os
import copy
import hydra
import hydra
import hydra.utils as hydra_utils
from pathlib import Path
from utils.print_fn import log
from jutils import slurm_utils


def update_pythonpath_relative_hydra():
    """Update PYTHONPATH to only have absolute paths."""
    # NOTE: We do not change sys.path: we want to update paths for future instantiations
    # of python using the current environment (namely, when submitit loads the job
    # pickle).
    try:
        original_cwd = Path(hydra_utils.get_original_cwd()).resolve()
    except (AttributeError, ValueError):
        # Assume hydra is not initialized, we don't need to do anything.
        # In hydra 0.11, this returns AttributeError; later it will return ValueError
        # https://github.com/facebookresearch/hydra/issues/496
        # I don't know how else to reliably check whether Hydra is initialized.
        return
    paths = []
    for orig_path in os.environ["PYTHONPATH"].split(":"):
        path = Path(orig_path)
        if not path.is_absolute():
            path = original_cwd / path
        paths.append(path.resolve())
    os.environ["PYTHONPATH"] = ":".join([str(x) for x in paths])
    log.info('PYTHONPATH: {}'.format(os.environ["PYTHONPATH"]))


class Worker:
    def checkpoint(self, *args: Any, **kwargs: Any):
        print('requeue!!!?????')
        return submitit.helpers.DelayedSubmission(self, *args, **kwargs)  # submits to requeuing    

    def __call__(self, origargs):
        """TODO: Docstring for __call__.

        :args: TODO
        :returns: TODO

        """
        main_function = import_module(origargs.worker)
        # main_function = import_module('train')
        # from train import main_function

        main_worker = main_function.main_function
        import numpy as np
        import torch.multiprocessing as mp
        import torch.utils.data.distributed
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

        main_args = copy.deepcopy(origargs)
        if main_args.environment.multiprocessing_distributed:
            mp.set_start_method('spawn')
        np.set_printoptions(precision=3)
        socket_name = os.popen(
            "ip r | grep default | awk '{print $5}'").read().strip('\n')
        print("Setting GLOO and NCCL sockets IFNAME to: {}".format(socket_name))
        os.environ["GLOO_SOCKET_IFNAME"] = socket_name

        if main_args.environment.slurm:
            job_env = submitit.JobEnvironment()
            main_args.environment.rank = job_env.global_rank
            main_args.environment.dist_url = f'tcp://{job_env.hostnames[0]}:{main_args.environment.port}'
        else:
            main_args.environment.port = random.randint(10000, 20000)
            main_args.environment.dist_url = f'tcp://{main_args.environment.node}:{main_args.environment.port}'
        print('Using url {}'.format(main_args.environment.dist_url))

        if main_args.environment.seed is not None:
            random.seed(main_args.environment.seed)
            torch.manual_seed(main_args.environment.seed)
            cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')
        if main_args.environment.gpu is not None:
            warnings.warn(
                'You have chosen a specific GPU. This will completely '
                'disable data parallelism.')

        if main_args.environment.dist_url == "env://" and main_args.environment.world_size == -1:
            main_args.environment.world_size = int(os.environ["WORLD_SIZE"])

        main_args.environment.distributed = main_args.environment.world_size > 1 or main_args.environment.multiprocessing_distributed
        ngpus_per_node = torch.cuda.device_count()
        if main_args.environment.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            main_args.environment.world_size = ngpus_per_node * main_args.environment.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function

            # main_worker(main_args.environment.gpu, ngpus_per_node, main_args)

            # spwan in main??
            mp.spawn(main_worker,
                     nprocs=ngpus_per_node,
                     args=(ngpus_per_node, main_args))
        else:
            # Simply call main_worker function
            main_worker(main_args.environment.gpu, ngpus_per_node, main_args)


@hydra.main(config_path='./configs/', config_name='volsdf_hoi')
def hydra_main(config):
    # TODO: change to hydra for better sweeping?
    update_pythonpath_relative_hydra()
    # DANGEROUS: set device !
    io_util.setup_config_for_hydra(config)

    if not config.environment.resume:
        log.warn('dangerous!! wipe out %s' % config.training.exp_dir)
        os.system('rm -r %s' % config.training.exp_dir)
    # add slurm 
    slurm_utils.slurm_wrapper_hydra(config, Worker())


if __name__ == '__main__':
    hydra_main()