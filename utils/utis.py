import hydra.utils as hydra_utils
import os
import logging as log
from pathlib import Path

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

