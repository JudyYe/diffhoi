def get_artnet(mode, cfg):
    if mode == 'gt':
        from .gt import get_artnet as build
    elif mode == 'learn':
        from .opt import get_artnet as build
    else:
        raise NotImplementedError(mode)
    return build(**cfg)
