def get_camera(args, **kwargs):
    if args.camera.mode == 'gt':
        from .gt import get_camera
    elif args.camera.mode == 'para':
        from .para import get_camera
    else:
        raise NotImplementedError(args.para.mode)
    return get_camera(args, **kwargs)
