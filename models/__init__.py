from .BayeSeg import build as build_BayeSeg
from .PUnet import build as build_PUnet
from .segmentation import build as build_Unet

def build_model(args):
    if args.model == 'BayeSeg':
        return build_BayeSeg(args)
    elif args.model == 'PUnet':
        return build_PUnet(args)
    elif args.model == 'Unet':
        return build_Unet(args)
    else:
        raise ValueError('invalid model:{}'.format(args.model))
