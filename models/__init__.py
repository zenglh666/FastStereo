from .stackhourglass import PSMNet as stackhourglass
from .fast import PSMNet as fast
from .fast2 import PSMNet as fast2

def get_model(args):
    if args.model == 'stackhourglass':
        model = stackhourglass(args)
    elif args.model == 'fast':
        model = fast(args)
    elif args.model == 'fast2':
        model = fast2(args)
    else:
        raise ValueError('no much model')
    return model
