from .stackhourglass import PSMNet as stackhourglass
from .fast import PSMNet as fast
from .fast2 import PSMNet as fast2
from .fast3 import PSMNet as fast3
from .fast4 import PSMNet as fast4
from .fast5 import PSMNet as fast5
from .fast6 import PSMNet as fast6
from .fasta1 import PSMNet as fasta1
from .fasta2 import PSMNet as fasta2
from .fasta3 import PSMNet as fasta3
from .fasta4 import PSMNet as fasta4
from .fastb1 import PSMNet as fastb1
from .fastb2 import PSMNet as fastb2
from .fastb3 import PSMNet as fastb3
from .fastb4 import PSMNet as fastb4
from .fastb5 import PSMNet as fastb5
from .fastb6 import PSMNet as fastb6
from .fastb7 import PSMNet as fastb7
from .fastb8 import PSMNet as fastb8
from .fastc1 import PSMNet as fastc1
from .fastc2 import PSMNet as fastc2
from .fastc3 import PSMNet as fastc3
from .fastc4 import PSMNet as fastc4
from .fastc5 import PSMNet as fastc5

def get_model(args):
    if args.model == 'stackhourglass':
        model = stackhourglass(args)
    elif args.model == 'fast':
        model = fast(args)
    elif args.model == 'fast2':
        model = fast2(args)
    elif args.model == 'fast3':
        model = fast3(args)
    elif args.model == 'fast4':
        model = fast4(args)
    elif args.model == 'fast5':
        model = fast5(args)
    elif args.model == 'fast6':
        model = fast6(args)
    elif args.model == 'fasta1':
        model = fasta1(args)
    elif args.model == 'fasta2':
        model = fasta2(args)
    elif args.model == 'fasta3':
        model = fasta3(args)
    elif args.model == 'fasta4':
        model = fasta4(args)
    elif args.model == 'fastb1':
        model = fastb1(args)
    elif args.model == 'fastb2':
        model = fastb2(args)
    elif args.model == 'fastb3':
        model = fastb3(args)
    elif args.model == 'fastb4':
        model = fastb4(args)
    elif args.model == 'fastb5':
        model = fastb5(args)
    elif args.model == 'fastb6':
        model = fastb6(args)
    elif args.model == 'fastb7':
        model = fastb7(args)
    elif args.model == 'fastb8':
        model = fastb8(args)
    elif args.model == 'fastb9':
        model = fastb8(args)
    elif args.model == 'fastc1':
        model = fastc1(args)
    elif args.model == 'fastc2':
        model = fastc2(args)
    elif args.model == 'fastc3':
        model = fastc3(args)
    elif args.model == 'fastc4':
        model = fastc4(args)
    elif args.model == 'fastc5':
        model = fastc5(args)
    else:
        raise ValueError('no much model')
    return model
