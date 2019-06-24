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
    else:
        raise ValueError('no much model')
    return model
