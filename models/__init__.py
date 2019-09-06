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
from .fastc6 import PSMNet as fastc6
from .fastc7 import PSMNet as fastc7
from .fastc8 import PSMNet as fastc8
from .fastc9 import PSMNet as fastc9
from .fastd1 import PSMNet as fastd1
from .fastd2 import PSMNet as fastd2
from .fastd3 import PSMNet as fastd3
from .fastd4 import PSMNet as fastd4
from .fastd5 import PSMNet as fastd5
from .fastd6 import PSMNet as fastd6
from .faste1 import PSMNet as faste1
from .faste2 import PSMNet as faste2
from .faste3 import PSMNet as faste3
from .faste4 import PSMNet as faste4
from .faste5 import PSMNet as faste5
from .faste6 import PSMNet as faste6

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
    elif args.model == 'fastc6':
        model = fastc6(args)
    elif args.model == 'fastc7':
        model = fastc7(args)
    elif args.model == 'fastc8':
        model = fastc8(args)
    elif args.model == 'fastc9':
        model = fastc9(args)
    elif args.model == 'fastd1':
        model = fastd1(args)
    elif args.model == 'fastd2':
        model = fastd2(args)
    elif args.model == 'fastd3':
        model = fastd3(args)
    elif args.model == 'fastd4':
        model = fastd4(args)
    elif args.model == 'fastd5':
        model = fastd5(args)
    elif args.model == 'fastd6':
        model = fastd6(args)
    elif args.model == 'faste1':
        model = faste1(args)
    elif args.model == 'faste2':
        model = faste2(args)
    elif args.model == 'faste3':
        model = faste3(args)
    elif args.model == 'faste4':
        model = faste4(args)
    elif args.model == 'faste5':
        model = faste5(args)
    elif args.model == 'faste6':
        model = faste6(args)
    else:
        raise ValueError('no such model')
    return model
