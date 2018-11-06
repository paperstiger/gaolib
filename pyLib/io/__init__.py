from .argument import getOnOffArgs
from .argument import getOnOffArgs as getArgs
from .fileio import ddctParse, ddctSave, getJson, dumpJson, assignModule, assignModuleByJson, getLogPath
from .printio import Mute
from .stringio import getNumber, numRe, joinNumber
from .codeio import load_source
from .codeio import load_source as loadSource
from .printcolor import print_yellow, print_purple, print_red, print_cyan, print_gray, print_green
try:
    from . import sharedmemory
    from .sharedmemory import npload
except:
    pass
