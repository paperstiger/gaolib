from .argument import getOnOffArgs
from .argument import getOnOffArgs as getArgs
from .fileio import ddctParse, ddctSave, getJson, dumpJson, assignModule, assignModuleByJson, getLogPath
from .printio import Mute
from .stringio import getNumber, numRe, joinNumber
try:
    import sharedmemory
except:
    pass
