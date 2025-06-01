
"""
# Common Utilities #

Useful type definitions and functions for wading through binary data.
"""

# imports
import os
import struct
import numpy as np

# sizes of data types
BYTE_SIZE = 1
SHORT_SIZE = 2
LONG_SIZE = 4
LLONG_SIZE = 8

# data types and their struct symbolcodes
FMT_SPEC = 0
FMT_SIZE = 1
# by default, we assume little-endian
SINGLE = ("<f", LONG_SIZE)
DOUBLE = ("<d", LLONG_SIZE)
UINT8 = ("<B", BYTE_SIZE)
UINT16 = ("<H", SHORT_SIZE)
UINT32 = ("<I", LONG_SIZE)
INT32 = ("<i", LONG_SIZE)
UINT64 = ("<Q", LLONG_SIZE)
# in needed cases, big-endian (gross!)
SINGLE_BE = (">f", LONG_SIZE)
DOUBLE_BE = (">d", LLONG_SIZE)

# numpy equivalent types
NP_TYPES = {
    SINGLE[FMT_SPEC]: np.float32, 
    DOUBLE[FMT_SPEC]: np.float64, 
    UINT8[FMT_SPEC]: np.uint8, 
    UINT16[FMT_SPEC]: np.uint16,
    UINT32[FMT_SPEC]: np.uint32, 
    INT32[FMT_SPEC]: np.int32, 
}

# to denote an empty byte or bit field
NONE_FIELD = (None, None)

# floating-point "not a number"
NAN = float("NaN")
EPS = np.finfo(float).eps

# a pointer object that can be used to keep track of position within a data block
class pointer(object):
    def __init__(self, value = 0):
        self.value = value
        return
    
    def getValue(self):
        return self.value
    
    def setValue(self, newValue):
        self.value = newValue
        return
    
    def add(self, delta):
        self.setValue(self.getValue() + delta)
        return
    
def bytes2Cstring(bs):
    return str(bs.split(b'\x00')[0], "utf-8")
    
def checkField(data, ptr, expected):
    actual = data[ptr.getValue(): ptr.getValue() + len(expected)]
    assert actual == expected, "At 0x%x, expected %s but got %s" % (ptr.getValue(), expected, actual)
    ptr.add(len(expected))
    return

def checkFieldFlat(data, ptrVal, expected):
    actual = data[ptrVal: ptrVal + len(expected)]
    assert actual == expected, "At 0x%x, expected %s but got %s" % (ptrVal, expected, actual)
    return

def getRaw(data, ptr, size):
    r = data[ptr.getValue(): ptr.getValue() + size]
    ptr.add(size)
    return r

def getRawFlat(data, ptrVal, size):
    return data[ptrVal: ptrVal + size]

# returns a field value and updates pointer object
def getField(data, ptr, fmt, size):
    r = struct.unpack(fmt, data[ptr.getValue(): ptr.getValue() + size])[0]
    # update pointer value
    ptr.add(size)
    return r

# returns a field value with a static offset
def getFieldFlat(data, ptrVal, fmt, size):
    return struct.unpack(fmt, data[ptrVal: ptrVal + size])[0]

def getFieldsStridedFlat(data, ptrVal, blk, N, fmt, size):
    r = [0] * N
    for idx in range(N):
        r[idx] = getFieldFlat(data, ptrVal + idx * blk, fmt, size)
        
    return r

def getBitField(bt, bitPos, fieldSize):
    return (bt >> bitPos) % (1 << fieldSize)

def getBitFieldFlat(data, ptrVal, bitPos, fieldSize):
    return getBitField(getFieldFlat(data, ptrVal, *UINT8), bitPos, fieldSize)

def getBitFieldsStridedFlat(data, ptrVal, blk, N, bitPos, fieldSize):
    r = [0] * N
    for idx in range(N):
        r[idx] = getBitFieldFlat(data, ptrVal + idx * blk, bitPos, fieldSize)
        
    return r

# versions of the above functions for multiprocessing
def initProcess(sharedData):
    global mySharedData
    mySharedData = sharedData
    return

def getFieldFlatShared(ptrVal, fmt, size):
    global mySharedData
    return getFieldFlat(mySharedData, ptrVal, fmt, size)

def getFieldsStridedFlatShared(ptrVal, blk, N, fmt, size):
    global mySharedData
    return getFieldsStridedFlat(mySharedData, ptrVal, blk, N, fmt, size)

def getBitFieldFlatShared(ptrVal, bitPos, fieldSize):
    global mySharedData
    return getBitFieldFlat(mySharedData, ptrVal, bitPos, fieldSize)

def getBitFieldsStridedFlatShared(ptrVal, blk, N, bitPos, fieldSize):
    global mySharedData
    return getBitFieldsStridedFlat(mySharedData, ptrVal, blk, N, bitPos, fieldSize)

def ternary(condition, valueIfTrue, valueIfFalse):
    if condition:
        return valueIfTrue
    else:
        return valueIfFalse

def intersperse(l, item):
    r = [item] * (2 * len(l) - 1)
    r[: : 2] = l
    return r

# collect all files in a directory that have a substring
# by default, get *all* files in directory
def collectFiles(inputFileDir, s = ""):
    return [fname for fname in os.listdir(inputFileDir) if os.path.isfile(os.path.join(inputFileDir, fname)) and s in fname]

def createOutputFileName(inputFiles, suffix, extension = None):
    if len(inputFiles) == 0: assert False
    r, ext = os.path.splitext(inputFiles[0])
    roots = [r]
    # strip extensions
    for inputFile in inputFiles[1: ]:
        r, ext_ = os.path.splitext(inputFile)
        if extension is None and ext != ext_: assert False
        roots.append(r)
        
    common_part = os.path.commonprefix(roots).strip('_')
    if extension is not None:
        ext = '.' + extension
        
    return '_'.join([common_part, suffix]) + ext

def printElapsedTime(tag, start, finish):
    print("%s finished in %0.6f s" % (tag, finish - start))
    return
