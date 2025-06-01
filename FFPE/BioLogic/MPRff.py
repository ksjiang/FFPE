
"""
# Biologic MPR File Parser #

Defines `fromFile()`, which parses data from `.mpr` files.
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import ctypes

import FFPE.Util.common as common

DATE_SIZE = 8
MODULE_SN_SIZE = 10
MODULE_LN_SIZE = 25

# headers
FILE_HEADER = b"BIO-LOGIC MODULAR FILE\x1a".ljust(0x30) + 4 * b"\x00"
HEADER = b"MODULE"
VMP_SET_SN = b"VMP Set".ljust(MODULE_SN_SIZE, b' ')
VMP_SET_LN = b"VMP settings".ljust(MODULE_LN_SIZE, b' ')
VMP_DATA_SN = b"VMP data".ljust(MODULE_SN_SIZE, b' ')
VMP_DATA_LN = b"VMP data".ljust(MODULE_LN_SIZE, b' ')
VMP_LOG_SN = b"VMP LOG".ljust(MODULE_SN_SIZE, b' ')
VMP_LOG_LN = b"VMP LOG".ljust(MODULE_LN_SIZE, b' ')

# byte value marking the beginning of the data stream
DATA_SENTINEL = 0x01

ACCEPTED_VERSIONS = [1101, 1146, 1152]

# attributes of a data column
COL_NAME = 0
COL_FMT = 1
COL_SIZE = 2
COL_FBIT = 3
COL_FSIZE = 4
VMP_DATA_FIELD_LIST = {
    1: ("mode", *common.NONE_FIELD, 0, 2), 
    2: ("ox/red", *common.NONE_FIELD, 2, 1), 
    3: ("error", *common.NONE_FIELD, 3, 1), 
    4: ("time", *common.DOUBLE, *common.NONE_FIELD), 
    5: ("control I", *common.SINGLE, *common.NONE_FIELD), 
    6: ("Ewe", *common.SINGLE, *common.NONE_FIELD), 
    7: ("dq", *common.DOUBLE, *common.NONE_FIELD), 
    8: ("I", *common.SINGLE, *common.NONE_FIELD), 
    0xb: ("<I>_mA", *common.DOUBLE, *common.NONE_FIELD), 
    0xd: ("Q-Q0", *common.DOUBLE, *common.NONE_FIELD), 
    0x13: ("control V", *common.SINGLE, *common.NONE_FIELD), 
    0x15: ("control change", *common.NONE_FIELD, 4, 1), 
    0x18: ("cycle number", *common.DOUBLE, *common.NONE_FIELD), 
    0x1f: ("Ns change", *common.NONE_FIELD, 5, 1), 
    0x20: ("freq", *common.SINGLE, *common.NONE_FIELD), 
    0x21: ("|Ewe|", *common.SINGLE, *common.NONE_FIELD), 
    0x22: ("|I|", *common.SINGLE, *common.NONE_FIELD), 
    0x23: ("angle(Z)", *common.SINGLE, *common.NONE_FIELD), 
    0x24: ("|Z|", *common.SINGLE, *common.NONE_FIELD), 
    0x25: ("Re(Z)", *common.SINGLE, *common.NONE_FIELD), 
    0x26: ("-Im(Z)", *common.SINGLE, *common.NONE_FIELD), 
    0x27: ("I range", *common.UINT16, *common.NONE_FIELD), 
    0x41: ("counter change", *common.NONE_FIELD, 7, 1), 
    0x4a: ("Energy", *common.DOUBLE, *common.NONE_FIELD), 
    0x46: ("P", *common.SINGLE, *common.NONE_FIELD), 
    0x4c: ("<I>", *common.SINGLE, *common.NONE_FIELD), 
    0x4d: ("<Ewe>", *common.SINGLE, *common.NONE_FIELD), 
    0x7b: ("Energy charge", *common.DOUBLE, *common.NONE_FIELD), 
    0x7c: ("Energy discharge", *common.DOUBLE, *common.NONE_FIELD), 
    0x7d: ("Capacitance charge", *common.DOUBLE, *common.NONE_FIELD), 
    0x7e: ("Capacitance discharge", *common.DOUBLE, *common.NONE_FIELD), 
    0x83: ("Ns", *common.UINT16, *common.NONE_FIELD), 
    0xa9: ("Cs", *common.SINGLE, *common.NONE_FIELD), 
    0xac: ("Cp", *common.SINGLE, *common.NONE_FIELD), 
    0x1b2: ("(Q-Q0)_C", *common.SINGLE, *common.NONE_FIELD), 
    0x1b6: ("Step time", *common.DOUBLE, *common.NONE_FIELD), 
    0x1d3: ("Q charge/discharge", *common.DOUBLE, *common.NONE_FIELD), 
    0x1d4: ("half cycle", *common.UINT32, *common.NONE_FIELD), 
    0x1d5: ("Z cycle", *common.UINT32, *common.NONE_FIELD), 
}

REST_MODE = 3

def parseVMPDataCode(code):
    assert code in VMP_DATA_FIELD_LIST, "Unknown data code 0x%x" % (code)
    return VMP_DATA_FIELD_LIST[code]

def parseVMPDataPoint(dataBlock, columns, offsets):
    assert len(columns) == len(offsets)
    myPtr = common.pointer()
    r = [0] * len(columns)
    # read flags
    flags = common.getField(dataBlock, myPtr, *common.UINT8)
    for i, column in enumerate(columns):
        if column[COL_FMT] is None:
            # flag value
            r[i] = common.getBitField(flags, column[COL_FBIT], column[COL_FSIZE])
        else:
            r[i] = common.getField(dataBlock, myPtr, column[COL_FMT], column[COL_SIZE])
            
    return r

# VMP sections -> Settings, Data, Log
class VMPsection(object):
    # initializes the object using header
    def __init__(self, datablock, version, section_sn = None, section_ln = None):
        assert version in ACCEPTED_VERSIONS, "Version %s not supported" % (version)
        self.softVersion = version
        ptr = common.pointer()
        common.checkField(datablock, ptr, HEADER)
        if section_sn is not None and section_ln is not None:
            common.checkField(datablock, ptr, section_sn)
            common.checkField(datablock, ptr, section_ln)
        else:
            common.getRaw(datablock, ptr, MODULE_SN_SIZE)
            common.getRaw(datablock, ptr, MODULE_LN_SIZE)
            
        # here, the different versions have different fields. for now, we try to guess which.
        if self.softVersion == 1152:
            # some field that is always -1
            common.getField(datablock, ptr, *common.UINT32)
            # read the next field as a qword integer
            self.dataSize = common.getField(datablock, ptr, *common.UINT64)
        else:
            self.dataSize = common.getField(datablock, ptr, *common.UINT32)
            
        self.secVersion = common.getField(datablock, ptr, *common.UINT32)
        # get the date string
        self.date = str(common.getRaw(datablock, ptr, DATE_SIZE), "utf-8")
        # data follows
        self.data = common.getRaw(datablock, ptr, self.dataSize)
        self.end = ptr.getValue()
        return
    
    def getEnd(self):
        return self.end
    
class VMPsettings(VMPsection):
    def __init__(self, datablock, version):
        # pass to super
        super(VMPsettings, self).__init__(datablock, version, VMP_SET_SN, VMP_SET_LN)
        # do something with the setting information
        return
    
class VMPlog(VMPsection):
    def __init__(self, datablock, version):
        # pass to super
        super(VMPlog, self).__init__(datablock, version, VMP_LOG_SN, VMP_LOG_LN)
        # do something with the log information
        return
    
class VMPdata(VMPsection):
    def __init__(self, datablock, version):
        super(VMPdata, self).__init__(datablock, version, VMP_DATA_SN, VMP_DATA_LN)
        ptr = common.pointer()
        # get the datafields
        self.numDataPts = common.getField(self.data, ptr, *common.UINT32)
        if self.softVersion == 1152:
            self.numCols = common.getField(self.data, ptr, *common.UINT16)
        else:
            self.numCols = common.getField(self.data, ptr, *common.UINT8)
            
        self.colList = []
        self.hasFlags = False
        for i in range(self.numCols):
            colData = parseVMPDataCode(common.getField(self.data, ptr, *common.UINT16))
            self.colList.append(colData)
            if colData[COL_FMT] is None:
                self.hasFlags = True
           
        # the data sentinel will not always assume a fixed value, making this unreliable
        '''
        while common.getField(self.data, ptr, *common.UINT8) != DATA_SENTINEL:
            continue
            
        '''
        
        # instead, we calculate the start of data based on the data size. assumes full blocks.
        # first, create numpy ndarray data structure
        self.record_spec = []
        if self.hasFlags:
            self.record_spec.append(("flags", np.uint8))
            
        for col in self.colList:
            if col[COL_FMT] is None:
                # flag value - skip
                continue
            else:
                self.record_spec.append((col[COL_NAME], common.NP_TYPES[col[COL_FMT]]))
                
        # then, compute offsets
        data_offs_rel_content_start = self.dataSize - self.numDataPts * np.dtype(self.record_spec).itemsize
        if self.softVersion == 1101:
            # there is a bug in this version where the last record is not transferred
            ptr.setValue(data_offs_rel_content_start - np.dtype(self.record_spec).itemsize)
        else:
            ptr.setValue(data_offs_rel_content_start)
            
        self.ptr = ptr
        return
    
    def parse(self):
        self.dataList = {}
        for col in self.colList:
            self.dataList[col[COL_NAME]] = []
            
        # profiling
        startTime = time.perf_counter()
        for i in range(self.numDataPts):
            # read flags
            if self.hasFlags:
                flags = common.getField(self.data, self.ptr, *common.UINT8)
                
            for col in self.colList:
                if col[COL_FMT] is None:
                    # flag value
                    self.dataList[col[COL_NAME]].append(common.getBitField(flags, col[COL_FBIT], col[COL_FSIZE]))
                else:
                    self.dataList[col[COL_NAME]].append(common.getField(self.data, self.ptr, col[COL_FMT], col[COL_SIZE]))
                    
        finishTime = time.perf_counter()
        common.printElapsedTime("Parse", startTime, finishTime)
        return
    
    # version that takes advantage of numpy's frombuffer function
    def parseVec(self):
        # profiling
        startTime = time.perf_counter()
#        print(self.record_spec, np.dtype(self.record_spec).itemsize, self.numDataPts)
        data = np.frombuffer(self.data, dtype = np.dtype(self.record_spec), count = self.numDataPts, offset = self.ptr.getValue())
        dataframe = pd.DataFrame(data = data, columns = [_[0] for _ in self.record_spec])
        
        self.dataList = dict()
        for col in self.colList:
            if col[COL_FMT] is None:
                # flag value
                self.dataList[col[COL_NAME]] = common.getBitField(np.array(dataframe["flags"]), col[COL_FBIT], col[COL_FSIZE])
                
            else:
                self.dataList[col[COL_NAME]] = np.array(dataframe[col[COL_NAME]])
                
        finishTime = time.perf_counter()
        common.printElapsedTime("Parse", startTime, finishTime)
        return
    
    # parallelized version
    def parseMP(self):
        self.dataList = {}            
        # compute the offsets of fields in the block
        self.blockOffsets = [0] * self.numCols
        
        if self.hasFlags:
            szPrev = common.BYTE_SIZE      #size of flags
        else:
            szPrev = 0
            
        for i, col in enumerate(self.colList):
            # check if flag - offset 0
            if col[COL_FMT] is None:
                continue
            
            # otherwise...
            self.blockOffsets[i] = self.blockOffsets[i - 1] + szPrev
            szPrev = col[COL_SIZE]
            
        # total size of each block
        self.dataBlockSize = self.blockOffsets[-1] + szPrev
        # offset to the actual data blocks
        dataOffset = self.ptr.getValue()
        # create a shared mapping so we don't have to always copy the data
        # no need to lock, as we are only reading from it
        sharedData = mp.Array(ctypes.c_char, self.data, lock = False)
        # start up a pool
        print("Spinning up a pool with %d workers" % (mp.cpu_count()))
        with mp.Pool(mp.cpu_count(), initializer = common.initProcess, initargs = (sharedData, )) as pool:
            # profiling
            startTime = time.perf_counter()
            # for each data column, read values from the data array in parallel
            for col, blockOffs in zip(self.colList, self.blockOffsets):
                if col[COL_FMT] is None:
                    # flag value
                    fbit, fsize = col[COL_FBIT], col[COL_FSIZE]
                    self.dataList[col[COL_NAME]] = pool.starmap(common.getBitFieldFlatShared, [(dataOffset + idx * self.dataBlockSize, fbit, fsize) for idx in range(self.numDataPts)])
                else:
                    fmt, size = col[COL_FMT], col[COL_SIZE]
                    self.dataList[col[COL_NAME]] = pool.starmap(common.getFieldFlatShared, [(dataOffset + idx * self.dataBlockSize + blockOffs, fmt, size) for idx in range(self.numDataPts)])
            
            finishTime = time.perf_counter()
            
        common.printElapsedTime("Parse", startTime, finishTime)
        # update pointer now that everything has been processed
        self.ptr.add(self.numDataPts * self.dataBlockSize)
        return
    
    def getDataFrame(self, mp = False):
        if mp == False:
            # use the fastest parse mode
            self.parseVec()
        else:
            if mp == "MP":
                self.parseMP()
            elif mp == "Vec":
                self.parseVec()
            else:
                print("Warning: Unrecognized parse mode %s. Falling back to default parse method." % (mp))
                self.parse()
                
        return pd.DataFrame(self.dataList)
    
def fromFile(fileName, version):
    ptr = common.pointer()
    with open(fileName, "rb") as f:
        contents = f.read()
        common.checkField(contents, ptr, FILE_HEADER)
        
    # assume that we start with VMP settings section
    x = VMPsettings(contents[ptr.getValue(): ], version)
    ptr.add(x.getEnd())
    y = VMPdata(contents[ptr.getValue(): ], version)
    return x, y
