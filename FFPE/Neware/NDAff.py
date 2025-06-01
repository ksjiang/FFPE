
"""
# Neware NDA File Parser #

Defines `fromFile()`, which parses data from `.nda` files.
"""

import time
import numpy as np
import pandas as pd

import FFPE.Util.common as common
import FFPE.Util.converters as conv
import FFPE.Neware.constants as constants

FILE_HEADER = b"NEWARE"
YEAR_SLEN, MONTH_SLEN, DAY_SLEN = 4, 2, 2
NDA_VERSION_STRING_OFFS = 0x70
NDA_VERSION_STRING_SLEN = 30
CHANNEL_INFO_OFFS = 0x82b
USERNAME_OFFS = 0x876
USERNAME_LEN = 15
BATCH_LEN = 20
MEMO_LEN = 100
STATUS_SUCCESS = 0

NDA_RECORD_INFO_SPEC = [
    ("status", np.uint8), 
    ("record_no", np.uint32), 
    ("cycle number", np.uint32), 
    ("Ns", np.uint8), 
    ("mode", np.uint8), 
    ("step_time", np.uint32), 
    ("Ewe", np.int32), 
    ("current", np.int32), 
    ("temperature", np.int64), 
    ("Q charge/discharge", np.int64), 
    ("Energy charge/discharge", np.int64), 
    ("clock_time", np.uint64), 
    ("checksum", np.uint32)
]

NDA_RECORD_INFO = np.dtype(NDA_RECORD_INFO_SPEC)

# accumulates a series by referencing step changes
# expects numpy array
def accumulateSeriesSteps(series, change_indices):
    # new series
    r = np.zeros(len(series))
    # accumulator and mark
    a = 0.
    mark = 0
    for change_index in change_indices:
        r[mark: change_index + 1] = a + series[mark: change_index + 1]
        # update accumulator and mark
        a += series[change_index]
        mark = change_index + 1
        
    # finish the series
    r[mark: ] = a + series[mark: ]
    return r

def fromFile(fileName):
    ptr = common.pointer()
    with open(fileName, "rb") as f:
        contents = f.read()
        
    # check magic bytes
    common.checkField(contents, ptr, FILE_HEADER)
    # initialize header info structure
    header_info = dict()
    # get date YYYYMMDD
    date = dict()
    date["year"] = int(common.getRaw(contents, ptr, YEAR_SLEN))
    date["month"] = int(common.getRaw(contents, ptr, MONTH_SLEN))
    date["day"] = int(common.getRaw(contents, ptr, DAY_SLEN))
    header_info["date"] = date
    
    ptr.setValue(NDA_VERSION_STRING_OFFS)
    header_info["sw_version"] = common.bytes2Cstring(common.getRaw(contents, ptr, NDA_VERSION_STRING_SLEN))
    
    ptr.setValue(CHANNEL_INFO_OFFS)
    header_info["machine"] = common.getField(contents, ptr, *common.UINT8)
    header_info["hw_version"] = common.getField(contents, ptr, *common.UINT8)
    
    ptr.setValue(USERNAME_OFFS)
    header_info["username"] = common.bytes2Cstring(common.getRaw(contents, ptr, USERNAME_LEN))
    header_info["batch"] = common.bytes2Cstring(common.getRaw(contents, ptr, BATCH_LEN))
    header_info["memo"] = common.bytes2Cstring(common.getRaw(contents, ptr, MEMO_LEN))
    
    # this brings us to the step definitions
    # extract the step definitions from header
    # TODO: there should be a better way to figure out where this header ends
    step_infos = [None]
    step_id = 1
    while True:
        assert step_id < 0xff, "Too many steps encountered, possible parsing error"
        # initialize step info dict
        step_info = dict()
        if common.getFieldFlat(contents, ptr.getValue(), *common.UINT8) != step_id:
            break
        
        # update pointer
        ptr.add(common.UINT8[1])
        step_info["Ns"] = step_id
        step_type = common.getField(contents, ptr, *common.UINT8)
        assert step_type < len(constants.STEP_TYPES) and constants.STEP_TYPES[step_type] is not None, "Unrecognized step type"
        step_info["mode"] = constants.STEP_TYPES[step_type]
        if step_info["mode"] in ["CC_charge", "CC_discharge"]:
            step_info["current"] = conv.UA2MA(common.getField(contents, ptr, *common.INT32))
            step_info["time"] = common.getField(contents, ptr, *common.INT32)
            step_info["voltage"] = conv.TMV2V(common.getField(contents, ptr, *common.INT32))
            ptr.add(2 * common.INT32[1])
        elif step_info["mode"] == "Rest":
            step_info["time"] = common.getField(contents, ptr, *common.INT32)
            ptr.add(4 * common.INT32[1])
        elif step_info["mode"] == "Loop":
            step_info["target"] = common.getField(contents, ptr, *common.INT32)
            step_info["repeats"] = common.getField(contents, ptr, *common.INT32)
            ptr.add(3 * common.INT32[1])
        elif step_info["mode"] == "Stop":
            # no parameters
            ptr.add(5 * common.INT32[1])
            
        step_infos.append(step_info)
        step_id += 1
        
    # this brings us to the data records
    # the pointer object holds the offset to the data records
    # use numpy frombuffer with this offset to efficiently extract data
    startTime = time.perf_counter()
    data = np.frombuffer(contents, dtype = NDA_RECORD_INFO, offset = ptr.getValue())
    dataframe = pd.DataFrame(data = data, columns = [_[0] for _ in NDA_RECORD_INFO_SPEC])
    # perform conversions on some of the columns
    # change type of step_time to float
    dataframe["step_time"] = dataframe["step_time"].astype(float)
    # convert voltage from tenths of mV to V
    dataframe["Ewe"] = conv.TMV2V(dataframe["Ewe"].astype(float))
    # convert current from uA to mA
    dataframe["current"] = conv.UA2MA(dataframe["current"].astype(float))
    # convert capacity from uAs to mAh
    dataframe["Q charge/discharge"] = conv.UAS2MAH(dataframe["Q charge/discharge"].astype(float))
    # convert energy from uWs to Wh
    dataframe["Energy charge/discharge"] = conv.UWS2WH(dataframe["Energy charge/discharge"].astype(float))
    # wish there was a way to verify the checksum, but for now just delete it
    del dataframe["checksum"]
    
    # remove invalid rows and reset the index (original index is preserved in record_no column)
    dataframe = dataframe.loc[dataframe["status"] == STATUS_SUCCESS].reset_index(drop = True)
    finishTime = time.perf_counter()
    common.printElapsedTime("Parse", startTime, finishTime)
    
    # now, we accumulate some variables, such as time and charge referenced to the beginning of the experiment
    # find step changes
    step_changes = constants.findStepChanges(np.array(dataframe["Ns"]))
    # find half cycle changes
    hc_changes = np.concatenate(([-1], constants.findHalfCycleChanges(np.array(dataframe["mode"]))), axis = 0)
    dataframe["half cycle"] = constants.convertIndices(np.arange(len(dataframe["Ns"])), hc_changes)
    dataframe["time"] = accumulateSeriesSteps(np.array(dataframe["step_time"]), step_changes)
    dataframe["Q-Q0"] = accumulateSeriesSteps(np.array(dataframe["Q charge/discharge"]) * np.where(np.array(dataframe["mode"]) == constants.STEP_TYPES.index("CC_discharge"), -1, 1), step_changes)
    
    return header_info, step_infos, dataframe
