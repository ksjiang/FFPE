
import time
import numpy as np
import pandas as pd
import FFPE.Util.common as common

FILE_HEADER = b"NEWARE"
YEAR_SLEN, MONTH_SLEN, DAY_SLEN = 4, 2, 2
BTS_VERSION_STRING_OFFS = 0x70
BTS_VERSION_STRING_SLEN = 30
CHANNEL_INFO_OFFS = 0x82b
USERNAME_OFFS = 0x876
USERNAME_LEN = 15
BATCH_LEN = 20
MEMO_LEN = 100
STATUS_SUCCESS = 0

UA2MA = lambda x: x * 1E-3
TMV2V = lambda x: x * 1E-4
UW2W = lambda x: x * 1E-6
S2HR = lambda x: x / 3600.
UAS2MAH = lambda x: S2HR(UA2MA(x))
UWS2WH = lambda x: S2HR(UW2W(x))

BTS_RECORD_INFO_SPEC = [
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

BTS_RECORD_INFO = np.dtype(BTS_RECORD_INFO_SPEC)
BTS_STEP_TYPES = [None, "CC_charge", "CC_discharge", None, "Rest", "Loop", "Stop"]

# returns the leading edge (smaller index) of changes
# expects numpy array, so convert using to_numpy() if using pandas
def findStepChanges(series):
    return np.where(np.diff(series) != 0)[0]

def findHalfStepChanges(series, triggers):
    changes = []
    for trigger in triggers:
        hot = np.where(series == trigger)[0]
        leading_edge_idcs = np.where(np.diff(hot) != 1)[0] + 1
        # subtract one to get leading edge index (don't subtract if already 0)
        hot_leading_edges = np.maximum(0, hot[leading_edge_idcs] - 1)
        # treat the first instance
        # subtract one to get leading edge index (don't subtract if already 0)
        first = max([0, hot[0] - 1])
        if first not in hot_leading_edges:
            hot_leading_edges = np.concatenate([[first], hot_leading_edges], axis = 0)
            
        changes.append(hot_leading_edges)
        
    return np.sort(np.concatenate(changes, axis = 0))

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

def fillCount(series_size, change_indices):
    # new series
    r = np.zeros(series_size, dtype = int)
    a = 0
    mark = 0
    for change_index in change_indices:
        r[mark: change_index + 1] = a
        a += 1
        mark = change_index + 1
        
    # finish the series
    r[mark: ] = a
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
    
    ptr.setValue(BTS_VERSION_STRING_OFFS)
    header_info["version"] = common.bytes2Cstring(common.getRaw(contents, ptr, BTS_VERSION_STRING_SLEN))
    
    ptr.setValue(CHANNEL_INFO_OFFS)
    header_info["machine"] = common.getField(contents, ptr, *common.UINT8)
    header_info["version"] = common.getField(contents, ptr, *common.UINT8)
    
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
        assert step_type < len(BTS_STEP_TYPES) and BTS_STEP_TYPES[step_type] is not None, "Unrecognized step type"
        step_info["mode"] = BTS_STEP_TYPES[step_type]
        if step_info["mode"] in ["CC_charge", "CC_discharge"]:
            step_info["current"] = UA2MA(common.getField(contents, ptr, *common.INT32))
            step_info["time"] = common.getField(contents, ptr, *common.INT32)
            step_info["voltage"] = TMV2V(common.getField(contents, ptr, *common.INT32))
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
    data = np.frombuffer(contents, dtype = BTS_RECORD_INFO, offset = ptr.getValue())
    dataframe = pd.DataFrame(data = data, columns = [_[0] for _ in BTS_RECORD_INFO_SPEC])
    # perform conversions on some of the columns
    # change type of step_time to float
    dataframe["step_time"] = dataframe["step_time"].astype(float)
    # convert voltage from tenths of mV to V
    dataframe["Ewe"] = TMV2V(dataframe["Ewe"].astype(float))
    # convert current from uA to mA
    dataframe["current"] = UA2MA(dataframe["current"].astype(float))
    # convert capacity from uAs to mAh
    dataframe["Q charge/discharge"] = UAS2MAH(dataframe["Q charge/discharge"].astype(float))
    # convert energy from uWs to Wh
    dataframe["Energy charge/discharge"] = UWS2WH(dataframe["Energy charge/discharge"].astype(float))
    # wish there was a way to verify the checksum, but for now just delete it
    del dataframe["checksum"]
    
    # remove invalid rows and reset the index (original index is preserved in record_no column)
    dataframe = dataframe.loc[dataframe["status"] == STATUS_SUCCESS].reset_index(drop = True)
    finishTime = time.perf_counter()
    common.printElapsedTime("Parse", startTime, finishTime)
    
    # now, we accumulate some variables, such as time and charge
    # referenced to the beginning of the experiment
    # find the rows where the step changes (leading edge)
    step_changes = findStepChanges(np.array(dataframe["Ns"]))
    # find where the current changes direction (occurs less frequently than step_changes)
    half_step_changes = findHalfStepChanges(np.array(dataframe["mode"]), (BTS_STEP_TYPES.index("CC_discharge"), BTS_STEP_TYPES.index("CC_charge")))
    dataframe["half cycle"] = fillCount(len(dataframe["Ns"]), half_step_changes)
    dataframe["time"] = accumulateSeriesSteps(np.array(dataframe["step_time"]), step_changes)
    dataframe["Q-Q0"] = accumulateSeriesSteps(np.array(dataframe["Q charge/discharge"]) * np.where(np.array(dataframe["mode"]) == BTS_STEP_TYPES.index("CC_discharge"), -1, 1), step_changes)
    
    return header_info, step_infos, dataframe
