
"""
# Neware NDAX File Parser #

Defines `fromFile()`, which parses data from `.ndax` files.
"""

import time
import numpy as np
import pandas as pd
import zipfile
import zlib
import xml.etree.ElementTree as ET
import datetime

import FFPE.Util.common as common
import FFPE.Util.converters as conv
import FFPE.Neware.constants as constants

# a page consists of a 2-byte header identifying the type of page, 2-byte header for the population count, a 128-byte (1024-bit) bitarray, 3960 bytes of arbitrary data, with validity defined by the bitmap, and a 4-byte CRC32 checksum
PAGE_SIZE = 0x1000

# defined in Common.Ndc.NdcDefine.NdcData
NDAX_PAGE_BITMAP_SZ = 128
NDAX_PAGE_DATA_SZ = 3960
NDAX_PAGE_INFO_SPEC = [
    ("type", np.uint16), 
    ("count", np.uint16), 
    ("bitmap", np.dtype((np.uint8, NDAX_PAGE_BITMAP_SZ))), 
    ("data", np.dtype((np.uint8, NDAX_PAGE_DATA_SZ))), 
    ("crc32", np.uint32), 
]

# defined in Common.Ndc.NdcDefine.DFDATA
NDAX_DATA_INFO_SPEC = [
    ("Ewe", np.float32), 
    ("current", np.float32), 
]

# defined in Common.Ndc.NdcDefine.StepDFDATA
NDAX_STEP_INFO_SPEC = [
    ("cycle number", np.uint32), 
    ("Ns", np.uint32),                      #this is really defined with the following field as an array of 5 uints, but for main data, only the first one is populated
    ("stepIDs", np.dtype((np.uint32, 4))), 
    ("mode", np.uint8), 
    ("start_time", np.uint64), 
    ("start_record_no", np.uint32), 
]

NDAX8_STEP_INFO_SPEC = [
    ("cycle number", np.uint32), 
    ("Ns", np.uint32),                      #this is really defined with the following field as an array of 5 uints, but for main data, only the first one is populated
    ("stepIDs", np.dtype((np.uint32, 4))), 
    ("mode", np.uint8), 
    ("start_time", np.uint64), 
    ("start_record_no", np.uint32), 
    ("padding", np.dtype((np.uint8, 63))),  #version 8 has some padding here
]

# defined in Common.Ndc.NdcDefine.DFDATARunInfo
NDAX_RUN_INFO_SPEC = [
    ("step_time", np.uint32),               #ms
    ("step_time_high", np.uint8),           #high byte of step time
    ("Q charge", np.float32),               #uAs
    ("Q discharge", np.float32),            #uAs
    ("Energy charge", np.float32),          #uWs
    ("Energy discharge", np.float32),       #uWs
    ("current range", np.float32), 
    ("work type", np.uint32), 
    ("delta t", np.uint32),                 #ms
    ("clock_time", np.uint32),              #Unix time
    ("step_no", np.uint32), 
    ("record_no", np.uint32), 
    ("clock_time_ms", np.uint16),           #ms
]

NDAX8_RUN_INFO_SPEC = [
    ("step_time", np.uint32),               #ms
    ("step_time_high", np.uint8),           #high byte of step time
    ("Q charge", np.float32),               #uAs
    ("Q discharge", np.float32),            #uAs
    ("Energy charge", np.float32),          #uWs
    ("Energy discharge", np.float32),       #uWs
    ("current range", np.float32), 
    ("work type", np.uint32), 
    ("delta t", np.uint32),                 #ms
    ("clock_time", np.uint32),              #Unix time
    ("step_no", np.uint32), 
    ("record_no", np.uint32), 
    ("clock_time_ms", np.uint16),           #ms
    ("padding", np.dtype((np.uint8, 53)))   #version 8 has some padding here
]

NDAX_PAGE_INFO = np.dtype(NDAX_PAGE_INFO_SPEC)
NDAX_STEP_INFO = np.dtype(NDAX_STEP_INFO_SPEC)
NDAX8_STEP_INFO = np.dtype(NDAX8_STEP_INFO_SPEC)
NDAX_DATA_INFO = np.dtype(NDAX_DATA_INFO_SPEC)
NDAX_RUN_INFO = np.dtype(NDAX_RUN_INFO_SPEC)
NDAX8_RUN_INFO = np.dtype(NDAX8_RUN_INFO_SPEC)
NDAX_LIMIT_TYPES = {
    "Time": "time", 
    "Curr": "current", 
    "Volt": "voltage", 
    "Stop_Volt": "stop_voltage", 
    "Stop_Curr": "stop_current", 
    "Start_Step": "target", 
    "Cycle_Count": "repeats", 
}

NDAX_STEP_INFO_PAGENUM = 8
NDAX_DATA_INFO_PAGENUM = 2
NDAX_RUN_INFO_PAGENUM = 0x13

UNIT_CONVERTERS = {
    "time": conv.MS2S, 
    "voltage": conv.TMV2V, 
}

def isPageValid(f):
    """
    Checks that a page is valid by computing and comparing the checksum.

    Parameters
    --------
    f : NDAX_PAGE_INFO
        The page to check

    Returns
    --------
    bool
        Whether the page is valid

    """
    return zlib.crc32(f.tobytes()[: -4]) == f["crc32"]

def extractAllFromPage(f, req_type, drop_invalid_records = True, last_page = False):
    """
    Extracts all records of type `req_type` from a page. This just pulls the data from the page; no guarantees are made about the validity of the page, which should be checked by the caller.

    Parameters
    --------
    f : NDAX_PAGE_INFO
        The page to parse

    req_type : numpy.dtype
        A datatype

    drop_invalid_records : bool
        Whether to silently drop invalid records. If `True` (the default), this function will only return the valid records, and the second return value will be empty. If `False`, this function will return all the records in the page, and the second return value will contain a boolean array specifying the validity of each record

    last_page : bool
        Whether `f` is the last page. This argument is not essential to specify if `drop_invalid` is `True`. When `drop_invalid` is `False`, this flag indicates whether trailing records (i.e., invalid records following the last valid record) should be silently dropped

    Returns
    --------
    Tuple[numpy.ndarray[numpy.dtype], numpy.ndarray[bool]]
        Two `numpy` arrays. The first is the array containing the data, with elements being of the requested type. The second is a boolean array specifying the validity of each record

    """
    num_items = NDAX_PAGE_DATA_SZ // req_type.itemsize
    data = np.frombuffer(f["data"], dtype = req_type, count = num_items)
    # not all of the data is valid. only return the ones indicated by the bitmap
    valid = np.unpackbits(f["bitmap"], bitorder = "little")[: num_items]
    # check that the number of set bits matches the count
    assert np.sum(valid) == f["count"]
    valid_bool = valid.astype("bool")
    if drop_invalid_records:
        parsed_contents = data[valid_bool]
        # all records are valid
        validity = np.full(parsed_contents.shape, True, dtype = bool)
    else:
        parsed_contents = data
        validity = valid_bool
        if last_page:
            valid_idcs = np.argwhere(validity)
            if valid_idcs.shape[0] == 0:
                # this page holds no valid records!
                return np.empty((0, ), dtype = req_type), np.empty((0, ), dtype = bool)
            
            # drop all records following the last valid record
            parsed_contents = parsed_contents[: valid_idcs[-1][0]]
            validity = validity[: valid_idcs[-1][0]]

    return parsed_contents, validity

def extractAllFromBuffer(buf, type_pagenum, req_type, validate_pages = True, drop_invalid_records = True):
    """
    Extracts all records of type `req_type` from a buffer containing a whole number of pages, usually the full contents of a file. The ID of the type should be specified in the `type_pagenum` argument; this is the 2-byte number that begins every page of the specified type.

    Parameters
    --------
    buf : bytes
        The data to parse

    type_pagenum : int
        The ID of the type

    req_type : numpy.dtype
        A datatype

    validate_pages : bool
        Whether to verify the checksum of each page for data integrity

    drop_invalid_records : bool
        Whether to silently drop invalid records. If `True` (the default), this function will return only the valid records. If `False`, this function will return all records up to and including the last valid frame, and the second return value will contain a boolean array corresponding to the validity of each record

    Returns
    --------
    Tuple[numpy.ndarray[numpy.dtype], np.ndarray]
        Two `numpy` arrays. The first is the array containing the data, with elements being of the requested type. The second is a boolean array specifying the validity of each record

    """
    step_data = np.frombuffer(buf, dtype = NDAX_PAGE_INFO)
    req_type_pages = step_data[step_data["type"] == type_pagenum]
    # check if empty
    if step_data.shape[0] == 0:
        return np.empty((0, ), dtype = req_type), np.empty((0, ), dtype = bool)
    
    # perform page validation if required
    if validate_pages: assert np.all([isPageValid(f) for f in req_type_pages], axis = 0)
    # process data
    page_results = [extractAllFromPage(f, req_type, drop_invalid_records = drop_invalid_records, last_page = False) for f in req_type_pages[: -1]] + [extractAllFromPage(req_type_pages[-1], req_type, drop_invalid_records = drop_invalid_records, last_page = True)]
    # concatenate results
    parsed_contents = np.concatenate([page_result[0] for page_result in page_results], axis = 0)
    validity = np.concatenate([page_result[1] for page_result in page_results], axis = 0)
    return parsed_contents, validity

def fromFile(fileName):
    # the new Neware file format splits steps and records into separate files
    # we must read each one separately and then aggregate the data
    with zipfile.ZipFile(fileName) as zf:
        # process version
        version_tree = ET.ElementTree(ET.fromstring(zf.open("VersionInfo.xml", 'r').read().decode("GB2312")))
        root = version_tree.getroot()
        config = root.find("config")
        assert config.attrib["type"] == "Version Info File", "Bad VersionInfo.xml file"
        zwj = config.find("ZwjVersion")
        if zwj.attrib["ZwjVersion"][: 5] == "4S_8.": ver: int = 8
        elif zwj.attrib["ZwjVersion"][: 5] == "4S_4.": ver: int = 4
        else: assert False, "Unrecognized version"

        # process the header
        header_info = dict()
        header_tree = ET.ElementTree(ET.fromstring(zf.open("TestInfo.xml", 'r').read().decode("GB2312")))
        root = header_tree.getroot()
        config = root.find("config")
        assert config.attrib["type"] == "Test Info File", "Bad TestInfo.xml file"
        header_info["sw_version"] = config.attrib["SortVersion"]
        testInfo_node = config.find("TestInfo")
        header_info["machine"] = int(testInfo_node.attrib["DevID"])
        header_info["hw_version"] = int(testInfo_node.attrib["DevType"])
        header_info["unit"] = int(testInfo_node.attrib["UnitID"])
        header_info["channel"] = int(testInfo_node.attrib["ChlID"])
        header_info["test_id"] = int(testInfo_node.attrib["TestID"])
        start_datetime = datetime.datetime.strptime(testInfo_node.attrib["StartTime"], "%Y-%m-%d %H:%M:%S")
        header_info["date"] = {
            "year": start_datetime.year, 
            "month": start_datetime.month, 
            "day": start_datetime.day, 
        }

        # process the step file
        step_infos = []
        with zf.open("Step.xml", 'r') as g:
            step_tree = ET.parse(g)
            root = step_tree.getroot()
            config = root.find("config")
            assert config.attrib["type"] == "Step File", "Bad Step.xml file"
            step_info_node = config.find("Step_Info")
            step_count = int(step_info_node.attrib["Num"])
            for i in range(step_count):
                # we build this dictionary for each step
                step_info = dict()
                step_id = i + 1
                step_info["Ns"] = step_id
                assert step_info_node[i].tag == "Step%d" % (step_id)
                assert int(step_info_node[i].attrib["Step_ID"]) == step_id
                step_type = int(step_info_node[i].attrib["Step_Type"])
                assert step_type < len(constants.STEP_TYPES) and constants.STEP_TYPES[step_type] is not None, "Unrecognized step type"
                step_info["mode"] = constants.STEP_TYPES[step_type]
                step_limits = step_info_node[i].find("Limit")
                if step_limits is not None:
                    main_limits = step_limits.find("Main")
                    if main_limits is not None:
                        for lim in main_limits:
                            assert lim.tag in NDAX_LIMIT_TYPES, "Unrecognized main limit %s" % (lim.tag)
                            limit_name = NDAX_LIMIT_TYPES[lim.tag]
                            raw_value = float(lim.attrib["Value"])
                            step_info[limit_name] = UNIT_CONVERTERS[limit_name](raw_value) if limit_name in UNIT_CONVERTERS else raw_value

                    other_limits = step_limits.find("Other")
                    if other_limits is not None:
                        for lim in other_limits:
                            assert lim.tag in NDAX_LIMIT_TYPES, "Unrecognized other limit %s" % (lim.tag)
                            limit_name = NDAX_LIMIT_TYPES[lim.tag]
                            step_info[limit_name] = int(lim.attrib["Value"])

                step_infos.append(step_info)

        # read the binary files into their respective formats
        startTime = time.perf_counter()
        if ver == 4: step_data, _ = extractAllFromBuffer(zf.open("data_step.ndc", 'r').read(), NDAX_STEP_INFO_PAGENUM, NDAX_STEP_INFO)
        elif ver == 8: step_data, _ = extractAllFromBuffer(zf.open("data_step.ndc", 'r').read(), NDAX_STEP_INFO_PAGENUM, NDAX8_STEP_INFO)
        data_data, valid_data = extractAllFromBuffer(zf.open("data.ndc", 'r').read(), NDAX_DATA_INFO_PAGENUM, NDAX_DATA_INFO, drop_invalid_records = False)
        if ver == 4: run_data, _ = extractAllFromBuffer(zf.open("data_runInfo.ndc", 'r').read(), NDAX_RUN_INFO_PAGENUM, NDAX_RUN_INFO)
        elif ver == 8: run_data, _ = extractAllFromBuffer(zf.open("data_runInfo.ndc", 'r').read(), NDAX_RUN_INFO_PAGENUM, NDAX8_RUN_INFO)
        
        dataframe = pd.DataFrame(data = data_data, columns = [field[0] for field in NDAX_DATA_INFO_SPEC])
        # voltage should be converted in version 4
        # current is already in mA - no need to convert
        if ver == 4: dataframe["Ewe"] = conv.TMV2V(np.array(dataframe["Ewe"]))
        # for the other tracked variables, we need to interpolate
        # the interpolant is just the record number
        dataframe["record_no"] = np.arange(1, data_data.shape[0] + 1, dtype = int)
        # annoyingly, there can be *skipped records*, which we must remove from the dataframe before continuing
        dataframe.drop(np.argwhere(~valid_data)[: , 0], inplace = True)
        # step time
        step_data["start_time"] = conv.MS2S(step_data["start_time"])
        dataframe["step_time"] = conv.MS2S(np.interp(np.array(dataframe["record_no"]), run_data["record_no"], (run_data["step_time_high"] * (1 << 32)) + run_data["step_time"]))
        # charge
        if ver == 4: dataframe["Q charge"] = conv.MAS2MAH(np.interp(np.array(dataframe["record_no"]), run_data["record_no"], run_data["Q charge"]))
        elif ver == 8: dataframe["Q charge"] = conv.AH2MAH(np.interp(np.array(dataframe["record_no"]), run_data["record_no"], run_data["Q charge"]))
        if ver == 4: dataframe["Q discharge"] = conv.MAS2MAH(np.interp(np.array(dataframe["record_no"]), run_data["record_no"], run_data["Q discharge"]))
        elif ver == 8: dataframe["Q discharge"] = conv.AH2MAH(np.interp(np.array(dataframe["record_no"]), run_data["record_no"], run_data["Q discharge"]))
        # energy
        if ver == 4: dataframe["Energy charge"] = conv.MWS2WH(np.interp(np.array(dataframe["record_no"]), run_data["record_no"], run_data["Energy charge"]))
        elif ver == 8: dataframe["Energy charge"] = np.interp(np.array(dataframe["record_no"]), run_data["record_no"], run_data["Energy charge"])
        if ver == 4: dataframe["Energy discharge"] = conv.MWS2WH(np.interp(np.array(dataframe["record_no"]), run_data["record_no"], run_data["Energy discharge"]))
        elif ver == 8: dataframe["Energy discharge"] = np.interp(np.array(dataframe["record_no"]), run_data["record_no"], run_data["Energy discharge"])
        # locate the step changes to find Ns and half cycle for each record; it will also come in handy for calculating the global time
        assert step_data["start_record_no"][0] == 1, "The first step has a starting reocrd number of %d, which is invalid (it should be 1)." % (step_data["start_record_no"][0])
        data_step_indices = common.convertIndices(np.array(dataframe["record_no"]), step_data["start_record_no"])
        dataframe["Ns"] = step_data["Ns"][data_step_indices]
        # elapsed time relative to beginning of experiment
        dataframe["time"] = step_data["start_time"][data_step_indices] + np.array(dataframe["step_time"])
        # charge relative to beginning of experiment
        dataframe["Q-Q0"] = common.convertStepToGlobal(np.array(dataframe["Q charge"]) - np.array(dataframe["Q discharge"]), step_data["start_record_no"].astype(int) - 1)      #in the "change_idcs" argument, subtract one to get indices instead of record numbers (which are 1-indexed)
        # find half cycle numbers (indices into step_data)
        hc_changes = common.findSegmentChanges(step_data["mode"], constants.HALF_STEP_TRIGGER_STEP_NUMS)
        dataframe["half cycle"] = common.convertIndices(np.arange(step_data.shape[0]), hc_changes)[data_step_indices]
        finishTime = time.perf_counter()
        common.printElapsedTime("Parse", startTime, finishTime)

    return header_info, step_infos, dataframe
