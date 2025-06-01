
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

NDAX_PAGE_BITMAP_SZ = 128
NDAX_PAGE_DATA_SZ = 3960
NDAX_PAGE_INFO_SPEC = [
    ("type", np.uint16), 
    ("count", np.uint16), 
    ("bitmap", np.dtype((np.uint8, NDAX_PAGE_BITMAP_SZ))), 
    ("data", np.dtype((np.uint8, NDAX_PAGE_DATA_SZ))), 
    ("crc32", np.uint32), 
]

NDAX_DATA_INFO_SPEC = [
    ("Ewe", np.float32), 
    ("current", np.float32), 
]

NDAX_STEP_INFO_SPEC = [
    ("cycle number", np.uint32), 
    ("Ns", np.uint32), 
    ("padding1", np.dtype((np.uint8, 16))), 
    ("mode", np.uint8), 
    ("start_time", np.uint32), 
    ("padding2", np.uint32), 
    ("start_record_no", np.uint32), 
]

NDAX_RUN_INFO_SPEC = [
    ("step_time", np.uint32),               #ms
    ("padding1", np.uint8),                 #always zero
    ("Q charge", np.float32),               #uAs
    ("Q discharge", np.float32),            #uAs
    ("Energy charge", np.float32),          #uWs
    ("Energy discharge", np.float32),       #uWs
    ("unk1", np.float32),                   #unknown field: always 1.0
    ("unk2", np.uint32),                    #unknown field: always 16
    ("delta t", np.uint32),                 #ms
    ("clock_time", np.uint32),              #Unix time
    ("step_no", np.uint32), 
    ("record_no", np.uint32), 
    ("unk3", np.uint16),                    #unknown field: varies in general, but constant for all records in the same step
]

NDAX_PAGE_INFO = np.dtype(NDAX_PAGE_INFO_SPEC)
NDAX_STEP_INFO = np.dtype(NDAX_STEP_INFO_SPEC)
NDAX_DATA_INFO = np.dtype(NDAX_DATA_INFO_SPEC)
NDAX_RUN_INFO = np.dtype(NDAX_RUN_INFO_SPEC)
NDAX_LIMIT_TYPES = {
    "Time": "time", 
    "Curr": "current", 
    "Stop_Volt": "voltage", 
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

def extractAllFromPage(f, req_type):
    """
    Extracts all records of type `req_type` from a page. This just pulls the data from the page; no guarantees are made about the validity of the page, which should be checked by the caller.

    Parameters
    --------
    f : NDAX_PAGE_INFO
        The page to parse

    req_type : numpy.dtype
        A datatype

    Returns
    --------
    numpy.ndarray[numpy.dtype]
        An array of the requested type

    """
    num_items = NDAX_PAGE_DATA_SZ // req_type.itemsize
    data = np.frombuffer(f["data"], dtype = req_type, count = num_items)
    # not all of the data is valid. only return the ones indicated by the bitmap
    valid = np.unpackbits(f["bitmap"], bitorder = "little")[: num_items]
    # check that the number of set bits matches the count
    assert np.sum(valid) == f["count"]
    return data[valid.astype("bool")]

def extractAllFromBuffer(buf, type_pagenum, req_type, validate_pages = True):
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

    Returns
    --------
    numpy.ndarray[numpy.dtype]
        An array of the requested type

    """
    step_data = np.frombuffer(buf, dtype = NDAX_PAGE_INFO)
    req_type_pages = step_data[step_data["type"] == type_pagenum]
    if validate_pages: assert np.all([isPageValid(f) for f in req_type_pages], axis = 0)
    parsed_contents = np.concatenate([extractAllFromPage(f, req_type) for f in req_type_pages], axis = 0)
    return parsed_contents

def fromFile(fileName):
    # the new Neware file format splits steps and records into separate files
    # we must read each one separately and then aggregate the data
    with zipfile.ZipFile(fileName) as zf:
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
        step_data = extractAllFromBuffer(zf.open("data_step.ndc", 'r').read(), NDAX_STEP_INFO_PAGENUM, NDAX_STEP_INFO)
        startTime = time.perf_counter()
        data_data = extractAllFromBuffer(zf.open("data.ndc", 'r').read(), NDAX_DATA_INFO_PAGENUM, NDAX_DATA_INFO)
        run_data = extractAllFromBuffer(zf.open("data_runInfo.ndc", 'r').read(), NDAX_RUN_INFO_PAGENUM, NDAX_RUN_INFO)
        
        dataframe = pd.DataFrame(data = data_data, columns = [field[0] for field in NDAX_DATA_INFO_SPEC])
        # voltage should be converted
        # current is already in mA - no need to convert
        dataframe["Ewe"] = conv.TMV2V(dataframe["Ewe"])
        # for the other tracked variables, we need to interpolate
        # the interpolant is just the record number
        dataframe["record_no"] = np.arange(1, data_data.shape[0] + 1, dtype = int)
        # step time
        step_data["start_time"] = conv.MS2S(step_data["start_time"])
        dataframe["step_time"] = conv.MS2S(np.interp(dataframe["record_no"], run_data["record_no"], run_data["step_time"]))
        # charge
        dataframe["Q charge"] = conv.MAS2MAH(np.interp(dataframe["record_no"], run_data["record_no"], run_data["Q charge"]))
        dataframe["Q discharge"] = conv.MAS2MAH(np.interp(dataframe["record_no"], run_data["record_no"], run_data["Q discharge"]))
        # energy
        dataframe["Energy charge"] = conv.MWS2WH(np.interp(dataframe["record_no"], run_data["record_no"], run_data["Energy charge"]))
        dataframe["Energy discharge"] = conv.MWS2WH(np.interp(dataframe["record_no"], run_data["record_no"], run_data["Energy discharge"]))
        # locate the step changes to find Ns and half cycle for each record; it will also come in handy for calculating the global time
        assert step_data["start_record_no"][0] == 1, "There are records that do not belong to any step"
        data_step_indices = constants.convertIndices(dataframe["record_no"], step_data["start_record_no"])
        dataframe["Ns"] = step_data["Ns"][data_step_indices]
        # global elapsed time relative to beginning of experiment
        dataframe["time"] = step_data["start_time"][data_step_indices] + dataframe["step_time"]
        # the step information does not include an equivalent "start charge", but we can easily calculate this by looking at the record just before the step
        # if the index is -1, it means the step is the beginning of the experiment
        # the conversion to type `int` is necessary because otherwise we will get a large positive index instead of -1
        prestep_indices = step_data["start_record_no"].astype(int) - 2
        qnets = dataframe["Q charge"] - dataframe["Q discharge"]
        final_qnets = np.zeros(step_data.shape[0])
        final_qnets[prestep_indices >= 0] = qnets[prestep_indices[prestep_indices >= 0]]
        start_qq0 = np.cumsum(final_qnets)
        dataframe["Q-Q0"] = start_qq0[data_step_indices] + qnets
        # find half cycle numbers (indices into step_data)
        hc_changes = np.concatenate(([-1], constants.findHalfCycleChanges(step_data["mode"])), axis = 0)
        dataframe["half cycle"] = constants.convertIndices(np.arange(step_data.shape[0]), hc_changes)[data_step_indices]
        finishTime = time.perf_counter()
        common.printElapsedTime("Parse", startTime, finishTime)

    return header_info, step_infos, dataframe
