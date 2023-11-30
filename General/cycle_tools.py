# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:36:28 2023

@author: Kyle
"""

import os
MYDIR = os.path.dirname(__file__)

import numpy as np
import pandas as pd

import sys
sys.path.append(os.path.join(MYDIR, "../General/"))
import common

sec2hr = lambda time_delta: time_delta / 3600.
specCapacity = lambda capacity, area: capacity / area

# common indices to reference first and last datapoints in some ordered series
FIRST, LAST = 0, -1


# custom exception type for incomplete cycling data
class MissingCycleDataException(Exception):
    pass


# base class for galvanostatic cycling data container
class GalvanostaticCyclingExperiment(object):
    def __init__(self, area):
        self.area = area
        self.metadata = None
        self.measurement_sequence = None
        return
    
    # these functions are "utility" functions that don't need an instantiation
    # =============================================================================================
    def capacityDiff(self, series):
        return series.iloc[LAST]["capacity"] - series.iloc[FIRST]["capacity"]
    
    def checkHalfCycle(self, series, throw_error = False):
        ok = False
        if len(series["capacity"]) > 0: ok = True
        if not ok and throw_error: raise MissingCycleDataException
        return ok
    
    def stitchHalfCycles(self, half_cycles, add_breaks = False):
        assert len(half_cycles) > 0, "Empty data"
        # get columns and check that data are homogeneous
        data_columns = half_cycles[FIRST].columns
        for half_cycle in half_cycles[FIRST + 1: ]:
            assert (half_cycle.columns == data_columns).all(), "Unmatching data columns"
            
        if add_breaks:
            # create the break item
            brk = {}
            for data_column in data_columns:
                brk[data_column] = [None]
                
            brk = pd.DataFrame(brk)
            # add breaks between every pair of data series
            half_cycles = common.intersperse(half_cycles, brk)
            
        return pd.concat(half_cycles, ignore_index = True)
    
    # =============================================================================================
    
    def getVvsTAfterStep(self, step_id, relative = True):
        cycleData = self.measurement_sequence.loc[self.measurement_sequence["Ns"] > step_id]
        t, V = np.array(cycleData["time"]), np.array(cycleData["Ewe"])
        t = sec2hr(t)
        if relative and len(t) > 0:
            t = t - t[FIRST]
            
        return pd.DataFrame({
                "time": t, 
                "voltage": V, 
                })
    
    def getCycleData_hc(self, cycle, half_cycle, include_rest):
        raise NotImplementedError
        
    def VvsT_hc(self, cycle, half_cycle, relative = False, include_rest = False):
        cycleData = self.getCycleData_hc(cycle, half_cycle, include_rest = include_rest)
        t, V = np.array(cycleData["time"]), np.array(cycleData["Ewe"])
        t = sec2hr(t)
        if relative and len(t) > 0:
            # calculate time relative to cycle start
            t = t - t[FIRST]
            
        return pd.DataFrame({
                "time": t, 
                "voltage": V, 
                })
    
    def VvsCapacity_hc(self, cycle, half_cycle, relative = True, include_rest = False):
        cycleData = self.getCycleData_hc(cycle, half_cycle, include_rest = include_rest)
        Q, V = np.array(cycleData["Q-Q0"]), np.array(cycleData["Ewe"])
        # compute specific capacity
        Q = specCapacity(Q, self.area)
        if relative and len(Q) > 0:
            Q = np.abs(Q - Q[FIRST])
            
        return pd.DataFrame({
                "capacity": Q, 
                "voltage": V, 
                })
        
        
# class representing constant-capacity cycling experiments
class MODE1CyclingExperiment(GalvanostaticCyclingExperiment):
    def __init__(self, area, REST, CYCLE_PLATING, CYCLE_STRIPPING):
        GalvanostaticCyclingExperiment.__init__(self, area)
        self.REST = REST
        self.CYCLE_PLATING = CYCLE_PLATING
        self.CYCLE_STRIPPING = CYCLE_STRIPPING
        return
    
    def getTimeSeries(self):
        return self.getVvsTAfterStep(self.REST[0])
    
    def calculate_CE(self):
        # we can only calculate CEs up to the last full cycle
        # TODO: there is probably a better way to do this, by extracting the total number of half cycles
        CEs = []
        cyc_num = 0
        while True:
            plating = self.VvsCapacity_hc(self.CYCLE_PLATING[0], self.CYCLE_PLATING[1] + 2 * cyc_num)
            if not self.checkHalfCycle(plating): break
            stripping = self.VvsCapacity_hc(self.CYCLE_STRIPPING[0], self.CYCLE_STRIPPING[1] + 2 * cyc_num)
            if not self.checkHalfCycle(stripping): break
            # calculate CE for this cycle
            CEs.append(self.capacityDiff(stripping) / self.capacityDiff(plating))
            cyc_num += 1
            
        return np.array(CEs)
        
    
# class representing PNNL cycling experiments
class PNNLCyclingExperiment(GalvanostaticCyclingExperiment):
    def __init__(self, area, REST, INITIAL_PLATING, INITIAL_STRIPPING, TEST_PLATING, SHORT_CYCLE_PLATING, SHORT_CYCLE_STRIPPING, TEST_STRIPPING, NUM_SHORT_CYCLES):
        GalvanostaticCyclingExperiment.__init__(self, area)
        self.REST = REST
        self.INITIAL_PLATING = INITIAL_PLATING
        self.INITIAL_STRIPPING = INITIAL_STRIPPING
        self.TEST_PLATING = TEST_PLATING
        self.SHORT_CYCLE_PLATING = SHORT_CYCLE_PLATING
        self.SHORT_CYCLE_STRIPPING = SHORT_CYCLE_STRIPPING
        self.TEST_STRIPPING = TEST_STRIPPING
        self.NUM_SHORT_CYCLES = NUM_SHORT_CYCLES
        return
    
    def getTimeSeries(self):
        return self.getVvsTAfterStep(self.REST[0])
    
    def calculate_CE(self):
        # calculate initial cycle CE
        initial_plating = self.VvsCapacity_hc(*self.INITIAL_PLATING)
        self.checkHalfCycle(initial_plating, throw_error = True)
        initial_stripping = self.VvsCapacity_hc(*self.INITIAL_STRIPPING)
        self.checkHalfCycle(initial_stripping, throw_error = True)
        initialCE = self.capacityDiff(initial_stripping) / self.capacityDiff(initial_plating)
        
        # calculate test cycle CE
        plating_caps, stripping_caps = [], []
        # test plating
        test_plating = self.VvsCapacity_hc(*self.TEST_PLATING)
        self.checkHalfCycle(test_plating, throw_error = True)
        plating_caps.append(self.capacityDiff(test_plating))
        #cycling
        for i in range(self.NUM_SHORT_CYCLES):
            short_plating = self.VvsCapacity_hc(self.SHORT_CYCLE_PLATING[0], self.SHORT_CYCLE_PLATING[1] + 2 * i)
            self.checkHalfCycle(short_plating, throw_error = True)
            plating_caps.append(self.capacityDiff(short_plating))
            short_stripping = self.VvsCapacity_hc(self.SHORT_CYCLE_STRIPPING[0], self.SHORT_CYCLE_STRIPPING[1] + 2 * i)
            self.checkHalfCycle(short_stripping, throw_error = True)
            stripping_caps.append(self.capacityDiff(short_stripping))
            
        # test stripping
        test_stripping = self.VvsCapacity_hc(*self.TEST_STRIPPING)
        self.checkHalfCycle(test_stripping, throw_error = True)
        stripping_caps.append(self.capacityDiff(test_stripping))
        testCE = sum(stripping_caps) / sum(plating_caps)
        return initialCE, testCE
    
    