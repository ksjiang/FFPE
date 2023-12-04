# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:02:41 2023

@author: Kyle
"""

import VMPff
import os

# this directory
MYDIR = os.path.dirname(__file__)

import sys
sys.path.append(os.path.join(MYDIR, "../General/"))
import cycle_tools

# class representing experiments recorded by Biologic
class BiologicExperiment(object):
    def __init__(self, hc = 0):
        self.hc = hc
        return
    
    # instantiate fromFile
    def fromFile(self, fileName):
        x, y = VMPff.fromFile(fileName)
        self.metadata = [x]
        self.measurement_sequence = y.getDataFrame(mp = "Vec")
        return
    
    def getCycleDataIdx_hc(self, cycle, half_cycle, include_rest):
        if include_rest:
            indices = self.measurement_sequence[(self.measurement_sequence["Ns"] == cycle) & (self.measurement_sequence["half cycle"] == self.hc + half_cycle)].index
        else:
            indices = self.measurement_sequence[(self.measurement_sequence["Ns"] == cycle) & (self.measurement_sequence["half cycle"] == self.hc + half_cycle) & (self.measurement_sequence["mode"] != VMPff.REST_MODE)].index
            
        return indices
    
    def getCycleData_hc(self, cycle, half_cycle, include_rest):
        return self.measurement_sequence.loc[self.getCycleDataIdx_hc(cycle, half_cycle, include_rest)]
    
    
class BiologicMODE1CyclingExperiment(BiologicExperiment, cycle_tools.MODE1CyclingExperiment):
    def __init__(self, area, hc = 0):
        BiologicExperiment.__init__(self, hc)
        cycle_tools.MODE1CyclingExperiment.__init__(self, area, REST = (0, 0), CYCLE_PLATING = (1, 0), CYCLE_STRIPPING = (2, 1))
        return
    
    
class BiologicPNNLCyclingExperiment(BiologicExperiment, cycle_tools.PNNLCyclingExperiment):
    def __init__(self, area, hc = 0):
        BiologicExperiment.__init__(self, hc)
        cycle_tools.PNNLCyclingExperiment.__init__(self, area, REST = (0, 0), INITIAL_PLATING = (1, 0), INITIAL_STRIPPING = (2, 1), TEST_PLATING = (3, 2), SHORT_CYCLE_PLATING = (4, 3), SHORT_CYCLE_STRIPPING = (5, 4), TEST_STRIPPING = (6, 23), NUM_SHORT_CYCLES = 10)
        return
    
    