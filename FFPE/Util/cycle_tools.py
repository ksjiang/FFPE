
"""
# Electrochemical Experiment #

Defines electrochemical experiments.
"""

import numpy as np
import pandas as pd

import FFPE.Util.common as common

sec2hr = lambda time_delta: time_delta / 3600.
specCapacity = lambda capacity, area: capacity / area

# common indices to reference first and last datapoints in some ordered series
FIRST, LAST = 0, -1


# custom exception type for incomplete cycling data
class MissingCycleDataException(Exception):
    pass


# base class for galvanostatic cycling data
class GalvanostaticCyclingExperiment(object):
    def __init__(self, area):
        self.area = area
        # these attributes must be instantiated by an instrument-specific method
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
    
    # selects the prefix of the data where the voltage is below a specified threshold. direction argument
    # refers to whether the threshold is a lower limit (0) or upper limit (1)
    def thresholdIdx(self, series, threshold, direction):
        if direction == 0:
            crossed = np.where(series < threshold)[0]
        elif direction == 1:
            crossed = np.where(series > threshold)[0]
        else:
            assert False, "direction argument should be 0 (lower) or 1 (upper)"
            
        if crossed.shape[0] == 0:
            # if the threshold is never crossed, just return the entire series
            return series.shape[0]
        else:
            return crossed[0]
    
    # =============================================================================================
    
    def getVvsTAfterStep(self, step_id = None, relative = True):
        if step_id is None:
            # return entire sequence
            cycleData = self.measurement_sequence
        else:
            cycleData = self.measurement_sequence.loc[self.measurement_sequence["Ns"] > step_id]
            
        t, V = np.array(cycleData["time"]), np.array(cycleData["Ewe"])
        t = sec2hr(t)
        if relative and len(t) > 0:
            t = t - t[FIRST]
            
        return pd.DataFrame({
                "time": t, 
                "voltage": V, 
                })
    
    # this is NOT DEFINED here. it depends on the instrument used to collect the data
    # so we defer to the child class to inherit this method from an instrument experiment parent class
    def getCycleData_hc(self, step_num, half_cycle, include_rest):
        raise NotImplementedError
        
    def VvsT_hc(self, step_num, half_cycle, relative = False, include_rest = False, Vcutoff = None):
        cycleData = self.getCycleData_hc(step_num, half_cycle, include_rest = include_rest)
        t, V = np.array(cycleData["time"]), np.array(cycleData["Ewe"])
        if Vcutoff is not None:
            firstCrossing = self.thresholdIdx(V, *Vcutoff)
            t, V = t[: firstCrossing], V[: firstCrossing]
            
        t = sec2hr(t)
        if relative and len(t) > 0:
            # calculate time relative to cycle start (should always be positive)
            t = t - t[FIRST]
            
        return pd.DataFrame({
                "time": t, 
                "voltage": V, 
                })
    
    def VvsCapacity_hc(self, step_num, half_cycle, relative = True, rectify = True, include_rest = False, Vcutoff = None):
        cycleData = self.getCycleData_hc(step_num, half_cycle, include_rest = include_rest)
        Q, V = np.array(cycleData["Q-Q0"]), np.array(cycleData["Ewe"])
        if Vcutoff is not None:
            firstCrossing = self.thresholdIdx(V, *Vcutoff)
            Q, V = Q[: firstCrossing], V[: firstCrossing]
            
        # compute specific capacity
        Q = specCapacity(Q, self.area)
        if len(Q) > 0:
            if relative: Q = Q - Q[FIRST]
            if rectify: Q = np.abs(Q)
            
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
    
    # default behavior is to get all data after the rest step
    def getTimeSeries(self, include_rest = False):
        si = self.REST[0] if not include_rest else None
        return self.getVvsTAfterStep(step_id = si)
    
    # definition of Coulombic efficiency for constant-capacity cycling
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
    
    def calculate_capacities(self, num_cycles):
        plating_caps = []
        stripping_caps = []
        for i in range(num_cycles):
            plating = self.VvsCapacity_hc(self.CYCLE_PLATING[0], self.CYCLE_PLATING[1] + 2 * i)
            if not self.checkHalfCycle(plating): assert False, "Incomplete experiment"
            plating_caps.append(self.capacityDiff(plating))
            stripping = self.VvsCapacity_hc(self.CYCLE_STRIPPING[0], self.CYCLE_STRIPPING[1] + 2 * i)
            if not self.checkHalfCycle(stripping): assert False, "Incomplete experiment"
            stripping_caps.append(self.capacityDiff(stripping))

        return np.array(plating_caps), np.array(stripping_caps)
    
    def calculate_capacity_loss(self, num_cycles):
        plating_caps, stripping_caps = self.calculate_capacities(num_cycles)
        return np.sum(plating_caps) - np.sum(stripping_caps)
    
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
    
    def getTimeSeries(self, include_rest = False):
        si = self.REST[0] if not include_rest else None
        return self.getVvsTAfterStep(step_id = si)
    
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
    
    
class FormationCyclingExperiment(GalvanostaticCyclingExperiment):
    def __init__(self, area, REST, FORMATION_CHARGE, FORMATION_DISCHARGE, NUM_FORMATION_CYCLES, CYCLE_CHARGE, CYCLE_DISCHARGE):
        GalvanostaticCyclingExperiment.__init__(self, area)
        self.REST = REST
        self.FORMATION_CHARGE = FORMATION_CHARGE
        self.FORMATION_DISCHARGE = FORMATION_DISCHARGE
        self.NUM_FORMATION_CYCLES = NUM_FORMATION_CYCLES
        self.CYCLE_CHARGE = CYCLE_CHARGE
        self.CYCLE_DISCHARGE = CYCLE_DISCHARGE
        return
    
    def getTimeSeries(self, include_rest = False):
        si = self.REST[0] if not include_rest else None
        return self.getVvsTAfterStep(step_id = si)
    
    def process_formation_cycles(self):
        """
        Extracts and processes information pertaining to the formation cycles.

        Parameters
        --------

        Returns
        --------
        pandas.Series
            A series containing 5 elements, described below.

                `formation_cycles_separate` : pandas.DataFrame
                    The voltage-vs-capacity dataframes for each half formation cycle, separated

                `formation_cycles_stitched` : pandas.DataFrame
                    The voltage-vs-capacity dataframes for each full formation cycle, with a break added between charge and discharge half cycles

                `formation_charge_cap` : numpy.ndarray
                    The charge capacity for each formation cycle

                `formation_discharge_cap` : numpy.ndarray
                    The discharge capacity for each formation cycle

                `formation_CE` : np.ndarray
                    The Coulombic efficiency for each formation cycle

        """
        formation_cycles_separate = [
            [
                self.VvsCapacity_hc(
                    self.FORMATION_CHARGE[0], self.FORMATION_CHARGE[1] + 2 * i
                ), 
                self.VvsCapacity_hc(
                    self.FORMATION_DISCHARGE[0], self.FORMATION_DISCHARGE[1] + 2 * i
                )
            ] for i in range(self.NUM_FORMATION_CYCLES)
        ]
        formation_cycles_stitched = [
            self.stitchHalfCycles(formation_cycles_separate[i], add_breaks = True) for i in range(self.NUM_FORMATION_CYCLES)
        ]
        formation_charge_cap = np.array([self.capacityDiff(cycle[0]) if cycle[0]["capacity"].shape[0] > 0 else np.NaN for cycle in formation_cycles_separate])
        formation_discharge_cap = np.array([self.capacityDiff(cycle[1]) if cycle[1]["capacity"].shape[0] > 0 else np.NaN for cycle in formation_cycles_separate])
        formation_CE = formation_discharge_cap / formation_charge_cap
        return formation_cycles_separate, formation_cycles_stitched, formation_charge_cap, formation_discharge_cap, formation_CE
    
    def process_test_cycles(self):
        """
        Extracts and processes information pertaining to the test cycles.

        Parameters
        --------

        Returns
        --------
        pandas.Series
            A series containing 5 elements, described below.

                `test_cycles_separate` : pandas.DataFrame
                    The voltage-vs-capacity dataframes for each half test cycle, separated

                `test_cycles_stitched` : pandas.DataFrame
                    The voltage-vs-capacity dataframes for each full test cycle, with a break added between charge and discharge half cycles

                `test_charge_cap` : numpy.ndarray
                    The charge capacity for each test cycle

                `test_discharge_cap` : numpy.ndarray
                    The discharge capacity for each test cycle

                `test_CE` : np.ndarray
                    The Coulombic efficiency for each test cycle

        """
        test_cycles_separate = []
        cyc_num = 0
        while True:
            test_cycle_charge = self.VvsCapacity_hc(self.CYCLE_CHARGE[0], self.CYCLE_CHARGE[1] + 2 * cyc_num)
            if not self.checkHalfCycle(test_cycle_charge): break
            test_cycle_discharge = self.VvsCapacity_hc(self.CYCLE_DISCHARGE[0], self.CYCLE_DISCHARGE[1] + 2 * cyc_num)
            if not self.checkHalfCycle(test_cycle_discharge): break
            test_cycles_separate.append([test_cycle_charge, test_cycle_discharge])
            cyc_num += 1
            
        test_cycles_stitched = [
            self.stitchHalfCycles(test_cycles_separate[i], add_breaks = True) for i in range(len(test_cycles_separate))
        ]
        test_charge_cap = np.array([self.capacityDiff(cycle[0]) for cycle in test_cycles_separate])
        test_discharge_cap = np.array([self.capacityDiff(cycle[1]) for cycle in test_cycles_separate])
        test_CE = test_discharge_cap / test_charge_cap
        return test_cycles_separate, test_cycles_stitched, test_charge_cap, test_discharge_cap, test_CE
    