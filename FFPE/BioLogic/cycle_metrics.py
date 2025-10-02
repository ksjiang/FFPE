
"""
# BioLogic Experiment #

Defines `BiologicExperiment`, a class representing experiments recorded on a BioLogic device.
"""

import FFPE.BioLogic.MPRff as MPRff
import FFPE.Util.cycle_tools as cycle_tools

# class representing experiments recorded by Biologic
class BiologicExperiment(object):
    def __init__(self, version = MPRff.ACCEPTED_VERSIONS[-1], hc = 0):
        # software version
        self.version = version
        self.hc = hc
        return
    
    # instantiate fromFile
    def fromFile(self, fileName):
        x, y = MPRff.fromFile(fileName, self.version)
        self.metadata = [x]
        # default to vectorized parsing, because it is FAST!
        self.measurement_sequence = y.getDataFrame(mp = "Vec")
        return
    
    def getCycleDataIdx_hc(self, step_num, half_cycle, include_rest):
        if include_rest:
            indices = self.measurement_sequence[(self.measurement_sequence["Ns"] == step_num) & (self.measurement_sequence["half cycle"] == self.hc + half_cycle)].index
        else:
            indices = self.measurement_sequence[(self.measurement_sequence["Ns"] == step_num) & (self.measurement_sequence["half cycle"] == self.hc + half_cycle) & (self.measurement_sequence["mode"] != MPRff.REST_MODE)].index
            
        return indices
    
    def getCycleData_hc(self, step_num, half_cycle, include_rest):
        return self.measurement_sequence.loc[self.getCycleDataIdx_hc(step_num, half_cycle, include_rest)]
    
    
class BiologicMODE1CyclingExperiment(BiologicExperiment, cycle_tools.MODE1CyclingExperiment):
    def __init__(self, area, version = MPRff.ACCEPTED_VERSIONS[-1], hc = 0):
        BiologicExperiment.__init__(self, version, hc)
        cycle_tools.MODE1CyclingExperiment.__init__(self, area, REST = (0, 0), CYCLE_PLATING = (1, 0), CYCLE_STRIPPING = (2, 1))
        return
    
    
class BiologicPNNLCyclingExperiment(BiologicExperiment, cycle_tools.PNNLCyclingExperiment):
    def __init__(self, area, version = MPRff.ACCEPTED_VERSIONS[-1], hc = 0):
        BiologicExperiment.__init__(self, version, hc)
        cycle_tools.PNNLCyclingExperiment.__init__(self, area, REST = (0, 0), INITIAL_PLATING = (1, 0), INITIAL_STRIPPING = (2, 1), TEST_PLATING = (3, 2), SHORT_CYCLE_STRIPPING = (4, 3), SHORT_CYCLE_PLATING = (5, 4), TEST_STRIPPING = (6, 23), NUM_SHORT_CYCLES = 10)
        return
    
class BiologicFormationCyclingExperiment(BiologicExperiment, cycle_tools.FormationCyclingExperiment):
    def __init__(self, area, num_formation, version = MPRff.ACCEPTED_VERSIONS[-1], hc = 0):
        BiologicExperiment.__init__(self, version, hc)
        cycle_tools.FormationCyclingExperiment.__init__(self, area, REST = (0, 0), FORMATION_CHARGE = (1, 0), FORMATION_DISCHARGE = (3, 1), NUM_FORMATION_CYCLES = num_formation, CYCLE_CHARGE = (6, 2 * num_formation), CYCLE_DISCHARGE = (8, 1 + 2 * num_formation))
        return
    