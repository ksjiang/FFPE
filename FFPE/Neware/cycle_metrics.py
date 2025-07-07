
"""
# Neware Experiment #

Defines `NewareExperiment`, a class representing experiments recorded on Neware cyclers.
"""

import os

import FFPE.Neware.NDAff as NDAff
import FFPE.Neware.NDAXff as NDAXff
import FFPE.Util.cycle_tools as cycle_tools

class NewareExperiment(object):
    def __init__(self):
        return
    
    # instantiate fromFile
    def fromFile(self, fileName):
        fileName_extension = os.path.splitext(fileName)[1]
        if fileName_extension == ".nda":
            x1, x2, Y = NDAff.fromFile(fileName)
        elif fileName_extension == ".ndax":
            x1, x2, Y = NDAXff.fromFile(fileName)
        else: assert False, "Unknown file extension %s" % (fileName_extension)
        self.metadata = [x1, x2]
        self.measurement_sequence = Y
        return
    
    def getCycleDataIdx_hc(self, step_num, half_cycle, include_rest):
        if include_rest:
            indices = self.measurement_sequence[((self.measurement_sequence["Ns"] == step_num) | (self.measurement_sequence["Ns"] == step_num + 1)) & (self.measurement_sequence["half cycle"] == half_cycle)].index
        else:
            indices = self.measurement_sequence[(self.measurement_sequence["Ns"] == step_num) & (self.measurement_sequence["half cycle"] == half_cycle)].index
            
        return indices
    
    def getCycleData_hc(self, step_num, half_cycle, include_rest):
        return self.measurement_sequence.loc[self.getCycleDataIdx_hc(step_num, half_cycle, include_rest)]
    
    
class NewareMODE1CyclingExperiment(NewareExperiment, cycle_tools.MODE1CyclingExperiment):
    def __init__(self, area):
        NewareExperiment.__init__(self)
        cycle_tools.MODE1CyclingExperiment.__init__(self, area, REST = (1, 0), CYCLE_PLATING = (2, 1), CYCLE_STRIPPING = (4, 2))
        return
    
    
class NewarePNNLCyclingExperiment(NewareExperiment, cycle_tools.PNNLCyclingExperiment):
    def __init__(self, area):
        NewareExperiment.__init__(self)
        cycle_tools.PNNLCyclingExperiment.__init__(self, area, REST = (1, 0), INITIAL_PLATING = (2, 1), INITIAL_STRIPPING = (4, 2), TEST_PLATING = (6, 3), SHORT_CYCLE_STRIPPING = (8, 4), SHORT_CYCLE_PLATING = (10, 5), TEST_STRIPPING = (13, 24), NUM_SHORT_CYCLES = 10)
        return
    

class NewareFormationCyclingExperiment(NewareExperiment, cycle_tools.FormationCyclingExperiment):
    def __init__(self, area, num_formation):
        NewareExperiment.__init__(self)
        cycle_tools.FormationCyclingExperiment.__init__(self, area, REST = (1, 0), FORMATION_CHARGE = (2, 1), FORMATION_DISCHARGE = (4, 2), NUM_FORMATION_CYCLES = num_formation, CYCLE_CHARGE = (7, 1 + 2 * num_formation), CYCLE_DISCHARGE = (9, 2 + 2 * num_formation))
        return
    
    