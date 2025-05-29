
import FFPE.Neware.NDAff as NDAff
import FFPE.Util.cycle_tools as cycle_tools

# class representing experiments recorded by MTI Cycler
class MTICycExperiment(object):
    def __init__(self):
        return
    
    # instantiate fromFile
    def fromFile(self, fileName):
        x1, x2, Y = NDAff.fromFile(fileName)
        self.metadata = [x1, x2]
        self.measurement_sequence = Y
        return
    
    def getCycleDataIdx_hc(self, cycle, half_cycle, include_rest):
        if include_rest:
            indices = self.measurement_sequence[((self.measurement_sequence["Ns"] == cycle) | (self.measurement_sequence["Ns"] == cycle + 1)) & (self.measurement_sequence["half cycle"] == half_cycle)].index
        else:
            indices = self.measurement_sequence[(self.measurement_sequence["Ns"] == cycle) & (self.measurement_sequence["half cycle"] == half_cycle)].index
            
        return indices
    
    def getCycleData_hc(self, cycle, half_cycle, include_rest):
        return self.measurement_sequence.loc[self.getCycleDataIdx_hc(cycle, half_cycle, include_rest)]
    
    
class MTICycMODE1CyclingExperiment(MTICycExperiment, cycle_tools.MODE1CyclingExperiment):
    def __init__(self, area):
        MTICycExperiment.__init__(self)
        cycle_tools.MODE1CyclingExperiment.__init__(self, area, REST = (1, 0), CYCLE_PLATING = (2, 1), CYCLE_STRIPPING = (4, 2))
        return
    
    
class MTICycPNNLCyclingExperiment(MTICycExperiment, cycle_tools.PNNLCyclingExperiment):
    def __init__(self, area):
        MTICycExperiment.__init__(self)
        cycle_tools.PNNLCyclingExperiment.__init__(self, area, REST = (1, 0), INITIAL_PLATING = (2, 1), INITIAL_STRIPPING = (4, 2), TEST_PLATING = (6, 3), SHORT_CYCLE_STRIPPING = (8, 4), SHORT_CYCLE_PLATING = (10, 5), TEST_STRIPPING = (13, 24), NUM_SHORT_CYCLES = 10)
        return
    
    