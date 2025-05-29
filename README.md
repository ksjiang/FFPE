
# VMPff: Fast File Parsing for Electrochemistry

VMPff is a collection of **parsers** for binary data written by electrochemical instruments. Its goal is to automate away the process of exporting data as text or spreadsheets.

## Usage

No installation is needed. Simply `git clone https://github.com/ksjiang/VMPff <PATH TO VMPff>`. If you are interested in exporting the full data in a file to a dataframe, import `<INSTRUMENT>.cycle_metrics` and instantiate an experiment object. For example, to extract the data from a Biologic MPR file, you can use

    import Biologic.cycle_metrics

    experiment = Biologic.cycle_metrics.BiologicExperiment()
    experiment.fromFile("<PATH TO MPR FILE>")
    
Then, the extracted data is available in `experiment.measurement_sequence`.

Usually, it is desirable to interpret the data in some way, depending on the specific experiment it is from. For this, more information about the experiment is needed. Such information can be organized as classes in an instrument-agnostic manner in `General/cycle_tools.py`.

### Galvanostatic Experiments

A base class for galvanostatic cycling experiments as well as child classes for two popular types of galvanostatic tests are defined in `General/cycle_tools.py`. To retrieve the voltage versus time curve for the entire experiment (excluding the initial rest step), use `getTimeSeries()`. To plot the voltage of a cell versus the capacity for a given set of cycles, starmap `VvsCapacity_hc()` over a list containing the desired step and cycle numbers, then apply `stitchHalfCycles()` on that list. Coulombic efficiency metrics can be calculated using `calculate_CE()`. To implement instrument-specific details, such as different cycle-counting schemes, the respective `cycle_metrics.py` under each instrument can be modified. User-accessible classes must inherit from both the experiment class as well as the instrument class (in the example above, `BiologicExperiment()`).
