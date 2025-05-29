
# FFPE: Fast File Parsing for Electrochemistry

FFPE is a collection of **parsers** for binary data written by electrochemical instruments. Its goal is to automate away the process of exporting data as text or spreadsheets without sacrificing ease of data analysis.

## Installation

Simply run `pip install git+https://github.com/ksjiang/FFPE.git@v0.0.1`.

## Usage

If you are interested in exporting the full data in a file to a dataframe, import `FFPE.<INSTRUMENT>.cycle_metrics` and instantiate an experiment object. For example, to extract the data from a BioLogic MPR file, you can use

    import FFPE.BioLogic.cycle_metrics

    experiment = FFPE.BioLogic.cycle_metrics.BiologicExperiment(<BioLogic software version>)
    experiment.fromFile("<PATH TO MPR FILE>")
    
Then, the extracted data is available in `experiment.measurement_sequence`.

Usually, it is desirable to interpret the data in some way, depending on the specific experiment it is from. For this, more information about the experiment is needed. Such information can be organized as classes in an instrument-agnostic manner in `FFPE/Util/cycle_tools.py`.

### Galvanostatic Experiments

A base class for galvanostatic cycling experiments as well as child classes for two popular types of galvanostatic tests are defined in `FFPE/Util/cycle_tools.py`. To retrieve the voltage versus time curve for the entire experiment (excluding the initial rest step), use `getTimeSeries()`. To plot the voltage of a cell versus the capacity for a given set of cycles, starmap `VvsCapacity_hc()` over a list containing the desired step and cycle numbers, then apply `stitchHalfCycles()` on that list. Coulombic efficiency metrics can be calculated using `calculate_CE()`. To implement instrument-specific details, such as different cycle-counting schemes, the respective `cycle_metrics.py` under each instrument can be modified. User-accessible classes must inherit from both the experiment class as well as the instrument class (in the example above, `BiologicExperiment()`).
