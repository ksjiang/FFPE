
"""
# Neware Constants #

A set of constants that are shared among Neware cyclers.
"""

import numpy as np

STEP_TYPES = [
    None, 
    "CC_charge", 
    "CC_discharge", 
    "CV_charge", 
    "Rest", 
    "Loop", 
    "Stop", 
    "CCCV_charge", 
    "CP_discharge", 
    "CP_charge", 
    None, 
    None, 
    "Pause", 
    None, 
    None, 
    "Pulse", 
    "SIM", 
    None, 
    "CV_discharge", 
    "CCCV_discharge", 
    "Control", 
    "OCV", 
    None, 
    None, 
    None, 
    "CPCV_discharge", 
    "CPCV_charge", 
]

DEFAULT_HALF_STEP_TRIGGERS = [
    "CC_charge", 
    "CC_discharge", 
    "CV_charge", 
    "CCCV_charge", 
    "CP_discharge", 
    "CP_charge", 
    "CV_discharge", 
    "CCCV_discharge", 
    "CPCV_discharge", 
    "CPCV_charge", 
]

def findStepChanges(step_series):
    """
    Finds the indices at which new steps occur.

    Parameters
    --------
    step_series : numpy.ndarray
        An array containing step type numbers

    Returns
    --------
    numpy.ndarray
        An array containing the indices of the step changes
    """
    return np.where(np.diff(step_series) != 0)[0]

def convertIndices(arr, change_idcs):
    """
    Computes the step indices of the corresponding global indices in `arr`, where the global indices of step changes are specified in the argument `change_idcs`.

    Parameters
    --------
    arr : numpy.ndarray
        An array containing global indices

    change_idcs : numpy.ndarray
        An array containing the global indices of step changes

    Returns
    --------
    numpy.ndarray
        An array of step indices
    """
    return np.searchsorted(change_idcs, arr, side = "right") - 1

def findHalfCycleChanges(step_series, triggers = None):
    """
    Finds the indices at which new half cycles occur. The names of the step types that trigger a half cycle change should be specified in the `triggers` argument. A "half cycle" is defined as the transition to a current-passing step from any other step, and is so named because it usually constitutes half of a full cycle.
    
    Note that multiple steps with the same current direction (e.g., `CC_charge` followed by a `CV_charge`) that are interrupted by any other step type still constitutes a half cycle change.

    Parameters
    --------
    step_series : numpy.ndarray
        An array containing step type numbers

    triggers : List[str] | None
        A list of step names that should trigger a half cycle change. If `None`, use the default triggers defined in this file

    Returns
    --------
    numpy.ndarray
        An array containing the indices of the half cycle changes
    """
    if triggers is None: triggers = DEFAULT_HALF_STEP_TRIGGERS
    changes = []
    for trigger in triggers:
        trigger_step_type_num = STEP_TYPES.index(trigger)
        hot_indices = np.where(step_series == trigger_step_type_num)[0]
        hot_indices_diff = np.diff(np.concatenate(([-1], hot_indices), axis = 0))
        changes.append(hot_indices[np.where(hot_indices_diff != 1)[0]])

    return np.sort(np.concatenate(changes, axis = 0))
