
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

HALF_STEP_TRIGGER_STEPS = [
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

CHARGE_STEPS = [
    "CC_charge", 
    "CV_charge", 
    "CCCV_charge", 
    "CP_charge", 
    "CPCV_charge", 
]

DISCHARGE_STEPS = [
    "CC_discharge", 
    "CP_discharge", 
    "CV_discharge", 
    "CCCV_discharge", 
    "CPCV_discharge", 
]

HALF_STEP_TRIGGER_STEP_NUMS = [STEP_TYPES.index(s) for s in HALF_STEP_TRIGGER_STEPS]
CHARGE_STEP_NUMS = [STEP_TYPES.index(s) for s in CHARGE_STEPS]
DISCHARGE_STEP_NUMS = [STEP_TYPES.index(s) for s in DISCHARGE_STEPS]
