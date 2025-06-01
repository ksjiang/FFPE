
'''
# Unit Converters #

A collection of useful functions for converting between units found on battery testing devices.
'''

# current
UA2MA = lambda x: x * 1E-3

# voltage
TMV2V = lambda x: x * 1E-4

# power
UW2W = lambda x: x * 1E-6
MW2W = lambda x: x * 1E-3

# time
MS2S = lambda x: x * 1E-3
S2HR = lambda x: x / 3600.

# capacity
UAS2MAH = lambda x: S2HR(UA2MA(x))
MAS2MAH = lambda x: S2HR(x)

# energy
UWS2WH = lambda x: S2HR(UW2W(x))
MWS2WH = lambda x: S2HR(MW2W(x))
