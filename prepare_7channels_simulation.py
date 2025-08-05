import matplotlib.pyplot as plt
import numpy as np
import joblib
from utils import describe_dict  # type: ignore

"""# Comparison of multivariate vs. bivariate approach to connectivity estimation
## Exemplary data.
First, let's simulate a multichannel system. We load a fragment of a single-channel EEG containing alpha rhythm and create additional channels by making noisy copies with a time lag.
"""

#!wget www.fuw.edu.pl/~jarekz/HeidelbergSchool/EEEG_alpha.joblib

#load data from matlab EEG_alpha.joblib
data = joblib.load('EEG_alpha.joblib')
describe_dict(data)

Fs = data['Fs']

"""Set the lag between the signals in samples:"""
dt = 1



"""Extract a fragment of the EEG data."""
idx_t0 = 80
N = 2000
x0 = data['EEG'][data['channels']['O1'], idx_t0      : idx_t0    +N+2*dt]
x0 = x0 / np.std(x0) # normalize the signal


"""Create noisy copies of the signal."""

N = x0.shape[0]
A = 1 # amplitude of noise proportional to the amplitude of signal
CHECK_DELAYS = False # if True then insert mmarkers and switch off noise
if CHECK_DELAYS:
    x0*=0
    A = 0 

x0[10] = 5 # This is a marker to visualize the delays between the channels better.
x1 = x0 + A*np.random.randn(N)
if CHECK_DELAYS:
    x1[20] = 5 # This is a marker to visualize the delays between the channels better.
x2 = x1 + A*np.random.randn(N)
x3 = x0 + A*np.random.randn(N)
if CHECK_DELAYS:
    x3[20] = 7 # This is a marker to visualize the delays between the channels better.
x4 = x3 + A*np.random.randn(N)
x5 = x3 + A*np.random.randn(N)
x6 = A*np.random.randn(N)

"""Form a multichannel array - rows represent the channels and introduce delays."""




sim1 = np.vstack((
                    x0[2*dt:     ],
                    x1[1*dt:-1*dt],
                    x2[    :-2*dt],
                    x3[1*dt:-1*dt],
                    x4[    :-2*dt],
                    x5[    :-2*dt],
                    x6[2*dt:     ]
))



N_chan, N_samp = sim1.shape

"""The simulation sim1 follows the scheme:
Let's see the first twenty samples. Observe the marker between the sample 7 and 10 and compare its position with the introduced delays.
"""

idx = np.arange(0,30,1)
Min = np.min(np.min(sim1[:,idx]))
Max = np.max(np.max(sim1[:,idx]))
# create figure with N_chan axes
fig, ax = plt.subplots(N_chan, 1, sharex=True)
for i in range(N_chan):
    ax[i].stem(sim1[i, idx])
    ax[i].set_ylabel(i)
    ax[i].set_ylim([Min, Max])
ax[N_chan-1].set_xlabel('Time [samples]')
plt.show()

# Pack the simulated data into a dictionary for saving
# This is useful for later use.
sim = {'EEG': sim1, 'Fs': Fs, 'channels': {'O': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6}}
joblib.dump(sim, 'simulated_7_channels.joblib')

