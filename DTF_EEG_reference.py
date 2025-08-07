


import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import joblib

#! pip install networkx

import networkx as nx
from utils import describe_dict


# Local application/library imports
from mtmvar import mvar_plot_dense, DTF_multivariate, multivariate_spectra # type: ignore
# Load the EEG data

data = joblib.load('EEG_alpha.joblib')
print('data contents:')
describe_dict(data)
N_chan = data['EEG'].shape[0]
Fs = data['Fs'].item()
print('Sampling frequency: ', Fs)
print('Number of channels: ', N_chan)
print('Channel names: ', data['ChanNames'])


"""## We estimate the model order for all channels at once. Note the optimal order printed below.
"""
f = np.arange(1,30, 0.1)
N_f = f.shape[0]
max_p = 25
p_opt = None
crit_type ='AIC'
S_multivariate = multivariate_spectra(data['EEG'], f, Fs, max_p, p_opt = p_opt, crit_type = crit_type)
DTF = DTF_multivariate(data['EEG'], f, Fs, max_p, p_opt = p_opt, crit_type = crit_type)
mvar_plot_dense(S_multivariate, DTF,   f, 'From ', 'To ', data['ChanNames'],'Multivariate DTF - linked ears reference' ,'sqrt')
#plt.show()

# check the effects of reference
# change the reference to common average
data_car = data['EEG'] - np.mean(data['EEG'], axis=0, keepdims=True) 
# after this operation data_car has reduced degrees of freedom - we need to exclude one channel from the analysis
# the best candidate is T3 as in the previous analysis it din't show any significant connectivity
# we will replace T3 with random noise
data_car[data['ChanNames'].index('T3'),:] = np.random.randn(*data_car[data['ChanNames'].index('T3'),:].shape) # replace T3 with random noise
S_multivariate_CAR = multivariate_spectra(data_car, f, Fs, max_p, p_opt = p_opt, crit_type = crit_type)
DTF_CAR = DTF_multivariate(data_car, f, Fs, max_p, p_opt = p_opt, crit_type = crit_type)
mvar_plot_dense(S_multivariate_CAR, DTF_CAR,   f, 'From ', 'To ', data['ChanNames'],'Multivariate DTF - CAR reference' ,'sqrt')
plt.show()