import numpy as np
import pandas as pd
import scipy.signal as signal
import matplotlib.pyplot as plt

def countCorr(x, ip, iwhat):
    '''
    Internal procedure 
    '''
    sdt = np.shape(x)
    m = sdt[0]
    n = sdt[1]
    if len(sdt) > 2:
        trials = sdt[2]
    else:
        trials = 1
    mip = m * ip
    rleft = np.zeros((mip, mip))
    rright = np.zeros((mip, m))
    r = np.zeros((m, m))
    rleft_tot = np.zeros((mip, mip))
    rright_tot = np.zeros((mip, m))
    r_tot = np.zeros((m, m))

    for trial in range(trials):
        for k in range(ip):
            if iwhat == 1:
                corrscale = 1 / n
                nn = n - k - 1
                r[:, :] = np.dot(x[:, :nn, trial], x[:, k + 1:k + nn + 1, trial].T) * corrscale
            elif iwhat == 2:
                corrscale = 1 / (n - k)
                nn = n - k - 1
                r[:, :] = np.dot(x[:, :nn, trial], x[:, k + 1:k + nn + 1, trial].T) * corrscale
        
            rright[k * m:k * m + m, :] = r[:, :]
        
            if k < ip:
                for i in range(1, ip - k):
                    rleft[(k + i) * m:(k + i) * m + m, (i - 1) * m:(i - 1) * m + m] = r[:, :]
                    rleft[(i - 1) * m:(i - 1) * m + m, (k + i) * m:(k + i) * m + m] = r[:, :].T
        
        corrscale = 1 / n
        r[:, :] = np.dot(x[:, :, trial], x[:, :, trial].T) * corrscale
    
        for k in range(ip):
            rleft[k * m:k * m + m, k * m:k * m + m] = r[:, :]
    
        rleft_tot = rleft_tot + rleft
        rright_tot = rright_tot + rright
        r_tot = r_tot + r

    if trials > 1:
        rleft_tot = rleft_tot / trials
        rright_tot = rright_tot / trials
        r_tot = r_tot / trials

    return rleft_tot, rright_tot, r_tot

def AR_coeff(dat0, p=5, method=1):
    '''
    Estimates AR model coefficients
    for multivariate/multitrial data.

    dat - input time series of shape (trials,chans,timepoints)
    p - model order
    method - unused

    returns:
    ARr - model coefficients of size (p,chans,chans)
    Vr - residual data variance matrix
    '''
    
    ds = dat0.shape
    if len(ds) < 3:
        dat = np.zeros((ds[0], ds[1], 1))
        dat[:, :, 0] = dat0[:, :]
    else:
        dat = dat0
    chans = ds[0]

    rleft, rright, r = countCorr(dat, p, 1)
    xres = np.linalg.solve(rleft, rright)  
    xres= xres.T  
    Vr = -np.dot(xres, rright) + r
    AR = np.reshape(xres, ( chans, p,chans)) #(chans, p, chans))# ( chans, chans,p,)) #
    AR = AR.transpose((0, 2, 1))
    return AR, Vr


def mvar_H(Ar, f, Fs):
    """
    Calculate the transfer function H from multivariate autoregressive coefficients Ar.
    
    Parameters:
    Ar : numpy.ndarray
        AR coefficient matrix with shape (chan, chan, p), where p is model order.
    f : numpy.ndarray
        Frequency vector.
    Fs : float
        Sampling frequency.

    Returns:
    H : numpy.ndarray
        Transfer function matrix with shape (chan, chan, len(f)).
    A_out : numpy.ndarray
        Frequency-dependent AR matrix with shape (chan, chan, len(f)).
    """
    p = Ar.shape[2]
    Nf = len(f)
    chan = Ar.shape[0]
    
    H = np.zeros((chan, chan, Nf), dtype=complex)
    A_out = np.zeros((chan, chan, Nf), dtype=complex)

    z = np.zeros((p, Nf), dtype=complex)
    for m in range(1, p + 1):
        z[m-1, :] = np.exp(-m * 2 * np.pi * 1j * f / Fs)

    for fi in range(Nf):
        A = np.eye(chan, dtype=complex)
        for m in range(p):
            A -= Ar[:, :,m] * z[m, fi].item()
        H[:, :, fi] = np.linalg.inv(A)
        A_out[:, :, fi] = A

    return H, A_out





def mvar_plot(onDiag, offDiag, f, xlab, ylab, ChanNames, Top_title, scale='linear'):
    """
    Plot MVAR results using bar plots for diagonal (auto) and off-diagonal (cross) terms.

    Parameters:
    onDiag : np.ndarray
        Auto components (shape: N_chan x N_chan x len(f))
    offDiag : np.ndarray
        Cross components (shape: N_chan x N_chan x len(f))
    f : np.ndarray
        Frequency vector
    xlab : str
        Label for x-axis
    ylab : str
        Label for y-axis
    ChanNames : list of str
        Names of channels
    Top_title : str
        Main plot title
    scale : str
        'linear', 'sqrt', or 'log'
    """
    onDiag = np.abs(onDiag)
    offDiag = np.abs(offDiag)

    if scale == 'sqrt':
        onDiag = np.sqrt(onDiag)
        offDiag = np.sqrt(offDiag)
    elif scale == 'log':
        onDiag = np.log(onDiag + 1e-12)  # Avoid log(0)
        offDiag = np.log(offDiag + 1e-12)

    N_chan = onDiag.shape[0]

    # Zero-out irrelevant parts
    for i in range(N_chan):
        for j in range(N_chan):
            if i != j:
                onDiag[i, j, :] = 0
            else:
                offDiag[i, i, :] = 0

    MaxonDiag = np.max(onDiag)
    MaxoffDiag = np.max(offDiag)

    fig, axs = plt.subplots(N_chan, N_chan, figsize=(10,10), constrained_layout=True)
    for i in range(N_chan):
        for j in range(N_chan):
            ax = axs[i, j] if N_chan > 1 else axs
            if i != j:
                y = np.real(offDiag[i, j, :])
                ax.plot(f, offDiag[i, j, :])
                ax.fill_between(f, y, 0, color='skyblue', alpha=0.4)
                ax.set_ylim([0, MaxoffDiag])
            else:
                y = np.real(onDiag[i, j, :])
                ax.plot(f, y, color=[0.7, 0.7, 0.7])
                ax.fill_between(f, y, 0, color=[0.7, 0.7, 0.7], alpha=0.4)
                ax.set_ylim([0, MaxonDiag])

            if i == N_chan - 1:
                ax.set_xlabel(f"{xlab}{ChanNames[j]}")
            if j == 0:
                ax.set_ylabel(f"{ylab}{ChanNames[i]}")

    axs[0, 0].set_title(Top_title)
    plt.tight_layout()

def mvar_plot_dense(onDiag, offDiag, f, xlab, ylab,ChanNames , Top_title, scale='linear'):
    """
    Plot MVAR results for diagonal (auto) and off-diagonal (cross) terms.

    Parameters:
    onDiag : np.ndarray
        Auto components (shape: N_chan x N_chan x len(f))
    offDiag : np.ndarray
        Cross components (shape: N_chan x N_chan x len(f))
    f : np.ndarray
        Frequency vector
    xlab : str
        Label for x-axis
    ylab : str
        Label for y-axis
    ChanNames : list of str
        Names of channels
    Top_title : str
        Main plot title
    scale : str
        'linear', 'sqrt', or 'log'
    """
    import numpy as np
    import matplotlib.pyplot as plt

    onDiag = np.abs(onDiag)
    offDiag = np.abs(offDiag)

    if scale == 'sqrt':
        onDiag = np.sqrt(onDiag)
        offDiag = np.sqrt(offDiag)
    elif scale == 'log':
        onDiag = np.log(onDiag + 1e-12)  # Avoid log(0)
        offDiag = np.log(offDiag + 1e-12)

    N_chan = onDiag.shape[0]

    # Zero-out irrelevant parts
    for i in range(N_chan):
        for j in range(N_chan):
            if i != j:
                onDiag[i, j, :] = 0
            else:
                offDiag[i, i, :] = 0

    MaxonDiag = np.max(onDiag)
    MaxoffDiag = np.max(offDiag)

    fig, axs = plt.subplots(N_chan, N_chan, figsize=(8,8),
                            gridspec_kw={'wspace': 0, 'hspace': 0})

    for i in range(N_chan):
        for j in range(N_chan):
            ax = axs[i, j] if N_chan > 1 else axs
            if i != j:
                y = np.real(offDiag[i, j, :])
                ax.plot(f, offDiag[i, j, :])
                ax.fill_between(f, y, 0, color='skyblue', alpha=0.4)
                ax.set_ylim([0, MaxoffDiag])
                ax.set_yticks([0, MaxoffDiag // 2])
            else:
                y = np.real(onDiag[i, j, :])
                ax.plot(f, y, color=[0.7, 0.7, 0.7])
                ax.fill_between(f, y, 0, color=[0.7, 0.7, 0.7], alpha=0.4)
                ax.set_ylim([0, MaxonDiag])
                ax.set_yticks([0, MaxonDiag // 2])

            ax.set_xticks([f[0], int(f[len(f)//2]) ])
            ax.tick_params(labelleft=(j == 0), labelbottom=(i == N_chan - 1))

            if i == N_chan - 1:
                ax.set_xlabel(f"{xlab}{ ChanNames[j]}")
            if j == 0:
                ax.set_ylabel(f"{ylab}{ChanNames[i]}")

    if N_chan > 1:
        axs[0, 0].set_title(Top_title)
    else:
        axs.set_title(Top_title)
    plt.tight_layout()

def mvar_criterion(dat, max_p, crit_type='AIC', do_plot=False):
    """
    Compute model order selection criteria (AIC, HQ, SC) for MVAR modeling.

    Parameters:
    dat : np.ndarray
        Input data of shape (channels, samples).
    max_p : int
        Maximum model order to evaluate.
    crit_type : str
        Criterion type: 'AIC', 'HQ', or 'SC'.
    do_plot : bool
        Whether to plot the criterion values.

    Returns:
    crit : np.ndarray
        Criterion values for each model order.
    p_range : np.ndarray
        Evaluated model order range (1:max_p).
    p_opt : int
        Optimal model order (minimizing the criterion).
    """
    k, N = dat.shape
    p_range = np.arange(1, max_p + 1)
    crit = np.zeros(max_p)

    for p in p_range:
        _, Vr = AR_coeff(dat, p)  # You must define or import AR_coeff
        if crit_type == 'AIC':
            crit[p-1] = np.log(np.linalg.det(Vr)) + 2 * p * k**2 / N
        elif crit_type == 'HQ':
            crit[p-1] = np.log(np.linalg.det(Vr)) + 2 * np.log(np.log(N)) * p * k**2 / N
        elif crit_type == 'SC':
            crit[p-1] = np.log(np.linalg.det(Vr)) + np.log(N) * p * k**2 / N
        else:
            raise ValueError("Invalid criterion type. Choose from 'AIC', 'HQ', 'SC'.")

    p_opt = p_range[np.argmin(crit)]
    if do_plot:
        plt.figure()
        plt.plot(p_range, crit, marker='o')
        plt.plot(p_opt, np.min(crit), 'ro')
        plt.xlabel('Model order p')
        plt.ylabel(f'{crit_type} criterion')
        plt.title(f'MVAR Model Order Selection ({crit_type}). The best order = {p_opt}')
        plt.grid(True)
        plt.show()

    return crit, p_range, p_opt


def mvar_spectra(H, V, f  ):
    """
    Calculate the MVAR spectra from transfer function H and noise covariance V. 
    Parameters:
    H : np.ndarray
        Transfer function matrix with shape (chan, chan, len(f)).
    V : np.ndarray      
        Noise covariance matrix with shape (chan, chan).
    f : np.ndarray  
        Frequency vector.           
    Returns:    
    S : np.ndarray
        MVAR spectra with shape (chan, chan, len(f)).
    """ 
    N_chan, _, N_f = H.shape
    # check if f has the same length as the third dimension of H
    if len(f) != N_f:
        raise ValueError("Frequency vector f must match the third dimension of H.") 
    S_multivariate = np.zeros((N_chan,N_chan, N_f), dtype=np.complex128) #initialize the multivariate spectrum
    for fi in range(N_f): #compute spectrum for all channels
        S_multivariate[:,:,fi] = H[:,:,fi].dot(V.dot(H[:,:,fi].T)) 
    return S_multivariate, f

     

