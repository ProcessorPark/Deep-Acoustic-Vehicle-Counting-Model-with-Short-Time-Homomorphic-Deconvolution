## convert from MATLAB function to Python function

import numpy as np
from scipy.signal import stft, hann
from scipy.fft import fft, ifft

import torch

def standardization(data):
    mean_data = np.mean(data)
    std_data = np.std(data)
    data = (data - mean_data) / std_data
    return data

def sum_2sig_with_delay(x, delay):  
    # input: 4ch signal / output: 6ch sum signal
    y = np.zeros((6, x.shape[1]))
    
    y[0, :] = x[0, :]
    y[0, delay:] += x[1, :-delay]
    
    y[1, :] = x[0, :]
    y[1, delay:] += x[2, :-delay]
    
    y[2, :] = x[0, :]
    y[2, delay:] += x[3, :-delay]
    
    y[3, :] = x[1, :]
    y[3, delay:] += x[2, :-delay]
    
    y[4, :] = x[1, :]
    y[4, delay:] += x[3, :-delay]
    
    y[5, :] = x[2, :]
    y[5, delay:] += x[3, :-delay]
    
    return y

def feature_spectrogram(sig, fs, fframe):
    # input: 4ch signal / output: 4ch image like
    num_feat1_ch = 4
    feat1 = []

    for ch_nn in range(num_feat1_ch):
        f, t, S = stft(sig[ch_nn, :] + np.finfo(float).eps, fs, window=hann(fframe, sym=False), nperseg=fframe, noverlap=fframe//2, nfft=fframe, return_onesided=True)
        
        # Convert to magnitude in dB
        spectrum = 10 * np.log10(np.abs(S)**2 + np.finfo(float).eps)
        
        # Select a portion of the spectrum
        mesh_xlim = slice(0, fframe//8)
        spectrum = spectrum[mesh_xlim, :]

        # Append to feature list
        feat1.append(spectrum)
    
    feat1 = np.stack(feat1, axis=-1)

    return feat1

def sthd(x, fs, fframe):
    # MATLAB windata = [1; 2*ones(fframe/2-1,1); ones(1-rem(fframe,2),1); zeros(fframe/2-1,1)]';
    windata = np.concatenate(([1], 2 * np.ones(fframe//2 - 1), np.ones(1 - fframe % 2), np.zeros(fframe//2 - 1)))

    # MATLAB windata2 = [hann(fframe/2); hann(fframe/2)];
    windata2 = np.concatenate((hann(fframe//2), hann(fframe//2)))

    # Compute STFT
    f, t, S = stft(x, fs, window=hann(fframe), nperseg=fframe, noverlap=fframe//2, nfft=fframe, return_onesided=False)  # 'FrequencyRange', 'twosided'
    
    # Step-by-step processing
    temp1 = np.abs(S)
    temp2 = np.log(temp1 + np.finfo(float).eps)  # Add eps to avoid log(0)
    temp3 = np.real(ifft(temp2, axis=0))
    temp4 = temp3 * windata[:, np.newaxis]
    temp5 = np.real(fft(temp4, axis=0))
    temp6 = np.exp(temp5)
    temp7 = np.real(ifft(temp6, axis=0))
    HDout = temp7 * windata2[:, np.newaxis]
    
    return HDout

def feature_sthd(sig, fs, fframe, delay):
    # input: 4ch signal / output: 6ch image like
    num_feat2_ch = 6
    feat2 = []

    sum_sig = sum_2sig_with_delay(sig, delay)
    
    for ch_nn in range(num_feat2_ch):
        sthd_output = sthd(sum_sig[ch_nn, :], fs, fframe)
        
        mesh_xlim = slice(fframe//4 - fframe//16, fframe//4 + fframe//16)
        sthd_output  = sthd_output [mesh_xlim, :]
        
        if len(feat2) == 0:
            feat2 = sthd_output [:, :, np.newaxis]
        else:
            feat2 = np.concatenate((feat2, sthd_output [:, :, np.newaxis]), axis=2)
    
    return feat2

# GPU version
def standardization_tensor(data):
    mean_data = torch.mean(data)
    std_data = torch.std(data)
    data = (data - mean_data) / std_data
    return data

def sum_2sig_with_delay_tensor(x, delay):  
    # input: 4ch signal / output: 6ch sum signal
    y = torch.zeros(6, x.shape[1], dtype=x.dtype, device=x.device)
    
    y[0, :] = x[0, :]
    y[0, delay:] += x[1, :-delay]
    
    y[1, :] = x[0, :]
    y[1, delay:] += x[2, :-delay]
    
    y[2, :] = x[0, :]
    y[2, delay:] += x[3, :-delay]
    
    y[3, :] = x[1, :]
    y[3, delay:] += x[2, :-delay]
    
    y[4, :] = x[1, :]
    y[4, delay:] += x[3, :-delay]
    
    y[5, :] = x[2, :]
    y[5, delay:] += x[3, :-delay]
    
    return y

def feature_spectrogram_tensor(sig, fs, fframe, device):
    # input: 4ch signal / output: 4ch image like    
    sig_tensor = torch.tensor(sig, dtype=torch.float32, device=device)            # Convert numpy array to PyTorch tensor
    
    num_feat1_ch = 4
    feat1 = []
    
    for ch_nn in range(num_feat1_ch):
        # Perform STFT using PyTorch
        window = torch.hann_window(fframe, periodic=True).to(device)
        S = torch.stft(sig_tensor[ch_nn, :] + torch.finfo(torch.float32).eps,
                     n_fft=fframe, hop_length=fframe//2, window=window, onesided=True, return_complex=True)
        
        # Convert to magnitude in dB
        magnitude = torch.abs(S)**2
        spectrum = 10 * torch.log10(magnitude + torch.finfo(torch.float32).eps)
        
        # Select a portion of the spectrum
        mesh_xlim = slice(0, fframe//8)
        spectrum = spectrum[mesh_xlim, :]

        # Append to feature list
        feat1.append(spectrum)  # Append PyTorch tensor
    
    feat1 = torch.stack(feat1, axis=-1)
    
    return feat1

def sthd_tensor(x_tensor, fs, fframe, device):    
    # Define window data
    windata = torch.cat((torch.tensor([1.], device=device), 2 * torch.ones(fframe//2 - 1, device=device),
                         torch.ones(1 - fframe % 2, device=device), torch.zeros(fframe//2 - 1, device=device)))
    
    windata2 = torch.cat((torch.hann_window(fframe//2, periodic=True).to(device), torch.hann_window(fframe//2, periodic=True).to(device)))
    
    # Compute STFT
    S = torch.stft(x_tensor + torch.finfo(torch.float32).eps, n_fft=fframe, hop_length=fframe//2, window=windata, onesided=False, return_complex=True)
    
    # Step-by-step processing
    temp1 = torch.abs(S)
    temp2 = torch.log(temp1 + torch.finfo(torch.float32).eps)
    temp3 = torch.real(torch.fft.ifft(temp2, axis=0))
    temp4 = temp3 * windata.view(-1, 1)
    temp5 = torch.real(torch.fft.fft(temp4, axis=0))
    temp6 = torch.exp(temp5)
    temp7 = torch.real(torch.fft.ifft(temp6, axis=0))
    HDout = temp7 * windata2.view(-1, 1)
    
    return HDout

def feature_sthd_tensor(sig, fs, fframe, delay, device):
    # input: 4ch signal / output: 6ch image like
    sig_tensor = torch.tensor(sig, dtype=torch.float32, device=device)     # Convert numpy array to PyTorch tensor    

    num_feat2_ch = 6
    feat2 = None

    sum_sig = sum_2sig_with_delay_tensor(sig_tensor, delay)
    
    for ch_nn in range(num_feat2_ch):
        sthd_output = sthd_tensor(sum_sig[ch_nn, :], fs, fframe, device)
        
        mesh_xlim = slice(fframe//4 - fframe//16, fframe//4 + fframe//16)
        sthd_output  = sthd_output [mesh_xlim, :]
        
        if feat2 is None:
            feat2 = sthd_output.unsqueeze(2)
        else:
            feat2 = torch.cat((feat2, sthd_output.unsqueeze(2)), dim=2)
    
    return feat2