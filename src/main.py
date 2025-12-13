
# ------------------- GLOBAL IMPORTS -------------------
import os
import math, random

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', font_scale=2)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ------------------- PROJECT-SPECIFIC IMPORTS -------------------
from src.utils import CachedPulseDataset, sr, fc, doppler_range, c, compute_fft
from src.model import UNet1D_DenoiseOnly

# ------------------- PULSE TRAIN PARAMETERS -------------------
fc            = 3.5e6
sr            = 10e6
num_pulses    = 5
PRI           = .12e-3
pulse_len     = 0.1e-3
v_peak_range  = (1e-3, 8e-3)
snr_range     = (1, 3)
c             = 1500

V_max = 5
f_d_max = (fc*((2*V_max / c)+1))-fc
doppler_range = (-f_d_max, f_d_max)

V_peak = .5e-3
SPIKE_FREQ_POOL = np.linspace(3e6, 10e6, 200)
SPIKE_AMP_POOL  = np.linspace(0.1*V_peak, .9*V_peak, 2000)
SPUR_FREQ_POOL  = np.linspace(2e6, 4e6, 200)
SPUR_AMP_POOL   = np.linspace(0.001*V_peak, .9*V_peak, 100)

print(f_d_max)
num_samples = int(PRI * num_pulses * sr)
print(num_samples)

# ------------------- DATASET -------------------
batch_size = 64
train_dataset = CachedPulseDataset(N=30000, cache_file="train_16k.pt", regenerate=True)
val_dataset   = CachedPulseDataset(N=3000,  cache_file="val_2k.pt",   regenerate=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# ------------------- MODEL -------------------
model = UNet1D_DenoiseOnly(in_ch=2, out_ch=2, features=32)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model on {device}")

# ------------------- TRAINING PARAMETERS -------------------
num_epochs = 60
lr = 0.5e-4
pulse_weight = 20.0
lambda_stft = 1.0

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

train_losses = []
val_losses   = []
rmse_fft     = []

# ------------------- TRAINING LOOP -------------------
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for x, y, _ in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)

        # === Masked time-domain loss ===
        mag_clean = torch.sqrt(y[:,0,:]**2 + y[:,1,:]**2)
        pulse_mask = (mag_clean > 0.01).float()
        se = (y_pred - y) ** 2
        se = se.mean(dim=1)
        weights = 1.0 + pulse_weight * pulse_mask
        loss_time = (se * weights).sum() / weights.sum()

        # === STFT loss ===
        y_complex      = torch.complex(y[:,0,:], y[:,1,:])
        y_pred_complex = torch.complex(y_pred[:,0,:], y_pred[:,1,:])

        spec_clean = torch.stft(y_complex, n_fft=512, hop_length=128, win_length=512,
                                window=torch.hann_window(512, device=device), return_complex=True)
        spec_pred  = torch.stft(y_pred_complex, n_fft=512, hop_length=128, win_length=512,
                                window=torch.hann_window(512, device=device), return_complex=True)

        loss_stft = F.mse_loss(torch.abs(spec_pred), torch.abs(spec_clean))

        loss = loss_time + lambda_stft * loss_stft

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # ------------------- VALIDATION -------------------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)

            mag_clean = torch.sqrt(y[:,0,:]**2 + y[:,1,:]**2)
            pulse_mask = (mag_clean > 0.01).float()
            se = (y_pred - y) ** 2
            se = se.mean(dim=1)
            weights = 1.0 + pulse_weight * pulse_mask
            loss_time = (se * weights).sum() / weights.sum()

            y_complex      = torch.complex(y[:,0,:], y[:,1,:])
            y_pred_complex = torch.complex(y_pred[:,0,:], y_pred[:,1,:])
            spec_clean = torch.stft(y_complex, n_fft=512, hop_length=128, win_length=512,
                                    window=torch.hann_window(512, device=device), return_complex=True)
            spec_pred  = torch.stft(y_pred_complex, n_fft=512, hop_length=128, win_length=512,
                                    window=torch.hann_window(512, device=device), return_complex=True)
            loss_stft = F.mse_loss(torch.abs(spec_pred), torch.abs(spec_clean))

            loss = loss_time + lambda_stft * loss_stft
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1:3d}  Train: {train_loss:.4f}  Val: {val_loss:.4f}")

# ------------------- PLOT TRAINING/VALIDATION LOSS -------------------
plt.figure(figsize=(10,6))
plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
plt.title('Training vs Validation Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------- PLOT EXAMPLE SIGNALS -------------------
model.eval()
with torch.no_grad():
    x_batch, y_clean_batch, _ = next(iter(val_loader))
    x_batch = x_batch.to(device)
    y_pred_batch = model(x_batch).cpu().numpy()

    idx = 0
    noisy   = x_batch[idx].cpu().numpy()
    clean   = y_clean_batch[idx].numpy()
    denoised = y_pred_batch[idx]

t = np.arange(noisy.shape[1]) * 1e6 / sr

noisy_real, noisy_imag = noisy[0], noisy[1]
clean_real, clean_imag = clean[0], clean[1]
denoised_real, denoised_imag = denoised[0], denoised[1]

fig, axs = plt.subplots(3, 1, figsize=(18, 11))
axs[0].plot(t, noisy_real, color='red', label='Noisy Real', linewidth=1.5, alpha=0.9)
axs[0].plot(t, noisy_imag, color='orange', label='Noisy Imag', linewidth=1.5, alpha=0.9)
axs[0].set_title("1. Noisy Input (Real + Imaginary)", fontsize=16, pad=15)
axs[0].set_ylabel("Amplitude")
axs[0].legend()
axs[0].grid(True, alpha=0.3)
axs[0].set_xlim(0, t[-1])

axs[2].plot(t, clean_real, color='blue', label='Clean Real', linewidth=2.5)
axs[2].plot(t, clean_imag, color='cyan', label='Clean Imag', linewidth=2.5, alpha=0.8)
axs[2].set_title("2. Ground Truth Clean Pulse Train", fontsize=16, pad=15)
axs[2].set_ylabel("Amplitude")
axs[2].legend()
axs[2].grid(True, alpha=0.3)
axs[2].set_xlim(0, t[-1])

axs[1].plot(t, denoised_real, color='green', label='Denoised Real', linewidth=2.8)
axs[1].plot(t, denoised_imag, color='lime', label='Denoised Imag', linewidth=2.8, alpha=0.8)
axs[1].set_title("3. Model Output vs Ground Truth", fontsize=16, pad=15)
axs[1].set_xlabel("Time (μs)")
axs[1].set_ylabel("Amplitude")
axs[1].legend()
axs[1].grid(True, alpha=0.3)
axs[1].set_xlim(0, t[-1])

plt.tight_layout()
plt.show()

# ------------------- EVALUATE DENOISER -------------------
def evaluate_denoiser(loader, fs=sr, num_test_samples=300, top_k=3):
    model.eval()
    scale = loader.dataset.scale_factor

    all_errors_noisy = []
    all_errors_denoised = []
    all_data = []

    with torch.no_grad():
        count = 0
        for batch_noisy, batch_clean, batch_fd in loader:
            batch_noisy = batch_noisy.to(device)
            batch_pred = model(batch_noisy).cpu()
            batch_noisy = batch_noisy.cpu()
            batch_clean = batch_clean.cpu()
            batch_fd = batch_fd.cpu()

            for i in range(batch_noisy.size(0)):
                if count >= num_test_samples:
                    break

                noisy   = batch_noisy[i].numpy()
                clean   = batch_clean[i].numpy()
                denoised = batch_pred[i].numpy()

                noisy_c = noisy[0] + 1j*noisy[1]
                clean_c = clean[0] + 1j*clean[1]
                denoised_c = denoised[0] + 1j*denoised[1]

                noisy_c   *= scale
                clean_c   *= scale
                denoised_c *= scale
                denoised_c -= np.mean(denoised_c)

                CLEAN_MAG, FREQ = compute_fft(clean_c, fs, window=True, zero_pad_factor=8, doppler_range=doppler_range)
                NOISY_MAG, _ = compute_fft(noisy_c, fs, window=True, zero_pad_factor=8, doppler_range=doppler_range)
                DENOISED_MAG, _ = compute_fft(denoised_c, fs, window=True, zero_pad_factor=8, doppler_range=doppler_range)

                f_peak_clean    = FREQ[np.argmax(CLEAN_MAG)]
                f_peak_noisy    = FREQ[np.argmax(NOISY_MAG)]
                f_peak_denoised = FREQ[np.argmax(DENOISED_MAG)]

                f_shift_true_hz = batch_fd[i].item() * 1000.0
                v_true = f_shift_true_hz * c / (2 * fc)
                v_fft_noisy = f_peak_noisy * c / (2 * fc)
                v_fft_denoised = f_peak_denoised * c / (2 * fc)

                all_errors_noisy.append((v_fft_noisy - v_true)**2)
                all_errors_denoised.append((v_fft_denoised - v_true)**2)

                all_data.append({
                    'v_true': v_true,
                    'v_fft_noisy': v_fft_noisy,
                    'v_fft_denoised': v_fft_denoised,
                    'clean_c': clean_c,
                    'noisy_c': noisy_c,
                    'denoised_c': denoised_c,
                    'FREQ': FREQ,
                    'DENOISED_MAG': DENOISED_MAG
                })

                count += 1
            if count >= num_test_samples:
                break

    rmse_noisy = np.sqrt(np.mean(all_errors_noisy))
    rmse_denoised = np.sqrt(np.mean(all_errors_denoised))
    print(f"RMSE over {num_test_samples} samples:")
    print(f"  Noisy input   : {rmse_noisy:.6f} m/s")
    print(f"  Denoised output: {rmse_denoised:.6f} m/s")

    best_indices = np.argsort(all_errors_denoised)[:top_k]

    for idx in best_indices:
        data = all_data[idx]
        t = np.arange(len(data['clean_c'])) / fs

        fig, axs = plt.subplots(2, 2, figsize=(18, 10.5))

        axs[0,0].plot(t, data['clean_c'].real, label='Real', linewidth=2)
        axs[0,0].plot(t, data['clean_c'].imag, label='Imag', alpha=0.7, linewidth=2)
        axs[0,0].set_title("Clean Baseband Signal", fontsize=14)
        axs[0,0].legend()
        axs[0,0].grid(True, alpha=0.3)

        axs[0,1].plot(t, data['noisy_c'].real, color='red', label='Real', linewidth=1.8)
        axs[0,1].plot(t, data['noisy_c'].imag, color='orange', label='Imag', alpha=0.7, linewidth=1.8)
        axs[0,1].set_title("Noisy Input", fontsize=14)
        axs[0,1].legend()
        axs[0,1].grid(True, alpha=0.3)

        axs[1,0].plot(t, data['denoised_c'].real, color='green', label='Real', linewidth=2.2)
        axs[1,0].plot(t, data['denoised_c'].imag, color='lime', label='Imag', alpha=0.8, linewidth=2.2)
        axs[1,0].set_title(f"Denoised (Model)\nVelocity: {data['v_fft_denoised']:+.3f} m/s", fontsize=14)
        axs[1,0].legend()
        axs[1,0].grid(True, alpha=0.3)

        axs[1,1].plot(data['FREQ']/1e3, data['DENOISED_MAG'], color='green', linewidth=2)
        axs[1,1].axvline(data['v_fft_denoised']*2*fc/c/1e3, color='orange', linestyle='--', linewidth=2)
        axs[1,1].set_title(f"FFT Denoised → {data['v_fft_denoised']:+.3f} m/s\nTrue: {data['v_true']:+.3f} m/s", fontsize=14)
        axs[1,1].set_xlabel("Frequency (kHz)")
        axs[1,1].grid(True, alpha=0.3)

        plt.suptitle(f"Best Example — True Velocity: {data['v_true']:+.3f} m/s", fontsize=16)
        plt.tight_layout()
        plt.show()

# Run evaluation
evaluate_denoiser(val_loader, fs=sr, num_test_samples=300, top_k=3)

