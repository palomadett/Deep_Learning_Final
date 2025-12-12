#Needed Constants--------------------------------------------------------------------------------------------------------
-

# Signal Paramters
fc            = 3.5e6  #carrier frequency
sr            = 10e6 #sample rate (Hz)
num_pulses    = 5 #pulses per train
PRI           = .12e-3 #pulse repetition interval (sec)
pulse_len     = 0.1e-3 #single pulse length (sec)
num_samples = PRI * num_pulses * sr #number of samples per pulse train
v_peak_range = (1e-3, 8e-3) # amplitude of pulse in volts
snr_range     = (1, 3) #random SNR range for noisy signals
c = 1500 # Speed of Sound in water = 1500 m/s

V_max = 5 #Max speed of vehicle, 5 m/s
f_d_max = (fc*((2*V_max / c)+1))-fc #Max doppler frequency shift depending on the max speed of the vehicle
doppler_range = (-f_d_max, f_d_max)#Doppler shift range (Hz)

V_peak =.5e-3
# Spike frequencies (PCB NUM)
SPIKE_FREQ_POOL = np.linspace(3e6, 10e6, 200)
# Spike amplitudes (0.1–50% of pulse)
SPIKE_AMP_POOL = np.linspace(0.1*V_peak, .9*V_peak, 2000)

# Spur frequencies (clock / SMPS artifacts)
SPUR_FREQ_POOL = np.linspace(2e6, 4e6, 200)

# Spur amplitudes (0.1–5% of pulse)
SPUR_AMP_POOL = np.linspace(0.001*V_peak, .9*V_peak, 100)
#--------------------------------------------------------------------------------------------------------------------------
def generate_clean_pulse_train(
    fs=sr,
    PRI=PRI,
    pulse_len=pulse_len,
    fc=fc,
    num_pulses=num_pulses,
    phase_random=True,
    f_shift=None,
    v_peak=None  # Allow None for randomization
):
    Npri = int(PRI * fs) #Pulse repetition interval in number of samples
    Npulse = int(pulse_len * fs) #pulse len in # of samples
    total_len = num_pulses * Npri #number of samples per pulse train (computed this earlier to print but i dont deel like passing it into this fucntion )
    signal = np.zeros(total_len, dtype=np.float32) #initialize signals

    if f_shift is None:
        f_shift = np.random.uniform(doppler_range[0], doppler_range[1])
    if v_peak is None:
        v_peak = np.random.uniform(v_peak_range[0], v_peak_range[1])  # Assume v_peak_range = (1e-3, 8e-3)
    # ... (rest same, use v_peak in carrier)


    '''Sim poulse train:
    start with array of zeros
    then fill in wiht puls '''
    for p in range(num_pulses):
        start = p * Npri #Pulses start index
        end = start + Npulse #pulse end index
        phase = np.random.uniform(0, 2*np.pi)
        t = np.arange(Npulse) / fs #time array
        f_doppler = f_shift + fc

        carrier = v_peak * np.cos(2*np.pi*f_doppler*t + phase) #pulse just a plain sin wave
        #velocity = c * (1 - doppler_hz/ fc)A

        signal[start:end] += carrier

    return signal, f_shift

''' Add a scattering effect and add noise to the pulse '''
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def colored_noise(x, cutoff=0.05, order=3):
    """Low-pass filter white noise to make it colored (smooth)."""
    b, a = butter(order, cutoff, btype='low')
    return lfilter(b, a, x)
#---------------------------------------------------------------------------------------------------------------------------------------------------------------

def add_scattering_and_noise(
    clean,
    fs=sr,
    snr_db=snr_range,
    spike_freqs=SPIKE_FREQ_POOL,
    spike_amps=SPIKE_AMP_POOL,
    spur_freqs=SPUR_FREQ_POOL,
    spur_amps=SPUR_AMP_POOL,
    scattering_strength=0.3,
    spike_rate=1000,
    spike_decay_us=70,
    num_spurs=5
):
    N = len(clean)
    t = np.arange(N) / fs

    scattered = np.copy(clean)
    for p in range(num_pulses):
        start = p * int(PRI * fs)
        end = start + int(pulse_len * fs)
        if end > len(clean):
            break
        Npulse = end - start
        amps = np.cumsum(np.random.randn(Npulse))
        amps = 1 + scattering_strength * amps / np.std(amps) if np.std(amps) > 0 else 1
        phase_noise = np.cumsum(np.random.randn(Npulse)) * 0.5
        scattered[start:end] *= amps * np.cos(phase_noise)
    # ... (rest same)

    # --- Burst spikes / multipath ---
    spike_signal = np.zeros(N, dtype=np.float32)
    duration = N / fs
    n_spikes = np.random.poisson(spike_rate * duration)

    for _ in range(n_spikes):
        t0 = np.random.uniform(0, duration)
        idx0 = int(t0 * fs)
        if idx0 >= N:
            continue
        t_rel = t[idx0:] - t0
        spike_freq_i = np.random.choice(spike_freqs)
        spike_amp_i  = np.random.choice(spike_amps) * np.random.uniform(0.5, 1.5)
        env = np.exp(-t_rel / (spike_decay_us * 1e-6))
        spike_signal[idx0:] += spike_amp_i * env * np.sin(2*np.pi*spike_freq_i*t_rel)

    #Periodic spurs (drifting frequency + amplitude)
    spur_signal = np.zeros(N, dtype=np.float32)

    # Randomly select a subset of spurs to add
    if len(spur_freqs) > 0 and len(spur_amps) > 0:
        selected_indices = np.random.choice(min(len(spur_freqs), len(spur_amps)), size=num_spurs, replace=True)
        selected_freqs = spur_freqs[selected_indices]
        selected_amps = spur_amps[selected_indices % len(spur_amps)]  # Cycle amps if fewer

        for f, a in zip(selected_freqs, selected_amps):
            f_i = f * np.random.uniform(0.95, 1.05)
            a_i = a * np.random.uniform(0.5, 1.5)
            phase = np.random.uniform(0, 2*np.pi)
            spur_signal += a_i * np.sin(2*np.pi*f_i*t + phase)

    # -Band-limited (colored) Gaussian noise
    sig_rms = np.sqrt(np.mean(scattered**2))
    snr_lin = 10**(snr_db / 10)
    noise_rms = sig_rms / np.sqrt(snr_lin)
    awgn = noise_rms * colored_noise(np.random.randn(N), cutoff=0.05)

    #Combine everything
    noisy = scattered + spike_signal + spur_signal + awgn
    return noisy.astype(np.float32)
  #---------------------------------------------------------------------------------------------------------------------------------------------------------------


def compute_fft(signal, fs, window=True, zero_pad_factor=4, doppler_range=None):
    """
    Compute FFT of a complex signal with optional windowing, zero-padding,
    and bandpass filtering in the frequency domain.

    Args:
        signal : np.array
            Complex basebanded signal
        fs : float
            Sampling frequency
        window : bool
            Apply Hann window
        zero_pad_factor : int
            Multiply signal length by this for zero-padding
        doppler_range : tuple or None
            Tuple of (f_min, f_max) in Hz to keep. Frequencies outside this range
            are set to zero.

    Returns:
        mag : np.array
            FFT magnitude
        freqs : np.array
            Corresponding frequency axis (Hz)
    """
    N = len(signal)

    # Apply window
    if window:
        win = np.hanning(N)
        sig_win = signal * win
    else:
        sig_win = signal

    # Zero-pad to increase frequency resolution
    Nfft = zero_pad_factor * N
    F = np.fft.fftshift(np.fft.fft(sig_win, n=Nfft))
    freqs = np.fft.fftshift(np.fft.fftfreq(Nfft, 1/fs))

    # Apply Doppler bandpass filter
    if doppler_range is not None:
        f_min, f_max = doppler_range
        mask = (freqs >= f_min) & (freqs <= f_max)
        F = F * mask  # zero out frequencies outside range

    # Normalize magnitude
    mag = np.abs(F) / N

    return mag, freqs
#-------------------------------------------------------------------------------------------------------------------------------------------
def plot_noisy_clean_with_fft(loader, fs=sr, num_examples=3):
    scale = loader.dataset.scale_factor  # ← GLOBAL SCALE FACTOR

    # Get a single batch
    for batch_noisy, batch_clean, batch_V in loader:
        break

    for i in range(min(num_examples, batch_noisy.size(0))):
        # Reconstruct normalized complex signals
        noisy = batch_noisy[i].squeeze().numpy()
        clean = batch_clean[i].squeeze().numpy()

        noisy_c = noisy[0] + 1j * noisy[1]
        clean_c = clean[0] + 1j * clean[1]

        # ------------ UN-NORMALIZE HERE ------------
        noisy_c = noisy_c * scale
        clean_c = clean_c * scale
        # -------------------------------------------

        # Convert fd back to Hz
        f_shift_true = batch_V[i].item() * 1e3

        # Compute FFT
        CLEAN, FREQ = compute_fft(clean_c, fs, window=True,
                                  zero_pad_factor=4, doppler_range=doppler_range)
        NOISY, FREQ = compute_fft(noisy_c, fs, window=True,
                                  zero_pad_factor=4, doppler_range=doppler_range)

        # Determine FFT peaks
        f_peak_clean = FREQ[np.argmax(CLEAN)]
        f_peak_noisy = FREQ[np.argmax(NOISY)]

        v_clean = f_peak_clean * c / (2 * fc)
        v_noisy = f_peak_noisy * c / (2 * fc)
        v_true  = f_shift_true * c / (2 * fc)

        print(f"True v: {v_true:.3f}, FFT clean v: {v_clean:.3f}, FFT noisy v: {v_noisy:.3f}")

        # Time axis
        t = np.arange(len(clean_c)) / fs

        # --- Plots ---
        fig, axs = plt.subplots(2, 2, figsize=(18,10.5))

        axs[0,0].plot(t, clean_c.real)
        axs[0,0].plot(t, clean_c.imag, alpha=0.7)
        axs[0,0].set_title("Clean Baseband Signal (UN-NORMALIZED)")
        axs[0,0].grid(True, alpha=0.3)

        axs[0,1].plot(t, noisy_c.real)
        axs[0,1].plot(t, noisy_c.imag, alpha=0.7)
        axs[0,1].set_title("Noisy Baseband Signal (UN-NORMALIZED)")
        axs[0,1].grid(True, alpha=0.3)

        axs[1,0].plot(FREQ, CLEAN)
        axs[1,0].set_title(f"FFT Clean (v={v_clean:.3f})")
        axs[1,0].grid(True)

        axs[1,1].plot(FREQ, NOISY)
        axs[1,1].set_title(f"FFT Noisy (v={v_noisy:.3f})")
        axs[1,1].grid(True)

        plt.tight_layout()
        plt.show()
        print(v_clean, v_noisy, v_true)
