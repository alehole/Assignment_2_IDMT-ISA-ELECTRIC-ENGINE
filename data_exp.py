import os
Fs=44100
dt=1.0/Fs

def main():
    n_examples()

    plt_wav_n("IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine1_good", 1)
    plt_wav_n("IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine2_broken", 1)
    plt_wav_n("IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine3_heavyload", 1)

    plt_amp_hist("IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine1_good", 1)
    plt_amp_hist("IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine2_broken", 1)
    plt_amp_hist("IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine3_heavyload", 1)

    statistics("IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine1_good", 1)
    statistics("IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine2_broken", 1)
    statistics("IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine3_heavyload", 1)

    plt_fft_wav_n("IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine1_good", 22)
    plt_fft_wav_n("IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine2_broken", 22)
    plt_fft_wav_n("IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine3_heavyload", 22)

def n_examples():
    train_cut_good_path = "IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine1_good"
    train_cut_good_items = os.listdir(train_cut_good_path)
    # print(train_cut_good_items)
    files = [f for f in os.listdir(train_cut_good_path) if os.path.isfile(os.path.join(train_cut_good_path, f))]
    print("Number of good examples :", len(files))

    train_cut_broken_path = "IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine2_broken"
    train_cut_broken_items = os.listdir(train_cut_broken_path)
    # print(train_cut_broken_items)
    files = [f for f in os.listdir(train_cut_broken_path) if os.path.isfile(os.path.join(train_cut_broken_path, f))]
    print("Number of broken examples :", len(files))

    train_cut_heavy_load_path = "IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine3_heavyload"
    train_cut_heavy_load_items = os.listdir(train_cut_heavy_load_path)
    # print(train_cut_heavy_load_items)
    files = [f for f in os.listdir(train_cut_heavy_load_path) if os.path.isfile(os.path.join(train_cut_heavy_load_path, f))]
    print("Number of heavy load examples :", len(files))

def get_wav(folder_path,n):
    import numpy as np
    import wave

    # Get only wav files
    wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    wav_files.sort()

    # Check if n is within range
    if n < 1 or n > len(wav_files):
        print(f"Invalid number. Choose between 1 and {len(wav_files)}")
        return None

    # Select nth file (1-based index)
    file_name = wav_files[n-1]
    my_wav_file = os.path.join(folder_path, file_name)

    with wave.open(my_wav_file, "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio_array = np.frombuffer(frames, dtype=np.int16)

    return audio_array, file_name

def plt_wav_n(folder_path,n):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    audio_array, file_name = get_wav(folder_path, n)
    time = np.arange(len(audio_array)) / Fs
    plt.figure()
    plt.plot(time, audio_array)
    plt.title(f"Audio file: {os.path.basename(folder_path) +"/"+ file_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.tight_layout()
    plt.show()

def plt_amp_hist(folder_path,n):
    import matplotlib.pyplot as plt
    import os

    audio_array, file_name = get_wav(folder_path, n)

    plt.figure()
    plt.hist(audio_array, bins=50)
    plt.title(f"Audio file: {os.path.basename(folder_path) + "/" + file_name}")
    plt.title("Amplitude distribution")
    plt.xlabel("Amplitude")
    plt.ylabel("Count")
    plt.grid()
    plt.tight_layout()
    plt.show()

def statistics(folder_path,n):
    import numpy as np
    import os
    def safe_rms(x):
        if x.size == 0:
            return np.nan
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)  # replace bad values
        return np.sqrt(np.mean(x.astype(np.float64) ** 2))
    audio_array, file_name = get_wav(folder_path, n)

    mean_val = np.mean(audio_array)
    var_val = np.var(audio_array)
    std_val = np.std(audio_array)
    rms_val = safe_rms(audio_array)
    ptp_val = np.ptp(audio_array)  # max - min

    print(os.path.basename(folder_path) + "/" + file_name)
    print("Mean value:", mean_val)
    print("Var:", var_val)
    print("Standard deviation:", std_val)
    print("RMS value:", rms_val)
    print("PTP value:", ptp_val)


def plt_fft_wav_n(folder_path,n):
    from scipy.fft import fft, ifft, fftfreq
    import matplotlib.pyplot as plt
    import numpy as np

    audio_array, file_name = get_wav(folder_path, n)
    # Perform the Fast Fourier Transform (FFT)
    yf = fft(audio_array)
    # Calculate the frequencies corresponding to the FFT output
    n_samples = len(audio_array)
    xf = fftfreq(n_samples, dt)[:n_samples // 2]  # Only consider positive frequencies for real signals

    plt.plot(xf, 2.0 / n_samples * np.abs(yf[:n_samples // 2]))
    plt.title(f"FFT Audio file: {os.path.basename(folder_path) + "/" + file_name}")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude |Y(f)|')
    plt.grid(True)
    plt.show()

