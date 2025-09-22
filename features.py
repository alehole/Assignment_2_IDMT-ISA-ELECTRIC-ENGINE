import numpy as np
import glob, os



## Load datasets
def load_dataset(paths_and_labels, pattern="*.wav"):
    X, y = [], []
    for label, path in paths_and_labels:
        for fp in glob.glob(os.path.join(path, pattern)):
            feats = features(fp)
            if feats is not None and np.all(np.isfinite(feats)) and feats is not np.nan:
                X.append(feats)
                y.append(label)
    return np.vstack(X), np.array(y)

## load wav files
def get_wav(folder_path):
    import wave

    with wave.open(folder_path, "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio_array = np.frombuffer(frames, dtype=np.int16)
    return audio_array

## Extract features
def features(folder_path):
    def safe_rms(x):
        if x.size == 0:
            return np.nan
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)  # replace bad values
        return np.sqrt(np.mean(x.astype(np.float64) ** 2))

    audio_array = get_wav(folder_path)
    if audio_array.size == 0 or len(audio_array) == 0:
        return np.nan

    mean_val = np.mean(audio_array)
    var_val = np.var(audio_array)
    std_val = np.std(audio_array)
    rms_val = safe_rms(audio_array)
    ptp_val = np.ptp(audio_array)  # max - min
    zcr_val = float(((audio_array[:-1] * audio_array[1:]) < 0).mean()) # 2) Zero-crossing rate (proxy for dominant freq)

    # Next step , include frequency information
    return np.array([mean_val, rms_val, zcr_val ], dtype=float)
