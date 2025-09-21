import os

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


def plt_wav_n(folder_path,n):
    import numpy as np
    import wave
    import matplotlib.pyplot as plt
    import os

    # Get only wav files
    wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    wav_files.sort()

    # Check if n is within range
    if n < 1 or n > len(wav_files):
        print(f"Invalid number. Choose between 1 and {len(wav_files)}")
        return

    # Select nth file (1-based index)
    file_name = wav_files[n-1]
    my_wav_file = os.path.join(folder_path, file_name)

    with wave.open(my_wav_file, "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio_array = np.frombuffer(frames, dtype=np.int16)

    plt.figure()
    plt.plot(audio_array)
    plt.title(f"Audio file: {os.path.basename(folder_path) +"/"+ file_name}")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.show()
