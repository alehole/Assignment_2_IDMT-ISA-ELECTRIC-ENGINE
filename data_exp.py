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

def plt_wav():
    import numpy as np
    import wave  # wave library. if you are using pip, try "pip install wave"
    import matplotlib.pyplot as plt

    good_folder_path = "IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine1_good/"  # update according your data folder path
    file_name = "pure_0.wav"  # audio file. Hint: You can use os.listdir(folder_path) to list all the files in the folder.

    # Specify the audio file, for example:
    my_wav_file = good_folder_path + file_name  # update it for your file.

    # Read the wav. file (my_wav_file) as an array (audio_array):
    with wave.open(my_wav_file) as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio_array = np.frombuffer(frames, dtype=np.int16)

    plt.figure()
    plt.plot(audio_array)
    plt.title("Example of an audio file")
    plt.show()