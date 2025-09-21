"""
This an example of reading an audio file (.wav) as a numpy array.
In this example, we will use the "wave" library to read the audio file.
If you are using pip for installing libraries, you could try to install it by "pip install wave".
"""

import numpy as np
import wave  # wave library. if you are using pip, try "pip install wave"
import matplotlib.pyplot as plt

good_folder_path = "data/IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine1_good/"  # update according your data folder path
file_name = "pure_0.wav" # audio file. Hint: You can use os.listdir(folder_path) to list all the files in the folder.

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
