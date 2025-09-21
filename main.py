import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import data_exp

if __name__ == '__main__':
    fs=44100

    data_exp.n_examples()

    data_exp.plt_wav_n("IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine1_good",1)
    data_exp.plt_wav_n("IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine2_broken", 1)
    data_exp.plt_wav_n("IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine3_heavyload", 1)

    data_exp.plt_amp_hist("IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine1_good",1)
    data_exp.plt_amp_hist("IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine2_broken", 1)
    data_exp.plt_amp_hist("IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine3_heavyload", 1)