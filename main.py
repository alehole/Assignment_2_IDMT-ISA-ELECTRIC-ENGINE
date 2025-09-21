import matplotlib.pyplot as plt
import numpy as np
import os
if __name__ == '__main__':
    fs=44100

    train_cut_good_path="IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine1_good"
    train_cut_good_items=os.listdir(train_cut_good_path)
    print(train_cut_good_items)