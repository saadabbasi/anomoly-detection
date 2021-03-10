import os
import glob

import numpy as np
from tqdm import tqdm
import wavutils
from skimage.util.shape import view_as_windows
import math
from pathlib import Path

# make folder spectograms
# make folders spectograms/normal
# make folders spectograms/abnormal

# for each machine id
# read wave files
# make mel-spectrogram
# crop to 32 x 128 windows (make a parameter)
# do for all wav files in the folder
# save numpy file

n_mels=128
frames=5
n_fft=1024
hop_length=512
power=1.0

def generate_spectograms(target_dir,
                        sound_type="normal",
                        window_size = 32,
                        ext="wav"):

    def crop_spectograms():
        crops_per_file = int(math.floor(spectograms.shape[1]/window_size))
        number_of_crops = len(spectograms)*crops_per_file
        cropped_spectograms = np.zeros((number_of_crops, window_size, spectograms.shape[2]))
        for n in range(len(spectograms)):
            arr_out = view_as_windows(spectograms[n], (32,128), 32)
            cropped_spectograms[n*crops_per_file:(n+1)*crops_per_file] = np.squeeze(arr_out)
        return cropped_spectograms


    os.makedirs("spectograms", exist_ok=True)
    os.makedirs("spectograms/normal", exist_ok=True)
    os.makedirs("spectograms/abnormal", exist_ok=True)
    
    dirs = sorted(glob.glob(os.path.abspath(f"{target_dir}/*/*/*")))[11:12]

    for fpath in tqdm(dirs):
        files = sorted(glob.glob(os.path.abspath(f"{fpath}/{sound_type}/*.{ext}")))
        spectograms = wavutils.list_to_vector_array(files,
                                                    msg="Computing Mel-Spectograms",
                                                    n_mels=n_mels,
                                                    frames=frames,
                                                    n_fft=n_fft,
                                                    hop_length=hop_length,
                                                    power=power)

        cropped_spectograms = crop_spectograms()

        fpath = Path(fpath)
        fpath = fpath.parts
        sound_lvl = fpath[-3]
        machine_typ = fpath[-2]
        machine_id = fpath[-1]
        npy_path = os.path.join("spectograms", f"{sound_type}", f"{sound_type}_{sound_lvl}_{machine_typ}_{machine_id}.npy")
        np.save(npy_path, cropped_spectograms)

generate_spectograms('/data/mimii_dataset',sound_type="normal")
generate_spectograms('/data/mimii_dataset',sound_type="abnormal")