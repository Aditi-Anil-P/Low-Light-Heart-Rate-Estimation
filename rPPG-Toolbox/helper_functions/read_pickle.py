import pickle
import numpy as np
from scipy.signal import welch

pickle_path = 'runs/exp/UBFC-rPPG_SizeW72_SizeH72_ClipLength180_DataTypeDiffNormalized_Standardized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse/saved_test_outputs/UBFC_UBFC_tscan_outputs.pickle'

with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

fs = data['fs']
print(f'Sampling Rate: {fs} Hz')
print(f'Label Type: {data["label_type"]}')
print(f'\n--- Estimated BPM per subject ---')

for subject_id, chunks in data['predictions'].items():
    # Concatenate all chunks for this subject
    all_chunks = []
    for chunk_id, signal in chunks.items():
        all_chunks.append(signal.numpy().flatten())
    full_signal = np.concatenate(all_chunks)
    
    # Estimate HR using FFT
    freqs, psd = welch(full_signal, fs=fs, nperseg=256)
    hr_mask = (freqs >= 40/60) & (freqs <= 180/60)
    peak_freq = freqs[hr_mask][np.argmax(psd[hr_mask])]
    bpm = peak_freq * 60
    print(f'{subject_id}: {bpm:.1f} BPM')