import pickle
import numpy as np
from scipy.signal import welch, butter, filtfilt
import os
import warnings

# Try to import torch to safely convert tensors back to numpy arrays
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

warnings.filterwarnings("ignore")

def calculate_bpm_from_signal(signal, fs=30.0):
    """Calculates the BPM from a raw BVP signal using Welch's method (FFT)."""
    signal = np.array(signal).flatten()
    
    # Bandpass filter between 0.75 Hz (45 BPM) and 2.5 Hz (150 BPM)
    b, a = butter(1, [0.75 / (fs / 2), 2.5 / (fs / 2)], btype='band')
    
    try:
        filtered_signal = filtfilt(b, a, signal)
    except ValueError:
        filtered_signal = signal
        
    nperseg = min(len(filtered_signal), 256) 
    f, Pxx = welch(filtered_signal, fs, nperseg=nperseg)
    
    peak_idx = np.argmax(Pxx)
    peak_f = f[peak_idx]
    
    bpm = peak_f * 60.0
    return bpm

def analyze_rppg_pickle(file_path, fs=30.0):
    parent_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    file_name = os.path.basename(file_path)
    print(f"\n{'='*80}\nAnalyzing: {parent_dir} -> {file_name}\n{'='*80}")
    
    try:
        # 1. Load using standard pickle (which we know works for your files)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        if not isinstance(data, dict):
            print(f"Unexpected data type: {type(data)}. Expected a dictionary.")
            return

        # 2. Find the correct key containing the predictions
        pred_key = next((k for k in ['predictions', 'predict_v', 'predict_hr'] if k in data), None)

        if not pred_key:
            print(f"Could not find a recognized prediction key. Available keys: {list(data.keys())}")
            return

        predictions_dict = data[pred_key]
        
        # Determine how to iterate over the predictions
        if isinstance(predictions_dict, dict):
            items_to_process = list(predictions_dict.items())
        else:
            subjects = [f"Subject_{i+1}" for i in range(len(predictions_dict))]
            items_to_process = list(zip(subjects, predictions_dict))

        print(f"Extracted {len(items_to_process)} subject records. Calculating BPM...")
        print("-" * 80)

        # 3. Process each subject's data
        for subj_name, chunks in items_to_process:
            subj_name_clean = str(subj_name).replace("subject", "Subj_")
            full_signal = []

            # A. If the signal is divided into chunks (dictionary of tensors)
            if isinstance(chunks, dict):
                for chunk_idx in sorted(chunks.keys()):
                    piece = chunks[chunk_idx]
                    if HAS_TORCH and torch.is_tensor(piece):
                        piece = piece.cpu().numpy() # Move to CPU and convert to numpy
                    full_signal.append(np.array(piece).flatten())
                full_signal = np.concatenate(full_signal)

            # B. If it's a list of tensors
            elif isinstance(chunks, list):
                for piece in chunks:
                    if HAS_TORCH and torch.is_tensor(piece):
                        piece = piece.cpu().numpy()
                    full_signal.append(np.array(piece).flatten())
                full_signal = np.concatenate(full_signal)
                
            # C. Direct array/tensor fallback
            else:
                if HAS_TORCH and torch.is_tensor(chunks):
                    chunks = chunks.cpu().numpy()
                full_signal = np.array(chunks).flatten()

            # 4. Calculate and print BPM
            if full_signal.size == 1:
                # If the network already output a single BPM scalar
                bpm = float(full_signal[0])
                print(f"Subject: {subj_name_clean:<40} | Estimated BPM: {bpm:.2f}")
            else:
                # If it's a raw waveform array, apply FFT to find BPM
                bpm = calculate_bpm_from_signal(full_signal, fs=fs)
                print(f"Subject: {subj_name_clean:<40} | Estimated BPM (from signal): {bpm:.2f}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error reading {file_name}: {e}")

if __name__ == "__main__":
    file_list = ["Add output file names you want to extract bpm from"]
    
    # 30 fps is standard for MMPD/UBFC
    frames_per_second = 30.0 
    
    for f_path in file_list:
        analyze_rppg_pickle(f_path, fs=frames_per_second)