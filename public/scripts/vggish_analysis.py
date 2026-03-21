import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from google.colab import files
import os

print("Loading VGGish from TF Hub...")
model = hub.load('https://tfhub.dev/google/vggish/1')

print("Upload your 5 clips:")
uploaded = files.upload()

for clip_path in uploaded.keys():
    print(f"\n{'='*70}")
    print(f"PROCESSING: {clip_path}")
    print(f"{'='*70}")
    
    sr, wav_data = wavfile.read(clip_path)
    if len(wav_data.shape) > 1:
        wav_data = np.mean(wav_data, axis=1)
    if sr != 16000:
        print(f"Resampling from {sr} to 16kHz...")
        from scipy.signal import resample
        num_samples = int(len(wav_data) * 16000 / sr)
        wav_data = resample(wav_data, num_samples)
    
    wav_data = wav_data.astype(np.float32) / 32768.0 if wav_data.dtype == np.int16 else wav_data
    
    embeddings = model(wav_data)
    norms = np.linalg.norm(embeddings, axis=1)
    time_axis = np.arange(len(norms)) * 0.96
    
    peak_frame = np.argmax(norms)
    peak_time = peak_frame * 0.96
    peak_value = norms[peak_frame]
    
    print(f"Peak at: {peak_time:.2f} seconds (norm = {peak_value:.3f})")
    
    base_name = os.path.splitext(clip_path)[0]
    output_png = f"{base_name}_VGGISH.png"
    
    plt.figure(figsize=(11, 5))
    plt.plot(time_axis, norms, color='purple', linewidth=2.5, marker='o', markersize=4)
    plt.axvline(peak_time, color='red', linestyle='--', linewidth=2, label=f'Peak @ {peak_time:.2f}s')
    plt.title(f"VGGish Acoustic Activation\n{clip_path}\nPeak @ {peak_time:.2f}s | {peak_value:.3f}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Activation Strength")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"Image saved: {output_png}")

print("All done. Download the PNGs from Files tab.")