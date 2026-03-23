#!/usr/bin/env python3
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch

try:
    extractor
except NameError:
    from beats_trainer import BEATsFeatureExtractor
    extractor = BEATsFeatureExtractor()
    print("Re-initialized extractor")

beats_model = extractor.model
beats_model.eval()
device = next(beats_model.parameters()).device
print(f"Using device: {device}")

file_paths = [
    "/content/Distress_Event_1_023810_2MIN.wav",
    "/content/Distress_Event_2_024230_2MIN.wav",
    "/content/Distress_Event_3_024258_2MIN.wav",
    "/content/Impact_Event_024244_2MIN.wav",
    "/content/Prime_Suspect_024212_2MIN.wav"
]

your_notes_times = [43, 48, 54, 56, 60, 62, 65, 74, 78, 79, 88, 89, 90, 93, 95, 96, 104, 105, 109]

all_spikes = []
overlap_summary = []

for filepath in file_paths:
    filename = os.path.basename(filepath)
    print(f"\n{'═'*70}")
    print(f"Processing: {filename}")
    print(f"{'═'*70}")

    waveform, sr = torchaudio.load(filepath)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    waveform = waveform.to(device)
    duration = waveform.shape[1] / 16000.0
    
    with torch.no_grad():
        output = beats_model.extract_features(waveform)
        embeds = output[0] if isinstance(output, tuple) else output
        if embeds.dim() == 3 and embeds.shape[0] == 1:
            embeds = embeds.squeeze(0)
        embeds = embeds.cpu().numpy()
    
    activation = np.mean(np.abs(embeds), axis=1)
    time_axis = np.linspace(0, duration, len(activation))

    plt.figure(figsize=(14, 6))
    plt.plot(time_axis, activation, color='darkblue', linewidth=2.5)
    plt.title(f"BEATs Activation - {filename}")
    for t in your_notes_times:
        if t <= duration:
            plt.axvline(t, color='red', ls='--', alpha=0.6)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Mean Abs Latent Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"/content/{filename}_beats.png", dpi=150)
    plt.close()

    peak_idx = np.argmax(activation)
    peak_time = time_axis[peak_idx]
    print(f"Strongest spike: ~{peak_time:.1f}s (value {activation[peak_idx]:.3f})")

    mean_a = np.mean(activation)
    std_a = np.std(activation)
    thresh = mean_a + 1.8 * std_a
    spike_mask = activation > thresh
    spike_times = time_axis[spike_mask]
    
    for t in spike_times:
        all_spikes.append({
            'file': filename,
            'time_sec': round(t, 1),
            'activation': round(activation[int((t / duration) * len(activation))], 4)
        })

    matched = [round(t, 1) for t in spike_times if any(abs(t - nt) < 2.0 for nt in your_notes_times)]
    overlap_summary.append({
        'file': filename,
        'detected': len(spike_times),
        'matched': len(matched),
        'matched_times': matched
    })

df = pd.DataFrame(all_spikes).sort_values(['file', 'time_sec'])
df.to_csv('/content/beats_full_spikes.csv', index=False)
print("\nSaved spikes: /content/beats_full_spikes.csv")
print(df.head(20))

print("\nOverlap Summary:")
for item in overlap_summary:
    print(f"{item['file']}: {item['matched']}/{item['detected']} matched")
    if item['matched_times']:
        print(f"  {item['matched_times']}")