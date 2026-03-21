# ================================================
# LAION-CLAP Chunked Heatmap – Your Final Working Version
# ================================================

!pip install -q transformers torch torchaudio librosa matplotlib numpy tqdm

import torch
from transformers import ClapModel, ClapProcessor
import librosa
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from tqdm import tqdm

model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to("cuda" if torch.cuda.is_available() else "cpu")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")

text_descriptions = [
    "loud male voice saying no or stop urgently early in the clip",
    "fast conflicted talking with faint female yelling crying or pleading in background",
    "faint muffled female screaming or distress cries",
    "males talking getting progressively louder and more aggressive",
    "possible male pleading or pain yelling",
    "sporadic loud yelling or talking bursts",
    "very loud triple bang or impact sound around 43 seconds",
    "loud muffled sound or thud with unclear angry or scared voices in background",
    "continuous nonstop loud yelling and screaming from multiple people",
    "one male yelling deeply while others try to cover or overpower his voice",
    "urgent male voice repeatedly saying stop stop stop",
    "faint female crying or yelling overpowered by loud excited males",
    "male yelling spikes with increasing loudness",
    "female vocal spikes or short cries",
    "loud indistinct sounds or spikes",
    "intense loud arguing or confrontation",
    "loud male spike possibly yelling",
    "loud males talking with female screaming crying in the background",
    "ending with male voices talking and yelling",
    "overall chaotic multi-person distress yelling screaming and impacts",
    "normal quiet or low-level background night sounds",
]

print("Queries loaded")

uploaded = files.upload()
clip_name = list(uploaded.keys())[0]

audio, sr = librosa.load(clip_name, sr=48000, mono=True)
duration = len(audio) / sr
print(f"Duration: {duration:.1f}s")

chunk_duration = 15.0
overlap = 5.0
hop = chunk_duration - overlap
num_chunks = int(np.ceil((duration - chunk_duration) / hop)) + 1

results = []

for i in tqdm(range(num_chunks)):
    start_sec = i * hop
    end_sec = min(start_sec + chunk_duration, duration)
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    chunk_audio = audio[start_sample:end_sample]

    if len(chunk_audio) < 16000:
        continue

    inputs = processor(text=text_descriptions, audio=[chunk_audio], return_tensors="pt", padding=True, sampling_rate=48000)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    audio_emb = outputs.audio_embeds / outputs.audio_embeds.norm(dim=-1, keepdim=True)
    text_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
    sim = (audio_emb @ text_emb.T).squeeze().cpu().numpy()

    results.append({
        "start_sec": start_sec,
        "end_sec": end_sec,
        "mid_sec": (start_sec + end_sec) / 2,
        "similarities": sim
    })

times = [r["mid_sec"] for r in results]
sim_matrix = np.array([r["similarities"] for r in results])

plt.figure(figsize=(16, 10))
plt.imshow(sim_matrix.T, aspect='auto', cmap='Purples', origin='lower',
           extent=[0, duration, 0, len(text_descriptions)])
plt.colorbar(label='Cosine Similarity')
plt.yticks(np.arange(len(text_descriptions)) + 0.5, [f"{i+1}. {d[:60]}..." for i, d in enumerate(text_descriptions)], fontsize=9)
plt.xlabel("Time in clip (seconds)")
plt.ylabel("Query Description")
plt.title(f"LAION-CLAP Similarity Over Time – {clip_name}")

# Your key lines
plt.axvline(43, color='red', ls='--', alpha=0.7, label="~43s triple bang")
plt.axvline(56, color='orange', ls='--', alpha=0.7, label="56–78s nonstop yelling")
plt.axvline(78, color='lime', ls='--', alpha=0.7, label="78–79s stop stop stop")
plt.axvline(88, color='cyan', ls='--', alpha=0.7, label="88–89s female spikes")
plt.axvline(95, color='purple', ls='--', alpha=0.7, label="95–103s intense arguing")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("Done. Dark bands at your times = strong matches.")