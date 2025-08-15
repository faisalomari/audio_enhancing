# 🎙️ YouTube Lecture Audio Cleaner & Enhancer

This repository provides a **complete pipeline** for downloading lecture audio from YouTube, cleaning up noise (especially in old recordings), enhancing the speaker’s voice, and normalizing loudness to **studio/YouTube-ready levels**.

## 📂 Repository Structure

```
.
├── ytmp3.py             # Download MP3 audio from YouTube URLs
├── audio_enhancer.py    # Full audio processing pipeline (denoise → enhance → normalize)
├── requirements.txt     # All dependencies
└── README.md
```

---

## 🚀 Features

1. **Download YouTube audio**  
   - Using `yt-dlp`, grabs the best available audio stream and saves as MP3.

2. **Noise cleaning (Denoising)**  
   - Bandpass filtering (keep only relevant speech frequencies).
   - Hum removal at 50 Hz + harmonics.
   - Automatic notch filters for narrow noise peaks.
   - Optional spectral gating to reduce constant hiss.

3. **Voice enhancement**  
   - **Low-shelf boost** for warmth (low frequencies).
   - **Presence boost** for clarity (upper mids).
   - **De-esser** to reduce harsh "s" sounds.
   - **Compressor** to balance loud and quiet parts.
   - **Harmonic exciter** to add brightness and intelligibility.

4. **Loudness normalization**  
   - Targets **−14 LUFS** for YouTube and streaming platforms.
   - Ensures **true peaks ≤ −1 dBFS** to prevent distortion.
   - Matches loudness standards for a professional final sound.

5. **Batch processing** with `tqdm` progress bars.

---

## 🛠 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**requirements.txt** should include:
```
yt-dlp
pydub
pyloudnorm
tqdm
numpy
scipy
librosa
soundfile
```

---

## 📥 Usage

### 1. Download audio from YouTube
Edit `urls` inside `ytmp3.py` or provide them dynamically.

Run:
```bash
python ytmp3.py
```
This will save files like:
```
lectures/1.mp3
lectures/2.mp3
...
```

---

### 2. Process audio (clean → enhance → normalize)

Example:
```bash
python audio_enhancer.py
```

The pipeline will:
- Load `.mp3` files.
- Apply denoising filters.
- Enhance voice clarity.
- Normalize loudness to −14 LUFS.
- Save both `_cleaned.wav` and `_enhanced.wav` versions.

Output:
```
lectures/1_cleaned.wav
lectures/1_enhanced.wav
...
```

---

## ⚙️ Processing Pipeline

**Step 1: Bandpass Filtering**  
Keeps only frequencies in the **80–8000 Hz** range (speech range).

**Step 2: Hum Removal**  
Removes electrical hum at 50 Hz and its harmonics.

**Step 3: Automatic Notch Filtering**  
Finds and removes narrowband noise peaks in the spectrum.

**Step 4: Spectral Gating (Optional)**  
Reduces constant background hiss based on a noise profile.

**Step 5: Voice Enhancement**  
- Low-shelf boost at 120 Hz for warmth.
- Presence boost at 3500 Hz for clarity.
- De-esser between 6500–9000 Hz to reduce harsh "S" sounds.
- Compression for dynamic balance.
- Harmonic exciter for richer sound.

**Step 6: Loudness Normalization**  
- Matches loudness to **−14 LUFS** (YouTube standard).
- Ensures no peaks above **−1 dBFS**.

---

## 📊 Example Batch Run with Progress Bar

```python
from tqdm import tqdm

FILE_NAMES = ["lectures/1", "lectures/2", "lectures/3"]

for name in tqdm(FILE_NAMES, desc="Processing files", unit="file"):
    input_path = name + ".mp3"
    cleaned_path = name + "_cleaned.wav"
    enhanced_path = name + "_enhanced.wav"
    denoise_then_enhance()
```

---

## 🎯 Why This Exists

Old recordings (like lectures from the 1990s) often have:
- Low volume
- Electrical hum
- Hiss in high frequencies
- Poor clarity

This pipeline automates the **post-production** process to make them sound **clear, loud, and professional**.

---

## 📜 License

MIT License. See `LICENSE` file for details.

---

## 🙌 Credits

- [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) for downloading YouTube audio.
- [`pydub`](https://github.com/jiaaro/pydub) for audio editing.
- [`pyloudnorm`](https://github.com/csteinmetz1/pyloudnorm) for loudness normalization.
- [`librosa`](https://librosa.org) for audio analysis.
