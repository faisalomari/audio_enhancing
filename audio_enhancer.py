# audio_denoise_enhance.py
# -*- coding: utf-8 -*-
"""
One-pass pipeline: DENOISE -> ENHANCE

Hardcoded settings from your two commands:
- Denoise:
  - bandpass: 80–8000 Hz
  - hum: 50 Hz, harmonics=6
  - auto_notch: on, auto_peaks=6, auto_prom_db=12
  - spectral gating: profile @ 5..7s (requires noisereduce if installed)
- Enhance:
  - low_shelf: +3 dB @ 120 Hz
  - presence: +4 dB @ 3500 Hz
  - deesser: 6500–9000 Hz, max 3 dB reduction
  - compressor: threshold −18 dBFS, ratio 3:1, makeup +4 dB, attack 5 ms, release 100 ms
  - harmonic exciter: on, mix 0.08, harmonics=3

Outputs:
- cleaned.wav
- enhanced.wav
"""

import warnings
import numpy as np
import soundfile as sf
from scipy import signal
import librosa
import pyloudnorm as pyln
from tqdm import tqdm

# =========================
# CONFIG — edit as needed
# =========================

# Denoise settings
BANDPASS = (80.0, 8000.0)
HUM_BASE = 50.0
HUM_HARMONICS = 6
NOTCH_Q = 30.0
AUTO_NOTCH = True
AUTO_PEAKS = 6
AUTO_PROM_DB = 12.0
AUTO_MINF = 40.0
AUTO_MAXF = 12000.0
GATING_MODE = "profile"     # "off" | "auto" | "profile"
PROFILE_START = 5.0         # seconds
PROFILE_DUR = 2.0           # seconds
NOISEREDUCE_PROP_DEC = 0.8  # spectral gating strength (0..1)

# Enhancement settings
LOW_SHELF = (120.0, 3.0)    # (freq, gain dB)
PRESENCE = (3500.0, 4.0)    # (center freq, gain dB)
DEESSER = (6500.0, 9000.0, 3.0)  # (low, high, max_reduction_db)
COMPRESS = (-18.0, 3.0, 4.0, 0.005, 0.100)  # (thr_db, ratio, makeup_db, attack_s, release_s)
EXCITER_ON = True
EXCITER_MIX = 0.08
EXCITER_HARMONICS = 3


# =========================
# Utilities
# =========================
def load_audio(path, sr=None, mono=True):
    y, sr = librosa.load(path, sr=sr, mono=mono)
    return np.ascontiguousarray(y.astype(np.float32)), sr

def save_wav(path, y, sr):
    sf.write(path, y, sr, subtype="FLOAT")

def time_to_samples(t, sr): return int(round(t * sr))

def extract_segment(y, sr, start_time, duration):
    n0 = max(0, time_to_samples(start_time, sr))
    n1 = min(len(y), n0 + time_to_samples(duration, sr))
    return y[n0:n1]

def welch_psd(y, sr, nperseg=8192):
    freqs, psd = signal.welch(y, fs=sr, nperseg=nperseg, noverlap=nperseg//2)
    return freqs, psd

def suggest_notch_freqs(freqs, psd, max_peaks=8, min_freq=40, max_freq=12000, prominence_db=12.0):
    psd_db = 10 * np.log10(psd + 1e-20)
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    f = freqs[mask]
    p = psd_db[mask]
    peaks, props = signal.find_peaks(p, prominence=prominence_db)
    if len(peaks) == 0:
        return []
    prominences = props.get("prominences", np.zeros_like(peaks))
    order = np.argsort(prominences)[::-1]
    top_idx = order[:max_peaks]
    return [float(f[peaks[i]]) for i in top_idx]

def iir_notch_sos(freq, sr, Q=30.0):
    w0 = freq / (sr / 2.0)
    if w0 >= 1.0:
        raise ValueError(f"Notch freq {freq} >= Nyquist.")
    b, a = signal.iirnotch(w0, Q)
    return signal.tf2sos(b, a)

def butter_bandpass_sos(lowcut, highcut, sr, order=4):
    if lowcut <= 0: lowcut = 1.0
    if highcut >= sr/2: highcut = sr/2 - 1.0
    return signal.butter(order, [lowcut, highcut], btype="bandpass", fs=sr, output="sos")

def cascade_sos(sos_list):
    if not sos_list: return None
    sos = sos_list[0]
    for s in sos_list[1:]:
        sos = np.vstack([sos, s])
    return sos

def apply_sos(y, sos):
    if sos is None: return y
    return signal.sosfiltfilt(sos, y, axis=-1)

def hum_harmonics(freq, sr, max_freq=None, count=6):
    if max_freq is None:
        max_freq = sr/2 - 100
    out = []
    f = freq
    while f < max_freq and len(out) < count:
        out.append(f); f += freq
    return out

def try_noisereduce(y, sr, noise_clip=None, prop_decrease=0.8):
    try:
        import noisereduce as nr
    except Exception:
        warnings.warn("noisereduce not installed. Skipping spectral gating.", RuntimeWarning)
        return y
    if noise_clip is not None and len(noise_clip) > 0:
        y_dnr = nr.reduce_noise(y=y, sr=sr, y_noise=noise_clip, prop_decrease=prop_decrease)
    else:
        y_dnr = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=prop_decrease)
    return y_dnr.astype(np.float32)

# ====== Enhancement tools ======
def design_low_shelf(fc, gain_db, sr, Q=0.707):
    A  = 10**(gain_db/40); w0 = 2*np.pi*fc/sr
    alpha = np.sin(w0)/(2*Q); cosw0 = np.cos(w0)
    b0 =    A*((A+1) - (A-1)*cosw0 + 2*np.sqrt(A)*alpha)
    b1 =  2*A*((A-1) - (A+1)*cosw0)
    b2 =    A*((A+1) - (A-1)*cosw0 - 2*np.sqrt(A)*alpha)
    a0 =        (A+1) + (A-1)*cosw0 + 2*np.sqrt(A)*alpha
    a1 =   -2*((A-1) + (A+1)*cosw0)
    a2 =        (A+1) + (A-1)*cosw0 - 2*np.sqrt(A)*alpha
    b = np.array([b0, b1, b2], dtype=np.float64)/a0
    a = np.array([1.0, a1/a0, a2/a0], dtype=np.float64)
    return signal.tf2sos(b, a)

def design_high_shelf(fc, gain_db, sr, Q=0.707):
    A  = 10**(gain_db/40); w0 = 2*np.pi*fc/sr
    alpha = np.sin(w0)/(2*Q); cosw0 = np.cos(w0)
    b0 =    A*((A+1) + (A-1)*cosw0 + 2*np.sqrt(A)*alpha)
    b1 = -2*A*((A-1) + (A+1)*cosw0)
    b2 =    A*((A+1) + (A-1)*cosw0 - 2*np.sqrt(A)*alpha)
    a0 =        (A+1) - (A-1)*cosw0 + 2*np.sqrt(A)*alpha
    a1 =    2*((A-1) - (A+1)*cosw0)
    a2 =        (A+1) - (A-1)*cosw0 - 2*np.sqrt(A)*alpha
    b = np.array([b0, b1, b2], dtype=np.float64)/a0
    a = np.array([1.0, a1/a0, a2/a0], dtype=np.float64)
    return signal.tf2sos(b, a)

def design_peaking(fc, gain_db, Q, sr):
    A  = 10**(gain_db/40); w0 = 2*np.pi*fc/sr
    alpha = np.sin(w0)/(2*Q); cosw0 = np.cos(w0)
    b0 = 1 + alpha*A; b1 = -2*cosw0; b2 = 1 - alpha*A
    a0 = 1 + alpha/A; a1 = -2*cosw0; a2 = 1 - alpha/A
    b = np.array([b0, b1, b2], dtype=np.float64)/a0
    a = np.array([1.0, a1/a0, a2/a0], dtype=np.float64)
    return signal.tf2sos(b, a)

def sos_apply(y, sos): return signal.sosfiltfilt(sos, y)

def cascade(sos_list):
    if not sos_list: return None
    out = sos_list[0]
    for s in sos_list[1:]:
        out = np.vstack([out, s])
    return out

def compressor(y, sr, threshold_db=-18.0, ratio=3.0, makeup_db=0.0, attack=0.005, release=0.100):
    eps = 1e-12
    win = max(1, int(0.010 * sr))
    rms = np.sqrt(signal.convolve(y**2, np.ones(win)/win, mode='same') + eps)
    lvl_db = 20*np.log10(rms + eps)
    knee_db = 6.0; thr = threshold_db
    over = lvl_db - thr
    gain_red_db = np.zeros_like(over)
    idx1 = over <= -knee_db/2; gain_red_db[idx1] = 0.0
    idx2 = (over > -knee_db/2) & (over < knee_db/2)
    x = (over[idx2] + knee_db/2) / knee_db
    comp = x**2 * (1 - 1/ratio)
    gain_red_db[idx2] = -comp * (over[idx2])
    idx3 = over >= knee_db/2
    gain_red_db[idx3] = -(over[idx3] - over[idx3]/ratio)
    target_gain = 10**((gain_red_db)/20)
    out_gain = np.zeros_like(target_gain)
    atk = np.exp(-1.0/(attack*sr)); rel = np.exp(-1.0/(release*sr))
    g = 1.0
    for i in range(len(target_gain)):
        tg = target_gain[i]
        if tg < g: g = atk*g + (1-atk)*tg
        else:      g = rel*g + (1-rel)*tg
        out_gain[i] = g
    makeup = 10**(makeup_db/20)
    return (y * out_gain * makeup).astype(np.float32)

def deesser(y, sr, low=6000, high=9000, max_reduction_db=6.0, attack=0.003, release=0.050):
    sos_bp = signal.butter(4, [low, high], btype='band', fs=sr, output='sos')
    s = signal.sosfiltfilt(sos_bp, y)
    win = max(1, int(0.005 * sr))
    env = np.sqrt(signal.convolve(s**2, np.ones(win)/win, mode='same') + 1e-12)
    ctrl = env / (np.max(env) + 1e-9)
    atk = np.exp(-1.0/(attack*sr)); rel = np.exp(-1.0/(release*sr))
    g = 0.0; sm = np.zeros_like(ctrl)
    for i in range(len(ctrl)):
        if ctrl[i] > g: g = atk*g + (1-atk)*ctrl[i]
        else:           g = rel*g + (1-rel)*ctrl[i]
        sm[i] = g
    reduction = 10**(-(sm * max_reduction_db)/20)
    return (y * reduction).astype(np.float32)

def harmonic_exciter(y, sr, mix=0.08, max_harm=3, strength=[1.0, 0.5, 0.3]):
    fmin, fmax = 70.0, 350.0
    try:
        f0, vflag, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, frame_length=2048, sr=sr)
    except Exception as e:
        warnings.warn(f"pyin failed ({e}); skipping exciter.")
        return y
    hop = 512; t = np.arange(len(y))/sr
    y_out = np.zeros_like(y, dtype=np.float32)
    frame_times = np.arange(len(f0)) * (hop/sr)
    win = max(1, int(0.020*sr))
    rms = np.sqrt(signal.convolve(y**2, np.ones(win)/win, mode='same') + 1e-12)
    for i, f in enumerate(f0):
        if not np.isfinite(f) or f <= 0 or (vflag is not None and vflag[i] < 0.5):
            continue
        t0 = frame_times[i]; t1 = t0 + (hop/sr)
        n0 = int(t0*sr); n1 = min(len(y), int(t1*sr))
        if n1 <= n0: continue
        tt = t[n0:n1]; amp = np.median(rms[n0:n1])
        sig = np.zeros_like(tt, dtype=np.float32)
        for h in range(1, int(max_harm)+1):
            k = h-1
            s = strength[k] if k < len(strength) else strength[-1] * (0.6**k)
            sig += (s * np.sin(2*np.pi*h*f*tt)).astype(np.float32)
        sig /= max(np.max(np.abs(sig)), 1e-6)
        sig *= amp
        y_out[n0:n1] += sig.astype(np.float32)
    sos_lp = signal.butter(4, 7000, btype='low', fs=sr, output='sos')
    y_out = signal.sosfiltfilt(sos_lp, y_out)
    return (y + mix*y_out).astype(np.float32)


def loudness_normalize_and_limit(y, sr, target_lufs=-14.0, true_peak_db=-1.0):
    """
    Normalize to target LUFS and ensure peaks do not exceed a true-peak ceiling.
    Simple post-normalization peak trim (no lookahead limiter artifacts).
    """
    # LUFS normalization
    meter = pyln.Meter(sr)  # K-weighted
    loudness = meter.integrated_loudness(y.astype(np.float64))  # pyloudnorm expects float64
    y_norm = pyln.normalize.loudness(y.astype(np.float64), loudness, target_lufs).astype(np.float32)

    # True-peak ceiling (approx with sample peak; conservative ceiling prevents clipping)
    ceiling_linear = 10 ** (true_peak_db / 20.0)  # e.g., -1 dBFS -> ~0.8913
    peak = float(np.max(np.abs(y_norm))) if y_norm.size else 0.0
    if peak > ceiling_linear and peak > 0:
        y_norm = (y_norm * (ceiling_linear / peak)).astype(np.float32)

    return y_norm

# =========================
# Pipeline
# =========================
def denoise_then_enhance():
    # ---- Load ----
    y, sr = load_audio(input_path, sr=None, mono=True)

    # ---- Analysis for auto-notch (middle 30s) ----
    total_dur = len(y) / sr
    start_time = max(0.0, total_dur/2 - 15.0)  # 30s window
    seg = extract_segment(y, sr, start_time, 30.0)
    freqs, psd = welch_psd(seg, sr, nperseg=8192)

    # ---- Build denoise filter cascade ----
    sos_list = []
    # Bandpass
    sos_list.append(butter_bandpass_sos(BANDPASS[0], BANDPASS[1], sr, order=4))
    # Hum + harmonics
    if HUM_BASE and HUM_BASE > 0:
        for f0 in hum_harmonics(HUM_BASE, sr, max_freq=sr/2 - 200, count=HUM_HARMONICS):
            try:
                sos_list.append(iir_notch_sos(f0, sr, Q=NOTCH_Q))
            except ValueError:
                pass
    # Auto-notch peaks
    if AUTO_NOTCH:
        suggested = suggest_notch_freqs(freqs, psd,
                                        max_peaks=AUTO_PEAKS,
                                        min_freq=AUTO_MINF,
                                        max_freq=AUTO_MAXF,
                                        prominence_db=AUTO_PROM_DB)
        for f0 in suggested:
            try:
                sos_list.append(iir_notch_sos(f0, sr, Q=NOTCH_Q))
            except ValueError:
                pass

    sos = cascade_sos(sos_list)
    y_filt = apply_sos(y, sos)

    # ---- Spectral gating ----
    if GATING_MODE != "off":
        noise_clip = None
        if GATING_MODE == "profile":
            noise_clip = extract_segment(y_filt, sr, PROFILE_START, PROFILE_DUR)
        y_clean = try_noisereduce(y_filt, sr, noise_clip=noise_clip, prop_decrease=NOISEREDUCE_PROP_DEC)
    else:
        y_clean = y_filt

    # ---- Save intermediate ----
    save_wav(cleaned_path, y_clean, sr)

    # ---- Enhancement EQ ----
    sos_eq = []
    if LOW_SHELF is not None:
        fc, g = LOW_SHELF
        sos_eq.append(design_low_shelf(fc, g, sr, Q=0.707))
    if PRESENCE is not None:
        fc, g = PRESENCE
        sos_eq.append(design_peaking(fc, g, Q=1.0, sr=sr))
    sos_eq = cascade(sos_eq)
    if sos_eq is not None:
        y_enh = sos_apply(y_clean, sos_eq)
    else:
        y_enh = y_clean

    # ---- De-esser ----
    if DEESSER is not None:
        low, high, maxdb = DEESSER
        y_enh = deesser(y_enh, sr, low=low, high=high, max_reduction_db=maxdb)

    # ---- Exciter ----
    if EXCITER_ON:
        y_enh = harmonic_exciter(y_enh, sr, mix=EXCITER_MIX, max_harm=EXCITER_HARMONICS)

    # ---- Compressor ----
    if COMPRESS is not None:
        thr, ratio, makeup, att, rel = COMPRESS
        y_enh = compressor(y_enh, sr, threshold_db=thr, ratio=ratio, makeup_db=makeup, attack=att, release=rel)

    # ---- Loudness normalize for YouTube & peak ceiling ----
    # Target: -14 LUFS, ceiling: -1 dBFS
    y_enh = loudness_normalize_and_limit(y_enh, sr, target_lufs=-14.0, true_peak_db=-1.0)

    # ---- Save final ----
    save_wav(enhanced_path, y_enh, sr)

    print("---- Done ----")
    print(f"Input        : {input_path}")
    print(f"Cleaned WAV  : {cleaned_path}")
    print(f"Enhanced WAV : {enhanced_path}")


FILE_NAMES = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
]
DIR_PATH = "lectures/"

for name in tqdm(FILE_NAMES, desc="Processing files", unit="file"):
    input_path = DIR_PATH + name + ".mp3"
    cleaned_path = DIR_PATH + name + "_cleaned.wav"
    enhanced_path = DIR_PATH + name + "_enhanced.wav"
    denoise_then_enhance()
