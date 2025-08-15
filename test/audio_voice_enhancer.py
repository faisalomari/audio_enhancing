# audio_voice_enhancer.py
# -*- coding: utf-8 -*-
"""
Voice Enhancement Toolkit (post-cleaning)
----------------------------------------
Adds presence/low-end warmth, compresses dynamics, and (optionally) synthesizes
subtle harmonics locked to the speaker's f0 to increase clarity/intelligibility.

Usage example (after you've produced cleaned.wav):
    python audio_voice_enhancer.py --in cleaned.wav --low_shelf 120 3 --presence 3500 4 --deesser 6500 9000 3 \
        --compress -18 3 4 0.005 0.100 --exciter on --exciter_mix 0.08 --exciter_harmonics 3 --export enhanced.wav

Requires:
    numpy, scipy, soundfile, librosa, matplotlib (plots optional)
"""

import argparse
import numpy as np
import soundfile as sf
from scipy import signal
import librosa
import warnings
import matplotlib.pyplot as plt


def load_audio(path, sr=None, mono=True):
    y, sr = librosa.load(path, sr=sr, mono=mono)
    return np.ascontiguousarray(y.astype(np.float32)), sr


def save_wav(path, y, sr):
    sf.write(path, y, sr, subtype="FLOAT")


# -------------------- EQ filters --------------------

def design_low_shelf(fc, gain_db, sr, Q=0.707):
    """RBJ audio EQ cookbook low-shelf in SOS form."""
    A  = 10**(gain_db/40)
    w0 = 2*np.pi*fc/sr
    alpha = np.sin(w0)/(2*Q)
    cosw0 = np.cos(w0)

    b0 =    A*((A+1) - (A-1)*cosw0 + 2*np.sqrt(A)*alpha)
    b1 =  2*A*((A-1) - (A+1)*cosw0)
    b2 =    A*((A+1) - (A-1)*cosw0 - 2*np.sqrt(A)*alpha)
    a0 =        (A+1) + (A-1)*cosw0 + 2*np.sqrt(A)*alpha
    a1 =   -2*((A-1) + (A+1)*cosw0)
    a2 =        (A+1) + (A-1)*cosw0 - 2*np.sqrt(A)*alpha

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1/a0, a2/a0], dtype=np.float64)
    sos = signal.tf2sos(b, a)
    return sos

def design_high_shelf(fc, gain_db, sr, Q=0.707):
    A  = 10**(gain_db/40)
    w0 = 2*np.pi*fc/sr
    alpha = np.sin(w0)/(2*Q)
    cosw0 = np.cos(w0)

    b0 =    A*((A+1) + (A-1)*cosw0 + 2*np.sqrt(A)*alpha)
    b1 = -2*A*((A-1) + (A+1)*cosw0)
    b2 =    A*((A+1) + (A-1)*cosw0 - 2*np.sqrt(A)*alpha)
    a0 =        (A+1) - (A-1)*cosw0 + 2*np.sqrt(A)*alpha
    a1 =    2*((A-1) - (A+1)*cosw0)
    a2 =        (A+1) - (A-1)*cosw0 - 2*np.sqrt(A)*alpha

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1/a0, a2/a0], dtype=np.float64)
    sos = signal.tf2sos(b, a)
    return sos

def design_peaking(fc, gain_db, Q, sr):
    A  = 10**(gain_db/40)
    w0 = 2*np.pi*fc/sr
    alpha = np.sin(w0)/(2*Q)
    cosw0 = np.cos(w0)

    b0 = 1 + alpha*A
    b1 = -2*cosw0
    b2 = 1 - alpha*A
    a0 = 1 + alpha/A
    a1 = -2*cosw0
    a2 = 1 - alpha/A

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    a = np.array([1.0, a1/a0, a2/a0], dtype=np.float64)
    sos = signal.tf2sos(b, a)
    return sos

def sos_apply(y, sos):
    return signal.sosfiltfilt(sos, y)

def cascade(sos_list):
    if not sos_list:
        return None
    out = sos_list[0]
    for s in sos_list[1:]:
        out = np.vstack([out, s])
    return out

# -------------------- Compressor --------------------

def compressor(y, sr, threshold_db=-18.0, ratio=3.0, makeup_db=0.0, attack=0.005, release=0.100):
    """Simple feed-forward compressor with RMS detector and soft knee."""
    eps = 1e-12
    # RMS detector window ~10ms
    win = max(1, int(0.010 * sr))
    rms = np.sqrt(signal.convolve(y**2, np.ones(win)/win, mode='same') + eps)

    # Convert to dB
    lvl_db = 20*np.log10(rms + eps)
    knee_db = 6.0
    thr = threshold_db

    # Gain computer with soft knee
    over = lvl_db - thr
    gain_red_db = np.zeros_like(over)

    # Below knee start
    idx1 = over <= -knee_db/2
    gain_red_db[idx1] = 0.0

    # Knee region
    idx2 = (over > -knee_db/2) & (over < knee_db/2)
    x = (over[idx2] + knee_db/2) / knee_db  # 0..1
    comp = x**2 * (1 - 1/ratio)
    gain_red_db[idx2] = -comp * (over[idx2])

    # Above knee
    idx3 = over >= knee_db/2
    gain_red_db[idx3] = -(over[idx3] - over[idx3]/ratio)

    # Attack/Release smoothing on linear gain
    target_gain = 10**((gain_red_db)/20)
    out_gain = np.zeros_like(target_gain)
    atk = np.exp(-1.0/(attack*sr))
    rel = np.exp(-1.0/(release*sr))
    g = 1.0
    for i in range(len(target_gain)):
        tg = target_gain[i]
        if tg < g:
            g = atk*g + (1-atk)*tg
        else:
            g = rel*g + (1-rel)*tg
        out_gain[i] = g

    makeup = 10**(makeup_db/20)
    return (y * out_gain * makeup).astype(np.float32)

# -------------------- De-esser (dynamic high-band tame) --------------------

def deesser(y, sr, low=6000, high=9000, max_reduction_db=6.0, attack=0.003, release=0.050):
    # Bandpass sibilance region
    sos_bp = signal.butter(4, [low, high], btype='band', fs=sr, output='sos')
    s = signal.sosfiltfilt(sos_bp, y)
    # Envelope
    win = max(1, int(0.005 * sr))
    env = np.sqrt(signal.convolve(s**2, np.ones(win)/win, mode='same') + 1e-12)
    # Normalize control to 0..1
    ctrl = env / (np.max(env) + 1e-9)
    # Attack/release
    atk = np.exp(-1.0/(attack*sr))
    rel = np.exp(-1.0/(release*sr))
    g = 0.0
    sm = np.zeros_like(ctrl)
    for i in range(len(ctrl)):
        if ctrl[i] > g:
            g = atk*g + (1-atk)*ctrl[i]
        else:
            g = rel*g + (1-rel)*ctrl[i]
        sm[i] = g
    reduction = 10**(-(sm * max_reduction_db)/20)
    return (y * reduction).astype(np.float32)

# -------------------- Harmonic exciter (f0-based) --------------------

def harmonic_exciter(y, sr, mix=0.08, max_harm=3, strength=[1.0, 0.5, 0.3]):
    """
    Estimate f0 with librosa.pyin, synthesize sinusoids at f0, 2f0, 3f0 under voiced frames,
    amplitude-shaped by local RMS, then gently lowpass to avoid harshness. Mix at low level.
    """
    fmin, fmax = 70.0, 350.0  # adult speech range (adjust if needed)
    try:
        f0, vflag, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, frame_length=2048, sr=sr)
    except Exception as e:
        warnings.warn(f"pyin failed ({e}); skipping exciter.")
        return y

    hop = 512
    t = np.arange(len(y))/sr
    y_out = np.zeros_like(y, dtype=np.float32)

    # Frame-wise synthesis
    frame_times = np.arange(len(f0)) * (hop/sr)
    # Local RMS window ~20 ms
    win = max(1, int(0.020*sr))
    rms = np.sqrt(signal.convolve(y**2, np.ones(win)/win, mode='same') + 1e-12)

    for i, f in enumerate(f0):
        if not np.isfinite(f) or f <= 0 or (vflag is not None and vflag[i] < 0.5):
            continue
        # frame boundaries
        t0 = frame_times[i]
        t1 = t0 + (hop/sr)
        n0 = int(t0*sr)
        n1 = min(len(y), int(t1*sr))
        if n1 <= n0:
            continue
        tt = t[n0:n1]
        amp = np.median(rms[n0:n1])
        # Synthesize harmonics
        sig = np.zeros_like(tt, dtype=np.float32)
        for h in range(1, int(max_harm)+1):
            k = h-1
            if k < len(strength):
                s = strength[k]
            else:
                s = strength[-1] * (0.6**k)
            sig += (s * np.sin(2*np.pi*h*f*tt)).astype(np.float32)
        # Normalize and scale by local amplitude
        sig /= max(np.max(np.abs(sig)), 1e-6)
        sig *= amp
        y_out[n0:n1] += sig.astype(np.float32)

    # Gentle lowpass to tame harshness above ~7 kHz
    sos_lp = signal.butter(4, 7000, btype='low', fs=sr, output='sos')
    y_out = signal.sosfiltfilt(sos_lp, y_out)
    # Mix
    return (y + mix*y_out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="Enhance voice presence/low-end & clarity.")
    ap.add_argument("--in", dest="inp", required=True, help="Input WAV/MP3 (prefer the cleaned file)")
    ap.add_argument("--export", default="enhanced.wav", help="Output WAV path")
    ap.add_argument("--sr", type=int, default=None, help="Resample rate (default: keep)")

    # EQ
    ap.add_argument("--low_shelf", nargs=2, type=float, metavar=('FREQ_HZ','GAIN_DB'),
                    help="Low-shelf EQ: frequency (e.g., 120) and gain dB (e.g., 3)")
    ap.add_argument("--presence", nargs=2, type=float, metavar=('FREQ_HZ','GAIN_DB'),
                    help="Presence peaking EQ around 3–5 kHz: center freq and gain dB")
    ap.add_argument("--high_shelf", nargs=2, type=float, metavar=('FREQ_HZ','GAIN_DB'),
                    help="High-shelf boost/cut (use negative gain for taming harshness)")

    # De-esser
    ap.add_argument("--deesser", nargs=3, type=float, metavar=('LOW_HZ','HIGH_HZ','MAX_DB'),
                    help="Dynamic sibilance tame: band [low, high], max reduction dB")

    # Compressor
    ap.add_argument("--compress", nargs=5, type=float, metavar=('THR_DB','RATIO','MAKEUP_DB','ATTACK_S','RELEASE_S'),
                    help="Compressor params: threshold dBFS, ratio, makeup dB, attack s, release s")

    # Harmonic exciter
    ap.add_argument("--exciter", choices=["off","on"], default="off", help="Enable f0-locked harmonic exciter")
    ap.add_argument("--exciter_mix", type=float, default=0.08, help="Exciter mix (0..0.3 typical)")
    ap.add_argument("--exciter_harmonics", type=int, default=3, help="Number of harmonics to synthesize (1-5)")


    # Plot
    ap.add_argument("--plot", choices=["yes","no"], default="no")

    args = ap.parse_args()

    y, sr = load_audio(args.inp, sr=args.sr, mono=True)

    # EQ cascade
    sos_list = []

    if args.low_shelf is not None:
        fc, g = args.low_shelf
        sos_list.append(design_low_shelf(fc, g, sr, Q=0.707))

    if args.presence is not None:
        fc, g = args.presence
        # Q=1.0 wide presence bump
        sos_list.append(design_peaking(fc, g, Q=1.0, sr=sr))

    if args.high_shelf is not None:
        fc, g = args.high_shelf
        sos_list.append(design_high_shelf(fc, g, sr, Q=0.707))

    sos = cascade(sos_list)
    if sos is not None:
        y = sos_apply(y, sos)

    # De-esser
    if args.deesser is not None:
        low, high, maxdb = args.deesser
        y = deesser(y, sr, low=low, high=high, max_reduction_db=maxdb)

    # Exciter
    if args.exciter == "on":
        y = harmonic_exciter(y, sr, mix=args.exciter_mix, max_harm=args.exciter_harmonics)

    # Compressor
    if args.compress is not None:
        thr, ratio, makeup, att, rel = args.compress
        y = compressor(y, sr, threshold_db=thr, ratio=ratio, makeup_db=makeup, attack=att, release=rel)

    save_wav(args.export, y, sr)

    if args.plot == "yes":
        # Before/after magnitude response visualization for EQ only (if applied)
        if sos is not None:
            w, h = signal.sosfreqz(sos, worN=4096, fs=sr)
            plt.figure()
            plt.semilogx(w, 20*np.log10(np.maximum(np.abs(h), 1e-6)))
            plt.title("EQ magnitude response (dB)")
            plt.xlabel("Hz"); plt.ylabel("dB"); plt.grid(True, which='both', ls=':')
            plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
