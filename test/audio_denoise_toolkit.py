# audio_denoise_toolkit.py
# -*- coding: utf-8 -*-
"""
Audio Denoise Toolkit (90s lecture cleaner)
------------------------------------------
Usage (basic):
    python audio_denoise_toolkit.py --in input.mp3 --export cleaned.wav

Usage (with analysis of a 30s mid segment + plots + hum removal at 50 Hz):
    python audio_denoise_toolkit.py --in input.mp3 --segment_mid 1200 --segment_dur 30 \
        --hum 50 --harmonics 6 --plot yes --export cleaned.wav

Tip: Start by running with --plot yes to SEE the spectrum and spectrogram,
then refine options (e.g., add --auto_notch, adjust --bandpass).

Requires:
    - numpy, scipy, soundfile, librosa, matplotlib
    - (optional) noisereduce for spectral gating
"""

import os
import argparse
import warnings
import numpy as np
import soundfile as sf
from scipy import signal

# librosa for decoding mp3s robustly
import librosa

# plotting (only if --plot yes)
import matplotlib.pyplot as plt


def load_audio(path, sr=None, mono=True):
    """
    Load audio using librosa (works with mp3/wav/etc.).
    Returns y (float32) and sr (int). Preserves native sr if sr=None.
    """
    y, sr = librosa.load(path, sr=sr, mono=mono)
    # Ensure contiguous float32 for efficient filtering
    y = np.ascontiguousarray(y.astype(np.float32))
    return y, sr


def save_wav(path, y, sr):
    """Save audio as WAV (float32)."""
    sf.write(path, y, sr, subtype="FLOAT")


def time_to_samples(t, sr):
    return int(round(t * sr))


def extract_segment(y, sr, start_time, duration):
    """Extract a segment (start_time sec, duration sec). Clips safely."""
    n0 = max(0, time_to_samples(start_time, sr))
    n1 = min(len(y), n0 + time_to_samples(duration, sr))
    return y[n0:n1]


def plot_waveform(y, sr, title="Waveform"):
    t = np.arange(len(y)) / sr
    plt.figure()
    plt.plot(t, y)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()


def plot_spectrogram(y, sr, title="Spectrogram (dB)"):
    # STFT magnitude in dB
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) + 1e-10
    S_db = 20 * np.log10(S)
    freqs = np.linspace(0, sr/2, S.shape[0])
    times = np.arange(S.shape[1]) * (512 / sr)
    plt.figure()
    plt.imshow(S_db, origin="lower", aspect="auto",
               extent=[times[0], times[-1], freqs[0], freqs[-1]])
    plt.colorbar(label="dB")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.tight_layout()


def welch_psd(y, sr, nperseg=8192):
    """Compute Welch PSD for better frequency resolution on long audio."""
    freqs, psd = signal.welch(y, fs=sr, nperseg=nperseg, noverlap=nperseg//2)
    return freqs, psd


def suggest_notch_freqs(freqs, psd, max_peaks=8, min_freq=40, max_freq=12000,
                        prominence_db=12.0):
    """
    Suggest narrowband peaks to notch out.
    We detect peaks in dB domain and return their frequencies.
    """
    psd_db = 10 * np.log10(psd + 1e-20)
    # Limit to band of interest
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    f = freqs[mask]
    p = psd_db[mask]

    # Find peaks
    peaks, props = signal.find_peaks(p, prominence=prominence_db)
    # Sort peaks by prominence descending
    if len(peaks) == 0:
        return []
    prominences = props.get("prominences", np.zeros_like(peaks))
    order = np.argsort(prominences)[::-1]
    top_idx = order[:max_peaks]
    return [float(f[peaks[i]]) for i in top_idx]


def iir_notch_sos(freq, sr, Q=30.0):
    """
    Design a single notch (band-stop) IIR filter in SOS form.
    Q ~ 30 is narrow. Increase for narrower notches.
    """
    w0 = freq / (sr / 2.0)  # normalize to Nyquist
    if w0 >= 1.0:
        raise ValueError(f"Notch freq {freq} >= Nyquist, cannot design.")
    b, a = signal.iirnotch(w0, Q)
    sos = signal.tf2sos(b, a)
    return sos


def butter_bandpass_sos(lowcut, highcut, sr, order=4):
    """Speech-friendly bandpass (e.g., 80–8000 Hz)."""
    if lowcut <= 0:
        lowcut = 1.0
    if highcut >= sr/2:
        highcut = sr/2 - 1.0
    sos = signal.butter(order, [lowcut, highcut], btype="bandpass", fs=sr, output="sos")
    return sos


def cascade_sos(sos_list):
    """Concatenate multiple SOS filters into one cascade (stack rows)."""
    if not sos_list:
        return None
    sos = sos_list[0]
    for s in sos_list[1:]:
        sos = np.vstack([sos, s])
    return sos


def apply_sos(y, sos):
    if sos is None:
        return y
    # zero-phase (no latency) filtering
    return signal.sosfiltfilt(sos, y, axis=-1)


def hum_harmonics(freq, sr, max_freq=None, count=6):
    """
    Generate a list of harmonic frequencies for a base hum (50 or 60Hz).
    Stops before Nyquist / max_freq.
    """
    if max_freq is None:
        max_freq = sr/2 - 100
    out = []
    f = freq
    while f < max_freq and len(out) < count:
        out.append(f)
        f += freq
    return out


def try_noisereduce(y, sr, noise_clip=None, prop_decrease=0.8):
    """
    Optional spectral gating with noisereduce if installed.
    If noise_clip is provided (noise-only segment), use it as a noise profile.
    """
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


def main():
    ap = argparse.ArgumentParser(description="Denoise old lecture audio (mp3/wav)")
    ap.add_argument("--in", dest="inp", required=True, help="Input audio file (mp3/wav/etc.)")
    ap.add_argument("--export", default="cleaned.wav", help="Output WAV path")
    ap.add_argument("--sr", type=int, default=None, help="Target sample rate (default: keep native)")

    # Analysis window
    ap.add_argument("--segment_mid", type=float, default=None,
                    help="Center time (sec) of the 30s segment to analyze (or custom --segment_dur).")
    ap.add_argument("--segment_dur", type=float, default=30.0,
                    help="Duration (sec) of analysis segment.")

    # Filtering options
    ap.add_argument("--bandpass", nargs=2, type=float, default=None,
                    help="Speech bandpass: low high (Hz), e.g., 80 8000")
    ap.add_argument("--hum", type=float, default=None,
                    help="Base hum frequency (e.g., 50 or 60). Adds notch at harmonics.")
    ap.add_argument("--harmonics", type=int, default=6, help="Number of hum harmonics to notch.")
    ap.add_argument("--notchQ", type=float, default=30.0, help="Q for iirnotch (higher=narrower).")

    # Automatic narrowband peak notching from PSD
    ap.add_argument("--auto_notch", action="store_true", help="Auto-detect narrowband noise peaks and notch them.")
    ap.add_argument("--auto_peaks", type=int, default=6, help="Max peaks to notch when --auto_notch.")
    ap.add_argument("--auto_prom_db", type=float, default=12.0, help="Prominence dB threshold for peak detection.")
    ap.add_argument("--auto_minfreq", type=float, default=40.0, help="Min freq for auto notch search.")
    ap.add_argument("--auto_maxfreq", type=float, default=12000.0, help="Max freq for auto notch search.")

    # Spectral gating (noisereduce)
    ap.add_argument("--gating", choices=["off", "auto", "profile"], default="off",
                    help="Spectral gating: off | auto | profile (uses noise-only profile segment).")
    ap.add_argument("--profile_start", type=float, default=0.0,
                    help="Start time for noise-only profile (sec), used if gating=profile.")
    ap.add_argument("--profile_dur", type=float, default=2.0,
                    help="Duration for noise-only profile (sec).")

    # Plots
    ap.add_argument("--plot", choices=["yes", "no"], default="no", help="Show plots.")

    args = ap.parse_args()

    # Load audio
    y, sr = load_audio(args.inp, sr=args.sr, mono=True)

    # ----- Analysis segment -----
    if args.segment_mid is not None:
        start_time = max(0.0, args.segment_mid - args.segment_dur/2)
    else:
        # Default to the middle of the full file
        total_dur = len(y) / sr
        start_time = max(0.0, total_dur/2 - args.segment_dur/2)

    seg = extract_segment(y, sr, start_time, args.segment_dur)

    # Compute PSD for the analysis segment
    freqs, psd = welch_psd(seg, sr, nperseg=8192)

    # Auto notch suggestion
    suggested = []
    if args.auto_notch:
        suggested = suggest_notch_freqs(freqs, psd, max_peaks=args.auto_peaks,
                                        min_freq=args.auto_minfreq, max_freq=args.auto_maxfreq,
                                        prominence_db=args.auto_prom_db)

    # Build filter cascade
    sos_list = []

    # Optional bandpass
    if args.bandpass is not None:
        low, high = args.bandpass
        sos_list.append(butter_bandpass_sos(low, high, sr, order=4))

    # Hum + harmonics
    if args.hum is not None and args.hum > 0:
        hums = hum_harmonics(args.hum, sr, max_freq=sr/2 - 200, count=args.harmonics)
        for f0 in hums:
            try:
                sos_list.append(iir_notch_sos(f0, sr, Q=args.notchQ))
            except ValueError:
                pass  # skip if above Nyquist

    # Auto-detected narrow peaks
    for f0 in suggested:
        try:
            sos_list.append(iir_notch_sos(f0, sr, Q=args.notchQ))
        except ValueError:
            pass

    sos = cascade_sos(sos_list)

    # Apply filters
    y_filt = apply_sos(y, sos)

    # Optional spectral gating
    if args.gating != "off":
        noise_clip = None
        if args.gating == "profile":
            noise_clip = extract_segment(y_filt, sr, args.profile_start, args.profile_dur)
        y_clean = try_noisereduce(y_filt, sr, noise_clip=noise_clip, prop_decrease=0.8)
    else:
        y_clean = y_filt

    # Export
    save_wav(args.export, y_clean, sr)

    # Print a tiny report
    print("---- Denoise Report ----")
    print(f"Input: {args.inp}")
    print(f"Sample rate: {sr} Hz")
    if args.bandpass is not None:
        print(f"Bandpass: {args.bandpass[0]:.0f}-{args.bandpass[1]:.0f} Hz")
    if args.hum is not None:
        print(f"Hum base: {args.hum} Hz | Harmonics: {args.harmonics}")
    if args.auto_notch:
        print("Auto-notch suggested frequencies (Hz):", [round(f, 1) for f in suggested])
    print(f"Spectral gating: {args.gating}")
    print(f"Exported: {args.export}")

    # Plots
    if args.plot == "yes":
        plot_waveform(seg, sr, title=f"Waveform (segment @ {start_time:.1f}s, {args.segment_dur:.0f}s)")
        plot_spectrogram(seg, sr, title="Spectrogram of analysis segment (dB)")

        # PSD plot
        plt.figure()
        psd_db = 10 * np.log10(psd + 1e-20)
        plt.semilogx(freqs, psd_db)
        plt.xlabel("Frequency (Hz, log scale)")
        plt.ylabel("PSD (dB)")
        plt.title("Welch PSD of analysis segment")
        plt.grid(True, which="both", ls=":")
        plt.tight_layout()

        # Show
        plt.show()


if __name__ == "__main__":
    main()
