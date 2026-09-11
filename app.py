"""Defence Speech Enhancement System - Streamlit demonstration app."""
from __future__ import annotations

import io
import wave
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


Region = Tuple[int, int]


def read_wav_bytes(data: bytes) -> Tuple[int, np.ndarray]:
    """Read a mono 16-bit PCM WAV from bytes and return (sample_rate, float signal)."""
    if not data:
        raise ValueError("The uploaded file is empty.")
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels, sample_width, sample_rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError("The file is not a valid WAV file or is corrupted.") from exc
    if channels != 1:
        raise ValueError("Please upload a mono WAV file.")
    if sample_width != 2:
        raise ValueError("Please upload a 16-bit PCM WAV file.")
    if sample_rate <= 0 or not frames:
        raise ValueError("The WAV file contains no audio samples.")
    signal = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if signal.size == 0 or not np.all(np.isfinite(signal)):
        raise ValueError("The WAV file contains invalid audio samples.")
    return sample_rate, signal


def wav_bytes(sample_rate: int, signal: np.ndarray) -> bytes:
    """Encode a floating-point signal as mono 16-bit PCM WAV bytes."""
    clipped = np.clip(np.nan_to_num(signal), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


def detect_impulses(signal: np.ndarray, threshold_factor: float, sample_rate: int, min_gap_ms: float = 8.0) -> Tuple[np.ndarray, List[Region], float]:
    """Detect and group samples whose absolute amplitude exceeds an adaptive median threshold."""
    magnitude = np.abs(signal)
    baseline = float(np.median(magnitude))
    # A small floor avoids an unusably low threshold for near-silent signals.
    threshold = max(threshold_factor * max(baseline, 1e-4), 0.05)
    mask = magnitude > threshold
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return mask, [], threshold
    max_gap = max(1, int(min_gap_ms * 0.001 * sample_rate))
    regions: List[Region] = []
    start = previous = int(indices[0])
    for index in indices[1:]:
        index = int(index)
        if index - previous > max_gap:
            regions.append((start, previous + 1))
            start = index
        previous = index
    regions.append((start, previous + 1))
    return mask, regions, threshold


def suppress_impulses(signal: np.ndarray, regions: List[Region], strength: float) -> np.ndarray:
    """Interpolate across impulse regions where possible, otherwise attenuate them."""
    output = signal.copy()
    strength = float(np.clip(strength, 0.0, 1.0))
    n = len(output)
    for raw_start, raw_end in regions:
        start, end = max(0, raw_start), min(n, raw_end)
        if end <= start:
            continue
        left, right = max(0, start - 1), min(n - 1, end)
        if left < start and end < n:
            output[start:end] = np.linspace(output[left], output[right], end - start, endpoint=False)
        else:
            output[start:end] *= (1.0 - strength)
    return output


def estimate_reference(signal: np.ndarray) -> np.ndarray:
    """Create a documented fallback reference estimate; it is not a measured microphone signal."""
    # A moving-average trend emphasizes slowly varying correlated noise.
    window = max(3, min(401, (len(signal) // 200) * 2 + 1))
    kernel = np.ones(window, dtype=np.float32) / window
    trend = np.convolve(signal, kernel, mode="same")
    return signal - trend


def lms_anc(desired: np.ndarray, reference: np.ndarray, mu: float, filter_length: int) -> np.ndarray:
    """Run a normalized LMS adaptive noise canceller and return its error signal."""
    n = min(len(desired), len(reference))
    d, ref = desired[:n].astype(np.float32), reference[:n].astype(np.float32)
    weights = np.zeros(filter_length, dtype=np.float32)
    buffer = np.zeros(filter_length, dtype=np.float32)
    error = np.zeros(n, dtype=np.float32)
    for i in range(n):
        buffer[1:] = buffer[:-1]
        buffer[0] = ref[i]
        predicted = float(np.dot(weights, buffer))
        error[i] = d[i] - predicted
        norm = float(np.dot(buffer, buffer)) + 1e-6
        weights += (mu / norm) * error[i] * buffer
    return np.clip(error, -1.0, 1.0)


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal)))) if signal.size else 0.0


def plot_waveforms(sample_rate: int, signals: List[np.ndarray], labels: List[str], title: str):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for signal, label in zip(signals, labels):
        time = np.arange(len(signal)) / sample_rate
        ax.plot(time, signal, linewidth=0.7, label=label, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="Defence Speech Enhancement", page_icon="🔊", layout="wide")
    st.title("Defence Speech Enhancement System")
    st.caption("Adaptive LMS Noise Cancellation + Impulsive Noise Detection")
    st.markdown("A practical signal-processing demonstration for noisy defence-style environments. This system uses no trained neural network.")

    with st.sidebar:
        st.header("System Parameters")
        mu = st.slider("LMS step size μ", 0.001, 1.0, 0.08, 0.001, help="Normalized LMS adaptation step size.")
        filter_length = st.slider("Adaptive filter length", 8, 256, 64, 8)
        threshold_factor = st.slider("Impulsive threshold factor", 1.5, 20.0, 5.0, 0.5)
        suppression_strength = st.slider("Impulse suppression strength", 0.0, 1.0, 0.85, 0.05)
        st.divider()
        st.info("If no reference WAV is supplied, the app uses a high-pass residual of the input as a practical fallback estimate. It is not a physically measured microphone reference.")

    st.header("1. Upload Audio")
    noisy_file = st.file_uploader("Upload noisy speech WAV", type=["wav"])
    reference_file = st.file_uploader("Optional reference-noise WAV", type=["wav"])
    if noisy_file is None:
        st.info("Upload a mono 16-bit PCM WAV to begin. You can test immediately with data/demo_noisy_speech.wav.")
        with st.expander("How the System Works", expanded=True):
            st.markdown("1. The noisy speech signal is received.\n2. Sudden high-amplitude disturbances are detected as impulsive noise.\n3. Detected impulsive regions are suppressed.\n4. LMS adaptive filtering estimates and removes correlated noise.\n5. The resulting signal is presented as enhanced speech.\n6. The original and enhanced signals can be compared visually and audibly.")
        return

    try:
        sample_rate, noisy = read_wav_bytes(noisy_file.getvalue())
    except ValueError as exc:
        st.error(str(exc))
        return
    reference: Optional[np.ndarray] = None
    if reference_file is not None:
        try:
            ref_rate, reference = read_wav_bytes(reference_file.getvalue())
            if ref_rate != sample_rate:
                st.warning("Reference sample rate differs; the reference will be resampled by simple index alignment only. Prefer matching sample rates.")
        except ValueError as exc:
            st.error(f"Reference WAV error: {exc}")
            return

    st.audio(wav_bytes(sample_rate, noisy), format="audio/wav")
    st.header("2. Signal Analysis")
    mask, regions, threshold = detect_impulses(noisy, threshold_factor, sample_rate)
    suppressed = suppress_impulses(noisy, regions, suppression_strength)
    ref = reference if reference is not None else estimate_reference(noisy)
    lms_only = lms_anc(noisy, ref, mu, filter_length)
    combined = lms_anc(suppressed, ref, mu, filter_length)
    duration = len(noisy) / sample_rate
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sample rate", f"{sample_rate:,} Hz")
    c2.metric("Duration", f"{duration:.2f} s")
    c3.metric("Impulse regions", str(len(regions)))
    c4.metric("Input RMS", f"{rms(noisy):.4f}")
    st.caption(f"Adaptive amplitude threshold: {threshold:.4f}. RMS is calculated from the uploaded signal; no experimental improvement is fabricated.")
    if regions:
        st.dataframe({"Start (s)": [round(a / sample_rate, 4) for a, _ in regions], "End (s)": [round(b / sample_rate, 4) for _, b in regions]}, use_container_width=True, hide_index=True)
    else:
        st.success("No samples exceeded the configured impulsive threshold.")

    st.header("3. Enhancement Results")
    cols = st.columns(3)
    outputs = [("Original / Noisy", noisy), ("LMS ANC", lms_only), ("Proposed: Impulsive Detection + LMS ANC", combined)]
    for col, (label, signal) in zip(cols, outputs):
        with col:
            st.subheader(label)
            st.audio(wav_bytes(sample_rate, signal), format="audio/wav")
            st.caption(f"RMS: {rms(signal):.4f}")
    st.download_button("Download Enhanced WAV", wav_bytes(sample_rate, combined), "enhanced_defence_speech.wav", "audio/wav", type="primary")

    st.header("4. Waveform Comparison")
    fig = plot_waveforms(sample_rate, [noisy, lms_only, combined], [x[0] for x in outputs], "Original and Enhanced Waveforms")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    st.header("5. Impulsive Noise Detection")
    fig2, ax = plt.subplots(figsize=(12, 3.8))
    time = np.arange(len(noisy)) / sample_rate
    ax.plot(time, noisy, color="#263238", linewidth=0.7, label="Input waveform")
    for start, end in regions:
        ax.axvspan(start / sample_rate, end / sample_rate, color="#ef5350", alpha=0.28)
    ax.set_title("Detected impulsive regions (shaded)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig2, clear_figure=True)
    plt.close(fig2)

    with st.expander("How the System Works"):
        st.markdown("1. The noisy speech signal is received.\n2. Sudden high-amplitude disturbances are detected as impulsive noise.\n3. Detected impulsive regions are suppressed using interpolation where neighboring samples are available.\n4. LMS adaptive filtering estimates and removes correlated noise.\n5. The resulting signal is presented as enhanced speech.\n6. The original and enhanced signals can be compared visually and audibly.")
        st.warning("Objective-reference metrics such as SNR, RMSE against clean speech, accuracy, or percentage improvement are not shown unless clean ground-truth speech is supplied. The optional reference-noise upload is for LMS adaptation, not clean ground truth.")


if __name__ == "__main__":
    main()
