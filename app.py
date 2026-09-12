"""
Defence Speech Enhancement System
Streamlit demonstration app

Core pipeline:
Noisy Speech
    ↓
Impulsive Noise Detection
    ↓
Impulsive Noise Suppression
    ↓
Safe Adaptive LMS / NLMS Filtering
    ↓
Speech Preservation
    ↓
Enhanced Speech
"""

from __future__ import annotations

import io
import wave
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


Region = Tuple[int, int]


# ============================================================
# 1. WAV FILE READING
# ============================================================

def read_wav_bytes(data: bytes) -> Tuple[int, np.ndarray]:
    """Read mono 16-bit PCM WAV and return sample rate + float signal."""

    if not data:
        raise ValueError("The uploaded file is empty.")

    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())

    except (wave.Error, EOFError) as exc:
        raise ValueError(
            "The file is not a valid WAV file or is corrupted."
        ) from exc

    if channels != 1:
        raise ValueError("Please upload a mono WAV file.")

    if sample_width != 2:
        raise ValueError("Please upload a 16-bit PCM WAV file.")

    if sample_rate <= 0 or not frames:
        raise ValueError("The WAV file contains no audio samples.")

    signal = (
        np.frombuffer(frames, dtype="<i2")
        .astype(np.float32)
        / 32768.0
    )

    if signal.size == 0 or not np.all(np.isfinite(signal)):
        raise ValueError("The WAV file contains invalid audio samples.")

    return sample_rate, signal


# ============================================================
# 2. WAV FILE WRITING
# ============================================================

def wav_bytes(sample_rate: int, signal: np.ndarray) -> bytes:
    """Convert floating-point signal to mono 16-bit PCM WAV."""

    cleaned = np.nan_to_num(
        signal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    clipped = np.clip(cleaned, -1.0, 1.0)

    pcm = (clipped * 32767.0).astype("<i2")

    output = io.BytesIO()

    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm.tobytes())

    return output.getvalue()


# ============================================================
# 3. IMPULSIVE NOISE DETECTION
# ============================================================

def detect_impulses(
    signal: np.ndarray,
    threshold_factor: float,
    sample_rate: int,
    min_gap_ms: float = 8.0,
) -> Tuple[np.ndarray, List[Region], float]:
    """
    Detect sudden high-amplitude disturbances.

    Uses an adaptive median-based amplitude threshold.
    """

    magnitude = np.abs(signal)

    baseline = float(np.median(magnitude))

    threshold = max(
        threshold_factor * max(baseline, 1e-4),
        0.05,
    )

    mask = magnitude > threshold

    indices = np.flatnonzero(mask)

    if indices.size == 0:
        return mask, [], threshold

    max_gap = max(
        1,
        int(min_gap_ms * 0.001 * sample_rate),
    )

    regions: List[Region] = []

    start = previous = int(indices[0])

    for index in indices[1:]:

        index = int(index)

        if index - previous > max_gap:

            regions.append(
                (start, previous + 1)
            )

            start = index

        previous = index

    regions.append(
        (start, previous + 1)
    )

    return mask, regions, threshold


# ============================================================
# 4. IMPULSIVE NOISE SUPPRESSION
# ============================================================

def suppress_impulses(
    signal: np.ndarray,
    regions: List[Region],
    strength: float,
) -> np.ndarray:
    """
    Suppress detected impulsive regions.

    Interpolation is preferred when samples exist
    on both sides of the impulse.
    """

    output = signal.copy()

    strength = float(
        np.clip(strength, 0.0, 1.0)
    )

    n = len(output)

    for raw_start, raw_end in regions:

        start = max(0, raw_start)
        end = min(n, raw_end)

        if end <= start:
            continue

        left = max(0, start - 1)
        right = min(n - 1, end)

        if left < start and end < n:

            output[start:end] = np.linspace(
                output[left],
                output[right],
                end - start,
                endpoint=False,
            )

        else:

            output[start:end] *= (
                1.0 - strength
            )

    return output


# ============================================================
# 5. SAFE FALLBACK REFERENCE
# ============================================================

def estimate_reference(signal: np.ndarray) -> np.ndarray:
    """
    Create a conservative pseudo-reference.

    IMPORTANT:
    This is NOT a measured microphone noise reference.

    It is only a practical fallback for demonstration when
    the user does not upload an independent noise reference.
    """

    if len(signal) < 10:
        return np.zeros_like(signal)

    # Moving-average trend.
    window = max(
        5,
        min(
            401,
            (len(signal) // 200) * 2 + 1,
        ),
    )

    kernel = (
        np.ones(window, dtype=np.float32)
        / window
    )

    trend = np.convolve(
        signal,
        kernel,
        mode="same",
    )

    residual = signal - trend

    # Limit the pseudo-reference energy.
    # This prevents the LMS from aggressively cancelling speech.
    signal_rms = rms(signal)
    reference_rms = rms(residual)

    if reference_rms > 1e-8:

        max_reference_rms = max(
            signal_rms * 0.35,
            1e-4,
        )

        scale = min(
            1.0,
            max_reference_rms / reference_rms,
        )

        residual = residual * scale

    return residual.astype(np.float32)


# ============================================================
# 6. NORMALIZED LMS
# ============================================================

def lms_anc(
    desired: np.ndarray,
    reference: np.ndarray,
    mu: float,
    filter_length: int,
) -> np.ndarray:
    """
    Normalized LMS adaptive noise cancellation.

    Returns the LMS error signal.
    """

    n = min(
        len(desired),
        len(reference),
    )

    if n == 0:
        return np.array([], dtype=np.float32)

    d = desired[:n].astype(np.float32)

    ref = reference[:n].astype(np.float32)

    filter_length = int(
        max(1, filter_length)
    )

    # Keep adaptation stable.
    mu = float(
        np.clip(mu, 0.001, 0.5)
    )

    weights = np.zeros(
        filter_length,
        dtype=np.float32,
    )

    buffer = np.zeros(
        filter_length,
        dtype=np.float32,
    )

    error = np.zeros(
        n,
        dtype=np.float32,
    )

    for i in range(n):

        # Shift reference buffer.
        buffer[1:] = buffer[:-1]

        buffer[0] = ref[i]

        # Estimated correlated noise.
        predicted = float(
            np.dot(weights, buffer)
        )

        # LMS error.
        error_i = (
            d[i] - predicted
        )

        error[i] = error_i

        # NLMS normalization.
        norm = (
            float(np.dot(buffer, buffer))
            + 1e-6
        )

        weights += (
            (mu / norm)
            * error_i
            * buffer
        )

        # Prevent numerical runaway.
        weights = np.clip(
            weights,
            -2.0,
            2.0,
        )

    return np.clip(
        error,
        -1.0,
        1.0,
    )


# ============================================================
# 7. SPEECH-PRESERVATION / SAFE LMS BLENDING
# ============================================================

def safe_lms_output(
    desired: np.ndarray,
    lms_error: np.ndarray,
    correction_strength: float = 0.30,
) -> np.ndarray:
    """
    Protect speech from excessive LMS cancellation.

    Instead of blindly using the LMS error signal,
    calculate the estimated correction and apply only
    a controlled fraction of it.
    """

    desired = desired.astype(
        np.float32
    )

    lms_error = lms_error.astype(
        np.float32
    )

    n = min(
        len(desired),
        len(lms_error),
    )

    if n == 0:
        return desired.copy()

    d = desired[:n]

    e = lms_error[:n]

    # LMS estimated noise/correction.
    estimated_noise = d - e

    # Limit correction amplitude relative to input.
    input_rms = rms(d)

    if input_rms > 1e-8:

        max_correction = (
            input_rms * 0.60
        )

        estimated_noise = np.clip(
            estimated_noise,
            -max_correction,
            max_correction,
        )

    # Apply only part of the LMS correction.
    strength = float(
        np.clip(
            correction_strength,
            0.0,
            0.50,
        )
    )

    enhanced = (
        d
        - strength * estimated_noise
    )

    # Safety normalization.
    peak = np.max(
        np.abs(enhanced)
    )

    if peak > 0.98:
        enhanced = (
            enhanced / peak
        ) * 0.95

    return np.clip(
        enhanced,
        -1.0,
        1.0,
    )


# ============================================================
# 8. RMS
# ============================================================

def rms(signal: np.ndarray) -> float:

    if signal.size == 0:
        return 0.0

    return float(
        np.sqrt(
            np.mean(
                np.square(signal)
            )
        )
    )


# ============================================================
# 9. WAVEFORM PLOT
# ============================================================

def plot_waveforms(
    sample_rate: int,
    signals: List[np.ndarray],
    labels: List[str],
    title: str,
):

    fig, ax = plt.subplots(
        figsize=(12, 4.5)
    )

    for signal, label in zip(
        signals,
        labels,
    ):

        time = (
            np.arange(len(signal))
            / sample_rate
        )

        ax.plot(
            time,
            signal,
            linewidth=0.7,
            label=label,
            alpha=0.8,
        )

    ax.set_title(title)

    ax.set_xlabel(
        "Time (seconds)"
    )

    ax.set_ylabel(
        "Amplitude"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        loc="upper right"
    )

    fig.tight_layout()

    return fig


# ============================================================
# 10. MAIN STREAMLIT APPLICATION
# ============================================================

def main():

    st.set_page_config(
        page_title="Defence Speech Enhancement",
        page_icon="🔊",
        layout="wide",
    )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    st.title(
        "Defence Speech Enhancement System"
    )

    st.caption(
        "Adaptive LMS Noise Cancellation + "
        "Impulsive Noise Detection"
    )

    st.markdown(
        """
        A practical signal-processing demonstration
        for noisy defence-style environments.

        The current prototype uses adaptive DSP techniques
        and does not use a trained neural network.
        """
    )

    # --------------------------------------------------------
    # SIDEBAR PARAMETERS
    # --------------------------------------------------------

    with st.sidebar:

        st.header(
            "System Parameters"
        )

        # SAFER DEFAULT μ
        mu = st.slider(
            "LMS step size μ",
            0.001,
            0.30,
            0.01,
            0.001,
            help=(
                "Conservative normalized LMS "
                "adaptation step size."
            ),
        )

        filter_length = st.slider(
            "Adaptive filter length",
            8,
            256,
            64,
            8,
        )

        threshold_factor = st.slider(
            "Impulsive threshold factor",
            1.5,
            20.0,
            5.0,
            0.5,
        )

        suppression_strength = st.slider(
            "Impulse suppression strength",
            0.0,
            1.0,
            0.85,
            0.05,
        )

        st.divider()

        st.info(
            """
            If no independent reference-noise WAV is
            supplied, the system uses a conservative
            signal-derived pseudo-reference.

            This is only a demonstration fallback and
            is NOT a physically measured microphone
            noise reference.
            """
        )

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    st.header(
        "1. Upload Audio"
    )

    noisy_file = st.file_uploader(
        "Upload noisy speech WAV",
        type=["wav"],
    )

    reference_file = st.file_uploader(
        "Optional reference-noise WAV",
        type=["wav"],
    )

    if noisy_file is None:

        st.info(
            "Upload a mono 16-bit PCM WAV to begin."
        )

        with st.expander(
            "How the System Works",
            expanded=True,
        ):

            st.markdown(
                """
                1. Noisy defence speech is received.
                2. Impulsive noise is detected.
                3. Detected impulses are suppressed.
                4. Adaptive LMS filtering estimates correlated noise.
                5. Speech-preservation control prevents excessive cancellation.
                6. Enhanced speech is presented for comparison.
                """
            )

        return

    # --------------------------------------------------------
    # READ INPUT AUDIO
    # --------------------------------------------------------

    try:

        sample_rate, noisy = read_wav_bytes(
            noisy_file.getvalue()
        )

    except ValueError as exc:

        st.error(str(exc))

        return

    # --------------------------------------------------------
    # READ OPTIONAL REFERENCE
    # --------------------------------------------------------

    reference: Optional[np.ndarray] = None

    if reference_file is not None:

        try:

            ref_rate, reference = read_wav_bytes(
                reference_file.getvalue()
            )

            if ref_rate != sample_rate:

                st.warning(
                    "Reference sample rate differs "
                    "from the input. For best results, "
                    "use matching sample rates."
                )

            if len(reference) != len(noisy):

                min_len = min(
                    len(reference),
                    len(noisy),
                )

                reference = reference[
                    :min_len
                ]

                st.info(
                    "Reference and input lengths differ; "
                    "the common duration is used."
                )

        except ValueError as exc:

            st.error(
                f"Reference WAV error: {exc}"
            )

            return

    # --------------------------------------------------------
    # ORIGINAL AUDIO
    # --------------------------------------------------------

    st.audio(
        wav_bytes(
            sample_rate,
            noisy,
        ),
        format="audio/wav",
    )

    # --------------------------------------------------------
    # SIGNAL ANALYSIS
    # --------------------------------------------------------

    st.header(
        "2. Signal Analysis"
    )

    mask, regions, threshold = detect_impulses(
        noisy,
        threshold_factor,
        sample_rate,
    )

    # --------------------------------------------------------
    # IMPULSIVE NOISE SUPPRESSION
    # --------------------------------------------------------

    suppressed = suppress_impulses(
        noisy,
        regions,
        suppression_strength,
    )

    # --------------------------------------------------------
    # REFERENCE SELECTION
    # --------------------------------------------------------

    if reference is not None:

        # Real uploaded reference.
        ref = reference

        reference_mode = (
            "Measured reference-noise WAV"
        )

    else:

        # Safe fallback.
        ref = estimate_reference(
            suppressed
        )

        reference_mode = (
            "Signal-derived pseudo-reference "
            "(fallback)"
        )

    # --------------------------------------------------------
    # LMS ONLY
    # --------------------------------------------------------

    raw_lms = lms_anc(
        noisy,
        ref,
        mu,
        filter_length,
    )

    # SAFELY BLEND LMS CORRECTION.
    lms_only = safe_lms_output(
        noisy,
        raw_lms,
        correction_strength=0.25,
    )

    # --------------------------------------------------------
    # PROPOSED COMBINED OUTPUT
    # --------------------------------------------------------

    raw_combined = lms_anc(
        suppressed,
        ref,
        mu,
        filter_length,
    )

    combined = safe_lms_output(
        suppressed,
        raw_combined,
        correction_strength=0.25,
    )

    # --------------------------------------------------------
    # SIGNAL METRICS
    # --------------------------------------------------------

    duration = (
        len(noisy)
        / sample_rate
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Sample rate",
        f"{sample_rate:,} Hz",
    )

    c2.metric(
        "Duration",
        f"{duration:.2f} s",
    )

    c3.metric(
        "Impulse regions",
        str(len(regions)),
    )

    c4.metric(
        "Input RMS",
        f"{rms(noisy):.4f}",
    )

    st.caption(
        f"Adaptive amplitude threshold: "
        f"{threshold:.4f}"
    )

    st.caption(
        f"LMS reference mode: {reference_mode}"
    )

    # --------------------------------------------------------
    # IMPULSE TABLE
    # --------------------------------------------------------

    if regions:

        st.dataframe(
            {
                "Start (s)": [
                    round(
                        a / sample_rate,
                        4,
                    )
                    for a, _ in regions
                ],
                "End (s)": [
                    round(
                        b / sample_rate,
                        4,
                    )
                    for _, b in regions
                ],
            },
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.success(
            "No samples exceeded the configured "
            "impulsive threshold."
        )

    # --------------------------------------------------------
    # ENHANCEMENT RESULTS
    # --------------------------------------------------------

    st.header(
        "3. Enhancement Results"
    )

    cols = st.columns(3)

    outputs = [
        (
            "Original / Noisy",
            noisy,
        ),
        (
            "LMS ANC",
            lms_only,
        ),
        (
            "Proposed: Impulsive Detection + LMS ANC",
            combined,
        ),
    ]

    for col, (label, signal) in zip(
        cols,
        outputs,
    ):

        with col:

            st.subheader(label)

            st.audio(
                wav_bytes(
                    sample_rate,
                    signal,
                ),
                format="audio/wav",
            )

            st.caption(
                f"RMS: {rms(signal):.4f}"
            )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    st.download_button(
        "Download Enhanced WAV",
        wav_bytes(
            sample_rate,
            combined,
        ),
        "enhanced_defence_speech.wav",
        "audio/wav",
        type="primary",
    )

    # --------------------------------------------------------
    # WAVEFORM COMPARISON
    # --------------------------------------------------------

    st.header(
        "4. Waveform Comparison"
    )

    fig = plot_waveforms(
        sample_rate,
        [
            noisy,
            lms_only,
            combined,
        ],
        [
            x[0]
            for x in outputs
        ],
        "Original and Enhanced Waveforms",
    )

    st.pyplot(
        fig,
        clear_figure=True,
    )

    plt.close(fig)

    # --------------------------------------------------------
    # IMPULSIVE NOISE DETECTION
    # --------------------------------------------------------

    st.header(
        "5. Impulsive Noise Detection"
    )

    fig2, ax = plt.subplots(
        figsize=(12, 3.8)
    )

    time = (
        np.arange(len(noisy))
        / sample_rate
    )

    ax.plot(
        time,
        noisy,
        linewidth=0.7,
        label="Input waveform",
    )

    for start, end in regions:

        ax.axvspan(
            start / sample_rate,
            end / sample_rate,
            alpha=0.28,
        )

    ax.set_title(
        "Detected impulsive regions"
    )

    ax.set_xlabel(
        "Time (seconds)"
    )

    ax.set_ylabel(
        "Amplitude"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    st.pyplot(
        fig2,
        clear_figure=True,
    )

    plt.close(fig2)

    # --------------------------------------------------------
    # HOW SYSTEM WORKS
    # --------------------------------------------------------

    with st.expander(
        "How the System Works"
    ):

        st.markdown(
            """
            ### Processing Pipeline

            **1. Noisy Speech Input**

            Defence-style speech is received as a WAV signal.

            **2. Impulsive Noise Detection**

            Sudden high-amplitude disturbances are detected
            using an adaptive amplitude threshold.

            **3. Impulsive Noise Suppression**

            Detected impulse regions are attenuated or
            interpolated using neighboring samples.

            **4. Adaptive LMS Noise Cancellation**

            Normalized LMS estimates correlated noise and
            continuously adapts the filter coefficients.

            **5. Speech Preservation**

            The LMS correction is controlled so that
            excessive cancellation of speech is avoided.

            **6. Enhanced Speech**

            The final signal combines impulsive-noise
            suppression with safe adaptive filtering.
            """
        )

        st.info(
            """
            The current prototype is a software validation
            of the core signal-processing approach.

            A physically measured reference microphone,
            real-time microphone input and embedded
            hardware implementation are future extensions.
            """
        )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
