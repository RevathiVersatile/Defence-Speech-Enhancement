# Defence Speech Enhancement System

## Problem statement
Defence communication environments can contain machinery, vehicle, helicopter-like low-frequency sound, background noise, and sudden high-amplitude disturbances. These conditions can reduce speech intelligibility. This project demonstrates a transparent signal-processing solution rather than claiming that a neural network was trained.

## Objective
Build an interactive Streamlit demo that detects impulsive noise, suppresses it with interpolation or attenuation, and applies an adaptive Least Mean Squares (LMS) noise canceller. The app compares the noisy input, LMS-only output, and the combined proposed method.

## Proposed solution and architecture

```text
Noisy Speech
    ↓
Preprocessing and WAV validation
    ↓
Impulsive Noise Detection
    ↓
Impulsive Noise Suppression
    ↓
Adaptive LMS ANC
    ↓
Enhanced Speech
    ↓
Visualization + Audio Output
```

Processing pipeline: **noisy WAV → impulsive detection → impulse suppression → LMS adaptive noise cancellation → waveform comparison and audio download**.

## Technologies used

- Python 3.9+
- Streamlit
- NumPy
- Matplotlib
- Python's standard `wave` module for dependency-light WAV validation and encoding

No external API key, machine-learning framework, or trained model is required.

## Algorithm explanation

The detector computes `abs(signal)`, estimates a robust baseline with the median, and marks samples above `threshold_factor × median(abs(signal))` with a small safety floor. Nearby marked samples are grouped into time regions. Each detected region is interpolated from its neighboring samples when possible; boundary regions are attenuated instead of deleted.

The LMS stage maintains adaptive filter weights and a reference buffer. For each sample, it calculates `y[n] = w[n]ᵀx[n]`, computes `e[n] = d[n] - y[n]`, and updates the weights using a normalized form of `w[n+1] = w[n] + μe[n]x[n]`. When a reference-noise WAV is not supplied, the app creates a documented high-pass residual estimate from the input. This fallback is useful for demonstration but is **not** a physically measured reference microphone signal.

## How to run locally

From the project root on Windows:

```text
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

On macOS/Linux, activate with `source .venv/bin/activate`. Then open the URL shown by Streamlit. Upload `data/demo_noisy_speech.wav` to test immediately.

## Streamlit Community Cloud deployment

1. Create a GitHub repository and upload this complete project folder.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect the GitHub repository.
4. Select `app.py` as the main file.
5. Click **Deploy**. No secrets are required.

## Demo instructions

Upload `data/demo_noisy_speech.wav`, leave the sensible defaults in the sidebar, and review the detected regions, three audio players, waveform comparison, shaded detector plot, and WAV download. The supplied recording is synthetic demonstration data, not an actual defence recording.

## Metrics and scientific honesty

The app reports sample rate, duration, impulse-region count, and RMS calculated from the current upload. It does not invent SNR, RMSE against clean speech, accuracy, percentage improvement, or experimental claims. Objective-reference metrics require clean ground-truth speech and are intentionally not displayed without it.

## Limitations

The fallback reference is an estimate, not a measured noise channel. The LMS result depends on the reference quality and parameter choices. Simple median-amplitude detection can confuse loud speech with impulses. The application currently accepts mono 16-bit PCM WAV files and does not resample mismatched reference files with a high-quality resampler.

## Future enhancements

Future work could add calibrated multi-channel microphone references, robust resampling, speech-activity-aware detection, objective evaluation on a labelled dataset, intelligibility metrics, and field recordings collected with appropriate permissions.

## Project structure

```text
Defence_Speech_Enhancement/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── demo_noisy_speech.wav
│   └── README.txt
├── results/
│   └── README.txt
└── docs/
    └── DEPLOYMENT.md
```
