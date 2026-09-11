# Deployment Guide

## Local Windows run

1. Extract the ZIP file.
2. Open Command Prompt or PowerShell in the extracted `Defence_Speech_Enhancement` folder.
3. Create a virtual environment:

```text
python -m venv .venv
```

4. Activate it:

```text
.venv\Scripts\activate
```

5. Install dependencies:

```text
pip install -r requirements.txt
```

6. Start the application:

```text
streamlit run app.py
```

7. Open the local URL shown in the terminal and upload `data/demo_noisy_speech.wav`.

## Streamlit Community Cloud

1. Extract the ZIP.
2. Create a new GitHub repository.
3. Upload all project files, preserving the folders.
4. Open [Streamlit Community Cloud](https://share.streamlit.io/).
5. Connect your GitHub account and repository.
6. Select `app.py` as the deployment entry file.
7. Click **Deploy** and wait for the app to start.

No API keys or secrets are needed.
