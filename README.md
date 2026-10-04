# ASL Sign Language Recognition AI

An AI-powered application for recognizing isolated American Sign Language (ASL) signs using uploaded videos and live webcam input. The project combines computer vision, hand landmark extraction, and deep learning to classify signs and compose recognized signs into phrases.

## Project Overview

Developed as part of an AI/ML internship project, this application uses MediaPipe to extract hand landmarks and an LSTM-based neural network to recognize selected ASL signs.
## 🚀 Live Demo

**Try the ASL Sign Language Recognition AI here:**

👉 [Launch ASL Sign Language AI](https://asl-sign-language-recognition-ai-zgzbl2g4aeoqvv2hjscxvd.streamlit.app/)

Explore the application to upload an ASL video and view sign predictions, confidence scores, and hand-detection diagnostics.



### Key Features

- Recognizes selected ASL signs from uploaded videos.
- Supports real-time webcam-based sign recognition.
- Extracts hand landmarks using MediaPipe.
- Uses an LSTM neural network for sequence classification.
- Displays predicted signs and confidence scores.
- Supports phrase composition from recognized signs.
- Provides a Streamlit graphical user interface.

## Technology Stack

- Python
- TensorFlow / Keras
- MediaPipe
- OpenCV
- Streamlit
- streamlit-webrtc
- NumPy
- LSTM (Long Short-Term Memory)

## Dataset and Model

The project uses a selected subset of 20 signs from the ASL Citizen dataset.

| Dataset Split | Videos |
|---|---:|
| Training | 383 |
| Validation | 79 |
| Testing | 298 |
| **Total** | **760** |

### Evaluation Results

- Held-out test accuracy: **75.34%**
- Macro F1-score: **74.35%**

The confidence score shown for an individual prediction is different from overall model accuracy. Results may vary depending on lighting, camera angle, hand visibility, and signer variation.

## Project Structure

```text
ASL_Sign_Language_AI/
├── app/
│   └── app.py
├── data/
│   └── metadata/
├── outputs/
│   ├── classification_report.txt
│   ├── evaluation_summary.json
│   └── training_history.json
├── src/
│   ├── extract_landmarks.py
│   ├── prepare_sequences_active.py
│   ├── train_model_active.py
│   ├── evaluate_model_active.py
│   └── realtime_inference.py
├── .gitignore
└── README.md
```

Additional scripts and evaluation files are included in the repository.

## Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd ASL_Sign_Language_AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the required Python packages using the project's dependency list.

### 4. Prepare the model and data

The trained model, MediaPipe task file, and dataset are not included in this repository because of their size and applicable dataset licensing restrictions.

Place the required files in their expected local directories before running the application.

### 5. Run the application

```bash
streamlit run app/app.py
```

## Dataset and Licensing

This project uses the ASL Citizen dataset for research and educational development. Refer to the original dataset source and its license before downloading, using, or redistributing any dataset content.

The dataset itself is not redistributed in this repository.

## Limitations

- This project recognizes isolated signs; it is not a continuous ASL translation system.
- Recognition performance depends on video quality, lighting, hand visibility, and signer variation.
- The model is trained on a selected subset of signs and is not a complete ASL interpreter.
- Predictions are not guaranteed to be correct in every real-world situation.

## Future Improvements

- Expand the vocabulary and signer diversity.
- Improve recognition under varied lighting and camera conditions.
- Explore continuous sign recognition and sentence-level translation.
- Improve real-time inference speed and robustness.

## Author

Developed as an AI/ML internship project.
