# Alzheimer Detection

An educational deep-learning application for classifying brain images into
`Non Demented` or `Demented` using a PyTorch CNN and Gradio.

## Project files

- `app.py` - Gradio inference application
- `alzmodel.pth` - trained Alzheimer classification model
- `Model Training.txt` - training workflow and notebook reference
- `requirements.txt` - Python dependencies

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Upload an MRI image in the web interface to receive the model prediction and
an informational recommendation.

## Disclaimer

This project is for educational and research purposes only. Its predictions
are not medical advice or a diagnosis. Consult a qualified healthcare
professional for medical interpretation.
