# EmotionSense-AI

EmotionSense-AI is a facial-expression recognition project built with Python, Keras, and TensorFlow. It trains a convolutional neural network (CNN) on 48x48 grayscale face images and predicts one of seven emotions:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The main workflow is available in [`main.ipynb`](./main.ipynb). The repository also contains a saved model that can be loaded without retraining.

## Project contents

| Path | Description |
| --- | --- |
| [`main.ipynb`](./main.ipynb) | End-to-end data preparation, model definition, evaluation, and prediction notebook |
| [`emotiondetector.json`](./emotiondetector.json) | Saved Keras model architecture |
| [`emotiondetector.h5`](./emotiondetector.h5) | Saved model weights |
| [`images/train/`](./images/train/) | Training images arranged by emotion |
| [`images/test/`](./images/test/) | Test images arranged by emotion |
| [`dataset.zip`](./dataset.zip) | Compressed dataset archive |
| [`main.py`](./main.py) | Small Python scratch file; it is not the application entry point |

## How the workflow works

1. Images are read from `images/train/<emotion>/` and `images/test/<emotion>/`.
2. Image paths and labels are collected into pandas data frames.
3. Images are loaded as 48x48 grayscale arrays and reshaped to `(48, 48, 1)`.
4. Pixel values are normalized to the `[0, 1]` range.
5. Emotion labels are converted to integer and one-hot encoded values.
6. A CNN with convolution, max-pooling, dropout, and dense layers is trained/evaluated.
7. The architecture and weights are saved to `emotiondetector.json` and `emotiondetector.h5`.
8. A saved model can classify a new 48x48 face image and return the predicted emotion.

## Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- TensorFlow/Keras
- pandas
- NumPy
- tqdm
- matplotlib
- scikit-learn

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow pandas numpy tqdm matplotlib scikit-learn jupyter
```

## Run the notebook

From the repository root:

```bash
jupyter notebook main.ipynb
```

Run the notebook cells in order. If starting with a fresh kernel, create the training features after defining `extract_features` and before normalizing the data:

```python
train_features = extract_features(train["image"])
test_features = extract_features(valid_images)
```

The notebook evaluates the model, exports the architecture and weights, reloads them, and demonstrates predictions and a confusion matrix.

## Dataset layout

The expected directory structure is:

```text
images/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

If the image directories are missing, extract [`dataset.zip`](./dataset.zip) and preserve this layout. Images used by the notebook must be 48x48 pixels.

## Loading the saved model

The saved architecture and weights can be loaded with Keras:

```python
from keras.models import model_from_json

with open("emotiondetector.json", "r", encoding="utf-8") as model_file:
    model = model_from_json(model_file.read())

model.load_weights("emotiondetector.h5")
```

For prediction, preprocess an input image as grayscale, resize it to 48x48, reshape it to `(1, 48, 48, 1)`, and divide pixel values by `255.0`. Map the output index using:

```python
labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
```

## Notes

- Run notebook commands from the repository root so relative paths resolve correctly.
- The notebook filters test images whose dimensions do not match the model input shape.
- No automated test suite or command-line prediction interface is currently included.
- Predictions are intended for experimentation and educational use; they should not be treated as a definitive assessment of a person's emotional state.

## License

No license file is currently included. Add a license before distributing or reusing this project.
