# Facial Emotion Recognition with AffectNet, MediaPipe, and Transfer Learning

This project builds an end-to-end facial emotion recognition pipeline for still images. It combines AffectNet-based datasets, MediaPipe Face Detection, and transfer learning with pretrained CNN backbones to classify facial expressions into eight emotion classes.

## Overview

The current system expands the earlier 7-class MobileNetV2 baseline into an 8-class setup by adding the `contempt` class. It uses two dataset sources:

- `mstjebashazida/affectnet`
- `fatihkgg/affectnet-yolo-format`

The final emotion classes are:

```text
anger, contempt, disgust, fear, happy, neutral, sad, surprise
```

Three main transfer-learning models are trained and evaluated on the same dataset split:

- MobileNetV2
- EfficientNetB0
- ResNet50

An additional ResEmoteNet notebook is included as an experimental model notebook, but the final artifact comparison focuses on MobileNetV2, EfficientNetB0, and ResNet50.

## Current Results

All three main models are evaluated on the same 8-class test set with 7,869 samples.

| Model | Test Samples | Test Accuracy |
|---|---:|---:|
| MobileNetV2 | 7,869 | 61.43% |
| EfficientNetB0 | 7,869 | 64.77% |
| ResNet50 | 7,869 | 64.98% |

ResNet50 has the highest saved test accuracy. EfficientNetB0 is very close, so it may be the better practical option when accuracy, model size, memory usage, and inference speed are considered together. MobileNetV2 remains useful as the lightweight baseline.

## Dataset

The indexed dataset contains 52,454 images from two sources:

| Source | Images |
|---|---:|
| AffectNet | 30,626 |
| AffectNet YOLO-format | 21,828 |

Class distribution:

| Class | Images |
|---|---:|
| anger | 6,164 |
| contempt | 6,112 |
| disgust | 5,438 |
| fear | 6,520 |
| happy | 7,424 |
| neutral | 7,921 |
| sad | 8,836 |
| surprise | 4,039 |

The dataset is split with stratified sampling:

| Split | Samples |
|---|---:|
| Training | 36,717 |
| Validation | 7,868 |
| Testing | 7,869 |

Class weighting is used during training to reduce the impact of class imbalance.

## Data Processing Pipeline

The notebooks support multiple dataset formats:

- class-labeled image folders
- AffectNet annotation-style records
- YOLO-format label files

For YOLO-format data, the pipeline reads the class ID and bounding box, converts normalized YOLO coordinates into pixel coordinates, crops the face region, and uses the cropped face as classifier input.

The common image pipeline is:

```text
Input image
-> RGB conversion
-> face crop when bounding boxes are available
-> resize to 224 x 224
-> cast to float32
-> model-specific preprocessing
-> TensorFlow dataset batching and prefetching
```

For inference, MediaPipe Face Detection is used to locate the face region before classification. This helps the classifier focus on facial expression rather than background content. The detected face box is converted to pixel coordinates, expanded with a small margin, cropped, resized to 224 x 224, and passed to the selected CNN model.

## Model Approach

The main models use transfer learning from ImageNet-pretrained CNN backbones.

Shared structure:

```text
Input 224 x 224 x 3
-> model-specific preprocessing
-> pretrained CNN backbone, include_top=False
-> global average pooling
-> dropout and/or batch normalization
-> Dense(8), softmax
```

Model strengths:

- MobileNetV2: lightweight and fast baseline.
- EfficientNetB0: strong balance between accuracy and efficiency.
- ResNet50: deeper architecture with higher representation capacity.

Training is performed in two stages:

1. Freeze the pretrained backbone and train the custom classification head.
2. Unfreeze the final portion of the backbone and fine-tune with a lower learning rate.

Training configuration:

| Model | Batch Size | Head Epochs | Fine-Tune Epochs | Head LR | Fine-Tune LR | Unfrozen Backbone Portion |
|---|---:|---:|---:|---:|---:|---|
| MobileNetV2 | 32 | 20 | 25 | 0.001 | 0.00001 | final 30% |
| EfficientNetB0 | 16 | 15 | 20 | 0.001 | 0.00001 | final 40% |
| ResNet50 | 32 | 15 | 20 | 0.001 | 0.00001 | final 40% |

The notebooks use `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau` during training.

## Project Structure

```text
Emotion Detect/
|-- app/
|-- archive/
|-- artifacts/
|   `-- models/
|       |-- EfficientNetB0/
|       |-- mobilenetv2/
|       |-- Resemotenet/
|       `-- Resnet50/
|-- data/
|-- docs/
|   |-- report/
|   |-- request.txt
|   |-- slide_content.md
|   `-- slide_image_candidates/
|-- notebooks/
|   |-- MobileNetV2_Face_Emotion_Detect.ipynb
|   |-- EfficientNetB0_Face_Emotion_Detect.ipynb
|   |-- ResNet50_Face_Emotion_Detect.ipynb
|   |-- ResEmoteNet_Face_Emotion_Detect.ipynb
|   `-- Evaluate_Artifact_Models.ipynb
|-- outputs/
|-- requirements.txt
`-- README.md
```

## Main Notebooks

| Notebook | Purpose |
|---|---|
| `notebooks/MobileNetV2_Face_Emotion_Detect.ipynb` | Train and evaluate the MobileNetV2 baseline. |
| `notebooks/EfficientNetB0_Face_Emotion_Detect.ipynb` | Train and evaluate the EfficientNetB0 model. |
| `notebooks/ResNet50_Face_Emotion_Detect.ipynb` | Train and evaluate the ResNet50 model. |
| `notebooks/ResEmoteNet_Face_Emotion_Detect.ipynb` | Experimental ResEmoteNet training notebook. |
| `notebooks/Evaluate_Artifact_Models.ipynb` | Load saved artifacts and evaluate the main models on the shared test split. |

## Saved Artifacts

Main model artifacts:

```text
artifacts/models/mobilenetv2/face_emotion_mobilenetv2_mediapipe_v2.keras
artifacts/models/EfficientNetB0/face_emotion_efficientnetb0_mediapipe.keras
artifacts/models/Resnet50/face_emotion_resnet50_mediapipe_v3.keras
```

Main metadata files:

```text
artifacts/models/mobilenetv2/metadata_mobilenetv2_mediapipe_v2.json
artifacts/models/EfficientNetB0/metadata_efficientnetb0_mediapipe.json
artifacts/models/Resnet50/metadata_resnet50_mediapipe_v3.json
```

## Evaluation

The evaluation notebook loads the saved `.keras` artifacts and evaluates them on the same deterministic test split. It reports:

- test accuracy
- precision, recall, and F1-score
- macro F1 and weighted F1
- confusion matrix
- normalized confusion matrix
- inference-time comparison
- sample predictions

The main evaluation notebook is:

```text
notebooks/Evaluate_Artifact_Models.ipynb
```

## Future Work

The next development step is to make the trained emotion recognition model easier to use outside the notebook environment.

- Complete a simple user interface for image-based emotion recognition.
- Allow users to upload an image from the interface.
- Send the uploaded image to a FastAPI inference backend.
- In FastAPI, load the saved model and run the full prediction pipeline:
  - face detection;
  - face cropping and resizing;
  - model-specific preprocessing;
  - emotion classification.
- Return structured API results, including the detected face region, predicted emotion, confidence score, and class-probability distribution.
- Display the FastAPI response clearly in the interface so users can view the prediction result without opening notebooks.

Planned deployment flow:

```text
User interface -> FastAPI backend -> saved emotion model -> prediction result
```

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

The notebooks also use TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, and MediaPipe. Some notebook environments may install missing helper packages inside the notebook itself.

## Notes

- Large datasets and trained model files may be stored outside Git or handled through Git LFS depending on the environment.
- The local `archive/` and `data/` folders are used for dataset storage and preprocessing.
- The final model should be selected by considering both accuracy and deployment constraints.
