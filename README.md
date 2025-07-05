# DeepFake Detection using ISTVT (Interpretable Spatio-Temporal Video Transformer)

This repository contains an implementation of the **ISTVT** deepfake detection model based on the paper:  
**"Interpretable Spatio-Temporal Video Transformer for DeepFake Detection" ** 

---

## Overview

This project classifies real vs fake videos using:
- **Xception** CNN for spatial feature extraction  
- A **Decomposed Spatial-Temporal Transformer** for modeling frame sequences  
- A lightweight **classifier** head for binary classification  
- Trained on 10 aligned face frames per video  

---

## Architecture


           +----------------+
           |   Video Clip   | (10 frames)
           +-------+--------+
                   |
         [Frame-wise Feature Extraction]
                   |
           +------------------------+
           |        Xception        |
           +-----------+------------+
                       |
            [Patch Tokenization]  
                ↑ Add spatial & temporal CLS tokens  
                ↑ + learnable position embeddings
                       |
        +--------------+---------------+
        |   Spatial Self-Attention     |
        +------------------------------+
        |   Temporal Self-Attention    |
        +------------------------------+
        |   [CLS] Token → Classifier   |
        +------------------------------+
---

## Dataset and Performance

### Dataset: FaceForensics++

We used a **subset of the FaceForensics++ dataset** for this project:

- **Real Videos**: 200 pristine real videos  
- **Fake Videos**: 200 synthetic deepfake videos  
- **Preprocessing**:
  - **Face alignment** using MTCNN
  - **10 face frames** extracted uniformly from each video
  - Frames saved as `.pt` tensors of shape `(10, 3, 224, 224)`

The extracted faces were preprocessed with data augmentation, normalized, and used directly for training a transformer-based model.

---

### Model Performance (Validation Set)

We trained the **ISTVT** model for **15 epochs** on this dataset with an 80/20 train/validation split.

| Metric         | Best Value Achieved |
|----------------|---------------------|
| **Accuracy**   | **91.25%**          |
| **Val Loss**   | **0.2214**          |
| **Epoch**      | 13                  |

> The best model was saved at **epoch 13**, where the model achieved **88.75–91.25% accuracy** consistently across multiple runs.

---





