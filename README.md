# 🕵️‍♂DeepFake Detection using ISTVT (Interpretable Spatio-Temporal Video Transformer)

This repository contains an implementation of the **ISTVT** deepfake detection model based on the paper:  
**"Interpretable Spatio-Temporal Video Transformer for DeepFake Detection" (CVPR 2022)**  
[[Paper Link](https://openaccess.thecvf.com/content/CVPR2022/html/Li_Interpretable_Spatio-Temporal_Video_Transformer_for_Deepfake_Detection_CVPR_2022_paper.html)]

---

## Overview

This project classifies real vs fake videos using:
- **Xception** CNN for spatial feature extraction  
- A **Decomposed Spatial-Temporal Transformer** for modeling frame sequences  
- A lightweight **classifier** head for binary classification  
- Trained on 10 aligned face frames per video  

---

## Architecture

```text
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
