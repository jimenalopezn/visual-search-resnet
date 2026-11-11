# Visual Information Retrieval API for Textile Catalog Matching

## Overview
Production-ready REST API built with **FastAPI**, **PyTorch (ResNet-50)**, and **Scikit-Learn** for real-time visual search and inventory matching. Engineered for Nua Collection, the system extracts feature vectors from user-uploaded images to query a textile catalog via cosine similarity, isolating structural patterns (grayscale embeddings) from color designs.


## Key Features & Achievements
* **Architected a dual-embedding extraction pipeline** using PyTorch ResNet-50, generating separate feature representations for pattern geometry (grayscale) and color aesthetics.
* **Engineered sub-second vector similarity retrieval** using Scikit-Learn cosine similarity matrices against pre-computed feature embeddings (`.npy`).
* **Implemented dynamic metadata filtering** via Pandas to restrict vector search spaces by style, pattern, and color attributes prior to matrix operations.
* **Built a lightweight, asynchronous API backend** using FastAPI, CORS middleware, and multipart file upload processing for seamless client integration.


## Technical Stack
* **Framework & API:** FastAPI, Uvicorn, CORS Middleware
* **Deep Learning & Computer Vision:** PyTorch, Torchvision (ResNet-50 Pretrained Backbone), Pillow (PIL), Torchvision Transforms
* **Vector Analytics & Processing:** Scikit-Learn (Cosine Similarity), NumPy, Pandas
* **Deployment & Storage:** CPU-Optimized Inference (`torch.no_grad()`), Embedded Vector Matrices (`.npy`)

