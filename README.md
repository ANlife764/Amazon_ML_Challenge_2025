# Amazon ML Challenge 2025

## Overview

This repository contains my solution for the Amazon Machine Learning Hackathon 2025, where the goal was to build a **Smart Product Pricing model** that predicts the optimal price of e-commerce products using their textual and visual data.

In e-commerce, determining the right price point is crucial for success. Our task was to predict product prices by analyzing descriptions, specifications, and other attributes from the provided dataset.

## Dataset

The dataset included:
- **catalog_content** – title, product description, and Item Pack Quantity (IPQ)
- **image_link** – URL of the product image  
- **price** – target variable for training data

## Approach

Our solution focused on capturing textual insights and structured patterns in the data through the following steps:

### 1. Text Preprocessing
- Cleaned and normalized product text (title + description + quantity)
- Applied TF-IDF vectorization to convert text into numerical features

### 2. Feature Engineering
- Combined structured features (like quantity) with TF-IDF text vectors

### 3. Modeling
- Used **LightGBM** for regression due to its speed, robustness, and ability to handle sparse data efficiently
- Tuned hyperparameters using validation splits

### 4. Evaluation
- Evaluated the model using **SMAPE** (Symmetric Mean Absolute Percentage Error)
- **Final SMAPE on leaderboard: 51.03**

## Repository Structure

| File | Description |
|------|-------------|
| `train_model.py` | Script to train the model and save outputs |
| `test_model.py` | Script to generate predictions for the test set |

## Results

- **Rank:** 1381 / 8000+ teams
- **Final SMAPE:** 51.03

## Learnings

This project provided valuable experience in:
- Practical text feature extraction using TF-IDF
- Model optimization with LightGBM
- Handling large-scale e-commerce datasets efficiently

## Team

**Team Oronyx**

## Tech Stack

- Python
- LightGBM
- Scikit-learn
- Pandas / NumPy
