# Ethereal Echo: Music Genre Classification

Music genre classification on the GTZAN dataset using classical machine learning models, dimensionality reduction techniques, and a small convolutional neural network.
All analysis and experiments are contained in the single Jupyter notebook: `notebooks/Project_Notebook.ipynb`.

---

## 1. Project Overview

The goal of this project is to:

* Extract meaningful audio features from songs in the **GTZAN** dataset.
* Train and compare multiple machine learning models for **10-class genre classification**.
* Explore the impact of **dimensionality reduction** (PCA, Kernel PCA, LDA).
* Experiment with a **2D CNN** on time–frequency representations (e.g., mel spectrograms).
* Analyse **feature importance** and **per-genre performance** of the best model (Logistic Regression).

All exploration, feature engineering, modelling, evaluation, and final discussion are implemented in `Project_Notebook.ipynb`.

---

## 2. Dataset

* Dataset: [GTZAN Music Genre Classification](https://huggingface.co/datasets/marsyas/gtzan)
* 1,000 audio clips
* Length: 30 seconds each
* 10 genres:

  ```text
  blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
  ```

> **Note:** The raw dataset is **not included** in this repository.
> Download GTZAN manually and place it in a suitable directory structure.

Example directory layout:

```
genres/
├─ blues/
├─ classical/
├─ country/
├─ disco/
├─ hiphop/
├─ jazz/
├─ metal/
├─ pop/
├─ reggae/
└─ rock/
```

Paths in the notebook use a Google Colab + Google Drive layout; adjust accordingly when running locally.

---

## 3. Feature Extraction

The notebook loads audio using **librosa** and extracts a range of features (~100 features), including:

* Zero Crossing Rate
* Harmonic / Perceptual features
* Tempo & Beat Frames
* Tonnetz
* Spectral Centroid
* Spectral Bandwidth
* Spectral Rolloff
* Spectral Contrast
* Spectral Flatness
* RMS Energy
* Chroma Features
* Fourier Transform & Spectrogram
* Mel-Spectrogram
* MFCCs

Each feature set is summarised using:

* Mean
* Variance
* Skewness
* Kurtosis

All features across all tracks are aggregated into a single DataFrame and saved to:

```
more_project_data.csv
```

This file is then used for training and evaluating models.

---

## 4. Data Preparation

From `more_project_data.csv`, the notebook:

* Drops non-feature columns (e.g., `filename`).

* Separates the target column `genre`.

* Converts all feature columns to numeric.

* Encodes genres using `LabelEncoder` with the predefined order:

  ```python
  GENRES = ["blues", "classical", "country", "disco", "hiphop",
            "jazz", "metal", "pop", "reggae", "rock"]
  ```

* Applies common preprocessing:

  * `StandardScaler` for scaling
  * `train_test_split` for train/test separation

---

## 5. Models

### 5.1 Classical Machine Learning Models

Implemented using **scikit-learn**:

* Logistic Regression (multinomial)
* Support Vector Machine (SVC)
* Stochastic Gradient Descent (hinge & log-loss)
* Gaussian Naive Bayes
* K-Nearest Neighbours (various `k`)
* XGBoost (`XGBClassifier`)
* Random Forest
* Multi-layer Perceptron (MLPClassifier)

A helper evaluation function:

* Trains the model
* Predicts on test set
* Computes accuracy + classification report
* Stores accuracy in a summary table

---

### 5.2 Dimensionality Reduction

The following DR techniques are applied to the feature set:

* **PCA**
* **Kernel PCA**
* **LDA**

Each transformed dataset is passed into the same models to compare the effect on accuracy.

---

### 5.3 2D Convolutional Neural Network (CNN)

A small CNN (Keras/TensorFlow) is trained on stacked 2D representations (mel-spectrogram + others).
Architecture includes:

* Conv2D + MaxPooling2D
* Batch Normalisation
* Dense layers with Softmax output

Training curves show:

* **Strong overfitting**
* Limited improvement on test accuracy due to small dataset

---

## 6. Results & Discussion

A summary DataFrame is created to compare all models:

* Logistic Regression and MLP are the **best-performing models** on the original features.
* PCA / KPCA sometimes improve performance, sometimes reduce it.
* LDA improves class separability but not universally across models.
* CNN severely overfits due to limited dataset size.

Per-genre analysis shows:

* **Rock** is consistently the hardest genre to classify.
* Some genres (e.g., classical, disco) are much easier.

Additional analysis includes:

* Inspecting logistic regression coefficients
* Examining confusion matrices
* Feature contribution analysis

---

## 7. How to Run This Project

### 7.1 Clone the Repository

```bash
git clone https://github.com/watarukeema/EtherealEcho.git
cd EtherealEcho
```

### 7.2 Set Up Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 7.3 Download the Dataset

Download GTZAN and ensure it matches the structure shown earlier.
Update file paths in the notebook if needed.

### 7.4 Run the Notebook

```bash
jupyter notebook
```

Open `notebooks/Project_Notebook.ipynb` and run cells sequentially.

---

## 8. Limitations & Future Work

* Limited dataset size → CNN overfitting
* Some genres remain difficult (e.g., rock)
* Requires better augmentation for deep learning

**Future extensions:**

* Ensemble models
* Stronger CNN regularisation
* Audio data augmentation (pitch shift, time stretch, noise)
* Pretrained models (Audio Spectrogram Transformers, YAMNet, etc.)

---

## 9. Project Context

This project was created for:

> **COMP9444 – Neural Networks and Deep Learning (UNSW)**
> Project Title: *Ethereal Echo*

For learning and portfolio use only.
Students taking COMP9444 in later terms should not reuse this work.

---

## 10. License

This project is released under the **MIT License**.

GTZAN dataset is subject to its own license and is not included in this repository.
