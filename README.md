# Master's projects - Data Processing, NLP, and Audio Signal Projects

This repository contains a collection of Python scripts and Jupyter notebooks focusing on Natural Language Processing (NLP), Information Retrieval (IR), and Audio Signal Processing. The project covers a range of algorithms including vector space modeling, n-gram analysis, next-token language modeling, time-series analysis, and 3D sound trajectory tracking.

---

## File Overview & Details

### 1. `Vector_model_for_information_retrieval.py`
* **Purpose:** Implements a Vector Space Model (VSM) for information retrieval over text document collections.
* **Text Preprocessing:** Uses stemming and token normalization to preprocess raw text documents and search queries.
* **Vectorization:** Computes Term Frequency-Inverse Document Frequency (TF-IDF) representation for vocabulary terms.
* **Similarity Matching:** Evaluates document relevance to search queries using Cosine Similarity.
* **Output:** Ranks and returns top matching documents based on computed similarity scores.

### 2. `Sound_trajectory.ipynb`
* **Purpose:** Analyzes multi-channel audio recordings to estimate and track sound source movement in 3D space.
* **Signal Processing:** Processes audio inputs received by a microphone array to extract spatial features.
* **Localization:** Applies Time Difference of Arrival (TDOA) or beamforming algorithms for positional estimation.
* **Trajectory Mapping:** Reconstructs continuous 3D coordinate trajectories over time.
* **Visualization:** Plots 3D spatial paths and spatial energy distribution of the audio source.

### 3. `Temperature_analysis.ipynb`
* **Purpose:** *[Pending details - send file to populate]*
* **Feature 1:** *[Pending details]*
* **Feature 2:** *[Pending details]*
* **Feature 3:** *[Pending details]*
* **Feature 4:** *[Pending details]*

### 4. `Text_ngrams.ipynb`
* **Purpose:** *[Pending details - send file to populate]*
* **Feature 1:** *[Pending details]*
* **Feature 2:** *[Pending details]*
* **Feature 3:** *[Pending details]*
* **Feature 4:** *[Pending details]*

### 5. `Next_token_prediction.ipynb`
* **Purpose:** *[Pending details - send file to populate]*
* **Feature 1:** *[Pending details]*
* **Feature 2:** *[Pending details]*
* **Feature 3:** *[Pending details]*
* **Feature 4:** *[Pending details]*

---

## Getting Started

### Prerequisites
* Python 3.8+
* Recommended packages: `numpy`, `scipy`, `pandas`, `matplotlib`, `nltk` (or `spacy`), `torch` / `tensorflow` (if neural models are used).

### Usage
Each notebook can be executed independently using Jupyter Notebook or JupyterLab:
```bash
jupyter notebook
```
