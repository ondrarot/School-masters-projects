# Master's projects - Data Processing, NLP, and Audio Signal Projects

This repository contains a collection of Python scripts and Jupyter notebooks focusing on Natural Language Processing (NLP), Information Retrieval (IR), and Audio Signal Processing. The project covers a range of algorithms including vector space modeling, next-token language modeling, time-series analysis, and 3D sound trajectory tracking.

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
- **Purpose:** Collection, preparation, and statistical analysis of daily temperature data in Pilsen for the years 2018–2025 to identify temperature-anomalous years and evaluate climate trends.
- **Feature 1:** **Data Collection and Cleaning:** Fetching daily temperatures from the `meteostat` library, handling missing values (NaN) by computing multi-year averages, and aligning leap years to a constant length of 365 days.
- **Feature 2:** **Visualization and Annual Course:** Plotting daily temperature curves for individual years, calculating a long-term "climate curve," and displaying average annual temperatures with 95% confidence intervals.
- **Feature 3:** **Hypothesis Testing:** Performing unpaired and paired statistical tests at a significance level of $\alpha = 0.01$ to identify years significantly different, warmer, or colder than the overall average.
- **Feature 4:** **Normality and Trend Verification:** Visual inspection of normal distribution of daily temperature deviations and evaluating the statistical significance of interannual temperature trends using linear regression.

### 4. `Next_token_prediction.ipynb`
* **Purpose:** Implementation of a Viterbi-based acoustic decoding algorithm for speech recognition and word sequence prediction using phonetic transcriptions and language model integration.
* **Feature 1:** **Vocabulary & Data Preprocessing:** Converting vocabulary files across encodings (Windows-1250 to UTF-8) and parsing Czech words into structured lists with their corresponding phonetic transcriptions.
* **Feature 2:** **Acoustic & Observation Model Loading:** Loading phoneme transition probabilities (self-stay vs. cross-phoneme transition) from acoustic model files and importing frame-level observation likelihood vectors.
* **Feature 3:** **Viterbi Algorithm & Dynamic Programming:** Initializing and iteratively computing path scores ($\Phi$) over time frames ($T=804$), handling within-word phoneme transitions as well as inter-word boundary transitions.
* **Feature 4:** **Language Model Integration & Parameters:** Parsing unigram probabilities from ARPA-formatted language models and configuring decoding hyperparameters like word insertion penalty ($L_\gamma$) and language model scale weight ($\beta$).

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
