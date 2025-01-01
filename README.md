# Bayesian-Yelp-Causal-Inference

## Overview
This project implements a Bayesian causal inference model to analyze the relationship between review length and sentiment polarity using the Yelp Polarity dataset. The model leverages pre-trained BERT embeddings to account for the content of reviews while estimating the causal effect of review length on sentiment.

## Features
- Extracts review content features using a pre-trained BERT model.
- Uses Bayesian inference with priors on hyperparameters for robust uncertainty quantification.
- Models the causal effect of review length on sentiment polarity while controlling for confounders.
- Provides visualizations of posterior distributions, predictions, and residuals.

## Model Description
The model uses:
- **Input Features**: 
  - `X`: Review lengths (word counts).
  - `Z`: Semantic features extracted from BERT embeddings.
- **Outcome**:
  - `Y`: Binary sentiment polarity (0 for negative, 1 for positive).
- **Bayesian Framework**:
  - Priors are placed on the model parameters (`alpha`, `beta`, `sigma`, and weights for BERT embeddings).
  - Posterior distributions are estimated using Markov Chain Monte Carlo (MCMC) sampling with the No-U-Turn Sampler (NUTS).

## Workflow
1. **Load Dataset**:
   - A small subset (1%) of the Yelp Polarity dataset is used for training and evaluation.
2. **Feature Extraction**:
   - Text content is converted into semantic features (`Z`) using BERT embeddings.
   - Review lengths (`X`) are calculated as the word count of each review.
3. **Causal Model**:
   - A Bayesian model is defined to estimate the effect of `X` (length) on `Y` (sentiment), controlling for `Z` (content).
4. **MCMC Sampling**:
   - Posterior distributions of parameters are estimated using MCMC with 1,000 samples and 200 warm-up steps.
5. **Evaluation**:
   - Predictions are compared to actual sentiments on a test set.
   - Visualizations include posterior distributions, residuals, and scatter plots.

## Visualizations
1. **Posterior Distributions**:
   - Shows the distributions of key model parameters (`alpha`, `sigma`, `beta`).
2. **Predictions vs Actual Sentiments**:
   - Scatter plot comparing actual vs predicted sentiment values.
3. **Residual Analysis**:
   - Histogram of residuals to assess model calibration.

## Results

The model was evaluated on a subset (1%) of the Yelp Polarity dataset, and the following key findings were observed:

### 1. Posterior Distributions
The posterior distributions of key model parameters (`alpha`, `sigma`, and `beta`) provide insights into the model's learned beliefs:

- **`alpha`**: Controls the prior strength in the generative process. The posterior distribution shows well-defined values, indicating the model learns this parameter effectively.
- **`sigma`**: Represents the noise or variability in sentiment predictions. A narrow distribution suggests the model captures the variability in the data accurately.
- **`beta`**: The causal effect of review length (`X`) on sentiment (`Y`). The posterior distribution of `beta` shows values concentrated near zero, indicating a negligible direct effect of review length on sentiment once the review content (`Z`) is accounted for.

---

### 2. Actual vs Predicted Sentiments
- A scatter plot compares the **true sentiments** (`Y_test`) against the **predicted sentiments**.
- Predictions are well-separated into positive (1) and negative (0), showcasing the model's ability to classify sentiment correctly.
- The decision boundary at 0.5 clearly divides positive and negative predictions, validating the binary classification approach.

---

### 3. Residual Analysis
- Residuals (difference between actual and predicted sentiments) are symmetrically distributed around zero.
- This symmetry suggests that the model's predictions are unbiased, with no systematic overestimation or underestimation of sentiment polarity.
- The narrow spread of residuals indicates good predictive accuracy.

---

### 4. Key Insights
- **Minimal Causal Effect**: The causal effect of review length on sentiment is negligible (`beta` close to zero). Sentiment polarity is primarily driven by the review content rather than its length.
- **Confounding Control**: The inclusion of BERT embeddings (`Z`) effectively accounts for confounders, isolating the true relationship between review length and sentiment.
- **Uncertainty Quantification**: The Bayesian framework captures uncertainty in parameter estimates, as demonstrated by the posterior distributions.

---

<table>
    <tr>
        <th>Posterior Distributions of Hyperparameters</th>
        <th>Actual vs Predicted Sentiments</th>
    </tr>
    <tr>
        <td>
            <img src="https://github.com/user-attachments/assets/8dab393f-1c69-46c1-b86f-58cfd298f90b" alt="Original Embeddings" width="789"/>
        </td>
        <td>
           <img width="789" alt="Screenshot 2024-12-28 at 11 40 40 PM" src="https://github.com/user-attachments/assets/2558eff5-56f6-425c-850b-3116cec1bb8a" />
        </td>
    </tr>
</table>

---

These results highlight the model's ability to classify sentiments effectively while providing credible causal insights into the relationship between review length and sentiment polarity.
