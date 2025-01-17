import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

np.random.seed(42)
torch.manual_seed(42)

def load_data(sample_ratio=0.01):
    """
    Load a subset of the Yelp Polarity dataset.
    Change `sample_ratio` to process more data once testing is complete.
    """
    print("Loading dataset...")
    split = f"train[:{int(sample_ratio * 100)}%]"
    if int(sample_ratio * 100) == 0:
        raise ValueError("sample_ratio too small, resulting in no data!")
    return load_dataset("yelp_polarity", split=split)

def extract_features(text, tokenizer, model):
    """
    Extract features from text using a pre-trained BERT model.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze(0).numpy()

def preprocess_data(dataset, tokenizer, model, pca_components=10):
    """
    Preprocess dataset to extract features, reduce dimensionality, and prepare labels.
    """
    Z, X, Y = [], [], []
    start_time = time.time()
    print("Extracting features...")
    for i, item in enumerate(dataset):
        try:
            embedding = extract_features(item['text'], tokenizer, model)
            Z.append(embedding)
            X.append(len(item['text'].split()))
            Y.append(1 if item['label'] == 1 else 0)
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue
    end_time = time.time()
    print(f"Feature extraction took {end_time - start_time:.2f} seconds")
    
    print("Performing PCA on BERT embeddings...")
    pca = PCA(n_components=pca_components)
    Z_reduced = pca.fit_transform(Z)
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.2f}")
    return np.array(Z_reduced), np.array(X), np.array(Y)

def causal_model(Z, X, Y=None):
    """
    alpha = pyro.sample("alpha", dist.Gamma(2.0, 2.0))
    mean_Y = alpha + beta * X + torch.sum(
    """
    n_samples, n_covariates = Z.shape
    alpha = pyro.sample("alpha", dist.Gamma(2.0, 2.0))
    sigma = pyro.sample("sigma", dist.HalfNormal(1.0))
    beta = pyro.sample("beta", dist.Normal(0., 1.))
    mean_Y = beta * X + torch.sum(
        Z * pyro.sample("weights", dist.Normal(0., 1.).expand([n_covariates]).to_event(1)),
        dim=1
    )
    with pyro.plate("data", n_samples):
        Y = pyro.sample("Y", dist.Normal(mean_Y, sigma), obs=Y)
    return Y

def diagnostics(mcmc):
    """
    Perform diagnostics on MCMC results, including R-hat and trace plots.
    """
    print("Performing diagnostics...")
    mcmc.summary()
    samples = mcmc.get_samples()
    
    # Generate trace plots
    for param, value in samples.items():
        plt.figure(figsize=(10, 4))
        plt.plot(value.numpy(), label=f"Trace plot for {param}")
        plt.legend()
        plt.show()
    
    return samples

def sensitivity_analysis(Z_test, X_test, samples, Y_test):
    """
    Perform sensitivity analysis to test robustness of causal estimates.
    """
    weights_mean = samples['weights'].mean(axis=0)
    beta_mean = samples['beta'].mean()

    # Predictions with slight perturbations in X
    perturbed_X = X_test + torch.normal(0, 0.1, size=X_test.size())
    predictions = beta_mean * perturbed_X + torch.sum(Z_test * weights_mean, axis=1)

    residuals = Y_test.numpy() - predictions.numpy()

    print("Sensitivity Analysis - Perturbed Results:")
    sns.histplot(residuals, kde=True, color='purple')
    plt.title('Residual Distribution (Perturbed Test Set)')
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Density')
    plt.show()

def visualize_results(samples, Y_test, predictions):
    """
    Visualize posterior distributions and residuals for the model's predictions.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(samples['alpha'].numpy(), kde=True, label='alpha', color='blue')
    sns.histplot(samples['sigma'].numpy(), kde=True, label='sigma', color='green')
    sns.histplot(samples['beta'].numpy(), kde=True, label='beta', color='red')
    plt.legend()
    plt.title('Posterior Distributions of Hyperparameters')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=Y_test.numpy(), y=predictions.numpy(), alpha=0.6)
    plt.axhline(0.5, color='red', linestyle='--', label='Decision Boundary (0.5)')
    plt.title('Actual vs Predicted Sentiments (Test Set)')
    plt.xlabel('Actual Sentiment')
    plt.ylabel('Predicted Sentiment')
    plt.legend()
    plt.show()

    residuals = Y_test.numpy() - predictions.numpy()
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='purple')
    plt.title('Residual Distribution (Test Set)')
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Density')
    plt.show()

def main():
    """
    Main pipeline for loading data, training, and validating the model.
    """
    dataset = load_data(sample_ratio=0.4) 
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()

    Z, X, Y = preprocess_data(dataset, tokenizer, model, pca_components=10)
    Z, X, Y = map(lambda arr: torch.tensor(arr, dtype=torch.float32), (Z, X, Y))

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X, Y, Z, train_size=0.8)

    print("Running MCMC...")
    nuts_kernel = NUTS(causal_model)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
    mcmc.run(Z_train, X_train, Y_train)
    
    samples = diagnostics(mcmc)

    weights_mean = samples['weights'].mean(axis=0)
    beta_mean = samples['beta'].mean()
    predictions = beta_mean * X_test + torch.sum(Z_test * weights_mean, axis=1)

    visualize_results(samples, Y_test, predictions)
    sensitivity_analysis(Z_test, X_test, samples, Y_test)

if __name__ == "__main__":
    main()