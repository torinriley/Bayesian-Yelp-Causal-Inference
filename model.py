import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
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
    """
    print("Loading dataset...")
    return load_dataset("yelp_polarity", split=f"train[:{int(sample_ratio * 100)}%]")

def extract_features(text, tokenizer, model):
    """
    Extract features from text using a pre-trained BERT model.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze(0).numpy()

def preprocess_data(dataset, tokenizer, model):
    """
    Preprocess the dataset to extract features and labels.
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
    return np.array(Z), np.array(X), np.array(Y)

def causal_model(Z, X, Y=None):
    """
    Define a Bayesian causal model.
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

def train_test_split(X, Y, Z, train_ratio=0.8):
    """
    Perform a train-test split on the dataset.
    """
    dataset_size = len(X)
    train_size = int(train_ratio * dataset_size)
    indices = torch.randperm(dataset_size)
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    return (X[train_indices], X[test_indices],
            Y[train_indices], Y[test_indices],
            Z[train_indices], Z[test_indices])

def visualize_results(samples, Y_test, predictions):
    """
    Visualize the results of the model's predictions.
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

def save_results_to_csv(dataset, Z, X, Y, predictions, samples, file_prefix="results"):
    """
    Save the raw data and results from the model to CSV files for further analysis.
    """
    # Save raw text, review length, and sentiment labels
    dataset_list = list(dataset)
    raw_data = pd.DataFrame({
        "text": [item["text"] for item in dataset_list[:len(X)]],
        "label": [item["label"] for item in dataset_list[:len(X)]],
        "review_length": X.numpy(),
        "sentiment_label": Y.numpy()
    })
    raw_data.to_csv(f"{file_prefix}_raw_data.csv", index=False)
    
    # Save BERT embeddings (Z)
    bert_embeddings = pd.DataFrame(Z.numpy())
    bert_embeddings.to_csv(f"{file_prefix}_bert_embeddings.csv", index=False)
    
    # Save predictions and residuals
    residuals = Y.numpy() - predictions.numpy()
    predictions_data = pd.DataFrame({
        "actual_sentiment": Y.numpy(),
        "predicted_sentiment": predictions.numpy(),
        "residual": residuals
    })
    predictions_data.to_csv(f"{file_prefix}_predictions.csv", index=False)
    
    # Save posterior samples
    posterior_samples = pd.DataFrame({
        "alpha": samples["alpha"].numpy(),
        "sigma": samples["sigma"].numpy(),
        "beta": samples["beta"].numpy()
    })
    posterior_samples.to_csv(f"{file_prefix}_posterior_samples.csv", index=False)
    
    print("Data and results saved to CSV files.")

def main():
    """
    Main pipeline for loading data, preprocessing, training the model, and saving results.
    """
    dataset = load_data(sample_ratio=0.01)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()

    Z, X, Y = preprocess_data(dataset, tokenizer, model)
    Z, X, Y = map(lambda arr: torch.tensor(arr, dtype=torch.float32), (Z, X, Y))

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(X, Y, Z)

    print("Running MCMC...")
    nuts_kernel = NUTS(causal_model)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=200)
    mcmc.run(Z_train, X_train, Y_train)
    
    samples = mcmc.get_samples()
    weights_mean = samples['weights'].mean(axis=0)
    beta_mean = samples['beta'].mean()
    
    predictions = beta_mean * X_test + torch.sum(Z_test * weights_mean, axis=1)
    save_results_to_csv(dataset, Z_test, X_test, Y_test, predictions, samples, file_prefix="test_results")
    visualize_results(samples, Y_test, predictions)

if __name__ == "__main__":
    main()
