import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import seaborn as sns
import matplotlib.pyplot as plt
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

    Parameters:
    sample_ratio (float): The ratio of the dataset to load (default is 0.01).

    Returns:
    Dataset: A subset of the Yelp Polarity dataset.
    """
    print("Loading dataset...")
    return load_dataset("yelp_polarity", split=f"train[:{int(sample_ratio * 100)}%]")

def extract_features(text, tokenizer, model):
    """
    Extract features from text using a pre-trained BERT model.

    Parameters:
    text (str): The input text.
    tokenizer (AutoTokenizer): The tokenizer for the BERT model.
    model (AutoModel): The pre-trained BERT model.

    Returns:
    np.ndarray: The extracted features as a numpy array.
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze(0).numpy()

def preprocess_data(dataset, tokenizer, model):
    """
    Preprocess the dataset to extract features and labels.

    Parameters:
    dataset (Dataset): The input dataset.
    tokenizer (AutoTokenizer): The tokenizer for the BERT model.
    model (AutoModel): The pre-trained BERT model.

    Returns:
    tuple: A tuple containing arrays for Z (features), X (text lengths), and Y (labels).
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

    Parameters:
    Z (torch.Tensor): The covariate matrix.
    X (torch.Tensor): The feature vector.
    Y (torch.Tensor, optional): The response vector.

    Returns:
    torch.Tensor: The response vector.
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

    Parameters:
    X (np.ndarray): The feature vector.
    Y (np.ndarray): The response vector.
    Z (np.ndarray): The covariate matrix.
    train_ratio (float): The ratio of the dataset to use for training (default is 0.8).

    Returns:
    tuple: A tuple containing the train and test splits for X, Y, and Z.
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

    Parameters:
    samples (dict): Dictionary containing posterior samples of the model parameters.
    Y_test (torch.Tensor): Actual sentiment values of the test set.
    predictions (torch.Tensor): Predicted sentiment values of the test set.

    Visualizations:
    - Posterior distributions of hyperparameters (alpha, sigma, beta).
    - Scatter plot of actual vs predicted sentiments.
    - Residual distribution of the test set.
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
    Main pipeline for loading data, preprocessing, training the model, and visualizing results.

    Steps:
    1. Load a small sample of the dataset.
    2. Initialize the tokenizer and model from the BERT base uncased model.
    3. Preprocess the data to obtain tensors for Z, X, and Y.
    4. Split the data into training and test sets.
    5. Run MCMC to sample from the posterior distribution of the model parameters.
    6. Compute predictions on the test set.
    7. Visualize the results.
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
    visualize_results(samples, Y_test, predictions)

if __name__ == "__main__":
    main()