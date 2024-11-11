import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from scipy.optimize import linear_sum_assignment
import argparse

def initialize_parameters(X, n_components):
    np.random.seed(0)
    n_samples, n_features = X.shape
    means = X[np.random.choice(n_samples, n_components, replace=False)]
    covariances = np.array([np.eye(n_features) for _ in range(n_components)])
    weights = np.ones(n_components) / n_components
    return means, covariances, weights

def gaussian_pdf(X, mean, covariance):
    n = X.shape[1]
    diff = X - mean
    inv_covariance = np.linalg.inv(covariance)
    exponent = np.einsum('ij,jk,ik->i', diff, inv_covariance, diff)
    return np.exp(-0.5 * exponent) / np.sqrt((2 * np.pi)**n * np.linalg.det(covariance))

def expectation(X, means, covariances, weights):
    n_samples, n_components = X.shape[0], means.shape[0]
    responsibilities = np.zeros((n_samples, n_components))
    for k in range(n_components):
        responsibilities[:, k] = weights[k] * gaussian_pdf(X, means[k], covariances[k])
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities

def maximization(X, responsibilities):
    n_samples, n_features = X.shape
    n_components = responsibilities.shape[1]
    weights = responsibilities.sum(axis=0) / n_samples
    means = np.dot(responsibilities.T, X) / responsibilities.sum(axis=0)[:, np.newaxis]
    covariances = np.zeros((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / responsibilities[:, k].sum()
    return means, covariances, weights

def train_gmm(X, n_components, max_iters=100, tol=1e-4):
    means, covariances, weights = initialize_parameters(X, n_components)
    log_likelihoods = []
    for i in range(max_iters):
        responsibilities = expectation(X, means, covariances, weights)
        means, covariances, weights = maximization(X, responsibilities)
        log_likelihood = np.sum(np.log(np.sum([w * gaussian_pdf(X, m, c) for w, m, c in zip(weights, means, covariances)], axis=0)))
        log_likelihoods.append(log_likelihood)
        if i > 0 and abs(log_likelihood - log_likelihoods[-2]) < tol:
            print(f"Converged at iteration {i}")
            break
    return means, covariances, weights, responsibilities

def calculate_accuracy_recall(y_true, y_pred, n_components):
    contingency_matrix = np.zeros((n_components, n_components), dtype=int)
    for i in range(len(y_true)):
        contingency_matrix[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    best_matching = [(row, col) for row, col in zip(row_ind, col_ind)]
    
    total_correct = sum(contingency_matrix[row, col] for row, col in best_matching)
    accuracy = total_correct / len(y_true)
    
    recall_values = []
    for row, col in best_matching:
        cluster_total = np.sum(contingency_matrix[:, col])
        recall = contingency_matrix[row, col] / cluster_total if cluster_total > 0 else 0
        recall_values.append(recall)
    recall = np.mean(recall_values)
    return accuracy, recall

def main():
    parser = argparse.ArgumentParser(description='GMM')
    parser.add_argument('--dataset',type=str,default='iris')
    args = parser.parse_args()
    if args.dataset == 'iris':
        data = load_iris()
    elif args.dataset == 'wine':
        data = load_wine()
    X = data.data
    y = data.target

    n_components = 3
    means, covariances, weights, responsibilities = train_gmm(X, n_components)

    predictions = np.argmax(responsibilities, axis=1)

    accuracy, recall = calculate_accuracy_recall(y, predictions, n_components)
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")

    print("Predictions:", predictions)
    print("True Labels:", y)

if __name__ == "__main__":
    main()
