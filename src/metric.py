import torch
import numpy as np
from scipy.linalg import sqrtm

# Function to extract features from Inception v3
def extract_features(images, model):
    # Ensure images are on the right device (CUDA or CPU)
    images = images.cuda() if torch.cuda.is_available() else images
    # Get the features (use the last pooling layer before classification)
    with torch.no_grad():
        features = model(images)
    return features


def calculate_fid(real_features, generated_features):
    # Calculate the mean and covariance of real and generated feature distributions
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(generated_features, axis=0)
    
    # Calculate covariance matrices
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(generated_features, rowvar=False)

    # Compute the Fréchet distance between real and generated distributions
    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real.dot(sigma_gen))

    # Numerically stable version of sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid
