"""
Feature extraction methods for the Who's That Pokémon project.

This module contains various feature extraction techniques for image classification.
"""

import numpy as np
import os
from PIL import Image
import imagehash
from sklearn.feature_extraction.image import PatchExtractor
from skimage.feature import hog
from skimage.transform import resize
from skimage.color import rgb2gray

def extract_color_histogram(image_path, bins=32):
    """
    Extract color histogram features from an image.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    bins : int, default=32
        Number of bins for each color channel histogram
        
    Returns:
    --------
    numpy.ndarray
        Normalized color histogram features (R, G, B concatenated)
    """
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Convert to numpy array
        img_array = np.array(img)
        
        # Extract histograms for each channel
        hist_r, _ = np.histogram(img_array[:,:,0].flatten(), bins=bins, range=(0, 256))
        hist_g, _ = np.histogram(img_array[:,:,1].flatten(), bins=bins, range=(0, 256))
        hist_b, _ = np.histogram(img_array[:,:,2].flatten(), bins=bins, range=(0, 256))
        
        # Concatenate the histograms
        hist_features = np.concatenate([hist_r, hist_g, hist_b])
        
        # Normalize the histogram
        hist_features = hist_features.astype('float')
        hist_features /= (hist_features.sum() + 1e-7)  # Add small value to avoid division by zero
        
        return hist_features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def extract_hog_features(image_path, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    """
    Extract Histogram of Oriented Gradients (HOG) features from an image.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    pixels_per_cell : tuple, default=(8, 8)
        Size (in pixels) of a cell
    cells_per_block : tuple, default=(2, 2)
        Number of cells in each block
    orientations : int, default=9
        Number of orientation bins
        
    Returns:
    --------
    numpy.ndarray
        HOG features
    """
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Convert to numpy array
        img_array = np.array(img)
        
        # Convert to grayscale
        gray_img = rgb2gray(img_array)
        
        # Extract HOG features
        hog_features = hog(gray_img, orientations=orientations, 
                          pixels_per_cell=pixels_per_cell,
                          cells_per_block=cells_per_block,
                          visualize=False)
        
        return hog_features
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def extract_image_hash(image_path, hash_size=8, hash_type='phash'):
    """
    Extract perceptual hash features from an image.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
    hash_size : int, default=8
        Hash size to use
    hash_type : str, default='phash'
        Type of hash to use ('phash', 'dhash', 'whash', 'average_hash')
        
    Returns:
    --------
    numpy.ndarray
        Hash as a binary array
    """
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Calculate hash based on type
        if hash_type == 'phash':
            hash_value = imagehash.phash(img, hash_size=hash_size)
        elif hash_type == 'dhash':
            hash_value = imagehash.dhash(img, hash_size=hash_size)
        elif hash_type == 'whash':
            hash_value = imagehash.whash(img, hash_size=hash_size)
        elif hash_type == 'average_hash':
            hash_value = imagehash.average_hash(img, hash_size=hash_size)
        else:
            raise ValueError(f"Unknown hash type: {hash_type}")
            
        # Convert hash to binary array
        hash_array = np.array([int(bit) for bit in str(hash_value)])
        
        return hash_array
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def extract_color_moments(image_path):
    """
    Extract color moments from an image (mean, standard deviation, skewness).
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
        
    Returns:
    --------
    numpy.ndarray
        Color moments features
    """
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Convert to numpy array
        img_array = np.array(img).astype('float')
        
        # Split the channels
        r_channel = img_array[:,:,0].flatten()
        g_channel = img_array[:,:,1].flatten()
        b_channel = img_array[:,:,2].flatten()
        
        # Calculate moments for each channel
        # 1st moment - Mean
        r_mean = np.mean(r_channel)
        g_mean = np.mean(g_channel)
        b_mean = np.mean(b_channel)
        
        # 2nd moment - Standard deviation
        r_std = np.std(r_channel)
        g_std = np.std(g_channel)
        b_std = np.std(b_channel)
        
        # 3rd moment - Skewness
        r_skewness = np.mean(((r_channel - r_mean) / (r_std + 1e-8)) ** 3)
        g_skewness = np.mean(((g_channel - g_mean) / (g_std + 1e-8)) ** 3)
        b_skewness = np.mean(((b_channel - b_mean) / (b_std + 1e-8)) ** 3)
        
        # Combine all features
        color_moments = np.array([
            r_mean, g_mean, b_mean,
            r_std, g_std, b_std,
            r_skewness, g_skewness, b_skewness
        ])
        
        return color_moments
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def extract_features_batch(image_paths, method='histogram', **kwargs):
    """
    Extract features from a batch of images using the specified method.
    
    Parameters:
    -----------
    image_paths : list
        List of paths to image files
    method : str, default='histogram'
        Feature extraction method to use ('histogram', 'hog', 'hash', 'color_moments')
    **kwargs : dict
        Additional parameters for the chosen feature extraction method
        
    Returns:
    --------
    numpy.ndarray
        Array of features for each image
    list
        List of paths for successfully processed images
    """
    features = []
    successful_paths = []
    
    # Choose feature extraction method
    if method == 'histogram':
        extract_func = extract_color_histogram
    elif method == 'hog':
        extract_func = extract_hog_features
    elif method == 'hash':
        extract_func = extract_image_hash
    elif method == 'color_moments':
        extract_func = extract_color_moments
    else:
        raise ValueError(f"Unknown feature extraction method: {method}")
    
    # Process each image
    for path in image_paths:
        feature_vector = extract_func(path, **kwargs)
        if feature_vector is not None:
            features.append(feature_vector)
            successful_paths.append(path)
    
    # Convert to numpy array
    if features:
        features = np.array(features)
    else:
        features = np.array([])
        
    return features, successful_paths 