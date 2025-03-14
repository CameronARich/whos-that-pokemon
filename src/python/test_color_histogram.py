#!/usr/bin/env python3

"""
Test script for color histogram feature extraction on Pokemon images.
This script demonstrates the extraction of color histograms from Pokemon images
and compares histograms between different images.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Function to extract color histograms from an image
def extract_color_histogram(image_path, bins=32):
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

# Function to calculate similarity between histograms
def calculate_histogram_similarity(hist1, hist2, method='correlation'):
    """Calculate similarity between two histograms using different metrics"""
    if hist1 is None or hist2 is None:
        return None
    
    if method == 'correlation':
        # Correlation coefficient (higher value means more similar)
        correlation = np.corrcoef(hist1, hist2)[0, 1]
        return correlation
    
    elif method == 'chi_square':
        # Chi-square distance (lower value means more similar)
        eps = 1e-10  # small value to avoid division by zero
        chi_square = np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + eps)) / 2.0
        return chi_square
    
    elif method == 'intersection':
        # Histogram intersection (higher value means more similar)
        intersection = np.sum(np.minimum(hist1, hist2))
        return intersection
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")

def main():
    # Set paths to test data
    # Get script directory and go up two levels to reach the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    data_dir = os.path.join(project_root, 'data')
    
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    
    # List contents of data directory to verify structure
    print("\nContents of data directory:")
    try:
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                print(f"  DIR: {item}")
            else:
                print(f"  FILE: {item}")
    except Exception as e:
        print(f"Error listing data directory: {e}")
    
    # Test with Kaggle dataset
    print("\nTesting with Kaggle dataset images...")
    kaggle_dir = os.path.join(data_dir, 'kaggle_data')
    
    if os.path.exists(kaggle_dir):
        print(f"Kaggle data directory exists: {kaggle_dir}")
        
        # Look for dataset directory
        dataset_dir = os.path.join(kaggle_dir, 'dataset')
        if os.path.exists(dataset_dir):
            print(f"Found dataset directory: {dataset_dir}")
            # List Pokemon directories
            pokemon_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
            if pokemon_dirs:
                print(f"Found {len(pokemon_dirs)} Pokemon directories: {', '.join(pokemon_dirs[:5])}...")
                # Test with 3 Pokemon
                for pokemon_dir in pokemon_dirs[:3]:
                    pokemon_path = os.path.join(dataset_dir, pokemon_dir)
                    print(f"\nTesting color histogram for {pokemon_dir}:")
                    
                    # Get image files in this directory
                    image_files = [f for f in os.listdir(pokemon_path) if f.endswith('.png') or f.endswith('.jpg')]
                    if not image_files:
                        print(f"No image files found in {pokemon_path}")
                        continue
                    
                    print(f"Found {len(image_files)} images: {', '.join(image_files[:5])}...")
                    
                    # Get full paths for first 2 images
                    img_paths = [os.path.join(pokemon_path, f) for f in image_files[:2]]
                    
                    # Extract histograms
                    print(f"Extracting histograms for {len(img_paths)} images...")
                    histograms = []
                    
                    for path in img_paths:
                        hist = extract_color_histogram(path)
                        if hist is not None:
                            histograms.append(hist)
                            print(f"  Extracted histogram with shape {hist.shape}")
                    
                    # Compare histograms if we have at least 2
                    if len(histograms) >= 2:
                        correlation = calculate_histogram_similarity(histograms[0], histograms[1], 'correlation')
                        chi_square = calculate_histogram_similarity(histograms[0], histograms[1], 'chi_square')
                        intersection = calculate_histogram_similarity(histograms[0], histograms[1], 'intersection')
                        
                        print(f"Histogram comparison results:")
                        print(f"  Correlation: {correlation:.4f} (higher is more similar)")
                        print(f"  Chi-Square: {chi_square:.4f} (lower is more similar)")
                        print(f"  Intersection: {intersection:.4f} (higher is more similar)")
            else:
                print("No Pokemon directories found in dataset directory")
        else:
            print(f"Dataset directory not found: {dataset_dir}")
            print("Listing contents of kaggle_data directory:")
            for item in os.listdir(kaggle_dir):
                item_path = os.path.join(kaggle_dir, item)
                if os.path.isdir(item_path):
                    print(f"  DIR: {item}")
                else:
                    print(f"  FILE: {item}")
    else:
        print(f"Kaggle data directory not found: {kaggle_dir}")
    
    # Test with sprite images
    print("\nTesting with sprite images...")
    
    # Define paths to sprite directories
    sprites_dir = os.path.join(data_dir, 'sprites')
    if not os.path.exists(sprites_dir):
        print(f"Sprites directory not found: {sprites_dir}")
    else:
        print(f"Sprites directory exists: {sprites_dir}")
        
        # List contents of sprites directory
        print("Contents of sprites directory:")
        for item in os.listdir(sprites_dir):
            item_path = os.path.join(sprites_dir, item)
            if os.path.isdir(item_path):
                print(f"  DIR: {item}")
            else:
                print(f"  FILE: {item}")
    
    gen1_path = os.path.join(sprites_dir, 'gen_1/main-sprites/red-blue')
    gen5_path = os.path.join(sprites_dir, 'gen_5/main-sprites/black-white')
    sugimori_path = os.path.join(data_dir, 'renders/sugimori')
    
    # Check if these directories exist
    print(f"\nGen 1 sprites directory exists: {os.path.exists(gen1_path)}")
    print(f"Gen 5 sprites directory exists: {os.path.exists(gen5_path)}")
    print(f"Sugimori renders directory exists: {os.path.exists(sugimori_path)}")
    
    # If gen1 path exists, list its contents
    if os.path.exists(gen1_path):
        print("\nGen 1 sprites files (first 5):")
        files = os.listdir(gen1_path)[:5]
        for file in files:
            print(f"  {file}")
    
    # Test Pokemon IDs
    test_ids = [25, 6, 150]  # Pikachu, Charizard, Mewtwo
    
    for pokemon_id in test_ids:
        print(f"\nTesting Pokemon #{pokemon_id}:")
        
        # Create paths for this Pokemon
        gen1_file = os.path.join(gen1_path, f"{pokemon_id}.png")
        gen5_file = os.path.join(gen5_path, f"{pokemon_id}.png")
        sugimori_file = os.path.join(sugimori_path, f"{pokemon_id}.png")
        
        # Check which images exist
        paths = {}
        if os.path.exists(gen1_file):
            paths['gen1'] = gen1_file
        if os.path.exists(gen5_file):
            paths['gen5'] = gen5_file
        if os.path.exists(sugimori_file):
            paths['sugimori'] = sugimori_file
            
        print(f"Path checks:")
        print(f"  Gen1: {gen1_file} - Exists: {os.path.exists(gen1_file)}")
        print(f"  Gen5: {gen5_file} - Exists: {os.path.exists(gen5_file)}")
        print(f"  Sugimori: {sugimori_file} - Exists: {os.path.exists(sugimori_file)}")
            
        if not paths:
            print(f"No sprite images found for Pokemon #{pokemon_id}")
            continue
            
        print(f"Found {len(paths)} sprite images")
        
        # Extract histograms
        histograms = {}
        for source, path in paths.items():
            hist = extract_color_histogram(path)
            if hist is not None:
                histograms[source] = hist
                print(f"  Extracted {source} histogram with shape {hist.shape}")
        
        # Compare histograms if we have at least 2
        if len(histograms) >= 2:
            print("Comparing histograms between different sources:")
            
            for source1 in histograms:
                for source2 in histograms:
                    if source1 < source2:  # Avoid duplicate comparisons
                        correlation = calculate_histogram_similarity(histograms[source1], histograms[source2], 'correlation')
                        chi_square = calculate_histogram_similarity(histograms[source1], histograms[source2], 'chi_square')
                        
                        print(f"  {source1} vs {source2}:")
                        print(f"    Correlation: {correlation:.4f}")
                        print(f"    Chi-Square: {chi_square:.4f}")

if __name__ == "__main__":
    main()
    print("\nTest completed successfully!") 