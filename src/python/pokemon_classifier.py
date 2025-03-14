#!/usr/bin/env python3

"""
Pokemon Classifier
-----------------
This script trains a classifier on the Kaggle Pokemon dataset using color histogram features
and tests it on the test dataset to identify Pokemon.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Feature extraction function
def extract_color_histogram(image_path, bins=32):
    """Extract color histogram features from an image."""
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

def load_dataset(dataset_dir, limit_classes=None, max_samples_per_class=None, verbose=True):
    """
    Load Pokemon images from the dataset directory and extract features.
    
    Args:
        dataset_dir: Directory containing Pokemon subdirectories
        limit_classes: Maximum number of classes to use (for quicker testing)
        max_samples_per_class: Maximum number of samples per class
        verbose: Whether to display progress
        
    Returns:
        features: List of feature vectors
        labels: List of corresponding labels
        class_names: List of class names (Pokemon names)
    """
    features = []
    labels = []
    class_names = []
    
    # Get list of Pokemon directories
    pokemon_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    if limit_classes:
        pokemon_dirs = pokemon_dirs[:limit_classes]
    
    if verbose:
        print(f"Loading data for {len(pokemon_dirs)} Pokemon classes...")
    
    # Process each Pokemon class
    for pokemon_dir in tqdm(pokemon_dirs, disable=not verbose):
        pokemon_path = os.path.join(dataset_dir, pokemon_dir)
        
        # Get image files
        image_files = [f for f in os.listdir(pokemon_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if max_samples_per_class and len(image_files) > max_samples_per_class:
            # Randomly sample if we have too many images
            np.random.shuffle(image_files)
            image_files = image_files[:max_samples_per_class]
        
        # Extract features for each image
        for img_file in image_files:
            img_path = os.path.join(pokemon_path, img_file)
            hist_features = extract_color_histogram(img_path)
            
            if hist_features is not None:
                features.append(hist_features)
                labels.append(pokemon_dir)
                if pokemon_dir not in class_names:
                    class_names.append(pokemon_dir)
    
    if verbose:
        print(f"Loaded {len(features)} images across {len(class_names)} classes")
    
    return np.array(features), np.array(labels), class_names

def train_and_evaluate_models(X_train, y_train, X_test, y_test, class_names):
    """
    Train and evaluate multiple classifiers.
    
    Args:
        X_train, y_train: Training data and labels
        X_test, y_test: Test data and labels
        class_names: List of class names
        
    Returns:
        Dictionary of trained models
    """
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(gamma='scale', probability=True),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        training_time = time.time() - start_time
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'training_time': training_time,
            'predictions': y_pred
        }
        
        # Print detailed classification report for best model
        if name == 'RandomForest':  # Usually the best performer for this task
            print("\nDetailed Classification Report:")
            print(classification_report(y_test, y_pred))
    
    return results

def predict_test_images(model, test_dir, class_names, label_encoder, top_k=3):
    """
    Predict the identity of Pokemon in the test dataset.
    
    Args:
        model: Trained classifier
        test_dir: Directory containing test images
        class_names: List of class names
        label_encoder: LabelEncoder used to transform class names
        top_k: Number of top predictions to return
        
    Returns:
        DataFrame with predictions
    """
    results = []
    pokemon_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    
    print(f"Making predictions on {len(pokemon_dirs)} Pokemon in test set...")
    
    for pokemon_dir in tqdm(pokemon_dirs):
        pokemon_path = os.path.join(test_dir, pokemon_dir)
        
        # Get image files
        image_files = [f for f in os.listdir(pokemon_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(pokemon_path, img_file)
            
            # Extract features
            features = extract_color_histogram(img_path)
            
            if features is not None:
                # Get prediction probabilities
                probs = model.predict_proba([features])[0]
                
                # Get indices of top k predictions
                top_indices = np.argsort(probs)[-top_k:][::-1]
                
                # Get corresponding class names and probabilities
                top_classes = [class_names[label_encoder.transform([class_names[i]])[0]] for i in top_indices]
                top_probs = [probs[i] for i in top_indices]
                
                results.append({
                    'true_pokemon': pokemon_dir,
                    'image_file': img_file,
                    'top1_prediction': top_classes[0],
                    'top1_confidence': top_probs[0],
                    'is_correct': top_classes[0] == pokemon_dir,
                    'top3_predictions': top_classes,
                    'top3_confidences': top_probs
                })
    
    results_df = pd.DataFrame(results)
    
    # Calculate accuracy metrics
    top1_accuracy = results_df['is_correct'].mean()
    top3_accuracy = results_df.apply(
        lambda row: row['true_pokemon'] in row['top3_predictions'], axis=1).mean()
    
    print(f"Test Set Results:")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Top-3 Accuracy: {top3_accuracy:.4f}")
    
    return results_df

def visualize_predictions(results_df, test_dir, num_examples=5):
    """Visualize some predictions to see what the model got right and wrong."""
    # Get some correct and incorrect predictions
    correct = results_df[results_df['is_correct']].sample(min(num_examples, len(results_df[results_df['is_correct']])))
    incorrect = results_df[~results_df['is_correct']].sample(min(num_examples, len(results_df[~results_df['is_correct']])))
    
    # Function to display a single prediction
    def display_prediction(row, ax, title_color):
        img_path = os.path.join(test_dir, row['true_pokemon'], row['image_file'])
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(f"True: {row['true_pokemon']}\nPred: {row['top1_prediction']} ({row['top1_confidence']:.2f})", 
                     color=title_color)
        ax.axis('off')
    
    # Display correct predictions
    if len(correct) > 0:
        fig, axes = plt.subplots(1, len(correct), figsize=(15, 3))
        if len(correct) == 1:
            axes = [axes]
        
        plt.suptitle("Correct Predictions", fontsize=16)
        for i, (_, row) in enumerate(correct.iterrows()):
            display_prediction(row, axes[i], 'green')
        plt.tight_layout()
        plt.show()
    
    # Display incorrect predictions
    if len(incorrect) > 0:
        fig, axes = plt.subplots(1, len(incorrect), figsize=(15, 3))
        if len(incorrect) == 1:
            axes = [axes]
        
        plt.suptitle("Incorrect Predictions", fontsize=16)
        for i, (_, row) in enumerate(incorrect.iterrows()):
            display_prediction(row, axes[i], 'red')
        plt.tight_layout()
        plt.show()

def main():
    # Path to data directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    
    dataset_dir = os.path.join(project_root, 'data/kaggle_data/dataset')
    test_dir = os.path.join(project_root, 'data/kaggle_data/test')
    
    # Check if directories exist
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
    
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        return
    
    print(f"Dataset directory: {dataset_dir}")
    print(f"Test directory: {test_dir}")
    
    # For quick testing, limit number of classes and samples per class
    # Set these to None for full training
    LIMIT_CLASSES = 50  # Set to None to use all classes
    MAX_SAMPLES_PER_CLASS = 10  # Set to None to use all samples
    
    # Load and extract features from the dataset
    print("Loading training dataset...")
    features, labels, class_names = load_dataset(
        dataset_dir, 
        limit_classes=LIMIT_CLASSES,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS
    )
    
    # Encode the labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Split into training and validation sets
    print("Splitting into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        features, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Validation set size: {X_val.shape[0]} samples")
    
    # Train and evaluate models
    print("\nTraining models...")
    results = train_and_evaluate_models(X_train, y_train, X_val, y_val, class_names)
    
    # Select the best model (you can change this to pick a different model)
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_model = results[best_model_name]['model']
    print(f"\nBest model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")
    
    # Predict on test data
    print("\nPredicting on test data...")
    predictions_df = predict_test_images(best_model, test_dir, class_names, label_encoder)
    
    # Visualize predictions
    print("\nVisualizing predictions...")
    visualize_predictions(predictions_df, test_dir)
    
    # Save results to CSV
    results_path = os.path.join(project_root, 'results', 'test_predictions.csv')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    predictions_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main() 