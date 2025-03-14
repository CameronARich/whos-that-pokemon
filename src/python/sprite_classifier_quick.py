#!/usr/bin/env python3

"""
Quick Pokemon Sprite and Render Classifier
-----------------------------------------
This script trains a classifier on a small subset of the Kaggle Pokemon dataset
and tests it on sprites from different generations and official renders.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

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

def load_dataset(dataset_dir, pokemon_list, max_samples_per_class=10, verbose=True):
    """
    Load specific Pokemon images from the dataset directory and extract features.
    
    Args:
        dataset_dir: Directory containing Pokemon subdirectories
        pokemon_list: List of Pokemon names to include
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
    
    if verbose:
        print(f"Loading data for {len(pokemon_list)} specific Pokemon...")
    
    # Process each Pokemon class
    for pokemon_name in tqdm(pokemon_list, disable=not verbose):
        pokemon_path = os.path.join(dataset_dir, pokemon_name)
        
        if not os.path.exists(pokemon_path):
            if verbose:
                print(f"Warning: Directory for {pokemon_name} not found")
            continue
        
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
                labels.append(pokemon_name)
                if pokemon_name not in class_names:
                    class_names.append(pokemon_name)
    
    if verbose:
        print(f"Loaded {len(features)} images across {len(class_names)} classes")
    
    return np.array(features), np.array(labels), class_names

def train_model(X_train, y_train, model_name="RandomForest"):
    """
    Train a machine learning model.
    
    Args:
        X_train, y_train: Training data and labels
        model_name: Name of the model to train
        
    Returns:
        Trained model
    """
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(gamma='scale', probability=True),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    if model_name not in models:
        print(f"Model {model_name} not found. Using RandomForest instead.")
        model_name = 'RandomForest'
    
    print(f"Training {model_name}...")
    start_time = time.time()
    
    # Train the model
    model = models[model_name]
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    
    return model

def get_test_sprites(project_root):
    """
    Get a selection of test sprites for specific Pokemon.
    
    Args:
        project_root: Path to project root directory
        
    Returns:
        Dictionary with sprite/render data
    """
    # Define paths
    gen1_path = os.path.join(project_root, 'data/sprites/gen_1/main-sprites/red-blue')
    gen5_path = os.path.join(project_root, 'data/sprites/gen_5/main-sprites/black-white')
    sugimori_path = os.path.join(project_root, 'data/renders/sugimori')
    
    # Test with these Pokemon names (first-gen Pokemon that should be available in all sources)
    test_pokemon = ["pikachu", "charizard", "mewtwo", "gengar", "snorlax", 
                    "blastoise", "venusaur", "mew", "lapras", "arcanine"]
    
    # Pokemon name to ID mapping for reference (optional)
    name_to_id = {
        "pikachu": 25, "charizard": 6, "mewtwo": 150, "gengar": 94, 
        "snorlax": 143, "blastoise": 9, "venusaur": 3, "mew": 151,
        "lapras": 131, "arcanine": 59
    }
    
    # Collect test files
    test_data = []
    
    # Get Pokemon sprites from each source
    for source, path in [('gen1', gen1_path), ('gen5', gen5_path), ('sugimori', sugimori_path)]:
        if not os.path.exists(path):
            print(f"Warning: {source} sprites directory not found")
            continue
            
        print(f"Found {source} sprites directory: {path}")
        
        for pokemon_name in test_pokemon:
            # Try both name.png and ID.png formats
            sprite_paths = [
                os.path.join(path, f"{pokemon_name}.png")
            ]
            
            # For Gen 1, also try numeric ID
            if source == 'gen1' and pokemon_name in name_to_id:
                sprite_paths.append(os.path.join(path, f"{name_to_id[pokemon_name]}.png"))
            
            # Use the first path that exists
            for sprite_path in sprite_paths:
                if os.path.exists(sprite_path):
                    test_data.append({
                        'id': name_to_id.get(pokemon_name, 0),
                        'name': pokemon_name,
                        'source': source,
                        'path': sprite_path
                    })
                    break
    
    print(f"Found {len(test_data)} test sprites across all sources")
    
    return test_data, test_pokemon

def predict_sprites(model, test_data, class_names, label_encoder):
    """
    Predict the identity of Pokemon sprites.
    
    Args:
        model: Trained classifier
        test_data: List of test sprite data
        class_names: List of class names
        label_encoder: LabelEncoder used for training
        
    Returns:
        Results DataFrame
    """
    results = []
    
    print("Predicting on test sprites...")
    
    for sprite in tqdm(test_data):
        # Extract features
        features = extract_color_histogram(sprite['path'])
        
        if features is None:
            continue
            
        # Get predictions
        probs = model.predict_proba([features])[0]
        top_3_indices = np.argsort(probs)[-3:][::-1]
        
        # Get top 3 predictions
        top_3_classes = [class_names[i] for i in top_3_indices]
        top_3_probs = [probs[i] for i in top_3_indices]
        
        # Check if correct
        is_correct = sprite['name'] == top_3_classes[0]
        in_top_3 = sprite['name'] in top_3_classes
        
        # Save result
        results.append({
            'id': sprite['id'],
            'true_name': sprite['name'],
            'source': sprite['source'],
            'top1_pred': top_3_classes[0],
            'top1_prob': top_3_probs[0],
            'is_correct': is_correct,
            'in_top_3': in_top_3,
            'top3_preds': top_3_classes,
            'top3_probs': top_3_probs
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def analyze_and_visualize_results(results_df, test_data):
    """
    Analyze and visualize the prediction results.
    
    Args:
        results_df: DataFrame with prediction results
        test_data: Original test data
    """
    # Overall accuracy
    accuracy = results_df['is_correct'].mean()
    top3_accuracy = results_df['in_top_3'].mean()
    
    print(f"\nOverall Results:")
    print(f"  Top-1 Accuracy: {accuracy:.4f}")
    print(f"  Top-3 Accuracy: {top3_accuracy:.4f}")
    
    # Accuracy by source
    print("\nResults by Source:")
    for source in results_df['source'].unique():
        source_df = results_df[results_df['source'] == source]
        source_acc = source_df['is_correct'].mean()
        source_top3 = source_df['in_top_3'].mean()
        print(f"  {source}:")
        print(f"    Top-1 Accuracy: {source_acc:.4f}")
        print(f"    Top-3 Accuracy: {source_top3:.4f}")
    
    # Visualize correct and incorrect predictions for each source
    for source in results_df['source'].unique():
        source_df = results_df[results_df['source'] == source]
        
        # Sample some correct and incorrect predictions
        correct = source_df[source_df['is_correct']].sample(min(3, len(source_df[source_df['is_correct']])))
        incorrect = source_df[~source_df['is_correct']].sample(min(3, len(source_df[~source_df['is_correct']])))
        
        # Function to display predictions
        def show_predictions(samples, title, color):
            if len(samples) == 0:
                return
                
            fig, axes = plt.subplots(1, len(samples), figsize=(15, 4))
            if len(samples) == 1:
                axes = [axes]
                
            plt.suptitle(f"{source.capitalize()} - {title}", fontsize=16)
            
            for i, (_, row) in enumerate(samples.iterrows()):
                # Find the image path for this sprite
                sprite = next((s for s in test_data if s['id'] == row['id'] and s['source'] == row['source']), None)
                if sprite is None:
                    continue
                    
                # Display the image
                img = Image.open(sprite['path'])
                axes[i].imshow(img)
                axes[i].set_title(f"True: {row['true_name']}\nPred: {row['top1_pred']}\nProb: {row['top1_prob']:.2f}", 
                                 color=color)
                axes[i].axis('off')
                
            plt.tight_layout()
            plt.show()
        
        # Show correct and incorrect predictions
        show_predictions(correct, "Correct Predictions", "green")
        show_predictions(incorrect, "Incorrect Predictions", "red")

def main():
    # Path to data directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../..'))
    
    dataset_dir = os.path.join(project_root, 'data/kaggle_data/dataset')
    
    # Check if directory exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
    
    print(f"Dataset directory: {dataset_dir}")
    
    # Get test sprites
    test_data, test_pokemon = get_test_sprites(project_root)
    
    # List of Pokemon to use for training (including our test Pokemon)
    pokemon_list = test_pokemon
    
    # Add some other common Pokemon for a bit more variety in the training set
    additional_pokemon = [
        "eevee", "jigglypuff", "bulbasaur", "squirtle", "charmander",
        "dragonite", "gyarados", "magikarp", "snorlax", "vaporeon", 
        "flareon", "jolteon", "psyduck", "vulpix", "ninetales",
        "poliwhirl", "growlithe", "machamp", "geodude", "abra"
    ]
    pokemon_list.extend([p for p in additional_pokemon if p not in pokemon_list])
    
    # Load dataset for these Pokemon only
    print("\nLoading training dataset...")
    features, labels, class_names = load_dataset(
        dataset_dir, 
        pokemon_list=pokemon_list,
        max_samples_per_class=15
    )
    
    # Encode the labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Train model
    print("\nTraining model...")
    model = train_model(features, encoded_labels, model_name="RandomForest")
    
    # Predict on test sprites
    print("\nPredicting on test sprites...")
    results_df = predict_sprites(model, test_data, class_names, label_encoder)
    
    # Analyze and visualize results
    print("\nAnalyzing results...")
    analyze_and_visualize_results(results_df, test_data)
    
    # Save results to CSV
    output_path = os.path.join(project_root, 'results', 'sprite_results.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main() 