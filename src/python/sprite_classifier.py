#!/usr/bin/env python3

"""
Pokemon Sprite and Render Classifier
-----------------------------------
This script trains a classifier on the Kaggle Pokemon dataset
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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Feature extraction function (same as pokemon_classifier.py)
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

def load_sprites_and_renders(project_root):
    """
    Load sprites and renders for testing.
    
    Args:
        project_root: Path to project root directory
        
    Returns:
        Dictionary with sprite/render data
    """
    # Define paths
    gen1_path = os.path.join(project_root, 'data/sprites/gen_1/main-sprites/red-blue')
    gen5_path = os.path.join(project_root, 'data/sprites/gen_5/main-sprites/black-white')
    sugimori_path = os.path.join(project_root, 'data/renders/sugimori')
    
    # Check if directories exist
    sprites_available = {'gen1': os.path.exists(gen1_path),
                         'gen5': os.path.exists(gen5_path),
                         'sugimori': os.path.exists(sugimori_path)}
    
    for source, available in sprites_available.items():
        if not available:
            print(f"Warning: {source} sprites directory not found")
    
    # Collect available sprite files
    sprites_data = {
        'gen1': [],
        'gen5': [],
        'sugimori': []
    }
    
    # Get Pokemon IDs and file paths
    for source, path in [('gen1', gen1_path), ('gen5', gen5_path), ('sugimori', sugimori_path)]:
        if not sprites_available[source]:
            continue
            
        files = [f for f in os.listdir(path) if f.lower().endswith('.png')]
        for file in files:
            # Extract Pokemon ID from filename
            pokemon_id = os.path.splitext(file)[0]
            
            # Check if it's a number (some sprites have suffixes like -east)
            if pokemon_id.isdigit():
                sprites_data[source].append({
                    'id': int(pokemon_id),
                    'path': os.path.join(path, file)
                })
    
    # Print summary
    print("\nSprites and renders summary:")
    for source, data in sprites_data.items():
        print(f"  {source}: {len(data)} images")
    
    return sprites_data

def map_ids_to_pokemon_names(project_root):
    """
    Create a mapping from Pokemon IDs to Pokemon names.
    
    Returns:
        Dictionary mapping IDs to names
    """
    # Try to find a mapping file or database
    # For now, we'll use a simplified approach for known Pokemon
    id_to_name = {
        1: "bulbasaur", 2: "ivysaur", 3: "venusaur", 4: "charmander", 5: "charmeleon",
        6: "charizard", 7: "squirtle", 8: "wartortle", 9: "blastoise", 10: "caterpie",
        11: "metapod", 12: "butterfree", 13: "weedle", 14: "kakuna", 15: "beedrill",
        16: "pidgey", 17: "pidgeotto", 18: "pidgeot", 19: "rattata", 20: "raticate",
        21: "spearow", 22: "fearow", 23: "ekans", 24: "arbok", 25: "pikachu",
        26: "raichu", 27: "sandshrew", 28: "sandslash", 29: "nidoran-f", 30: "nidorina",
        31: "nidoqueen", 32: "nidoran-m", 33: "nidorino", 34: "nidoking", 35: "clefairy",
        36: "clefable", 37: "vulpix", 38: "ninetales", 39: "jigglypuff", 40: "wigglytuff",
        41: "zubat", 42: "golbat", 43: "oddish", 44: "gloom", 45: "vileplume",
        46: "paras", 47: "parasect", 48: "venonat", 49: "venomoth", 50: "diglett",
        51: "dugtrio", 52: "meowth", 53: "persian", 54: "psyduck", 55: "golduck",
        56: "mankey", 57: "primeape", 58: "growlithe", 59: "arcanine", 60: "poliwag",
        61: "poliwhirl", 62: "poliwrath", 63: "abra", 64: "kadabra", 65: "alakazam",
        66: "machop", 67: "machoke", 68: "machamp", 69: "bellsprout", 70: "weepinbell",
        71: "victreebel", 72: "tentacool", 73: "tentacruel", 74: "geodude", 75: "graveler",
        76: "golem", 77: "ponyta", 78: "rapidash", 79: "slowpoke", 80: "slowbro",
        81: "magnemite", 82: "magneton", 83: "farfetchd", 84: "doduo", 85: "dodrio",
        86: "seel", 87: "dewgong", 88: "grimer", 89: "muk", 90: "shellder",
        91: "cloyster", 92: "gastly", 93: "haunter", 94: "gengar", 95: "onix",
        96: "drowzee", 97: "hypno", 98: "krabby", 99: "kingler", 100: "voltorb",
        101: "electrode", 102: "exeggcute", 103: "exeggutor", 104: "cubone", 105: "marowak",
        106: "hitmonlee", 107: "hitmonchan", 108: "lickitung", 109: "koffing", 110: "weezing",
        111: "rhyhorn", 112: "rhydon", 113: "chansey", 114: "tangela", 115: "kangaskhan",
        116: "horsea", 117: "seadra", 118: "goldeen", 119: "seaking", 120: "staryu",
        121: "starmie", 122: "mr-mime", 123: "scyther", 124: "jynx", 125: "electabuzz",
        126: "magmar", 127: "pinsir", 128: "tauros", 129: "magikarp", 130: "gyarados",
        131: "lapras", 132: "ditto", 133: "eevee", 134: "vaporeon", 135: "jolteon",
        136: "flareon", 137: "porygon", 138: "omanyte", 139: "omastar", 140: "kabuto",
        141: "kabutops", 142: "aerodactyl", 143: "snorlax", 144: "articuno", 145: "zapdos",
        146: "moltres", 147: "dratini", 148: "dragonair", 149: "dragonite", 150: "mewtwo",
        151: "mew"
    }
    
    return id_to_name

def evaluate_sprites_with_model(model, sprites_data, id_to_name, class_names, label_encoder):
    """
    Evaluate the model on sprites and renders.
    
    Args:
        model: Trained classifier
        sprites_data: Dictionary with sprite/render data
        id_to_name: Dictionary mapping Pokemon IDs to names
        class_names: List of class names in training data
        label_encoder: LabelEncoder used for training
        
    Returns:
        Dictionary with evaluation results
    """
    results = {}
    
    for source, sprites in sprites_data.items():
        print(f"\nEvaluating {source} sprites/renders...")
        source_results = []
        
        for sprite in tqdm(sprites):
            pokemon_id = sprite['id']
            
            # Skip if we don't have a name mapping
            if pokemon_id not in id_to_name:
                continue
                
            pokemon_name = id_to_name[pokemon_id]
            
            # Skip if the Pokemon is not in our training classes
            if pokemon_name not in class_names:
                continue
            
            # Extract features
            img_path = sprite['path']
            features = extract_color_histogram(img_path)
            
            if features is None:
                continue
                
            # Get prediction
            probs = model.predict_proba([features])[0]
            top_5_indices = np.argsort(probs)[-5:][::-1]
            
            # Get top 5 predictions
            top_5_classes = [class_names[i] for i in top_5_indices]
            top_5_probs = [probs[i] for i in top_5_indices]
            
            # Check if correct
            is_correct = pokemon_name == top_5_classes[0]
            in_top_5 = pokemon_name in top_5_classes
            
            # Save result
            source_results.append({
                'id': pokemon_id,
                'true_name': pokemon_name,
                'pred_name': top_5_classes[0],
                'confidence': top_5_probs[0],
                'is_correct': is_correct,
                'in_top_5': in_top_5,
                'top_5_preds': top_5_classes,
                'top_5_confs': top_5_probs
            })
        
        # Calculate metrics
        if source_results:
            df = pd.DataFrame(source_results)
            accuracy = df['is_correct'].mean()
            top5_accuracy = df['in_top_5'].mean()
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Top-5 Accuracy: {top5_accuracy:.4f}")
            print(f"  Evaluated {len(df)} sprites")
            
            results[source] = {
                'data': df,
                'accuracy': accuracy,
                'top5_accuracy': top5_accuracy
            }
        else:
            print(f"  No results for {source}")
    
    return results

def visualize_results(results, num_examples=4):
    """Visualize sprite classification results."""
    for source, result in results.items():
        if 'data' not in result or len(result['data']) == 0:
            continue
            
        df = result['data']
        
        # Sample correct and incorrect predictions
        correct = df[df['is_correct']].sample(min(num_examples, len(df[df['is_correct']])))
        incorrect = df[~df['is_correct']].sample(min(num_examples, len(df[~df['is_correct']])))
        
        # Function to display predictions
        def show_predictions(samples, title, color):
            if len(samples) == 0:
                return
                
            fig, axes = plt.subplots(1, len(samples), figsize=(16, 4))
            if len(samples) == 1:
                axes = [axes]
                
            plt.suptitle(f"{source.capitalize()} - {title}", fontsize=16)
            
            for i, (_, row) in enumerate(samples.iterrows()):
                img = Image.open(sprites_data[source][df[df['id'] == row['id']].index[0]]['path'])
                axes[i].imshow(img)
                axes[i].set_title(f"True: {row['true_name']}\nPred: {row['pred_name']}\nConf: {row['confidence']:.2f}", 
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
    
    # Load mapping from Pokemon IDs to names
    id_to_name = map_ids_to_pokemon_names(project_root)
    
    # Load sprites and renders
    sprites_data = load_sprites_and_renders(project_root)
    
    # For faster training, limit the number of classes and samples
    LIMIT_CLASSES = None  # Set to None to use all classes
    MAX_SAMPLES_PER_CLASS = 15  # Set to None to use all samples
    
    # Load and extract features from the dataset
    print("\nLoading training dataset...")
    features, labels, class_names = load_dataset(
        dataset_dir, 
        limit_classes=LIMIT_CLASSES,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS
    )
    
    # Encode the labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Train a model
    print("\nTraining model...")
    model = train_model(features, encoded_labels, model_name="RandomForest")
    
    # Evaluate on sprites and renders
    print("\nEvaluating on sprites and renders...")
    results = evaluate_sprites_with_model(
        model, 
        sprites_data, 
        id_to_name, 
        class_names, 
        label_encoder
    )
    
    # Visualize results
    print("\nVisualizing results...")
    visualize_results(results)
    
    # Save results to CSV
    print("\nSaving results...")
    for source, result in results.items():
        if 'data' in result:
            output_path = os.path.join(project_root, 'results', f'{source}_results.csv')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result['data'].to_csv(output_path, index=False)
            print(f"Saved {source} results to {output_path}")

if __name__ == "__main__":
    main() 