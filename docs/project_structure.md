# Who's That Pokémon? - Project Structure

## Overview

This document outlines the structure of the project and provides a roadmap for development.

## Project Goals

Based on the project aims document and class assignment, our goals are to:

1. Classify Pokémon images using traditional computer vision and machine learning techniques (NOT neural networks)
2. Compare how different sklearn computer vision methods perform in matching Pokémon
3. Analyze how generational differences affect classification accuracy
4. Present our findings in a comprehensive Jupyter notebook

## Project Structure

```
.
├── data/                      # Data directory
│   ├── kaggle_data/           # Primary dataset from Kaggle (1000 Pokémon species)
│   │   ├── dataset/           # Raw images organized by Pokémon
│   │   ├── train/             # Training split
│   │   ├── val/               # Validation split
│   │   ├── test/              # Test split
│   │   └── metadata.csv       # Metadata for all images (includes stats)
│   ├── sprites/               # (Optional) Official game sprites 
│   └── renders/               # (Optional) Official Ken Sugimori renders
│
├── docs/                      # Documentation
│   ├── class_assignment.md    # Original class assignment
│   └── project_aims.md        # Project goals and requirements
│
├── src/                       # Source code
│   ├── jupyter/               # Jupyter notebooks
│   │   └── pokemon_classifier.ipynb  # Main project notebook
│   └── python/                # Python modules and scripts
│       ├── feature_extraction.py     # Feature extraction methods
│       ├── model_training.py         # Model training functions
│       └── evaluation.py             # Evaluation metrics and visualization
│
└── requirements.txt           # Project dependencies
```

## Development Roadmap

### Phase 1: Data Exploration and Preparation

- [x] Set up the project environment
- [ ] Explore the dataset structure and metadata
- [ ] Visualize sample images from different Pokémon species
- [ ] Analyze class distribution and balance
- [ ] Verify the train/val/test splits

### Phase 2: Feature Extraction

- [ ] Implement color histogram extraction
- [ ] Implement shape/edge-based features (HOG, SIFT, etc.)
- [ ] Implement texture-based features
- [ ] Extract and save features for the whole dataset
- [ ] Analyze and compare different feature types

### Phase 3: Model Training and Evaluation

- [ ] Train classic ML models (SVM, Random Forest, KNN, etc.)
- [ ] Compare models using cross-validation
- [ ] Tune hyperparameters for the best models
- [ ] Evaluate performance on the test set
- [ ] Analyze misclassifications and challenging cases

### Phase 4: Analysis of Generational Differences

- [ ] Group Pokémon by generation
- [ ] Compare classification accuracy across generations
- [ ] Analyze if increased sprite fidelity in newer generations improves matching
- [ ] Evaluate color-based features vs. shape-based features for different generations

### Phase 5: Documentation and Presentation

- [ ] Document all methods and findings in the Jupyter notebook
- [ ] Create visualizations to illustrate key results
- [ ] Prepare the final presentation
- [ ] Submit the completed project 