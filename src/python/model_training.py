"""
Model training and evaluation functions for the Who's That Pokémon project.

This module contains functions for training and evaluating machine learning models
for Pokémon classification.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def prepare_data(features, labels, test_size=0.2, random_state=42, scale=True):
    """
    Prepare data for model training by splitting and scaling.
    
    Parameters:
    -----------
    features : numpy.ndarray
        Feature matrix where each row is a sample and each column is a feature
    labels : numpy.ndarray or list
        Target labels
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, default=42
        Controls the shuffling applied to the data before applying the split
    scale : bool, default=True
        Whether to scale the features
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, label_encoder, scaler)
    """
    # Encode categorical labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        features, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Scale features if requested
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, label_encoder, scaler

def train_knn_model(X_train, y_train, n_neighbors=5, weights='uniform', cv=5):
    """
    Train a K-Nearest Neighbors classifier.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training feature matrix
    y_train : numpy.ndarray
        Training target values
    n_neighbors : int, default=5
        Number of neighbors
    weights : str, default='uniform'
        Weight function ('uniform' or 'distance')
    cv : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    sklearn.neighbors.KNeighborsClassifier
        Trained KNN model
    float
        Cross-validated accuracy score
    """
    # Create model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    
    # Train model
    knn.fit(X_train, y_train)
    
    # Calculate cross-validation score
    cv_scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
    cv_accuracy = cv_scores.mean()
    
    return knn, cv_accuracy

def train_svm_model(X_train, y_train, C=1.0, kernel='rbf', gamma='scale', cv=5):
    """
    Train a Support Vector Machine classifier.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training feature matrix
    y_train : numpy.ndarray
        Training target values
    C : float, default=1.0
        Regularization parameter
    kernel : str, default='rbf'
        Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
    gamma : str or float, default='scale'
        Kernel coefficient
    cv : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    sklearn.svm.SVC
        Trained SVM model
    float
        Cross-validated accuracy score
    """
    # Create model
    svm = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
    
    # Train model
    svm.fit(X_train, y_train)
    
    # Calculate cross-validation score
    cv_scores = cross_val_score(svm, X_train, y_train, cv=cv, scoring='accuracy')
    cv_accuracy = cv_scores.mean()
    
    return svm, cv_accuracy

def train_random_forest_model(X_train, y_train, n_estimators=100, max_depth=None, cv=5):
    """
    Train a Random Forest classifier.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training feature matrix
    y_train : numpy.ndarray
        Training target values
    n_estimators : int, default=100
        Number of trees in the forest
    max_depth : int, default=None
        Maximum depth of the trees
    cv : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    sklearn.ensemble.RandomForestClassifier
        Trained Random Forest model
    float
        Cross-validated accuracy score
    """
    # Create model
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
    # Train model
    rf.fit(X_train, y_train)
    
    # Calculate cross-validation score
    cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
    cv_accuracy = cv_scores.mean()
    
    return rf, cv_accuracy

def tune_model_hyperparameters(X_train, y_train, model_type='knn', param_grid=None, cv=5):
    """
    Tune model hyperparameters using grid search.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training feature matrix
    y_train : numpy.ndarray
        Training target values
    model_type : str, default='knn'
        Type of model to tune ('knn', 'svm', 'random_forest')
    param_grid : dict, default=None
        Dictionary with parameters names as keys and lists of parameter settings
    cv : int, default=5
        Number of cross-validation folds
        
    Returns:
    --------
    sklearn estimator
        Best model found by grid search
    dict
        Best parameters
    float
        Best cross-validated accuracy score
    """
    # Create base model based on type
    if model_type == 'knn':
        model = KNeighborsClassifier()
        if param_grid is None:
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
    elif model_type == 'svm':
        model = SVC(probability=True)
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
            }
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    return best_model, best_params, best_score

def evaluate_model(model, X_test, y_test, label_encoder, top_n=1):
    """
    Evaluate a trained model on test data.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_test : numpy.ndarray
        Test feature matrix
    y_test : numpy.ndarray
        Test target values
    label_encoder : sklearn.preprocessing.LabelEncoder
        Encoder used to transform class labels
    top_n : int, default=1
        Calculate top-N accuracy
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Get probabilities for top-N accuracy
    y_proba = model.predict_proba(X_test)
    
    # Calculate various metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calculate top-N accuracy
    top_n_accuracy = 0
    if top_n > 1:
        for i, probs in enumerate(y_proba):
            top_indices = np.argsort(probs)[-top_n:]
            if y_test[i] in top_indices:
                top_n_accuracy += 1
        top_n_accuracy /= len(y_test)
    else:
        top_n_accuracy = accuracy
    
    # Store results
    results = {
        'accuracy': accuracy,
        f'top_{top_n}_accuracy': top_n_accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': report,
        'y_pred': y_pred,
        'y_test': y_test,
        'class_names': label_encoder.classes_
    }
    
    return results

def plot_confusion_matrix(results, figsize=(10, 10), top_n=20):
    """
    Plot confusion matrix from evaluation results.
    
    Parameters:
    -----------
    results : dict
        Evaluation results from evaluate_model function
    figsize : tuple, default=(10, 10)
        Figure size
    top_n : int, default=20
        Number of classes to include in plot (most frequent in test set)
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Get confusion matrix and class names
    conf_matrix = results['confusion_matrix']
    class_names = results['class_names']
    
    # Get most frequent classes in test set
    y_test = results['y_test']
    class_counts = np.bincount(y_test)
    top_indices = np.argsort(class_counts)[-top_n:]
    
    # Extract subset of confusion matrix for most frequent classes
    conf_matrix_subset = conf_matrix[top_indices][:, top_indices]
    class_names_subset = [class_names[i] for i in top_indices]
    
    # Plot confusion matrix
    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix_subset, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_subset, yticklabels=class_names_subset)
    plt.title('Confusion Matrix (Top {} Classes)'.format(top_n))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    return plt.gcf()

def save_model(model, save_dir, filename, scaler=None, label_encoder=None):
    """
    Save trained model and associated transformers.
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model to save
    save_dir : str
        Directory to save model in
    filename : str
        Base filename for model
    scaler : sklearn.preprocessing.StandardScaler, default=None
        Feature scaler
    label_encoder : sklearn.preprocessing.LabelEncoder, default=None
        Label encoder
        
    Returns:
    --------
    str
        Path to saved model file
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, f"{filename}.joblib")
    joblib.dump(model, model_path)
    
    # Save scaler if provided
    if scaler is not None:
        scaler_path = os.path.join(save_dir, f"{filename}_scaler.joblib")
        joblib.dump(scaler, scaler_path)
    
    # Save label encoder if provided
    if label_encoder is not None:
        encoder_path = os.path.join(save_dir, f"{filename}_encoder.joblib")
        joblib.dump(label_encoder, encoder_path)
    
    return model_path

def load_model(model_path, scaler_path=None, encoder_path=None):
    """
    Load trained model and associated transformers.
    
    Parameters:
    -----------
    model_path : str
        Path to saved model file
    scaler_path : str, default=None
        Path to saved scaler file
    encoder_path : str, default=None
        Path to saved label encoder file
        
    Returns:
    --------
    tuple
        (model, scaler, label_encoder)
    """
    # Load model
    model = joblib.load(model_path)
    
    # Load scaler if path provided
    scaler = None
    if scaler_path is not None and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    
    # Load label encoder if path provided
    label_encoder = None
    if encoder_path is not None and os.path.exists(encoder_path):
        label_encoder = joblib.load(encoder_path)
    
    return model, scaler, label_encoder 