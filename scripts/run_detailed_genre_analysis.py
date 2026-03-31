#!/usr/bin/python3

import argparse, logging, os, sys
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# local imports

from spectral_probing.utils.setup import *
from spectral_probing.utils.datasets import LabelledDataset
from spectral_probing.utils.training import classify_dataset
from spectral_probing.models.encoders import *
from spectral_probing.models.classifiers import *

def run_analysis(target_genre, confused_genres, band_filter):
    """
    Runs a detailed confusion matrix analysis for a specific genre and frequency band.

    Args:
        target_genre (str): The main genre to analyze.
        confused_genres (list): A list of genres that are often confused with the target genre.
        band_filter (str): The frequency band filter to apply (e.g., "band(0,4)").
    """

    # --- 1. Setup ---
    exp_path = f"results/detailed_analysis/{target_genre}_{band_filter.replace('(',')').replace(',','-').replace(')','')}"
    setup_experiment(exp_path, prediction=True)
    logging.info(f"Running detailed analysis for genre '{target_genre}' with filter '{band_filter}'")
    logging.info(f"Results will be saved to {exp_path}")

    model_path = "results/spectral_baseline/best.pt" # Assuming a pre-trained model exists
    if not os.path.exists(model_path):
        logging.error(f"[Error] No pre-trained model available at '{model_path}'. Exiting.")
        sys.exit(1)

    # --- 2. Data Loading and Filtering ---
    valid_data = LabelledDataset.from_path("data/fma_metadata/validation.csv")
    
    genres_to_include = [target_genre] + confused_genres
    
    # Filter the dataset
    filtered_inputs = []
    filtered_labels = []
    for i, label_list in enumerate(valid_data._labels):
        # Assuming one label per item
        label = label_list[0]
        if label in genres_to_include:
            filtered_inputs.append(valid_data._inputs[i])
            filtered_labels.append(valid_data._labels[i])

    filtered_dataset = LabelledDataset(filtered_inputs, filtered_labels)
    label_types = sorted(list(set(genres_to_include)))
    logging.info(f"Filtered dataset to {len(filtered_dataset)} samples from genres: {label_types}")


    # --- 3. Model and Filter Setup ---
    frq_filter = setup_filter(band_filter)
    encoder = PrismEncoder.load(
        model_path, frq_filter=frq_filter, frq_tuning=False,
        emb_tuning=False, emb_pooling=None, cache=({} if True else None)
    )
    classifier = MLPClassifier.load(model_path, emb_model=encoder)
    criterion = CrossEntropyLoss(label_types)


    # --- 4. Run Prediction ---
    stats = classify_dataset(
        classifier, criterion, None, filtered_dataset,
        batch_size=32, repeat_labels=False, mode='eval', return_predictions=True
    )

    # --- 5. Generate and Save Confusion Matrix ---
    true_labels = [item for sublist in filtered_dataset._labels for item in sublist]
    pred_labels_indices = [item for sublist in stats['predictions'] for item in sublist]
    
    idx_lbl_map = {idx: lbl for idx, lbl in enumerate(label_types)}
    pred_labels = [idx_lbl_map[p] for p in pred_labels_indices]

    cm = confusion_matrix(true_labels, pred_labels, labels=label_types)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_types, yticklabels=label_types, cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix for {target_genre} (Band: {band_filter})')
    
    # Save the plot
    plot_path = os.path.join(exp_path, "confusion_matrix.png")
    plt.savefig(plot_path)
    logging.info(f"Saved confusion matrix to {plot_path}")


if __name__ == '__main__':
    # Example Usage
    target_genre = 'Electronic'
    confused_genres = ['Rock', 'Pop'] # Genres often confused with Electronic
    
    # Example band filters to test
    band_filters = [
        "band(0,4)", 
        "band(4,8)", 
        "band(8,12)",
        "band(12,16)"
    ]

    for band_filter in band_filters:
        run_analysis(target_genre, confused_genres, band_filter)
