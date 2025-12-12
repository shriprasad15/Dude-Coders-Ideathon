#!/usr/bin/env python3
"""
Evaluate inference results against ground truth.
Ground truth: 
- Samples 1-2500 have solar (has_solar = True)
- Samples 2501-3000 do NOT have solar (has_solar = False)
"""
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate_results(results_path="outputs/results.json",
                     save_report=True):
    """Evaluate inference results against ground truth."""
    
    print("Loading results from:", results_path)
    with open(results_path) as f:
        results = json.load(f)
    
    # Create dataframe
    df = pd.DataFrame(results)
    
    # Handle sample_id types (could be int, float, or string)
    df['sample_id_int'] = df['sample_id'].apply(lambda x: int(float(str(x).replace('.0', ''))))
    
    # Ground truth: First 2500 should have solar, last 500 should not
    df['gt_has_solar'] = df['sample_id_int'].apply(lambda x: x <= 2500)
    
    # Get predictions and ground truth
    y_true = df['gt_has_solar'].values
    y_pred = df['has_solar'].values
    
    print('=' * 60)
    print('CLASSIFICATION REPORT (sklearn)')
    print('=' * 60)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, 
                                   target_names=['No Solar', 'Solar'],
                                   digits=4)
    print(report)
    
    print('=' * 60)
    print('CONFUSION MATRIX')
    print('=' * 60)
    cm = confusion_matrix(y_true, y_pred)
    print(f"                  Predicted")
    print(f"                  No Solar  Solar")
    print(f"Actual No Solar    {cm[0][0]:5d}    {cm[0][1]:5d}")
    print(f"Actual Solar       {cm[1][0]:5d}    {cm[1][1]:5d}")
    
  
    
    print('=' * 60)
    
    



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate inference results")
    parser.add_argument('--results', type=str, 
                        default="outputs/results.json",
                        help="Path to results JSON file")
    args = parser.parse_args()
    
    evaluate_results(args.results)
