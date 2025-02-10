#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

def main(input_file, output_dir):
    print(f"Reading data from {input_file}")
    data = pd.read_json(input_file)
    X = data.drop(columns=['id', 'qd_id', 'target_label'])
    y = data['target_label']
    X.fillna(0, inplace=True)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize classifier and cross-validation
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    
    print("Cross-validation scores:", scores)
    print("Mean CV accuracy:", scores.mean())
    
    # Train and evaluate
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a RandomForest classifier on JSON data.")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("output_dir", help="Output directory (currently unused)")
    args = parser.parse_args()
    main(args.input_file, args.output_dir)
