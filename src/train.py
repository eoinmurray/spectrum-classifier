import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def main(input_file, output_dir):
    print(f"Reading data from {input_file}")
    data = pd.read_json(input_file)
    X = data.drop(columns=['id', 'qd_id', 'target_label'])
    y = data['target_label']
    X.fillna(0, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # # Optional: Feature importance
    # feature_importances = pd.DataFrame({
    #     'Feature': X.columns,
    #     'Importance': clf.feature_importances_
    # }).sort_values(by='Importance', ascending=False)

    # print("Feature Importances:\n", feature_importances)


