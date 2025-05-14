import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
import xgboost as xgb
import joblib
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.decomposition import PCA
import os
import time
import warnings

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

DATA_FILE = 'expanded_dataset.csv'
PROCESSED_DATA_FILE = 'processed_data.pkl'
MODEL_FILE = 'final_model.pkl'
SCALER_FILE = 'scaler.pkl'
ENCODER_FILE = 'encoder.pkl'
HYPERPARAMETERS_FILE = 'best_hyperparameters.pkl'


def load_and_preprocess_data(file_path=DATA_FILE, save_path=PROCESSED_DATA_FILE, plot_data=False):
    """Loads, preprocesses, and visualizes the dataset."""

    if os.path.exists(save_path):
        print(f"Loading processed data from {save_path}")
        data = joblib.load(save_path)
        X_train, X_test, y_train, y_test, label_encoder, scaler = data['X_train'], data['X_test'], data['y_train'], data['y_test'], data['label_encoder'], data['scaler']
        return X_train, X_test, y_train, y_test, label_encoder, scaler

    print(f"Loading data from {file_path} and preprocessing...")
    data = pd.read_csv(file_path)
    X = data.drop(columns=['Mineral'])
    y = data['Mineral']

    # --- Data Analysis and Visualization ---
    print("\n--- Data Analysis ---")

    # Class Distribution Analysis (Added Here!)
    print("Class Distribution:\n", y.value_counts())

    print("\nDescriptive Statistics:\n", X.describe())
    print("\nCorrelation Matrix:\n", X.corr())

    if plot_data:
        plot_data_distributions(X, y)
        plot_data_relationships(X, y)

    # --- Preprocessing ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # Stratify

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = X_train.copy()  # Initialize X_test_scaled here
    X_test_scaled = scaler.transform(X_test)

    # --- Feature Selection ---
    selector = SelectKBest(mutual_info_classif, k='all')  # Or try f_classif, chi2
    X_train_selected = selector.fit_transform(X_train_scaled, y_train_encoded)
    X_test_selected = selector.transform(X_test_scaled)

    # --- Dimensionality Reduction ---
    pca = PCA(n_components=0.95)  # Retain 95% variance
    X_train_pca = pca.fit_transform(X_train_selected)
    X_test_pca = pca.transform(X_test_selected)

    # --- Imbalance Handling ---
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_pca, y_train_encoded)

    joblib.dump({
        'X_train': X_train_resampled,
        'X_test': X_test_pca,
        'y_train': y_train_resampled,
        'y_test': y_test_encoded,
        'label_encoder': label_encoder,
        'scaler': scaler
    }, save_path)

    return X_train_resampled, X_test_pca, y_train_resampled, y_test_encoded, label_encoder, scaler


def plot_data_distributions(X, y):
    """Visualizes the distribution of features."""

    for col in X.columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=X, x=col, hue=y, kde=True)
        plt.title(f'Distribution of {col} by Mineral Type')
        plt.show()


def plot_data_relationships(X, y):
    """Visualizes relationships between features."""

    for i, col1 in enumerate(X.columns):
        for j, col2 in enumerate(X.columns):
            if i < j:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=X, x=col1, y=col2, hue=y, alpha=0.5)
                plt.title(f'Scatter Plot of {col1} vs. {col2}')
                plt.show()


def create_model(model_type='xgboost', hyperparameters=None):
    """Creates a machine learning model."""

    if model_type == 'xgboost':
        model = xgb.XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', random_state=42,
                                  tree_method='hist')
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=42)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42, multi_class='ovr', solver='liblinear')
    elif model_type == 'naive_bayes':
        model = GaussianNB()
    elif model_type == 'svm':
        model = SVC(random_state=42, probability=True)  # probability=True for ROC AUC
    elif model_type == 'neural_network':
        model = MLPClassifier(random_state=42, max_iter=500)
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == 'knn':
        model = KNeighborsClassifier()
    elif model_type == 'lda':
        model = LinearDiscriminantAnalysis()
    elif model_type == 'qda':
        model = QuadraticDiscriminantAnalysis()
    elif model_type == 'gaussian_process':
        kernel = 1.0 * RBF(1.0)
        model = GaussianProcessClassifier(kernel=kernel, random_state=42)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    if hyperparameters:
        model.set_params(**hyperparameters)
    return model


def train_and_evaluate_model(X_train, X_test, y_train, y_test, label_encoder, model_type='xgboost',
                             hyperparameters=None, calibrate=False):
    """Trains and evaluates a model, with optional calibration."""

    start_time = time.time()
    model = create_model(model_type, hyperparameters)

    if calibrate and model_type in ['logistic_regression', 'naive_bayes', 'gradient_boosting', 'random_forest', 'svm']:
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)  # Or 'sigmoid'
        calibrated_model.fit(X_train, y_train)
        model = calibrated_model

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)

    # --- More Robust Evaluation ---
    print("\n--- Evaluation ---")
    print(f"Model: {model_type}")
    print(f"Training Time: {time.time() - start_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(report)

    # --- ROC AUC for Multiclass ---
    if len(label_encoder.classes_) > 2 and model_type not in ['naive_bayes', 'lda', 'qda', 'gaussian_process']:  # Models with issues with ROC AUC
        y_test_binarized = label_encoder.transform(y_test)
        try:
            roc_auc = roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr')
            print(f"ROC AUC (OVR): {roc_auc:.4f}")
        except ValueError as e:
            print(f"Error calculating ROC AUC: {e}")

    return model, accuracy, report


def tune_hyperparameters(X_train, y_train, model_type='xgboost', param_grid=None, cv=5, n_iter=10,
                        scoring='accuracy'):
    """Tunes hyperparameters for a given model."""

    start_time = time.time()
    model = create_model(model_type)
    if param_grid is None:
        param_grid = get_default_hyperparameter_grid(model_type)

    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    search = RandomizedSearchCV(model, param_grid, cv=cv, scoring=scoring, n_iter=n_iter, n_jobs=-1,
                              random_state=42)
    search.fit(X_train, y_train)

    print("\n--- Hyperparameter Tuning ---")
    print(f"Tuning Time: {time.time() - start_time:.2f} seconds")
    print("Best Hyperparameters:", search.best_params_)
    print("Best Score:", search.best_score_)
    return search.best_params_


def get_default_hyperparameter_grid(model_type='xgboost'):
    """Provides default hyperparameter grids for different models."""

    if model_type == 'xgboost':
        return {
            'n_estimators': [100, 500, 1000],
            'max_depth': [3, 6, 9, 12],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 10, 100]
        }
    elif model_type == 'random_forest':
        return {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
    elif model_type == 'gradient_boosting':
        return {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'subsample': [0.7, 0.8, 0.9],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif model_type == 'logistic_regression':
        return {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    elif model_type == 'naive_bayes':
        return {}  # No hyperparameters to tune
    elif model_type == 'svm':
        return {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 1]
        }
    elif model_type == 'neural_network':
        return {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'solver': ['adam', 'lbfgs'],
            'learning_rate': ['constant', 'adaptive']
        }
    elif model_type == 'decision_tree':
        return {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
    elif model_type == 'knn':
        return {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # Manhattan, Euclidean
        }
    elif model_type == 'lda':
        return {
            'solver': ['svd', 'lsqr', 'eigen']
        }
    elif model_type == 'qda':
        return {
            'reg_param': [0.0, 0.1, 0.2, 0.5]
        }
    elif model_type == 'gaussian_process':
        return {}
    else:
        raise ValueError(f"Invalid model type: {model_type}")


if __name__ == '__main__':
    # 1. Load and Preprocess Data
    X_train, X_test, y_train, y_test, label_encoder, scaler = load_and_preprocess_data(plot_data=True)

    # 2. Model Selection and Hyperparameter Tuning
    best_model_type = None
    best_accuracy = 0
    best_hyperparameters = {}
    all_results = {}

    for model_type in ['xgboost', 'random_forest', 'gradient_boosting', 'logistic_regression',
                       'naive_bayes', 'svm', 'neural_network', 'decision_tree', 'knn', 'lda', 'qda']:
        try:
            hyperparameters = tune_hyperparameters(X_train, y_train, model_type=model_type,
                                                n_iter=5)  # Reduced n_iter for speed
            model, accuracy, report = train_and_evaluate_model(X_train, X_test, y_train, y_test,
                                                            label_encoder, model_type=model_type,
                                                            hyperparameters=hyperparameters, calibrate=True)

            all_results[model_type] = {'accuracy': accuracy, 'report': report,
                                        'hyperparameters': hyperparameters}

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_type = model_type
                best_hyperparameters = hyperparameters

        except Exception as e:
            print(f"Error with {model_type}: {e}")

    print("\n--- Best Model ---")
    print(f"Best Model Type: {best_model_type}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Best Hyperparameters: {best_hyperparameters}")

    # 3. Final Model Training and Saving
    final_model = create_model(best_model_type, best_hyperparameters)
    final_model.fit(X_train, y_train)

    joblib.dump(final_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(label_encoder, ENCODER_FILE)
    joblib.dump(best_hyperparameters, HYPERPARAMETERS_FILE)

    print("\nFinal model and artifacts saved.")