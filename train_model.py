import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import random
import time
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

# Configuration
MALWARE_CATEGORIES = ["ransomware", "trojan", "worm", "virus", "spyware", "adware", "benign"]
NUM_SAMPLES = 500000
ZIP_RATIO = 0.2  # 20% of samples will be ZIP archives
MIN_CONFIDENCE_THRESHOLD = 0.85
OUTPUT_FILE = "training_report.txt"

# Malware type parameters
type_params = {
    "ransomware": {
        "entropy": (7.5, 8.5), "imports": (100, 300), "packed": 0.98,
        "size_range": (1000000, 8000000), "sections": (10, 25),
        "dll_count": (50, 150), "write_ops": (2000, 10000),
        "api_calls": (200, 500), "network_ops": (100, 300),
        "crypto_ops": (50, 200), "debug_info": 0.1
    },
    "trojan": {
        "entropy": (6.5, 7.8), "imports": (80, 250), "packed": 0.9,
        "size_range": (500000, 5000000), "sections": (8, 20),
        "dll_count": (40, 120), "write_ops": (1000, 6000),
        "api_calls": (150, 400), "network_ops": (80, 250),
        "crypto_ops": (20, 100), "debug_info": 0.3
    },
    "worm": {
        "entropy": (6.0, 7.5), "imports": (60, 200), "packed": 0.85,
        "size_range": (300000, 4000000), "sections": (7, 18),
        "dll_count": (30, 100), "write_ops": (800, 5000),
        "api_calls": (120, 350), "network_ops": (60, 200),
        "crypto_ops": (10, 50), "debug_info": 0.2
    },
    "virus": {
        "entropy": (7.0, 8.2), "imports": (90, 280), "packed": 0.95,
        "size_range": (800000, 6000000), "sections": (9, 22),
        "dll_count": (45, 130), "write_ops": (1500, 7000),
        "api_calls": (180, 450), "network_ops": (90, 280),
        "crypto_ops": (30, 120), "debug_info": 0.15
    },
    "spyware": {
        "entropy": (6.0, 7.2), "imports": (50, 180), "packed": 0.88,
        "size_range": (400000, 3000000), "sections": (6, 16),
        "dll_count": (25, 90), "write_ops": (600, 4000),
        "api_calls": (100, 300), "network_ops": (50, 180),
        "crypto_ops": (5, 30), "debug_info": 0.4
    },
    "adware": {
        "entropy": (5.5, 6.8), "imports": (40, 150), "packed": 0.8,
        "size_range": (200000, 2000000), "sections": (5, 12),
        "dll_count": (20, 70), "write_ops": (400, 3000),
        "api_calls": (80, 250), "network_ops": (30, 120),
        "crypto_ops": (2, 15), "debug_info": 0.5
    },
    "benign": {
        "entropy": (3.5, 5.5), "imports": (5, 60), "packed": 0.05,
        "size_range": (10000, 1000000), "sections": (1, 6),
        "dll_count": (1, 30), "write_ops": (0, 800),
        "api_calls": (0, 80), "network_ops": (0, 30),
        "crypto_ops": (0, 5), "debug_info": 0.9
    }
}

class EnhancedMalwareLoader:
    def generate_samples(self, count):
        """Generate samples including both PE files and ZIP archives"""
        samples = []
        pe_count = int(count * (1 - ZIP_RATIO))
        zip_count = count - pe_count
        
        # Generate regular PE samples
        samples.extend(self._generate_pe_samples(pe_count))
        
        # Generate ZIP archive samples containing malware
        samples.extend(self._generate_zip_samples(zip_count))
        
        return samples

    def _generate_pe_samples(self, count):
        """Generate PE file samples"""
        samples = []
        for _ in range(count):
            malware_type = random.choice(MALWARE_CATEGORIES)
            features = self._generate_pe_features(malware_type)
            samples.append({
                "tag": malware_type,
                "features": features,
                "is_archive": 0
            })
        return samples

    def _generate_zip_samples(self, count):
        """Generate ZIP archive samples containing malware"""
        samples = []
        for _ in range(count):
            malware_type = random.choice(MALWARE_CATEGORIES[:-1])  # Exclude benign
            pe_features = self._generate_pe_features(malware_type)
            
            samples.append({
                "tag": malware_type,
                "features": {
                    **self._get_archive_features(),
                    **pe_features,
                    "is_archive": 1
                }
            })
        return samples

    def _generate_pe_features(self, malware_type):
        """Generate features for a PE file"""
        params = type_params[malware_type]
        features = {
            "size": random.randint(*params["size_range"]),
            "entropy": random.uniform(*params["entropy"]),
            "imports": random.randint(*params["imports"]),
            "sections": random.randint(*params["sections"]),
            "dll_count": random.randint(*params["dll_count"]),
            "write_ops": random.randint(*params["write_ops"]),
            "is_packed": int(random.random() < params["packed"]),
            "api_calls": random.randint(*params["api_calls"]),
            "network_ops": random.randint(*params["network_ops"]),
            "has_crypto": int(random.random() < 0.5 and malware_type != "benign"),
            "has_debug": int(random.random() < params["debug_info"]),
            "is_archive": 0
        }
        
        # Calculate derived features
        features.update(self._calculate_derived_features(features))
        return features

    def _get_archive_features(self):
        """Features specific to ZIP archives"""
        return {
            "archive_entropy": random.uniform(7.0, 8.5),
            "contained_files": random.randint(1, 10),
            "contains_pe": 1,
            "contains_scripts": random.choice([0, 1])
        }

    def _calculate_derived_features(self, features):
        """Calculate ratio-based features"""
        return {
            "entropy_imports_ratio": features["entropy"] / max(1, features["imports"]),
            "size_sections_ratio": features["size"] / max(1, features["sections"]),
            "api_dll_ratio": features["api_calls"] / max(1, features["dll_count"]),
            "write_network_ratio": features["write_ops"] / max(1, features["network_ops"]),
            "crypto_ratio": features["has_crypto"] * features["entropy"]
        }

def optimize_hyperparameters(X_train, y_train):
    """Perform grid search to find optimal hyperparameters"""
    param_grid = {
        'xgbclassifier__n_estimators': [200, 300],
        'xgbclassifier__max_depth': [4, 6],
        'xgbclassifier__learning_rate': [0.05, 0.1],
        'xgbclassifier__subsample': [0.8, 0.9],
        'xgbclassifier__colsample_bytree': [0.8, 0.9],
        'xgbclassifier__reg_alpha': [0, 0.1],
        'xgbclassifier__reg_lambda': [0.1, 1]
    }
    
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgbclassifier', xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(MALWARE_CATEGORIES),
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=3,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def train_final_model(X_train, y_train, best_params):
    """Train final model with optimal parameters"""
    # Remove parameters that are already set explicitly
    params_to_remove = ['objective', 'num_class', 'random_state', 'n_jobs']
    for param in params_to_remove:
        best_params.pop(f'xgbclassifier__{param}', None)
    
    # Create new params dict without the pipeline prefix
    model_params = {k.replace('xgbclassifier__', ''): v for k, v in best_params.items()}
    
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgbclassifier', xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(MALWARE_CATEGORIES),
            random_state=42,
            n_jobs=-1,
            **model_params
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Prepare data for calibration
    X_res, y_res = pipeline.named_steps['smote'].fit_resample(X_train, y_train)
    xgb_model = pipeline.named_steps['xgbclassifier']
    
    # Calibration
    calibrated_model = CalibratedClassifierCV(
        xgb_model,
        method='isotonic',
        cv=3
    )
    calibrated_model.fit(X_res, y_res)
    
    return calibrated_model, xgb_model

def evaluate_model(model, X_test, y_test, le, feature_names):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Overall metrics
    print(f"\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # ZIP-specific metrics
    zip_indices = [i for i, x in enumerate(X_test) if x[feature_names.index("is_archive")] == 1]
    if zip_indices:
        zip_acc = accuracy_score(y_test[zip_indices], y_pred[zip_indices])
        print(f"\nZIP Archive Accuracy: {zip_acc:.4f} (n={len(zip_indices)})")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        sorted_idx = model.feature_importances_.argsort()
        plt.barh(np.array(feature_names)[sorted_idx], model.feature_importances_[sorted_idx])
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.close()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(le.classes_))
    plt.xticks(tick_marks, le.classes_, rotation=45)
    plt.yticks(tick_marks, le.classes_)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

def save_training_report(model, le, feature_names, best_params, start_time):
    """Generate detailed training report"""
    with open(OUTPUT_FILE, 'w') as f:
        f.write("=== Enhanced Malware Detection Training Report ===\n\n")
        f.write(f"Training completed at: {time.ctime()}\n")
        f.write(f"Total training time: {(time.time()-start_time)/60:.1f} minutes\n\n")
        
        f.write("=== Model Architecture ===\n")
        f.write("XGBoost Classifier with SMOTE balancing\n")
        f.write(f"Number of classes: {len(le.classes_)}\n")
        f.write(f"Number of features: {len(feature_names)}\n\n")
        
        f.write("=== Best Parameters ===\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
        
        f.write("\n=== Feature List ===\n")
        for feat in feature_names:
            f.write(f"- {feat}\n")

def main():
    print(f"[{time.ctime()}] Starting enhanced malware detection training")
    start_time = time.time()
    
    try:
        # Load and prepare data
        print("\n[+] Generating synthetic dataset...")
        loader = EnhancedMalwareLoader()
        samples = loader.generate_samples(NUM_SAMPLES)
        
        # Verify distribution
        class_dist = Counter([s["tag"] for s in samples])
        print("\nClass distribution:")
        for cls, count in class_dist.items():
            print(f"{cls}: {count} samples ({count/NUM_SAMPLES:.1%})")
        print(f"ZIP archives: {sum(s['features']['is_archive'] for s in samples)} samples")
        
        # Prepare features and labels
        feature_names = list(samples[0]["features"].keys())
        X = np.array([[s["features"][f] for f in feature_names] for s in samples])
        y = [s["tag"] for s in samples]
        
        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split data
        print("\n[+] Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )
        
        # Scale features
        print("[+] Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Optimize hyperparameters
        print("\n[+] Optimizing hyperparameters...")
        best_pipeline, best_params = optimize_hyperparameters(X_train_scaled, y_train)
        
        # Train final model
        print("\n[+] Training final model...")
        model, base_model = train_final_model(X_train_scaled, y_train, best_params)
        
        # Evaluate
        print("\n[+] Evaluating model...")
        evaluate_model(base_model, X_test_scaled, y_test, le, feature_names)
        
        # Save artifacts
        print("\n[+] Saving model artifacts...")
        joblib.dump({
            "model": model,
            "base_model": base_model,
            "scaler": scaler,
            "encoder": le,
            "features": feature_names,
            "min_confidence": MIN_CONFIDENCE_THRESHOLD,
            "best_params": best_params
        }, "enhanced_malware_model.pkl")
        
        # Generate report
        save_training_report(base_model, le, feature_names, best_params, start_time)
        
        print(f"\n[{time.ctime()}] Training completed successfully in {(time.time()-start_time)/60:.1f} minutes!")
        print("\nModel artifacts saved:")
        print("- enhanced_malware_model.pkl")
        print("- feature_importance.png")
        print("- confusion_matrix.png")
        print(f"- {OUTPUT_FILE} (training report)")
        
    except Exception as e:
        print(f"\n[!] Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()