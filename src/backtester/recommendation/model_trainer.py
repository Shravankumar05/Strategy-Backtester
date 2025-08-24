import pandas as pd
import numpy as np
import logging
import pickle
import os
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from sklearn.neural_network import MLPClassifier
    HAS_NEURAL_NETWORK = True
except ImportError:
    HAS_NEURAL_NETWORK = False

class ModelTrainer:

    def __init__(self, cache_dir: str = "recommendation_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_fitted = False

        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                },
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier,
                'params': {
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'random_state': 42
                },
                'param_grid': {
                    'max_depth': [5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10, 20]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'max_iter': 1000,
                    'random_state': 42
                },
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2']
                }
            }
        }

        if HAS_XGBOOST:
            self.model_configs['xgboost'] = {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': 42,
                    'eval_metric': 'mlogloss'
                },
                'param_grid': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            }

        if HAS_NEURAL_NETWORK:
            self.model_configs['neural_network'] = {
                'model': MLPClassifier,
                'params': {
                    'hidden_layer_sizes': (100, 50),
                    'max_iter': 500,
                    'random_state': 42,
                    'early_stopping': True
                },
                'param_grid': {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50), (100, 50, 25)],
                    'alpha': [0.0001, 0.001, 0.01]
                }
            }

    def train_models(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, use_grid_search: bool = False, cv_folds: int = 5) -> Dict[str, Any]:
        self.logger.info(f"Training models on {len(X)} samples with {len(X.columns)} features")
        y_encoded = self.label_encoder.fit_transform(y)
        self.feature_names = X.columns.tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded)

        training_results = {
            'model_performance': {},
            'feature_importance': {},
            'training_info': {
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
                'n_features': len(X.columns),
                'n_classes': len(self.label_encoder.classes_),
                'classes': self.label_encoder.classes_.tolist()
            }
        }

        for model_name, config in self.model_configs.items():
            try:
                self.logger.info(f"Training {model_name}...")

                if use_grid_search:

                    model = self._train_with_grid_search(
                        config, X_train, y_train, cv_folds
                    )
                else:

                    model = config['model'](**config['params'])
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds)

                self.models[model_name] = model
                training_results['model_performance'][model_name] = {
                    'test_accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_scores': cv_scores.tolist()
                }

                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    training_results['feature_importance'][model_name] = dict(
                        zip(self.feature_names, importance)
                    )
                elif hasattr(model, 'coef_'):

                    importance = np.abs(model.coef_).mean(axis=0)
                    training_results['feature_importance'][model_name] = dict(
                        zip(self.feature_names, importance)
                    )

                self.logger.info(f"{model_name} - Accuracy: {accuracy:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")

            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {e}")
                continue

        if len(self.models) > 1:
            ensemble_pred = self._create_ensemble_prediction(X_test)
            ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
            training_results['model_performance']['ensemble'] = {
                'test_accuracy': ensemble_accuracy
            }
            self.logger.info(f"Ensemble - Accuracy: {ensemble_accuracy:.3f}")

        self.is_fitted = True

        best_model_name = max(
            training_results['model_performance'].items(),
            key=lambda x: x[1]['test_accuracy']
        )[0]

        if best_model_name != 'ensemble':
            best_model = self.models[best_model_name]
            y_pred_best = best_model.predict(X_test)

            y_test_names = self.label_encoder.inverse_transform(y_test)
            y_pred_names = self.label_encoder.inverse_transform(y_pred_best)

            training_results['classification_report'] = classification_report(
                y_test_names, y_pred_names, output_dict=True
            )

        return training_results

    def _train_with_grid_search(self, config: Dict[str, Any], X_train: pd.DataFrame, y_train: np.ndarray, cv_folds: int):
        model = config['model']()
        grid_search = GridSearchCV(model, config['param_grid'], cv=cv_folds, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def _create_ensemble_prediction(self, X_test: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise ValueError("No trained models available for ensemble")

        predictions = []
        for model_name, model in self.models.items():
            pred = model.predict(X_test)
            predictions.append(pred)

        predictions = np.array(predictions)
        ensemble_pred = []

        for i in range(predictions.shape[1]):
            votes = predictions[:, i]
            unique, counts = np.unique(votes, return_counts=True)
            ensemble_pred.append(unique[np.argmax(counts)])

        return np.array(ensemble_pred)

    def predict(self, X: pd.DataFrame, use_ensemble: bool = True) -> Dict[str, Any]:
        if not self.is_fitted:
            raise ValueError("Models must be trained before making predictions")

        X = X[self.feature_names]

        results = {
            'individual_predictions': {},
            'prediction_probabilities': {}
        }

        for model_name, model in self.models.items():
            pred_encoded = model.predict(X)
            pred_strategy = self.label_encoder.inverse_transform(pred_encoded)
            results['individual_predictions'][model_name] = pred_strategy[0]

            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[0]
                prob_dict = dict(zip(
                    self.label_encoder.classes_,
                    probabilities
                ))
                results['prediction_probabilities'][model_name] = prob_dict

        if use_ensemble and len(self.models) > 1:
            ensemble_pred = self._create_ensemble_prediction(X)
            ensemble_strategy = self.label_encoder.inverse_transform(ensemble_pred)
            results['ensemble_prediction'] = ensemble_strategy[0]

            individual_preds = list(results['individual_predictions'].values())
            ensemble_confidence = individual_preds.count(results['ensemble_prediction']) / len(individual_preds)
            results['ensemble_confidence'] = ensemble_confidence

        return results

    def get_model_summary(self) -> Dict[str, Any]:

        if not self.is_fitted:
            return {"error": "No trained models available"}

        summary = {
            'available_models': list(self.models.keys()),
            'feature_count': len(self.feature_names),
            'strategy_classes': self.label_encoder.classes_.tolist(),
            'training_timestamp': datetime.now().isoformat()
        }

        return summary

    def save_models(self, filename: str = None) -> str:
        if not self.is_fitted:
            raise ValueError("No trained models to save")

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recommendation_models_{timestamp}.pkl"

        filepath = os.path.join(self.cache_dir, filename)

        model_data = {
            'models': self.models,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_configs': self.model_configs,
            'is_fitted': self.is_fitted,
            'save_timestamp': datetime.now()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Saved models to {filepath}")
        return filepath

    def load_models(self, filename: str) -> None:
        filepath = os.path.join(self.cache_dir, filename)
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.models = model_data['models']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']

        self.logger.info(f"Loaded models from {filepath}")

    def get_latest_models(self) -> bool:
        try:
            files = [f for f in os.listdir(self.cache_dir)
                    if f.startswith("recommendation_models_") and f.endswith(".pkl")]

            if not files:
                return False

            latest_file = max(files)
            self.load_models(latest_file)
            return True

        except Exception as e:
            self.logger.error(f"Failed to load latest models: {e}")
            return False