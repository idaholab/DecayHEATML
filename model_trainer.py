from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor,
                             GradientBoostingRegressor)
from sklearn.model_selection import GridSearchCV
from sklearn.base import TransformerMixin

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from gplearn.genetic import SymbolicRegressor
import pandas as pd
import numpy as np

from sklearn.model_selection import learning_curve

class ModelTrainer:
    """Handles model training and evaluation"""

    def __init__(self, data: pd.DataFrame):
        self.feature_order = ['time', 'power_density', 'humidity_content', 'air_ingress']
        self.data = data[self.feature_order + ['decay_heat']].copy()
        self.models = {}
        self.results = {}
        self._initialize_models()

    def _initialize_models(self):
        """Create all model configurations"""
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

        self.models = {
            'Polynomial (deg 2)': make_pipeline(PolynomialFeatures(2), LinearRegression()),
            'Polynomial (deg 3)': make_pipeline(PolynomialFeatures(3), LinearRegression()),
            'Polynomial (deg 4)': make_pipeline(PolynomialFeatures(4), LinearRegression()),
            # 'Polynomial (deg 5)': make_pipeline(PolynomialFeatures(5), LinearRegression()),
            # 'Polynomial (deg 6)': make_pipeline(PolynomialFeatures(6), LinearRegression()),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'SVR': make_pipeline(StandardScaler(), SVR()),
            'K-Neighbors': make_pipeline(StandardScaler(), KNeighborsRegressor()),
            'MLP': make_pipeline(
                StandardScaler(),
                MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42)
            ),
            'Symbolic': SymbolicRegressor(
                population_size=1000,
                generations=10,
                stopping_criteria=0.01,
                random_state=42
            ),
            # 'Gaussian Process': GaussianProcessRegressor(
            #     kernel=kernel,
            #     n_restarts_optimizer=10
            # )
        }

    def train_and_evaluate(self, test_size: float = 0.2) -> dict:
        """Run full training and evaluation pipeline"""
        X = self.data[self.feature_order]
        y = self.data['decay_heat']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        for name, model in self.models.items():
            self._train_single_model(name, model, X, y, X_train, y_train, X_test, y_test)

        return self.results

    def _train_single_model(self, name, model, X, y, X_train, y_train, X_test, y_test):
        """Enhanced training with visualization data collection"""
        try:
            # Cross-validation
            y_pred_cv = cross_val_predict(model, X, y, cv=5)

            # Learning curve calculation
            train_sizes, train_scores, val_scores = learning_curve(
                estimator=model,
                X=X,
                y=y,
                cv=5,
                train_sizes=np.linspace(0.1, 1.0, 5),
                scoring='neg_mean_squared_error'
            )

            # Final training
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            residuals = y_test - y_pred

            # Store comprehensive results
            self.results[name] = {
                'cv_mse': mean_squared_error(y, y_pred_cv),
                'cv_r2': r2_score(y, y_pred_cv),
                'test_mse': mean_squared_error(y_test, y_pred),
                'test_r2': r2_score(y_test, y_pred),
                'y_test': y_test,
                'y_pred': y_pred,
                'residuals': residuals,
                'learning_curve': {
                    'train_sizes': train_sizes,
                    'train_scores': -train_scores.mean(axis=1),
                    'test_scores': -val_scores.mean(axis=1)
                }
            }

        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            self.results[name] = None


class UniversalRegionalEngineer(TransformerMixin, BaseEstimator):
    """Regional features compatible with all model types with proper 2D output handling"""
    def __init__(self, n_regions=1):
        self.n_regions = n_regions
        self.boundaries_ = None

    def fit(self, X, y=None):
        if self.n_regions > 1:
            time_values = X[:, 0]  # First column is time
            self.boundaries_ = np.quantile(
                time_values,
                np.linspace(0, 1, self.n_regions + 1)
            )[1:-1]
        return self

    def transform(self, X):
        from sklearn.utils.validation import check_array

        X = check_array(X, ensure_2d=True)  # Ensure proper 2D array

        if self.n_regions == 1:
            return X  # Return original features

        time_col = X[:, 0]
        regions = np.digitize(time_col, self.boundaries_)

        # Add interaction terms between regions and features
        return np.hstack([
            X,
            regions.reshape(-1, 1),
            X * regions.reshape(-1, 1)
        ])

class EnhancedModelTrainer(ModelTrainer):
    def __init__(self, data: pd.DataFrame, max_regions=10):
        super().__init__(data)
        self.max_regions = max_regions
        self.best_configs = {}

    def train_and_evaluate(self, test_size: float = 0.2) -> dict:
        X = self.data[self.feature_order].values
        y = self.data['decay_heat'].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        for name, base_model in self.models.items():
            try:
                # Create pipeline with regional engineer
                pipe = make_pipeline(
                    UniversalRegionalEngineer(),
                    base_model
                )

                # Grid search for regions
                param_grid = {
                    'universalregionalengineer__n_regions': list(range(1, self.max_regions+1))
                }

                gs = GridSearchCV(
                    pipe,
                    param_grid,
                    cv=None,
                    scoring='neg_mean_squared_error',
                    error_score='raise',
                    n_jobs=-1
                )
                gs.fit(X_train, y_train)

                # Update the original model reference with fitted pipeline
                self.models[name] = gs.best_estimator_

                # Store results
                y_pred = self.models[name].predict(X_test)
                self.results[name] = {
                    'test_mse': mean_squared_error(y_test, y_pred),
                    'test_r2': r2_score(y_test, y_pred),
                    'optimal_regions': gs.best_params_['universalregionalengineer__n_regions'],
                    'model': self.models[name],
                    'y_test': y_test,
                    'y_pred': y_pred
                }

            except Exception as e:
                print(f"Training failed for {name}: {str(e)}")
                try:
                    base_model.fit(X_train, y_train)
                    self.models[name] = base_model  # Update with fitted base model
                    y_pred = base_model.predict(X_test)
                    self.results[name] = {
                        'test_mse': mean_squared_error(y_test, y_pred),
                        'test_r2': r2_score(y_test, y_pred),
                        'optimal_regions': 1,
                        'model': base_model,
                        'y_test': y_test,
                        'y_pred': y_pred
                    }
                except Exception as fallback_error:
                    print(f"Fallback failed for {name}: {str(fallback_error)}")
                    self.results[name] = None

        return self.results

    def get_optimal_regions(self, model_name: str) -> int:
        return self.best_configs.get(model_name, {}).get('regions', 1)

    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in trained results")

        model = self.results[model_name].get('model')
        if model is None:
            raise ValueError(f"Model {model_name} failed training")

        return model.predict(X)
