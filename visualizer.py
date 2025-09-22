import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ipywidgets import interact, widgets
from typing import Dict, List, Tuple
from sklearn.base import BaseEstimator
from sklearn.model_selection import learning_curve
from polynomial_fitter import PolynomialFitter

class Visualizer:
    """Handles interactive visualization of model predictions"""

    """Handles all visualization components including model evaluation plots"""

    def __init__(self, trainer):
        self.trainer = trainer
        self.models = trainer.models
        self.results = trainer.results
        self.fitter = None

    def create_model_evaluation_report(self, save_path: str = None):
        """Generate comprehensive evaluation plots for all models"""
        num_models = len(self.models)
        fig, axs = plt.subplots(num_models, 2, figsize=(25, 6*num_models))
        fig.subplots_adjust(hspace=0.4, wspace=0.3)

        for idx, (model_name, model) in enumerate(self.models.items()):
            result = self.results.get(model_name)
            if not result or 'y_test' not in result:
                continue

            # Get transformed values
            y_test = np.exp(result['y_test'])
            y_pred = np.exp(result['y_pred'])
            residuals = (y_test - y_pred) / y_test.mean()

            # Real vs Predicted plot
            self._plot_real_vs_predicted(axs[idx, 0], y_test, y_pred, model_name)

            # Residual plot
            self._plot_residuals(axs[idx, 1], y_pred, residuals, model_name)

            # Learning curve plot
            # self._plot_learning_curve(axs[idx, 2], model_name)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def _plot_real_vs_predicted(self, ax, y_true, y_pred, model_name):
        """Log-log plot of actual vs predicted values"""
        ax.loglog(y_true, y_pred, 'o', alpha=0.5, markersize=4)
        lims = [np.min([y_true.min(), y_pred.min()]),
               np.max([y_true.max(), y_pred.max()])]
        ax.loglog(lims, lims, '--', color='r', linewidth=1)
        ax.set(xlabel='True Decay Heat (W/cm³)',
              ylabel='Predicted Decay Heat (W/cm³)',
              title=f'{model_name}\nActual vs Predicted')
        ax.grid(True, which='both', linestyle='--')

    def _plot_residuals(self, ax, y_pred, residuals, model_name):
        """Standardized residual plot"""
        ax.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax.axhline(0, color='r', linestyle='--')
        ax.set(xlabel='Predicted Values',
              ylabel='Standardized Residuals',
              title=f'{model_name}\nResidual Analysis')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.5, 1.5)

    def _plot_learning_curve(self, ax, model_name):
        """Learning curve with training/validation MSE"""
        result = self.results.get(model_name)
        if not result or 'learning_curve' not in result:
            return

        lc_data = result['learning_curve']
        ax.plot(lc_data['train_sizes'], lc_data['train_scores'], 'o-',
               label='Training Error')
        ax.plot(lc_data['train_sizes'], lc_data['test_scores'], 'o-',
               label='Validation Error')

        ax.set(xlabel='Training Set Size',
              ylabel='Mean Squared Error',
              title=f'{model_name}\nLearning Curve')
        ax.legend()
        ax.grid(True)
        ax.set_yscale('log')

    def create_interactive_plot(self):
        """Interactive prediction plot with polynomial fitting"""
        @interact(
            model_name=widgets.Dropdown(
                options=list(self.models.keys()),
                value=list(self.models.keys())[0],  # Add default value
                description='Model:'
            ),
            power_density=widgets.FloatSlider(
                value=10.0,
                min=1.0,
                max=100.0,
                step=0.1,  # Correct parameter order
                description='Power Density:'
            ),
            air_ingress=widgets.FloatSlider(
                value=1e-9,
                min=0.0,
                max=0.1,
                step=1e-4,  # Explicit step size
                description='Air Ingress:'
            ),
            humidity_content=widgets.FloatSlider(
                value=1e-9,
                min=0.0,
                max=0.1,
                step=1e-4,
                description='Humidity Content:'
            ),
            polynomial_order=widgets.IntSlider(
                value=2,
                min=1,
                max=10,
                step=1,
                description='Poly Order:'
            ),
            num_regions=widgets.IntSlider(
                value=1,
                min=1,
                max=10,
                step=1,
                description='Num Regions:'
            )
        )
        def update_plot(model_name, power_density, air_ingress,
                    humidity_content, polynomial_order, num_regions):
            self._plot_predictions(
                model_name, power_density, air_ingress,
                humidity_content, polynomial_order, num_regions
            )

    def _plot_predictions(self, model_name, power_density, air_ingress,
                         humidity_content, poly_order, num_regions):
        model = self.models[model_name]
        time_steps, decay_heat_pred = self._generate_prediction(
            model, power_density, air_ingress, humidity_content
        )

        # Setup plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(time_steps, decay_heat_pred, label='Model Prediction')

        try:
            # Polynomial fitting
            fitter = PolynomialFitter(time_steps, np.log(decay_heat_pred))
            equations = fitter.fit_regions(num_regions, poly_order)

            # Plot fits
            self._plot_polynomial_fits(ax, fitter, time_steps)

            # Add equations
            self._display_equations(fig, equations)

        except ValueError as e:
            print(f"⚠️ Fitting Error: {str(e)}")
            self._display_error_message(ax)

        # Finalize plot
        ax.set_xlabel('Time [days]')
        ax.set_ylabel('Decay Heat [W/cm³]')
        ax.set_title(f'Decay Heat Prediction - {model_name}')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    def _generate_prediction(self, model, power_density, air_ingress, humidity_content):
        """Generate time steps and model predictions"""
        time_steps = self._create_time_steps()
        X_new = pd.DataFrame({
            'time': np.log(time_steps),
            'power_density': power_density,
            'humidity_content': humidity_content,
            'air_ingress': air_ingress
        })[self.trainer.feature_order]  # Maintain training feature order

        log_pred = model.predict(X_new)
        return time_steps, np.exp(log_pred)

    def _create_time_steps(self):
        """Generate logarithmic time steps for prediction"""
        steps = 10000
        end_time = 10e4 * 365
        return np.logspace(1e-4, np.log10(end_time), steps) - 1 + 1e-10

    def _plot_polynomial_fits(self, ax, fitter, time_steps):
        """Plot polynomial fits with R² in legend"""
        log_time = np.log(time_steps)

        for coeffs, (start_log, end_log), r2 in zip(
            fitter.coefficients,
            fitter.regions,
            fitter.r_squared
        ):
            # Generate fit points
            log_time_fit = np.linspace(start_log, end_log, 500)
            time_fit = np.exp(log_time_fit)
            log_decay_fit = np.polyval(coeffs, log_time_fit)
            decay_fit = np.exp(log_decay_fit)

            # Format time range
            t_start = np.exp(start_log)
            t_end = np.exp(end_log)

            ax.loglog(
                time_fit,
                decay_fit,
                '--',
                linewidth=1.5,
                label=f"Fit [{t_start:.1e}, {t_end:.1e}]\nR² = {r2:.2f}"
            )

    def _display_equations(self, fig, equations):
        """Show equations below plot"""
        equation_text = "\n\n".join(equations)
        fig.text(0.5, -0.15, equation_text,
                ha='center', va='top',
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8))
        plt.subplots_adjust(bottom=0.3)

    def _display_error_message(self, ax):
        """Show error message on plot"""
        ax.text(0.5, 0.5, 'Fit Failed - Adjust Parameters',
               ha='center', va='center',
               transform=ax.transAxes,
               color='red', fontsize=12)
