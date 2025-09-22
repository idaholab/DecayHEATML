import numpy as np
from typing import List, Tuple

class PolynomialFitter:
    """Handles polynomial fitting across multiple time regions with continuity constraints"""

    def __init__(self, time_steps: np.ndarray, log_decay_heat: np.ndarray):
        self.time_steps = time_steps
        self.log_decay_heat = log_decay_heat
        self.log_time = np.log(time_steps)
        self.equations = []
        self.coefficients = []
        self.regions = []
        self.r_squared = []
        self.continuity_weight = 1e6  # Weight for boundary constraints

    def fit_regions(self, num_regions: int, polynomial_order: int) -> List[str]:
        """Main fitting method with continuity enforcement"""
        self._validate_inputs(num_regions, polynomial_order)
        self._clear_previous_fit()

        if num_regions == 1:
            return self._fit_single_region(polynomial_order)

        return self._fit_multi_regions_with_continuity(num_regions, polynomial_order)

    def _validate_inputs(self, num_regions: int, order: int):
        """Ensure valid fitting parameters"""
        if num_regions < 1:
            raise ValueError("Number of regions must be ≥ 1")
        if order < 1:
            raise ValueError("Polynomial order must be ≥ 1")
        if len(self.time_steps) < 2:
            raise ValueError("Insufficient data points for fitting")

    def _fit_single_region(self, order: int) -> List[str]:
        """Fit polynomial to entire time range"""
        mask = (self.log_time >= self.log_time.min()) & \
               (self.log_time <= self.log_time.max())
        x_data = self.log_time[mask]
        y_data = self.log_decay_heat[mask]

        coeffs, r2 = self._fit_region(x_data, y_data, order)
        equation = self._format_equation(
            coeffs,
            np.exp(self.log_time.min()),
            np.exp(self.log_time.max()),
            order,
            r2
        )
        self.coefficients.append(coeffs)
        self.r_squared.append(r2)
        self.regions.append((self.log_time.min(), self.log_time.max()))
        return [equation]

    def _fit_multi_regions_with_continuity(self, num_regions: int, order: int) -> List[str]:
        """Fit regions with C⁰ continuity constraints"""
        split_points = np.linspace(
            self.log_time.min(),
            self.log_time.max(),
            num_regions + 1
        )[1:-1]
        regions = self._create_regions(split_points)
        equations = []
        prev_coeffs = None

        for i, (start_log, end_log) in enumerate(regions):
            region_mask = self._get_region_mask(start_log, end_log)
            x_data = self.log_time[region_mask]
            y_data = self.log_decay_heat[region_mask]
            constraints = 0

            # Add continuity constraint from previous region
            if i > 0 and prev_coeffs is not None:
                x_data, y_data, constraints = self._add_continuity_constraint(
                    x_data, y_data, start_log, prev_coeffs
                )

            coeffs, r2 = self._fit_region(x_data, y_data, order, constraints)
            equation = self._format_equation(
                coeffs,
                np.exp(start_log),
                np.exp(end_log),
                order,
                r2
            )

            self.coefficients.append(coeffs)
            self.r_squared.append(r2)
            self.regions.append((start_log, end_log))
            equations.append(equation)
            prev_coeffs = coeffs

        return equations

    def _add_continuity_constraint(self, x, y, boundary, prev_coeffs):
        """Augment data with boundary constraint from previous region"""
        y_boundary = np.polyval(prev_coeffs, boundary)
        return (
            np.append(x, boundary),
            np.append(y, y_boundary),
            1  # Number of constraints added
        )

    def _fit_region(self, x, y, order, constraints=0):
        """Weighted least squares fit with boundary emphasis"""
        weights = np.ones(len(x))
        if constraints > 0:
            # Apply high weight to constraint points at end
            weights[-constraints:] = self.continuity_weight

        coeffs = np.polyfit(x, y, order, w=weights)

        # Calculate R² on original data (excluding constraint points)
        valid_points = slice(None, len(y)-constraints) if constraints > 0 else slice(None)
        y_pred = np.polyval(coeffs, x[valid_points])
        y_true = y[valid_points]

        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - (ss_res/ss_tot) if ss_tot != 0 else 0

        return coeffs, r2

    def _create_regions(self, split_points: np.ndarray) -> List[Tuple[float, float]]:
        """Create time regions in log space"""
        return [(prev, curr) for prev, curr in zip(
            [self.log_time.min()] + list(split_points),
            list(split_points) + [self.log_time.max()]
        )]

    def _get_region_mask(self, start: float, end: float) -> np.ndarray:
        """Create boolean mask for region selection"""
        return (self.log_time >= start) & (self.log_time <= end)

    def _format_equation(self, coeffs: np.ndarray,
                        t_start: float, t_end: float,
                        order: int, r2: float) -> str:
        """Format equation with scientific notation"""
        terms = []
        for power, coeff in enumerate(coeffs[::-1]):
            exponent = power
            if np.isclose(coeff, 0):
                continue
            terms.append(f"{coeff:.3e}·(log(t))^{exponent}")

        return (
            f"Time Range [{t_start:.2e}, {t_end:.2e}]\n"
            f"log(Q) = {' + '.join(terms) if terms else '0'}\n"
            f"R² = {r2:.2f}"
        )

    def _clear_previous_fit(self):
        """Reset fitting state"""
        self.equations = []
        self.coefficients = []
        self.regions = []
        self.r_squared = []
