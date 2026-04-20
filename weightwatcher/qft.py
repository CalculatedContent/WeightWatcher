"""
Quantum Field Theory Module for WeightWatcher

This module implements theoretical foundations from quantum field theory and 
renormalization group theory to analyze neural network weight matrices.
"""

import numpy as np
import scipy.stats
from scipy import linalg
import matplotlib.pyplot as plt
from .RMT_Util import get_esd

class RGAnalyzer:
    """
    Renormalization Group Analyzer for neural network weight matrices.
    
    This class implements methods to analyze weight matrices through the lens of
    Wilson's exact renormalization group theory, focusing on critical points,
    scale invariance, and emergent properties.
    
    The analyzer provides tools to:
    1. Detect critical points in weight matrices
    2. Measure scale invariance properties
    3. Calculate fractal dimensions
    4. Map free energy landscapes
    5. Identify phase transitions
    """
    
    def __init__(self, temperature=1.0):
        """
        Initialize the RG Analyzer.
        
        Args:
            temperature: Temperature parameter for free energy calculations (default: 1.0)
        """
        self.results = {}
        self.temperature = temperature
        self.history = []  # For tracking evolution across training
        
    def analyze_critical_point(self, W):
        """
        Analyze how close a weight matrix is to a critical point.
        
        Args:
            W: Weight matrix as numpy array
            
        Returns:
            Dictionary with critical point metrics
        """
        # Basic implementation - will expand in subsequent steps
        evals = get_esd(W)
        
        # Calculate basic metrics
        power_law_exponent = self._estimate_power_law(evals)
        scale_invariance = self._measure_scale_invariance(W)
        
        results = {
            'power_law_exponent': power_law_exponent,
            'scale_invariance': scale_invariance,
            'is_near_critical': power_law_exponent < 2.5 and scale_invariance > 0.8
        }
        
        self.results = results
        return results
    
    def _estimate_power_law(self, eigenvalues):
        """Estimate power law exponent from eigenvalue distribution."""
        # Simple implementation - will be enhanced
        eigenvalues = np.abs(eigenvalues)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove zeros
        
        if len(eigenvalues) < 10:
            return np.nan
            
        log_values = np.log(eigenvalues)
        # Fit power law using basic approach
        n = len(log_values)
        indices = np.log(np.arange(1, n+1))
        slope, _, _, _ = np.linalg.lstsq(
            indices.reshape(-1, 1), 
            np.sort(log_values), 
            rcond=None
        )[0]
        
        return -slope  # Return power law exponent
    
    def _measure_scale_invariance(self, W):
        """
        Measure scale invariance properties of weight matrix.
        
        This method quantifies how the statistical properties of the weight matrix
        remain invariant under different scales, a key property near critical points.
        
        Args:
            W: Weight matrix as numpy array
            
        Returns:
            Float between 0 and 1 indicating degree of scale invariance
        """
        # Get singular values
        s = np.linalg.svd(W, compute_uv=False)
        
        # Calculate metrics at different scales
        scales = [0.25, 0.5, 0.75, 1.0]
        distributions = []
        
        for scale in scales:
            # Take a subset of singular values based on scale
            n_values = max(int(len(s) * scale), 5)
            subset = s[:n_values]
            
            # Normalize and store distribution
            if len(subset) > 0:
                normalized = subset / np.max(subset)
                distributions.append(normalized)
        
        # Calculate similarity between distributions at different scales
        similarities = []
        for i in range(len(distributions)-1):
            # Use KL divergence to compare distributions
            # First, we need to bin the distributions
            min_len = min(len(distributions[i]), len(distributions[i+1]))
            hist1, _ = np.histogram(distributions[i][:min_len], bins=10, range=(0,1), density=True)
            hist2, _ = np.histogram(distributions[i+1][:min_len], bins=10, range=(0,1), density=True)
            
            # Avoid division by zero
            hist1 = hist1 + 1e-10
            hist2 = hist2 + 1e-10
            
            # Normalize
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)
            
            # Calculate KL divergence
            kl_div = scipy.stats.entropy(hist1, hist2)
            
            # Convert to similarity (0 to 1)
            similarity = np.exp(-kl_div)
            similarities.append(similarity)
        
        # Return average similarity as scale invariance measure
        if similarities:
            return np.mean(similarities)
        else:
            return 0.0
            
    def compute_fractal_dimension(self, W, max_scales=20):
        """
        Compute the fractal dimension of the weight matrix.
        
        This method estimates the fractal dimension using a box-counting approach
        on the singular value distribution, which helps quantify the self-similarity
        properties of the weight matrix across scales.
        
        Args:
            W: Weight matrix as numpy array
            max_scales: Maximum number of scales to use
            
        Returns:
            Dictionary with fractal dimension metrics
        """
        # Get singular values
        s = np.linalg.svd(W, compute_uv=False)
        
        # Prepare for box counting
        scales = np.logspace(-1, 0, max_scales)
        counts = []
        
        # Normalize singular values to [0,1]
        if len(s) > 0:
            s_norm = s / np.max(s)
            
            # Perform box counting at different scales
            for scale in scales:
                # Create boxes of size 'scale'
                box_size = scale
                box_count = 0
                
                # Count boxes
                boxes = np.arange(0, 1 + box_size, box_size)
                hist, _ = np.histogram(s_norm, bins=boxes)
                box_count = np.sum(hist > 0)
                
                counts.append(box_count)
            
            # Calculate fractal dimension as the slope of log(count) vs log(1/scale)
            if len(counts) > 2 and np.min(counts) > 0:
                log_scales = -np.log(scales)
                log_counts = np.log(counts)
                
                # Linear regression to find slope
                slope, _, _, _, _ = scipy.stats.linregress(log_scales, log_counts)
                
                return {
                    'fractal_dimension': slope,
                    'r_squared': self._r_squared(log_scales, log_counts, slope),
                    'scales': scales,
                    'counts': counts
                }
        
        # Default return if calculation fails
        return {
            'fractal_dimension': np.nan,
            'r_squared': 0,
            'scales': scales,
            'counts': counts if 'counts' in locals() else []
        }
    
    def _r_squared(self, x, y, slope):
        """Calculate R-squared for the linear fit."""
        if len(x) != len(y) or len(x) < 2:
            return 0
            
        # Calculate intercept
        intercept = np.mean(y) - slope * np.mean(x)
        
        # Calculate predictions
        y_pred = slope * x + intercept
        
        # Calculate R-squared
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - y_pred)**2)
        
        if ss_total == 0:
            return 0
            
        return 1 - (ss_residual / ss_total)
    def map_free_energy_landscape(self, W):
        """
        Map the free energy landscape of the weight matrix.
        
        This method calculates the free energy based on the eigenvalue spectrum
        using concepts from statistical physics and renormalization group theory.
        
        Args:
            W: Weight matrix as numpy array
            
        Returns:
            Dictionary with free energy metrics
        """
        # Get eigenvalues
        if W.shape[0] > W.shape[1]:
            # Non-square matrix: use singular values
            s = np.linalg.svd(W, compute_uv=False)
            evals = np.concatenate([s**2, np.zeros(W.shape[0] - len(s))])
        else:
            # Use eigenvalues of W*W.T for stability
            evals = np.linalg.eigvalsh(W @ W.T)
        
        # Remove any negative eigenvalues (numerical errors)
        evals = evals[evals > 1e-10]
        
        if len(evals) == 0:
            return {
                'free_energy': np.nan,
                'entropy': np.nan,
                'energy': np.nan,
                'is_critical': False
            }
        
        # Calculate partition function Z
        Z = np.sum(np.exp(-evals / self.temperature))
        
        # Calculate free energy F = -T log(Z)
        free_energy = -self.temperature * np.log(Z)
        
        # Calculate energy E = sum(E_i * p_i)
        probabilities = np.exp(-evals / self.temperature) / Z
        energy = np.sum(evals * probabilities)
        
        # Calculate entropy S = -sum(p_i * log(p_i))
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        
        # Check for criticality using specific heat capacity
        # C = d²F/dT² = d²(-T log(Z))/dT²
        # We approximate this with finite differences
        delta_T = 0.01
        T_plus = self.temperature + delta_T
        T_minus = self.temperature - delta_T
        
        Z_plus = np.sum(np.exp(-evals / T_plus))
        Z_minus = np.sum(np.exp(-evals / T_minus))
        
        F_plus = -T_plus * np.log(Z_plus)
        F_minus = -T_minus * np.log(Z_minus)
        
        # Second derivative approximation
        specific_heat = (F_plus - 2*free_energy + F_minus) / (delta_T**2)
        
        # In critical systems, specific heat often diverges or peaks
        is_critical = specific_heat > 10.0  # Threshold to be tuned empirically
        
        results = {
            'free_energy': free_energy,
            'entropy': entropy,
            'energy': energy,
            'specific_heat': specific_heat,
            'is_critical': is_critical,
            'eigenvalue_power_law': self._estimate_power_law(evals)
        }
        
        return results
    
    def track_rg_flow(self, W, epoch=None):
        """
        Track the renormalization group flow of the weight matrix.
        
        This method analyzes the weight matrix and adds the results to the history,
        allowing for tracking the RG flow across training epochs.
        
        Args:
            W: Weight matrix as numpy array
            epoch: Training epoch (optional)
            
        Returns:
            Dictionary with all computed metrics
        """
        # Analyze critical point
        critical_metrics = self.analyze_critical_point(W)
        
        # Compute fractal dimension
        fractal_metrics = self.compute_fractal_dimension(W)
        
        # Map free energy landscape
        energy_metrics = self.map_free_energy_landscape(W)
        
        # Combine all metrics
        all_metrics = {
            **critical_metrics,
            **fractal_metrics,
            **energy_metrics,
            'epoch': epoch
        }
        
        # Add to history
        self.history.append(all_metrics)
        
        return all_metrics
    
    def visualize_rg_flow(self, figsize=(12, 10)):
        """
        Visualize the renormalization group flow across training.
        
        This method creates plots showing how various metrics evolve during training,
        providing insights into the approach to criticality.
        
        Args:
            figsize: Figure size as tuple (width, height)
            
        Returns:
            Matplotlib figure object
        """
        if not self.history:
            print("No history available. Run track_rg_flow first.")
            return None
        
        # Extract epochs and metrics
        epochs = [entry.get('epoch', i) for i, entry in enumerate(self.history)]
        power_laws = [entry.get('power_law_exponent', np.nan) for entry in self.history]
        scale_invs = [entry.get('scale_invariance', np.nan) for entry in self.history]
        fractal_dims = [entry.get('fractal_dimension', np.nan) for entry in self.history]
        free_energies = [entry.get('free_energy', np.nan) for entry in self.history]
        entropies = [entry.get('entropy', np.nan) for entry in self.history]
        
        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # Plot power law exponents
        ax = axes[0, 0]
        ax.plot(epochs, power_laws, 'o-', label='Power Law Exponent')
        ax.axhline(y=2.0, color='r', linestyle='--', label='Critical Value')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Power Law Exponent')
        ax.set_title('Evolution of Power Law Exponent')
        ax.legend()
        
        # Plot scale invariance
        ax = axes[0, 1]
        ax.plot(epochs, scale_invs, 'o-', label='Scale Invariance')
        ax.axhline(y=0.9, color='r', linestyle='--', label='Critical Threshold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Scale Invariance')
        ax.set_title('Evolution of Scale Invariance')
        ax.legend()
        
        # Plot fractal dimension
        ax = axes[1, 0]
        ax.plot(epochs, fractal_dims, 'o-', label='Fractal Dimension')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Fractal Dimension')
        ax.set_title('Evolution of Fractal Dimension')
        ax.legend()
        
        # Plot free energy
        ax = axes[1, 1]
        ax.plot(epochs, free_energies, 'o-', label='Free Energy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Free Energy')
        ax.set_title('Evolution of Free Energy')
        ax.legend()
        
        # Plot entropy
        ax = axes[2, 0]
        ax.plot(epochs, entropies, 'o-', label='Entropy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Entropy')
        ax.set_title('Evolution of Entropy')
        ax.legend()
        
        # Plot phase diagram
        ax = axes[2, 1]
        sc = ax.scatter(
            power_laws, 
            scale_invs, 
            c=free_energies, 
            cmap='viridis', 
            s=50, 
            alpha=0.7
        )
        ax.set_xlabel('Power Law Exponent')
        ax.set_ylabel('Scale Invariance')
        ax.set_title('Phase Diagram')
        plt.colorbar(sc, ax=ax, label='Free Energy')
        
        # Add critical region
        ax.axhspan(0.9, 1.0, alpha=0.2, color='red', label='Critical Region')
        ax.axvspan(1.8, 2.2, alpha=0.2, color='red')
        ax.legend()
        
        plt.tight_layout()
        return fig
    def detect_phase_transition(self, W_before, W_after):
        """
        Detect phase transitions between two weight matrices.
        
        This method analyzes two weight matrices (e.g., before and after training)
        to detect if a phase transition has occurred in the model's parameter space.
        
        Args:
            W_before: Weight matrix before (e.g., at epoch t)
            W_after: Weight matrix after (e.g., at epoch t+1)
            
        Returns:
            Dictionary with phase transition metrics
        """
        # Analyze both matrices
        metrics_before = self.track_rg_flow(W_before)
        metrics_after = self.track_rg_flow(W_after)
        
        # Calculate key differences
        power_law_diff = abs(metrics_after['power_law_exponent'] - metrics_before['power_law_exponent'])
        scale_inv_diff = abs(metrics_after['scale_invariance'] - metrics_before['scale_invariance'])
        fractal_dim_diff = abs(metrics_after['fractal_dimension'] - metrics_before['fractal_dimension'])
        free_energy_diff = abs(metrics_after['free_energy'] - metrics_before['free_energy'])
        
        # Check for phase transition
        # A phase transition is characterized by sudden changes in these metrics
        is_transition = (
            power_law_diff > 0.3 or  # Significant change in power law exponent
            scale_inv_diff > 0.2 or  # Significant change in scale invariance
            fractal_dim_diff > 0.2 or  # Significant change in fractal dimension
            free_energy_diff > 5.0  # Significant change in free energy
        )
        
        # Determine transition type
        transition_type = "none"
        if is_transition:
            if metrics_after['is_critical'] and not metrics_before['is_critical']:
                transition_type = "to_critical"
            elif not metrics_after['is_critical'] and metrics_before['is_critical']:
                transition_type = "from_critical"
            else:
                transition_type = "non_critical"
        
        return {
            'is_phase_transition': is_transition,
            'transition_type': transition_type,
            'power_law_diff': power_law_diff,
            'scale_invariance_diff': scale_inv_diff,
            'fractal_dimension_diff': fractal_dim_diff,
            'free_energy_diff': free_energy_diff,
            'before': metrics_before,
            'after': metrics_after
        }
    
    def analyze_correlation_length(self, W):
        """
        Analyze correlation length in the weight matrix.
        
        This method estimates the correlation length, which diverges at critical points,
        providing another measure of criticality in the network.
        
        Args:
            W: Weight matrix as numpy array
            
        Returns:
            Dictionary with correlation length metrics
        """
        # Calculate correlation matrix
        if W.shape[0] > 10000 or W.shape[1] > 10000:
            # For very large matrices, use sampling
            sample_size = min(5000, min(W.shape))
            row_idx = np.random.choice(W.shape[0], sample_size, replace=False)
            col_idx = np.random.choice(W.shape[1], sample_size, replace=False)
            W_sample = W[np.ix_(row_idx, col_idx)]
            corr_matrix = np.corrcoef(W_sample)
        else:
            corr_matrix = np.corrcoef(W)
        
        # Remove NaNs
        corr_matrix = np.nan_to_num(corr_matrix)
        
        # Calculate eigenvalues of correlation matrix
        try:
            evals = np.linalg.eigvalsh(corr_matrix)
            evals = evals[evals > 1e-10]  # Remove numerical zeros
        except np.linalg.LinAlgError:
            return {'correlation_length': np.nan, 'correlation_decay': np.nan}
        
        if len(evals) == 0:
            return {'correlation_length': np.nan, 'correlation_decay': np.nan}
        
        # Estimate correlation length from largest eigenvalue
        max_eval = np.max(evals)
        correlation_length = np.sqrt(max_eval)
        
        # Calculate correlation decay exponent
        # In critical systems, correlations decay as power laws
        sorted_evals = np.sort(evals)[::-1]  # Sort in descending order
        if len(sorted_evals) > 5:
            log_evals = np.log(sorted_evals[:5])
            log_indices = np.log(np.arange(1, 6))
            
            # Linear regression to find decay exponent
            slope, _, _, _, _ = scipy.stats.linregress(log_indices, log_evals)
            correlation_decay = -slope
        else:
            correlation_decay = np.nan
        
        return {
            'correlation_length': correlation_length,
            'correlation_decay': correlation_decay,
            'is_critical_correlation': correlation_length > 10.0 or correlation_decay < 0.5
        }
    
    def compute_universality_class(self, W):
        """
        Compute the universality class of the weight matrix.
        
        This method attempts to classify the weight matrix into known universality
        classes from statistical physics based on its critical exponents.
        
        Args:
            W: Weight matrix as numpy array
            
        Returns:
            Dictionary with universality class information
        """
        # Get critical exponents
        critical_metrics = self.analyze_critical_point(W)
        fractal_metrics = self.compute_fractal_dimension(W)
        correlation_metrics = self.analyze_correlation_length(W)
        
        # Extract key exponents
        power_law = critical_metrics.get('power_law_exponent', np.nan)
        fractal_dim = fractal_metrics.get('fractal_dimension', np.nan)
        corr_decay = correlation_metrics.get('correlation_decay', np.nan)
        
        # Classify based on known universality classes
        # These are approximate classifications based on theoretical values
        universality_class = "unknown"
        confidence = 0.0
        
        if not np.isnan(power_law) and not np.isnan(fractal_dim) and not np.isnan(corr_decay):
            # Mean Field Theory class
            if 1.8 < power_law < 2.2 and 1.9 < fractal_dim < 2.1 and 0.9 < corr_decay < 1.1:
                universality_class = "mean_field"
                confidence = 0.8
            # 2D Ising Model class
            elif 2.0 < power_law < 2.4 and 1.7 < fractal_dim < 1.9 and 0.7 < corr_decay < 0.9:
                universality_class = "2d_ising"
                confidence = 0.7
            # 3D Ising Model class
            elif 2.3 < power_law < 2.7 and 2.3 < fractal_dim < 2.7 and 0.5 < corr_decay < 0.7:
                universality_class = "3d_ising"
                confidence = 0.7
            # Random Matrix Theory class
            elif 2.7 < power_law < 3.3 and 1.9 < fractal_dim < 2.1:
                universality_class = "random_matrix"
                confidence = 0.6
        
        return {
            'universality_class': universality_class,
            'confidence': confidence,
            'power_law_exponent': power_law,
            'fractal_dimension': fractal_dim,
            'correlation_decay': corr_decay
        }
