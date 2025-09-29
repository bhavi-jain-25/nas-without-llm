"""
Multi-Objective Evaluator - Handles simultaneous optimization of multiple objectives
Addresses accuracy, latency, memory footprint, and energy efficiency
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
try:
    import psutil
except ImportError:
    psutil = None
import os


@dataclass
class ObjectiveWeights:
    """Weights for multi-objective optimization"""
    accuracy: float = 0.4
    latency: float = 0.2
    memory: float = 0.2
    energy: float = 0.2
    
    def normalize(self):
        """Normalize weights to sum to 1.0"""
        total = self.accuracy + self.latency + self.memory + self.energy
        if total > 0:
            self.accuracy /= total
            self.latency /= total
            self.memory /= total
            self.energy /= total


@dataclass
class ObjectiveValues:
    """Values for all objectives"""
    accuracy: float = 0.0
    latency: float = 0.0  # milliseconds
    memory: float = 0.0   # MB
    energy: float = 0.0   # relative energy units
    flops: float = 0.0
    params: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy,
            'latency': self.latency,
            'memory': self.memory,
            'energy': self.energy,
            'flops': self.flops,
            'params': self.params
        }


class ObjectiveEvaluator(ABC):
    """Abstract base class for objective evaluation"""
    
    @abstractmethod
    def evaluate(self, model: nn.Module, input_shape: Tuple[int, ...]) -> float:
        """Evaluate a specific objective"""
        pass


class AccuracyEvaluator(ObjectiveEvaluator):
    """Evaluates model accuracy using training-free metrics"""
    
    def __init__(self, use_zen_score: bool = True):
        self.use_zen_score = use_zen_score
    
    def evaluate(self, model: nn.Module, input_shape: Tuple[int, ...]) -> float:
        """Evaluate accuracy using training-free metrics"""
        if self.use_zen_score:
            return self._compute_zen_score(model, input_shape)
        else:
            return self._compute_entropy_score(model, input_shape)
    
    def _compute_zen_score(self, model: nn.Module, input_shape: Tuple[int, ...]) -> float:
        """Compute Zen-NAS score (gradient-based)"""
        try:
            model.eval()
            x = torch.randn(1, *input_shape)
            
            # Compute gradients
            x.requires_grad_(True)
            output = model(x)
            
            # Use gradient magnitude as proxy for expressiveness
            if isinstance(output, (list, tuple)):
                output = output[0]
            
            grad_norm = torch.autograd.grad(
                outputs=output.sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True
            )[0]
            
            score = grad_norm.norm().item()
            return min(max(score / 1000.0, 0.0), 1.0)  # Normalize
            
        except Exception as e:
            print(f"Zen score computation failed: {e}")
            return 0.5  # Default score
    
    def _compute_entropy_score(self, model: nn.Module, input_shape: Tuple[int, ...]) -> float:
        """Compute entropy-based score (DeepMAD style)"""
        try:
            model.eval()
            x = torch.randn(1, *input_shape)
            
            with torch.no_grad():
                output = model(x)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                
                # Compute entropy of output distribution
                probs = torch.softmax(output, dim=1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
                
                return min(max(entropy.item() / 10.0, 0.0), 1.0)  # Normalize
                
        except Exception as e:
            print(f"Entropy score computation failed: {e}")
            return 0.5  # Default score


class LatencyEvaluator(ObjectiveEvaluator):
    """Evaluates model inference latency"""
    
    def __init__(self, num_warmup: int = 10, num_runs: int = 100):
        self.num_warmup = num_warmup
        self.num_runs = num_runs
    
    def evaluate(self, model: nn.Module, input_shape: Tuple[int, ...]) -> float:
        """Evaluate inference latency in milliseconds"""
        try:
            model.eval()
            x = torch.randn(1, *input_shape)
            
            # Warmup
            with torch.no_grad():
                for _ in range(self.num_warmup):
                    _ = model(x)
            
            # Measure latency
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(self.num_runs):
                    _ = model(x)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            avg_latency = (end_time - start_time) * 1000 / self.num_runs  # Convert to ms
            return avg_latency
            
        except Exception as e:
            print(f"Latency evaluation failed: {e}")
            return 100.0  # Default high latency


class MemoryEvaluator(ObjectiveEvaluator):
    """Evaluates model memory footprint"""
    
    def evaluate(self, model: nn.Module, input_shape: Tuple[int, ...]) -> float:
        """Evaluate memory usage in MB"""
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            # Estimate memory usage (parameters + activations)
            param_memory = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
            
            # Estimate activation memory (rough approximation)
            x = torch.randn(1, *input_shape)
            with torch.no_grad():
                output = model(x)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                activation_memory = output.numel() * 4 / (1024 * 1024)
            
            total_memory = param_memory + activation_memory
            return total_memory
            
        except Exception as e:
            print(f"Memory evaluation failed: {e}")
            return 100.0  # Default high memory


class EnergyEvaluator(ObjectiveEvaluator):
    """Evaluates model energy efficiency"""
    
    def __init__(self, use_cpu_usage: bool = True):
        self.use_cpu_usage = use_cpu_usage
    
    def evaluate(self, model: nn.Module, input_shape: Tuple[int, ...]) -> float:
        """Evaluate energy efficiency (relative units)"""
        try:
            if self.use_cpu_usage:
                return self._evaluate_cpu_energy(model, input_shape)
            else:
                return self._evaluate_flops_energy(model, input_shape)
                
        except Exception as e:
            print(f"Energy evaluation failed: {e}")
            return 1.0  # Default energy usage
    
    def _evaluate_cpu_energy(self, model: nn.Module, input_shape: Tuple[int, ...]) -> float:
        """Evaluate energy using CPU usage"""
        if psutil is None:
            return 1.0  # Default energy usage
        process = psutil.Process(os.getpid())
        
        # Measure CPU usage during inference
        x = torch.randn(1, *input_shape)
        
        cpu_before = process.cpu_percent()
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        cpu_after = process.cpu_percent()
        
        energy_usage = max(cpu_after - cpu_before, 0.1)  # Avoid zero
        return energy_usage
    
    def _evaluate_flops_energy(self, model: nn.Module, input_shape: Tuple[int, ...]) -> float:
        """Evaluate energy using FLOPs as proxy"""
        try:
            from thop import profile
            x = torch.randn(1, *input_shape)
            flops, params = profile(model, inputs=(x,), verbose=False)
            
            # Convert FLOPs to relative energy units
            energy_usage = flops / 1e9  # Normalize
            return min(energy_usage, 10.0)  # Cap at reasonable value
            
        except ImportError:
            print("thop not available, using default energy evaluation")
            return 1.0


class MultiObjectiveEvaluator:
    """
    Evaluates multiple objectives simultaneously
    Provides weighted scoring and Pareto frontier analysis
    """
    
    def __init__(self, 
                 weights: Optional[ObjectiveWeights] = None,
                 target_accuracy: float = 0.8,
                 target_latency: float = 50.0,
                 target_memory: float = 50.0,
                 target_energy: float = 1.0):
        
        self.weights = weights or ObjectiveWeights()
        self.weights.normalize()
        
        self.targets = {
            'accuracy': target_accuracy,
            'latency': target_latency,
            'memory': target_memory,
            'energy': target_energy
        }
        
        # Initialize evaluators
        self.evaluators = {
            'accuracy': AccuracyEvaluator(),
            'latency': LatencyEvaluator(),
            'memory': MemoryEvaluator(),
            'energy': EnergyEvaluator()
        }
    
    def evaluate_architecture(self, 
                            model: nn.Module, 
                            input_shape: Tuple[int, ...],
                            compute_flops: bool = True) -> ObjectiveValues:
        """
        Evaluate all objectives for a given architecture
        
        Args:
            model: Neural network model
            input_shape: Input tensor shape (excluding batch dimension)
            compute_flops: Whether to compute FLOPs and parameters
            
        Returns:
            ObjectiveValues: All objective values
        """
        values = ObjectiveValues()
        
        # Evaluate each objective
        values.accuracy = self.evaluators['accuracy'].evaluate(model, input_shape)
        values.latency = self.evaluators['latency'].evaluate(model, input_shape)
        values.memory = self.evaluators['memory'].evaluate(model, input_shape)
        values.energy = self.evaluators['energy'].evaluate(model, input_shape)
        
        # Compute FLOPs and parameters if requested
        if compute_flops:
            try:
                from thop import profile
                x = torch.randn(1, *input_shape)
                flops, params = profile(model, inputs=(x,), verbose=False)
                values.flops = flops
                values.params = params
            except ImportError:
                # Fallback: estimate from model
                values.params = sum(p.numel() for p in model.parameters())
                values.flops = values.params * 2  # Rough estimate
        
        return values
    
    def compute_weighted_score(self, values: ObjectiveValues) -> float:
        """
        Compute weighted multi-objective score
        
        Args:
            values: Objective values
            
        Returns:
            score: Weighted score (higher is better)
        """
        # Normalize objectives to [0, 1] range
        normalized_accuracy = values.accuracy
        normalized_latency = max(0, 1 - values.latency / self.targets['latency'])
        normalized_memory = max(0, 1 - values.memory / self.targets['memory'])
        normalized_energy = max(0, 1 - values.energy / self.targets['energy'])
        
        # Compute weighted score
        score = (
            self.weights.accuracy * normalized_accuracy +
            self.weights.latency * normalized_latency +
            self.weights.memory * normalized_memory +
            self.weights.energy * normalized_energy
        )
        
        return score
    
    def is_pareto_optimal(self, 
                         candidate_values: ObjectiveValues,
                         existing_values: List[ObjectiveValues]) -> bool:
        """
        Check if candidate is Pareto optimal
        
        Args:
            candidate_values: Values to check
            existing_values: List of existing values
            
        Returns:
            is_optimal: True if Pareto optimal
        """
        for existing in existing_values:
            # Check if existing dominates candidate
            if (existing.accuracy >= candidate_values.accuracy and
                existing.latency <= candidate_values.latency and
                existing.memory <= candidate_values.memory and
                existing.energy <= candidate_values.energy and
                (existing.accuracy > candidate_values.accuracy or
                 existing.latency < candidate_values.latency or
                 existing.memory < candidate_values.memory or
                 existing.energy < candidate_values.energy)):
                return False
        
        return True
    
    def get_pareto_frontier(self, 
                          all_values: List[ObjectiveValues]) -> List[ObjectiveValues]:
        """
        Compute Pareto frontier from all objective values
        
        Args:
            all_values: List of all objective values
            
        Returns:
            pareto_frontier: List of Pareto optimal solutions
        """
        pareto_frontier = []
        
        for candidate in all_values:
            if self.is_pareto_optimal(candidate, all_values):
                pareto_frontier.append(candidate)
        
        return pareto_frontier
    
    def update_weights_for_evolution(self, 
                                   concept_guidance: Dict[str, any]) -> None:
        """
        Update objective weights based on concept evolution guidance
        
        Args:
            concept_guidance: Guidance from concept evolution tracker
        """
        focus_areas = concept_guidance.get('focus_areas', [])
        adaptation_strength = concept_guidance.get('adaptation_strength', 0.1)
        
        # Reset weights
        self.weights = ObjectiveWeights()
        
        # Adjust weights based on focus areas
        if 'accuracy' in focus_areas:
            self.weights.accuracy += adaptation_strength
        if 'efficiency' in focus_areas or 'latency' in focus_areas:
            self.weights.latency += adaptation_strength
        if 'memory' in focus_areas:
            self.weights.memory += adaptation_strength
        if 'energy' in focus_areas:
            self.weights.energy += adaptation_strength
        
        # Normalize weights
        self.weights.normalize()
    
    def get_evolution_recommendations(self, 
                                    current_values: ObjectiveValues,
                                    target_improvement: float = 0.1) -> Dict[str, str]:
        """
        Get recommendations for architecture evolution
        
        Args:
            current_values: Current objective values
            target_improvement: Target improvement factor
            
        Returns:
            recommendations: Dictionary of recommendations
        """
        recommendations = {}
        
        # Analyze which objectives need improvement
        if current_values.accuracy < self.targets['accuracy'] * (1 - target_improvement):
            recommendations['accuracy'] = 'increase_model_capacity'
        
        if current_values.latency > self.targets['latency'] * (1 + target_improvement):
            recommendations['latency'] = 'reduce_model_complexity'
        
        if current_values.memory > self.targets['memory'] * (1 + target_improvement):
            recommendations['memory'] = 'optimize_architecture_efficiency'
        
        if current_values.energy > self.targets['energy'] * (1 + target_improvement):
            recommendations['energy'] = 'reduce_computational_overhead'
        
        return recommendations
