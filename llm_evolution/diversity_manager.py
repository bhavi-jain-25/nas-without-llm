"""
Diversity Manager - Maintains population diversity in LLM-guided evolution
Addresses the challenge of maintaining diversity in search population
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import hashlib


@dataclass
class ArchitectureSignature:
    """Unique signature for architecture comparison"""
    hash: str
    layer_count: int
    total_params: int
    flops: float
    depth: int
    width: int


@dataclass
class DiversityMetrics:
    """Metrics for population diversity"""
    architectural_diversity: float
    performance_diversity: float
    functional_diversity: float
    overall_diversity: float


class DiversityManager:
    """
    Manages population diversity in LLM-guided evolution
    Prevents premature convergence and maintains exploration
    """
    
    def __init__(self, 
                 target_diversity: float = 0.7,
                 min_diversity_threshold: float = 0.3,
                 similarity_threshold: float = 0.8,
                 diversity_window: int = 10):
        
        self.target_diversity = target_diversity
        self.min_diversity_threshold = min_diversity_threshold
        self.similarity_threshold = similarity_threshold
        self.diversity_window = diversity_window
        
        # Population tracking
        self.population_history: List[Dict] = []
        self.architecture_signatures: List[ArchitectureSignature] = []
        self.performance_history: List[Dict] = []
        
        # Diversity metrics
        self.diversity_trend: List[float] = []
        self.convergence_warnings: List[str] = []
        
    def add_architecture(self, 
                        architecture: Dict, 
                        performance: Dict[str, float]) -> bool:
        """
        Add architecture to population and check diversity
        
        Args:
            architecture: Architecture configuration
            performance: Performance metrics
            
        Returns:
            accepted: True if architecture adds diversity
        """
        # Create signature
        signature = self._create_signature(architecture, performance)
        
        # Check if architecture is too similar to existing ones
        if self._is_too_similar(signature):
            return False
        
        # Add to population
        self.population_history.append(architecture)
        self.architecture_signatures.append(signature)
        self.performance_history.append(performance)
        
        # Update diversity metrics
        self._update_diversity_metrics()
        
        # Check for convergence
        self._check_convergence()
        
        return True
    
    def _create_signature(self, 
                         architecture: Dict, 
                         performance: Dict[str, float]) -> ArchitectureSignature:
        """Create unique signature for architecture"""
        # Create hash from architecture structure
        arch_str = json.dumps(architecture, sort_keys=True)
        arch_hash = hashlib.md5(arch_str.encode()).hexdigest()[:16]
        
        # Extract structural features
        structure_info = architecture.get('structure_info', [])
        layer_count = len(structure_info)
        
        # Calculate depth and width
        depth = layer_count
        width = 0
        if structure_info:
            width = max(layer.get('out', 0) for layer in structure_info)
        
        # Get performance features
        total_params = performance.get('params', 0)
        flops = performance.get('flops', 0)
        
        return ArchitectureSignature(
            hash=arch_hash,
            layer_count=layer_count,
            total_params=total_params,
            flops=flops,
            depth=depth,
            width=width
        )
    
    def _is_too_similar(self, signature: ArchitectureSignature) -> bool:
        """Check if architecture is too similar to existing ones"""
        if not self.architecture_signatures:
            return False
        
        # Calculate similarity to each existing architecture
        similarities = []
        for existing_sig in self.architecture_signatures:
            similarity = self._calculate_similarity(signature, existing_sig)
            similarities.append(similarity)
        
        # Check if too similar to any existing architecture
        max_similarity = max(similarities) if similarities else 0.0
        return max_similarity > self.similarity_threshold
    
    def _calculate_similarity(self, 
                            sig1: ArchitectureSignature, 
                            sig2: ArchitectureSignature) -> float:
        """Calculate similarity between two architecture signatures"""
        # Structural similarity
        layer_sim = 1.0 - abs(sig1.layer_count - sig2.layer_count) / max(sig1.layer_count, sig2.layer_count, 1)
        depth_sim = 1.0 - abs(sig1.depth - sig2.depth) / max(sig1.depth, sig2.depth, 1)
        width_sim = 1.0 - abs(sig1.width - sig2.width) / max(sig1.width, sig2.width, 1)
        
        # Performance similarity
        params_sim = 1.0 - abs(sig1.total_params - sig2.total_params) / max(sig1.total_params, sig2.total_params, 1)
        flops_sim = 1.0 - abs(sig1.flops - sig2.flops) / max(sig1.flops, sig2.flops, 1)
        
        # Hash similarity (exact match)
        hash_sim = 1.0 if sig1.hash == sig2.hash else 0.0
        
        # Weighted combination
        weights = [0.2, 0.2, 0.2, 0.15, 0.15, 0.1]  # Structural, performance, hash
        similarities = [layer_sim, depth_sim, width_sim, params_sim, flops_sim, hash_sim]
        
        return sum(w * s for w, s in zip(weights, similarities))
    
    def _update_diversity_metrics(self):
        """Update diversity metrics for current population"""
        if len(self.architecture_signatures) < 2:
            return
        
        # Calculate architectural diversity
        arch_diversity = self._calculate_architectural_diversity()
        
        # Calculate performance diversity
        perf_diversity = self._calculate_performance_diversity()
        
        # Calculate functional diversity
        func_diversity = self._calculate_functional_diversity()
        
        # Overall diversity
        overall_diversity = (arch_diversity + perf_diversity + func_diversity) / 3.0
        
        # Store metrics
        diversity_metrics = DiversityMetrics(
            architectural_diversity=arch_diversity,
            performance_diversity=perf_diversity,
            functional_diversity=func_diversity,
            overall_diversity=overall_diversity
        )
        
        self.diversity_trend.append(overall_diversity)
        
        # Keep only recent history
        if len(self.diversity_trend) > self.diversity_window:
            self.diversity_trend.pop(0)
    
    def _calculate_architectural_diversity(self) -> float:
        """Calculate architectural diversity in population"""
        if len(self.architecture_signatures) < 2:
            return 1.0
        
        # Calculate pairwise similarities
        similarities = []
        n = len(self.architecture_signatures)
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._calculate_similarity(
                    self.architecture_signatures[i],
                    self.architecture_signatures[j]
                )
                similarities.append(sim)
        
        # Diversity is inverse of average similarity
        avg_similarity = np.mean(similarities) if similarities else 0.0
        return 1.0 - avg_similarity
    
    def _calculate_performance_diversity(self) -> float:
        """Calculate performance diversity in population"""
        if len(self.performance_history) < 2:
            return 1.0
        
        # Extract performance metrics
        metrics = ['accuracy', 'latency', 'memory', 'energy']
        performance_matrix = []
        
        for perf in self.performance_history:
            row = [perf.get(metric, 0.0) for metric in metrics]
            performance_matrix.append(row)
        
        performance_matrix = np.array(performance_matrix)
        
        # Calculate coefficient of variation for each metric
        cv_scores = []
        for i in range(performance_matrix.shape[1]):
            col = performance_matrix[:, i]
            if np.std(col) > 0:
                cv = np.std(col) / np.mean(col)
                cv_scores.append(cv)
        
        # Average coefficient of variation
        avg_cv = np.mean(cv_scores) if cv_scores else 0.0
        return min(avg_cv, 1.0)  # Cap at 1.0
    
    def _calculate_functional_diversity(self) -> float:
        """Calculate functional diversity (different architectural patterns)"""
        if len(self.architecture_signatures) < 2:
            return 1.0
        
        # Group architectures by functional patterns
        pattern_groups = defaultdict(list)
        
        for i, sig in enumerate(self.architecture_signatures):
            # Create functional pattern key
            pattern_key = f"{sig.layer_count}_{sig.depth}_{sig.width}"
            pattern_groups[pattern_key].append(i)
        
        # Calculate diversity based on pattern distribution
        num_patterns = len(pattern_groups)
        total_architectures = len(self.architecture_signatures)
        
        if total_architectures == 0:
            return 0.0
        
        # Entropy-based diversity
        entropy = 0.0
        for pattern, indices in pattern_groups.items():
            p = len(indices) / total_architectures
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Normalize entropy
        max_entropy = np.log2(num_patterns) if num_patterns > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def _check_convergence(self):
        """Check for population convergence and generate warnings"""
        if len(self.diversity_trend) < 5:
            return
        
        # Check recent diversity trend
        recent_diversity = self.diversity_trend[-5:]
        trend_slope = np.polyfit(range(len(recent_diversity)), recent_diversity, 1)[0]
        
        # Check current diversity level
        current_diversity = self.diversity_trend[-1] if self.diversity_trend else 0.0
        
        # Generate warnings
        if current_diversity < self.min_diversity_threshold:
            warning = f"Low diversity detected: {current_diversity:.3f} < {self.min_diversity_threshold}"
            if warning not in self.convergence_warnings:
                self.convergence_warnings.append(warning)
        
        if trend_slope < -0.05:  # Declining diversity
            warning = f"Diversity declining: slope = {trend_slope:.3f}"
            if warning not in self.convergence_warnings:
                self.convergence_warnings.append(warning)
    
    def get_diversity_guidance(self) -> Dict[str, any]:
        """
        Get guidance for maintaining diversity
        
        Returns:
            guidance: Dictionary with diversity recommendations
        """
        if not self.diversity_trend:
            return {
                'diversity_level': 'unknown',
                'recommendation': 'explore',
                'diversity_boost': 0.0,
                'focus_areas': ['architectural_variation', 'performance_range']
            }
        
        current_diversity = self.diversity_trend[-1]
        
        # Determine diversity level
        if current_diversity > self.target_diversity:
            diversity_level = 'high'
            recommendation = 'maintain'
            diversity_boost = 0.0
            focus_areas = ['quality_improvement', 'performance_optimization']
        elif current_diversity > self.min_diversity_threshold:
            diversity_level = 'medium'
            recommendation = 'slight_boost'
            diversity_boost = 0.2
            focus_areas = ['architectural_variation', 'performance_range']
        else:
            diversity_level = 'low'
            recommendation = 'aggressive_boost'
            diversity_boost = 0.5
            focus_areas = ['architectural_innovation', 'diverse_patterns', 'exploration']
        
        # Check for convergence warnings
        if self.convergence_warnings:
            recommendation = 'emergency_diversity'
            diversity_boost = max(diversity_boost, 0.7)
            focus_areas = ['architectural_innovation', 'pattern_diversity', 'exploration']
        
        return {
            'diversity_level': diversity_level,
            'recommendation': recommendation,
            'diversity_boost': diversity_boost,
            'focus_areas': focus_areas,
            'current_diversity': current_diversity,
            'target_diversity': self.target_diversity,
            'warnings': self.convergence_warnings.copy()
        }
    
    def suggest_diversity_actions(self) -> List[Dict[str, any]]:
        """
        Suggest specific actions to improve diversity
        
        Returns:
            actions: List of diversity improvement actions
        """
        actions = []
        guidance = self.get_diversity_guidance()
        
        if guidance['diversity_boost'] > 0:
            # Suggest architectural variations
            actions.append({
                'type': 'architectural_variation',
                'description': 'Introduce new architectural patterns',
                'priority': 'high' if guidance['diversity_boost'] > 0.3 else 'medium',
                'parameters': {
                    'mutation_rate': guidance['diversity_boost'],
                    'exploration_bonus': 0.3
                }
            })
            
            # Suggest performance range expansion
            actions.append({
                'type': 'performance_range',
                'description': 'Explore different performance trade-offs',
                'priority': 'medium',
                'parameters': {
                    'latency_range': [10, 200],
                    'memory_range': [10, 500],
                    'accuracy_range': [0.6, 0.95]
                }
            })
            
            # Suggest functional pattern diversity
            actions.append({
                'type': 'functional_diversity',
                'description': 'Increase functional pattern diversity',
                'priority': 'high' if guidance['diversity_boost'] > 0.5 else 'medium',
                'parameters': {
                    'pattern_variation': 0.4,
                    'novelty_bonus': 0.2
                }
            })
        
        return actions
    
    def prune_population(self, 
                        max_size: int = 50,
                        diversity_weight: float = 0.7) -> List[int]:
        """
        Prune population to maintain diversity and size limits
        
        Args:
            max_size: Maximum population size
            diversity_weight: Weight for diversity in pruning decisions
            
        Returns:
            indices_to_keep: Indices of architectures to keep
        """
        if len(self.population_history) <= max_size:
            return list(range(len(self.population_history)))
        
        # Calculate scores for each architecture
        scores = []
        for i in range(len(self.population_history)):
            # Performance score
            perf_score = self._calculate_performance_score(i)
            
            # Diversity score
            div_score = self._calculate_diversity_score(i)
            
            # Combined score
            combined_score = (1 - diversity_weight) * perf_score + diversity_weight * div_score
            scores.append(combined_score)
        
        # Select top architectures
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        indices_to_keep = sorted_indices[:max_size]
        
        return indices_to_keep
    
    def _calculate_performance_score(self, index: int) -> float:
        """Calculate performance score for architecture"""
        if index >= len(self.performance_history):
            return 0.0
        
        perf = self.performance_history[index]
        
        # Normalize performance metrics
        accuracy = perf.get('accuracy', 0.0)
        latency = max(0, 1 - perf.get('latency', 100) / 100)  # Lower is better
        memory = max(0, 1 - perf.get('memory', 100) / 100)    # Lower is better
        energy = max(0, 1 - perf.get('energy', 1.0))         # Lower is better
        
        # Weighted performance score
        weights = [0.4, 0.2, 0.2, 0.2]
        scores = [accuracy, latency, memory, energy]
        
        return sum(w * s for w, s in zip(weights, scores))
    
    def _calculate_diversity_score(self, index: int) -> float:
        """Calculate diversity score for architecture"""
        if index >= len(self.architecture_signatures):
            return 0.0
        
        target_sig = self.architecture_signatures[index]
        
        # Calculate average similarity to other architectures
        similarities = []
        for i, sig in enumerate(self.architecture_signatures):
            if i != index:
                sim = self._calculate_similarity(target_sig, sig)
                similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        # Diversity score is inverse of average similarity
        avg_similarity = np.mean(similarities)
        return 1.0 - avg_similarity
    
    def save_state(self, filepath: str):
        """Save diversity manager state"""
        state = {
            'target_diversity': self.target_diversity,
            'min_diversity_threshold': self.min_diversity_threshold,
            'similarity_threshold': self.similarity_threshold,
            'diversity_window': self.diversity_window,
            'population_history': self.population_history,
            'architecture_signatures': [
                {
                    'hash': sig.hash,
                    'layer_count': sig.layer_count,
                    'total_params': sig.total_params,
                    'flops': sig.flops,
                    'depth': sig.depth,
                    'width': sig.width
                }
                for sig in self.architecture_signatures
            ],
            'performance_history': self.performance_history,
            'diversity_trend': self.diversity_trend,
            'convergence_warnings': self.convergence_warnings
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load diversity manager state"""
        import os
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.target_diversity = state.get('target_diversity', 0.7)
        self.min_diversity_threshold = state.get('min_diversity_threshold', 0.3)
        self.similarity_threshold = state.get('similarity_threshold', 0.8)
        self.diversity_window = state.get('diversity_window', 10)
        
        self.population_history = state.get('population_history', [])
        self.performance_history = state.get('performance_history', [])
        self.diversity_trend = state.get('diversity_trend', [])
        self.convergence_warnings = state.get('convergence_warnings', [])
        
        # Reconstruct architecture signatures
        self.architecture_signatures = []
        for sig_data in state.get('architecture_signatures', []):
            signature = ArchitectureSignature(
                hash=sig_data['hash'],
                layer_count=sig_data['layer_count'],
                total_params=sig_data['total_params'],
                flops=sig_data['flops'],
                depth=sig_data['depth'],
                width=sig_data['width']
            )
            self.architecture_signatures.append(signature)
