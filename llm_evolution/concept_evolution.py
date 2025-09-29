"""
Concept Evolution Tracker - Monitors and manages concept drift in NAS
Addresses the challenge of maintaining relevance as data distributions evolve
"""

import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import torch
import torch.nn as nn


@dataclass
class ConceptSnapshot:
    """Snapshot of concept state at a given time"""
    timestamp: int
    architecture_signature: str
    performance_metrics: Dict[str, float]
    data_distribution_stats: Dict[str, float]
    drift_score: float


class ConceptEvolutionTracker:
    """
    Tracks concept evolution and drift in neural architecture search
    Provides feedback for LLM-guided evolution decisions
    """
    
    def __init__(self, 
                 window_size: int = 10,
                 drift_threshold: float = 0.3,
                 adaptation_rate: float = 0.1):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.adaptation_rate = adaptation_rate
        
        # Concept history
        self.concept_history: deque = deque(maxlen=window_size)
        self.current_epoch = 0
        
        # Drift detection metrics
        self.performance_trend = []
        self.architecture_diversity = []
        self.data_complexity = []
        
    def update_concept_state(self, 
                           architecture: Dict,
                           performance: Dict[str, float],
                           data_stats: Optional[Dict] = None) -> float:
        """
        Update concept state and compute drift score
        
        Args:
            architecture: Current architecture configuration
            performance: Performance metrics (accuracy, latency, etc.)
            data_stats: Optional data distribution statistics
            
        Returns:
            drift_score: Computed concept drift score
        """
        # Create architecture signature
        arch_signature = self._create_architecture_signature(architecture)
        
        # Compute drift score based on multiple factors
        drift_score = self._compute_drift_score(performance, data_stats)
        
        # Create snapshot
        snapshot = ConceptSnapshot(
            timestamp=self.current_epoch,
            architecture_signature=arch_signature,
            performance_metrics=performance.copy(),
            data_distribution_stats=data_stats or {},
            drift_score=drift_score
        )
        
        # Update history
        self.concept_history.append(snapshot)
        self.current_epoch += 1
        
        # Update trend tracking
        self._update_trends(performance, drift_score)
        
        return drift_score
    
    def _create_architecture_signature(self, architecture: Dict) -> str:
        """Create a unique signature for architecture comparison"""
        # Extract key architectural features
        features = []
        
        if 'structure_info' in architecture:
            for layer in architecture['structure_info']:
                layer_features = [
                    layer.get('class', ''),
                    layer.get('in', 0),
                    layer.get('out', 0),
                    layer.get('k', 0),
                    layer.get('s', 0)
                ]
                features.extend(layer_features)
        
        return '_'.join(map(str, features))
    
    def _compute_drift_score(self, 
                           performance: Dict[str, float],
                           data_stats: Optional[Dict]) -> float:
        """Compute concept drift score from multiple indicators"""
        drift_components = []
        
        # 1. Performance degradation
        if len(self.concept_history) > 0:
            last_performance = self.concept_history[-1].performance_metrics
            perf_drift = self._compute_performance_drift(last_performance, performance)
            drift_components.append(perf_drift)
        
        # 2. Architecture complexity change
        arch_drift = self._compute_architecture_drift(performance)
        drift_components.append(arch_drift)
        
        # 3. Data distribution shift (if available)
        if data_stats and len(self.concept_history) > 0:
            data_drift = self._compute_data_drift(data_stats)
            drift_components.append(data_drift)
        
        # Combine components with weights
        weights = [0.4, 0.3, 0.3]  # Performance, Architecture, Data
        drift_score = sum(w * d for w, d in zip(weights, drift_components))
        
        return min(max(drift_score, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _compute_performance_drift(self, 
                                 old_perf: Dict[str, float],
                                 new_perf: Dict[str, float]) -> float:
        """Compute performance-based drift"""
        drift = 0.0
        count = 0
        
        for metric in ['accuracy', 'latency', 'memory', 'flops']:
            if metric in old_perf and metric in new_perf:
                # Normalize and compute relative change
                old_val = old_perf[metric]
                new_val = new_perf[metric]
                
                if old_val > 0:
                    relative_change = abs(new_val - old_val) / old_val
                    drift += relative_change
                    count += 1
        
        return drift / max(count, 1)
    
    def _compute_architecture_drift(self, performance: Dict[str, float]) -> float:
        """Compute architecture complexity drift"""
        # Use FLOPs and parameter count as complexity indicators
        complexity_indicators = ['flops', 'model_size', 'layers']
        complexity_score = 0.0
        
        for indicator in complexity_indicators:
            if indicator in performance:
                # Normalize to [0, 1] range
                normalized = min(performance[indicator] / 1e9, 1.0)  # Scale down
                complexity_score += normalized
        
        return complexity_score / len(complexity_indicators)
    
    def _compute_data_drift(self, data_stats: Dict) -> float:
        """Compute data distribution drift"""
        if not self.concept_history:
            return 0.0
        
        last_data_stats = self.concept_history[-1].data_distribution_stats
        if not last_data_stats:
            return 0.0
        
        drift = 0.0
        count = 0
        
        for key in data_stats:
            if key in last_data_stats:
                old_val = last_data_stats[key]
                new_val = data_stats[key]
                
                if old_val > 0:
                    relative_change = abs(new_val - old_val) / old_val
                    drift += relative_change
                    count += 1
        
        return drift / max(count, 1)
    
    def _update_trends(self, performance: Dict[str, float], drift_score: float):
        """Update trend tracking for evolution guidance"""
        self.performance_trend.append(performance.get('accuracy', 0.0))
        self.architecture_diversity.append(drift_score)
        
        # Keep only recent history
        if len(self.performance_trend) > self.window_size:
            self.performance_trend.pop(0)
            self.architecture_diversity.pop(0)
    
    def get_evolution_guidance(self) -> Dict[str, any]:
        """
        Provide guidance for LLM evolution decisions
        
        Returns:
            guidance: Dictionary with evolution recommendations
        """
        if len(self.concept_history) < 2:
            return {
                'drift_level': 'low',
                'recommendation': 'explore',
                'adaptation_strength': 0.1,
                'focus_areas': ['accuracy', 'efficiency']
            }
        
        # Analyze recent drift patterns
        recent_drift = [snapshot.drift_score for snapshot in list(self.concept_history)[-3:]]
        avg_drift = np.mean(recent_drift)
        
        # Determine evolution strategy
        if avg_drift > self.drift_threshold:
            drift_level = 'high'
            recommendation = 'adapt_aggressively'
            adaptation_strength = min(0.5, avg_drift * 2)
            focus_areas = ['robustness', 'generalization', 'efficiency']
        elif avg_drift > self.drift_threshold * 0.5:
            drift_level = 'medium'
            recommendation = 'adapt_moderately'
            adaptation_strength = avg_drift
            focus_areas = ['accuracy', 'efficiency', 'stability']
        else:
            drift_level = 'low'
            recommendation = 'explore'
            adaptation_strength = 0.1
            focus_areas = ['accuracy', 'innovation']
        
        # Performance trend analysis
        if len(self.performance_trend) >= 3:
            trend_slope = np.polyfit(range(len(self.performance_trend)), 
                                   self.performance_trend, 1)[0]
            if trend_slope < -0.01:
                recommendation = 'recover_performance'
                adaptation_strength = max(adaptation_strength, 0.3)
        
        return {
            'drift_level': drift_level,
            'recommendation': recommendation,
            'adaptation_strength': adaptation_strength,
            'focus_areas': focus_areas,
            'avg_drift': avg_drift,
            'trend_slope': trend_slope if len(self.performance_trend) >= 3 else 0.0,
            'concept_stability': 1.0 - avg_drift
        }
    
    def save_state(self, filepath: str):
        """Save concept evolution state"""
        state = {
            'window_size': self.window_size,
            'drift_threshold': self.drift_threshold,
            'adaptation_rate': self.adaptation_rate,
            'current_epoch': self.current_epoch,
            'concept_history': [
                {
                    'timestamp': s.timestamp,
                    'architecture_signature': s.architecture_signature,
                    'performance_metrics': s.performance_metrics,
                    'data_distribution_stats': s.data_distribution_stats,
                    'drift_score': s.drift_score
                }
                for s in self.concept_history
            ],
            'performance_trend': self.performance_trend,
            'architecture_diversity': self.architecture_diversity
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load concept evolution state"""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.window_size = state.get('window_size', 10)
        self.drift_threshold = state.get('drift_threshold', 0.3)
        self.adaptation_rate = state.get('adaptation_rate', 0.1)
        self.current_epoch = state.get('current_epoch', 0)
        
        # Reconstruct concept history
        self.concept_history.clear()
        for snapshot_data in state.get('concept_history', []):
            snapshot = ConceptSnapshot(
                timestamp=snapshot_data['timestamp'],
                architecture_signature=snapshot_data['architecture_signature'],
                performance_metrics=snapshot_data['performance_metrics'],
                data_distribution_stats=snapshot_data['data_distribution_stats'],
                drift_score=snapshot_data['drift_score']
            )
            self.concept_history.append(snapshot)
        
        self.performance_trend = state.get('performance_trend', [])
        self.architecture_diversity = state.get('architecture_diversity', [])
