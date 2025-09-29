"""
LLM-Guided Evolution Framework - Main orchestrator for concept evolution NAS
Combines all components: concept evolution, multi-objective optimization, LLM guidance, diversity management
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn

from .concept_evolution import ConceptEvolutionTracker
from .multi_objective import MultiObjectiveEvaluator, ObjectiveValues, ObjectiveWeights
from .llm_interface import LLMInterface, MockLLMInterface, EvolutionPrompt, ArchitectureMutation
from .diversity_manager import DiversityManager


@dataclass
class EvolutionConfig:
    """Configuration for LLM-guided evolution"""
    # Evolution parameters
    population_size: int = 20
    max_generations: int = 50
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    
    # LLM parameters
    llm_creativity: float = 0.7
    llm_safety_threshold: float = 0.8
    
    # Diversity parameters
    target_diversity: float = 0.7
    min_diversity_threshold: float = 0.3
    
    # Concept evolution parameters
    drift_threshold: float = 0.3
    adaptation_rate: float = 0.1
    
    # Multi-objective parameters
    accuracy_weight: float = 0.4
    latency_weight: float = 0.2
    memory_weight: float = 0.2
    energy_weight: float = 0.2
    
    # Search constraints
    max_layers: int = 50
    max_channels: int = 2048
    min_layers: int = 3
    min_channels: int = 8


@dataclass
class EvolutionResult:
    """Result of evolution process"""
    best_architecture: Dict
    best_performance: ObjectiveValues
    evolution_history: List[Dict]
    diversity_metrics: List[Dict]
    concept_evolution_log: List[Dict]
    total_time: float
    generations_completed: int


class LLMGuidedEvolution:
    """
    Main orchestrator for LLM-guided evolution framework
    Integrates concept evolution, multi-objective optimization, and diversity management
    """
    
    def __init__(self, 
                 config: EvolutionConfig,
                 llm_interface: Optional[LLMInterface] = None,
                 work_dir: str = "./llm_evolution_results"):
        
        self.config = config
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        
        # Initialize components
        self.llm_interface = llm_interface or MockLLMInterface(
            creativity_level=config.llm_creativity,
            safety_threshold=config.llm_safety_threshold
        )
        
        self.concept_tracker = ConceptEvolutionTracker(
            drift_threshold=config.drift_threshold,
            adaptation_rate=config.adaptation_rate
        )
        
        self.multi_objective_evaluator = MultiObjectiveEvaluator(
            weights=ObjectiveWeights(
                accuracy=config.accuracy_weight,
                latency=config.latency_weight,
                memory=config.memory_weight,
                energy=config.energy_weight
            )
        )
        
        self.diversity_manager = DiversityManager(
            target_diversity=config.target_diversity,
            min_diversity_threshold=config.min_diversity_threshold
        )
        
        # Evolution state
        self.population: List[Dict] = []
        self.population_performance: List[ObjectiveValues] = []
        self.evolution_history: List[Dict] = []
        self.current_generation = 0
        
        # Statistics
        self.stats = {
            'total_mutations': 0,
            'successful_mutations': 0,
            'diversity_violations': 0,
            'concept_drift_events': 0,
            'llm_guidance_calls': 0
        }
    
    def evolve(self, 
               initial_architecture: Dict,
               input_shape: Tuple[int, ...] = (3, 224, 224),
               model_builder: Optional[callable] = None) -> EvolutionResult:
        """
        Run LLM-guided evolution process
        
        Args:
            initial_architecture: Starting architecture
            input_shape: Input tensor shape
            model_builder: Function to build model from architecture
            
        Returns:
            EvolutionResult: Complete evolution results
        """
        start_time = time.time()
        
        print(f"Starting LLM-guided evolution with {self.config.population_size} population size")
        print(f"Target: {self.config.max_generations} generations")
        
        # Initialize population
        self._initialize_population(initial_architecture)
        
        # Evolution loop
        for generation in range(self.config.max_generations):
            self.current_generation = generation
            print(f"\n--- Generation {generation + 1}/{self.config.max_generations} ---")
            
            # Evaluate current population
            self._evaluate_population(input_shape, model_builder)
            
            # Update concept evolution tracking
            self._update_concept_evolution()
            
            # Get evolution guidance
            guidance = self._get_evolution_guidance()
            
            # Generate new architectures using LLM
            new_architectures = self._generate_new_architectures(guidance)
            
            # Apply diversity filtering
            filtered_architectures = self._apply_diversity_filtering(new_architectures)
            
            # Update population
            self._update_population(filtered_architectures)
            
            # Log generation results
            self._log_generation_results(generation)
            
            # Check convergence
            if self._check_convergence():
                print(f"Convergence detected at generation {generation + 1}")
                break
        
        # Final evaluation
        self._evaluate_population(input_shape, model_builder)
        
        # Create result
        total_time = time.time() - start_time
        result = self._create_evolution_result(total_time)
        
        # Save results
        self._save_results(result)
        
        return result
    
    def _initialize_population(self, initial_architecture: Dict):
        """Initialize population with base architecture"""
        self.population = [initial_architecture.copy()]
        self.population_performance = []
        
        # Generate initial population through mutations
        for i in range(self.config.population_size - 1):
            # Create simple mutations for initial diversity
            mutated_arch = self._create_simple_mutation(initial_architecture)
            self.population.append(mutated_arch)
        
        print(f"Initialized population with {len(self.population)} architectures")
    
    def _create_simple_mutation(self, architecture: Dict) -> Dict:
        """Create simple mutation for initial population"""
        mutated = architecture.copy()
        structure_info = mutated.get('structure_info', [])
        
        if structure_info:
            # Randomly modify a layer
            layer_idx = torch.randint(0, len(structure_info), (1,)).item()
            layer = structure_info[layer_idx]
            
            # Simple parameter modification
            if 'out' in layer:
                current_out = layer['out']
                new_out = int(current_out * torch.rand(1).item() * 0.4 + 0.8)
                layer['out'] = max(self.config.min_channels, 
                                 min(new_out, self.config.max_channels))
        
        return mutated
    
    def _evaluate_population(self, 
                           input_shape: Tuple[int, ...],
                           model_builder: Optional[callable]):
        """Evaluate all architectures in population"""
        self.population_performance = []
        
        for i, architecture in enumerate(self.population):
            try:
                # Build model
                if model_builder:
                    model = model_builder(architecture)
                else:
                    model = self._default_model_builder(architecture)
                
                # Evaluate objectives
                performance = self.multi_objective_evaluator.evaluate_architecture(
                    model, input_shape
                )
                
                self.population_performance.append(performance)
                
            except Exception as e:
                print(f"Evaluation failed for architecture {i}: {e}")
                # Create default performance
                default_perf = ObjectiveValues()
                self.population_performance.append(default_perf)
    
    def _default_model_builder(self, architecture: Dict) -> nn.Module:
        """Default model builder (placeholder)"""
        # This would integrate with TinyNAS model building
        # For now, return a simple placeholder
        class PlaceholderModel(nn.Module):
            def __init__(self, arch):
                super().__init__()
                self.arch = arch
                self.layers = nn.ModuleList([
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Linear(64, 1000)
                ])
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        return PlaceholderModel(architecture)
    
    def _update_concept_evolution(self):
        """Update concept evolution tracking"""
        if not self.population_performance:
            return
        
        # Find best architecture
        best_idx = self._find_best_architecture()
        best_arch = self.population[best_idx]
        best_perf = self.population_performance[best_idx]
        
        # Update concept tracker
        drift_score = self.concept_tracker.update_concept_state(
            best_arch, best_perf.to_dict()
        )
        
        # Log concept evolution
        if drift_score > self.config.drift_threshold:
            self.stats['concept_drift_events'] += 1
            print(f"Concept drift detected: {drift_score:.3f}")
    
    def _find_best_architecture(self) -> int:
        """Find best architecture in current population"""
        if not self.population_performance:
            return 0
        
        best_score = -float('inf')
        best_idx = 0
        
        for i, performance in enumerate(self.population_performance):
            score = self.multi_objective_evaluator.compute_weighted_score(performance)
            if score > best_score:
                best_score = score
                best_idx = i
        
        return best_idx
    
    def _get_evolution_guidance(self) -> Dict[str, Any]:
        """Get comprehensive evolution guidance"""
        # Get concept evolution guidance
        concept_guidance = self.concept_tracker.get_evolution_guidance()
        
        # Get diversity guidance
        diversity_guidance = self.diversity_manager.get_diversity_guidance()
        
        # Get multi-objective recommendations
        if self.population_performance:
            best_idx = self._find_best_architecture()
            best_perf = self.population_performance[best_idx]
            mo_recommendations = self.multi_objective_evaluator.get_evolution_recommendations(best_perf)
        else:
            mo_recommendations = {}
        
        # Combine guidance
        combined_guidance = {
            'concept_evolution': concept_guidance,
            'diversity': diversity_guidance,
            'multi_objective': mo_recommendations,
            'generation': self.current_generation,
            'population_size': len(self.population)
        }
        
        return combined_guidance
    
    def _generate_new_architectures(self, guidance: Dict[str, Any]) -> List[Dict]:
        """Generate new architectures using LLM guidance"""
        new_architectures = []
        
        # Determine number of new architectures to generate
        num_new = int(self.config.population_size * self.config.mutation_rate)
        
        for i in range(num_new):
            # Select parent architecture
            parent_idx = self._select_parent()
            parent_arch = self.population[parent_idx]
            
            # Create evolution prompt
            prompt = self._create_evolution_prompt(parent_arch, guidance)
            
            # Get LLM mutations
            mutations = self.llm_interface.generate_architecture_mutations(prompt)
            self.stats['llm_guidance_calls'] += 1
            
            # Apply mutations
            for mutation in mutations:
                try:
                    new_arch = self._apply_mutation(parent_arch, mutation)
                    if new_arch:
                        new_architectures.append(new_arch)
                        self.stats['successful_mutations'] += 1
                    self.stats['total_mutations'] += 1
                except Exception as e:
                    print(f"Mutation application failed: {e}")
        
        return new_architectures
    
    def _select_parent(self) -> int:
        """Select parent architecture for mutation"""
        if not self.population_performance:
            return torch.randint(0, len(self.population), (1,)).item()
        
        # Tournament selection
        tournament_size = min(3, len(self.population))
        candidates = torch.randint(0, len(self.population), (tournament_size,))
        
        best_score = -float('inf')
        best_idx = 0
        
        for idx in candidates:
            score = self.multi_objective_evaluator.compute_weighted_score(
                self.population_performance[idx]
            )
            if score > best_score:
                best_score = score
                best_idx = idx.item()
        
        return best_idx
    
    def _create_evolution_prompt(self, 
                               architecture: Dict, 
                               guidance: Dict[str, Any]) -> EvolutionPrompt:
        """Create evolution prompt for LLM"""
        # Get current performance
        arch_idx = self.population.index(architecture)
        performance = self.population_performance[arch_idx].to_dict()
        
        # Create constraints
        constraints = {
            'max_layers': self.config.max_layers,
            'max_channels': self.config.max_channels,
            'min_layers': self.config.min_layers,
            'min_channels': self.config.min_channels,
            'current_layers': len(architecture.get('structure_info', []))
        }
        
        return EvolutionPrompt(
            context=f"Generation {self.current_generation} of LLM-guided evolution",
            current_architecture=architecture,
            performance_metrics=performance,
            concept_guidance=guidance['concept_evolution'],
            diversity_requirements=guidance['diversity'],
            constraints=constraints
        )
    
    def _apply_mutation(self, 
                       architecture: Dict, 
                       mutation: ArchitectureMutation) -> Optional[Dict]:
        """Apply mutation to architecture"""
        if not self.llm_interface.validate_mutation(mutation, {}):
            return None
        
        mutated = architecture.copy()
        structure_info = mutated.get('structure_info', [])
        
        if mutation.mutation_type == 'add_layer':
            return self._apply_add_layer_mutation(mutated, mutation)
        elif mutation.mutation_type == 'modify_layer':
            return self._apply_modify_layer_mutation(mutated, mutation)
        elif mutation.mutation_type == 'remove_layer':
            return self._apply_remove_layer_mutation(mutated, mutation)
        
        return None
    
    def _apply_add_layer_mutation(self, 
                                architecture: Dict, 
                                mutation: ArchitectureMutation) -> Optional[Dict]:
        """Apply add layer mutation"""
        structure_info = architecture.get('structure_info', [])
        insertion_point = mutation.target_layer or len(structure_info)
        
        if insertion_point > len(structure_info):
            insertion_point = len(structure_info)
        
        # Create new layer
        new_layer = mutation.parameters.copy()
        
        # Insert layer
        structure_info.insert(insertion_point, new_layer)
        architecture['structure_info'] = structure_info
        
        return architecture
    
    def _apply_modify_layer_mutation(self, 
                                   architecture: Dict, 
                                   mutation: ArchitectureMutation) -> Optional[Dict]:
        """Apply modify layer mutation"""
        structure_info = architecture.get('structure_info', [])
        target_layer = mutation.target_layer
        
        if target_layer >= len(structure_info):
            return None
        
        # Apply modification
        layer = structure_info[target_layer]
        parameter = mutation.parameters['parameter']
        new_value = mutation.parameters['new_value']
        
        layer[parameter] = new_value
        
        return architecture
    
    def _apply_remove_layer_mutation(self, 
                                   architecture: Dict, 
                                   mutation: ArchitectureMutation) -> Optional[Dict]:
        """Apply remove layer mutation"""
        structure_info = architecture.get('structure_info', [])
        target_layer = mutation.target_layer
        
        if target_layer >= len(structure_info) or len(structure_info) <= self.config.min_layers:
            return None
        
        # Remove layer
        structure_info.pop(target_layer)
        architecture['structure_info'] = structure_info
        
        return architecture
    
    def _apply_diversity_filtering(self, new_architectures: List[Dict]) -> List[Dict]:
        """Apply diversity filtering to new architectures"""
        filtered_architectures = []
        
        for arch in new_architectures:
            # Create dummy performance for diversity check
            dummy_performance = ObjectiveValues()
            
            if self.diversity_manager.add_architecture(arch, dummy_performance.to_dict()):
                filtered_architectures.append(arch)
            else:
                self.stats['diversity_violations'] += 1
        
        return filtered_architectures
    
    def _update_population(self, new_architectures: List[Dict]):
        """Update population with new architectures"""
        if not new_architectures:
            return
        
        # Add new architectures
        self.population.extend(new_architectures)
        
        # Prune population if too large
        if len(self.population) > self.config.population_size * 2:
            indices_to_keep = self.diversity_manager.prune_population(
                max_size=self.config.population_size
            )
            
            self.population = [self.population[i] for i in indices_to_keep]
            self.population_performance = [self.population_performance[i] for i in indices_to_keep]
    
    def _log_generation_results(self, generation: int):
        """Log results for current generation"""
        if not self.population_performance:
            return
        
        # Find best architecture
        best_idx = self._find_best_architecture()
        best_perf = self.population_performance[best_idx]
        best_score = self.multi_objective_evaluator.compute_weighted_score(best_perf)
        
        # Calculate diversity metrics
        diversity_guidance = self.diversity_manager.get_diversity_guidance()
        
        # Log generation results
        generation_log = {
            'generation': generation,
            'best_score': best_score,
            'best_performance': best_perf.to_dict(),
            'population_size': len(self.population),
            'diversity_level': diversity_guidance.get('current_diversity', 0.0),
            'concept_drift': self.concept_tracker.concept_history[-1].drift_score if self.concept_tracker.concept_history else 0.0
        }
        
        self.evolution_history.append(generation_log)
        
        print(f"Generation {generation + 1} Results:")
        print(f"  Best Score: {best_score:.4f}")
        print(f"  Best Accuracy: {best_perf.accuracy:.4f}")
        print(f"  Best Latency: {best_perf.latency:.2f}ms")
        print(f"  Diversity: {diversity_guidance.get('current_diversity', 0.0):.3f}")
        print(f"  Population Size: {len(self.population)}")
    
    def _check_convergence(self) -> bool:
        """Check if evolution has converged"""
        if len(self.evolution_history) < 10:
            return False
        
        # Check if best score has plateaued
        recent_scores = [log['best_score'] for log in self.evolution_history[-10:]]
        score_variance = torch.var(torch.tensor(recent_scores)).item()
        
        if score_variance < 0.001:  # Very low variance
            return True
        
        # Check diversity
        diversity_guidance = self.diversity_manager.get_diversity_guidance()
        if diversity_guidance['current_diversity'] < 0.1:  # Very low diversity
            return True
        
        return False
    
    def _create_evolution_result(self, total_time: float) -> EvolutionResult:
        """Create final evolution result"""
        # Find best architecture
        best_idx = self._find_best_architecture()
        best_arch = self.population[best_idx]
        best_perf = self.population_performance[best_idx]
        
        # Create diversity metrics log
        diversity_metrics = []
        for log in self.evolution_history:
            diversity_metrics.append({
                'generation': log['generation'],
                'diversity_level': log['diversity_level']
            })
        
        # Create concept evolution log
        concept_evolution_log = []
        for snapshot in self.concept_tracker.concept_history:
            concept_evolution_log.append({
                'timestamp': snapshot.timestamp,
                'drift_score': snapshot.drift_score,
                'performance_metrics': snapshot.performance_metrics
            })
        
        return EvolutionResult(
            best_architecture=best_arch,
            best_performance=best_perf,
            evolution_history=self.evolution_history,
            diversity_metrics=diversity_metrics,
            concept_evolution_log=concept_evolution_log,
            total_time=total_time,
            generations_completed=self.current_generation + 1
        )
    
    def _save_results(self, result: EvolutionResult):
        """Save evolution results"""
        # Save main result
        result_path = os.path.join(self.work_dir, "evolution_result.json")
        with open(result_path, 'w') as f:
            json.dump({
                'best_architecture': result.best_architecture,
                'best_performance': result.best_performance.to_dict(),
                'total_time': result.total_time,
                'generations_completed': result.generations_completed,
                'stats': self.stats
            }, f, indent=2)
        
        # Save evolution history
        history_path = os.path.join(self.work_dir, "evolution_history.json")
        with open(history_path, 'w') as f:
            json.dump(result.evolution_history, f, indent=2)
        
        # Save concept evolution log
        concept_path = os.path.join(self.work_dir, "concept_evolution.json")
        with open(concept_path, 'w') as f:
            json.dump(result.concept_evolution_log, f, indent=2)
        
        print(f"Results saved to {self.work_dir}")
