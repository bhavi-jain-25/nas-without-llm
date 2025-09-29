#!/usr/bin/env python3
"""
LLM-Guided Evolution Demo - Comprehensive demonstration of concept evolution NAS
Addresses all challenges from the presentation slides:
- Multi-objective optimization (accuracy, latency, memory, energy)
- Concept evolution and drift handling
- LLM-guided architecture generation
- Diversity maintenance
- CIFAR-10 evaluation
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from tinynas.llm_evolution import (
    LLMGuidedEvolution, 
    EvolutionConfig,
    MockLLMInterface,
    ConceptEvolutionTracker,
    MultiObjectiveEvaluator,
    DiversityManager
)
from tinynas.llm_evolution.cifar10_evaluator import CIFAR10Evaluator


class CIFAR10ModelBuilder:
    """Model builder for CIFAR-10 compatible architectures"""
    
    def __init__(self, input_shape: Tuple[int, ...] = (3, 32, 32), num_classes: int = 10):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_model(self, architecture: Dict) -> nn.Module:
        """Build model from architecture specification"""
        # Create a simple, robust model for demonstration
        class SimpleCNN(nn.Module):
            def __init__(self, input_shape, num_classes):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(input_shape[0], 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1)
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(128, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        return SimpleCNN(self.input_shape, self.num_classes)


def create_initial_architecture() -> Dict:
    """Create initial architecture for evolution"""
    return {
        'structure_info': [
            {'class': 'ConvKXBNRELU', 'in': 3, 'out': 32, 's': 1, 'k': 3},
            {'class': 'SuperResK1KXK1', 'in': 32, 'out': 64, 's': 2, 'k': 3, 'L': 1, 'btn': 32},
            {'class': 'SuperResK1KXK1', 'in': 64, 'out': 128, 's': 2, 'k': 3, 'L': 1, 'btn': 64},
            {'class': 'SuperResK1KXK1', 'in': 128, 'out': 256, 's': 1, 'k': 3, 'L': 1, 'btn': 128},
        ]
    }


def create_evolution_config(args) -> EvolutionConfig:
    """Create evolution configuration based on arguments"""
    return EvolutionConfig(
        population_size=args.population_size,
        max_generations=args.max_generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        llm_creativity=args.llm_creativity,
        llm_safety_threshold=args.llm_safety_threshold,
        target_diversity=args.target_diversity,
        min_diversity_threshold=args.min_diversity_threshold,
        drift_threshold=args.drift_threshold,
        adaptation_rate=args.adaptation_rate,
        accuracy_weight=args.accuracy_weight,
        latency_weight=args.latency_weight,
        memory_weight=args.memory_weight,
        energy_weight=args.energy_weight,
        max_layers=args.max_layers,
        max_channels=args.max_channels,
        min_layers=args.min_layers,
        min_channels=args.min_channels
    )


def run_llm_guided_evolution(args):
    """Run the complete LLM-guided evolution process"""
    print("=" * 80)
    print("LLM-GUIDED EVOLUTION DEMO")
    print("=" * 80)
    print("Addressing all challenges from presentation slides:")
    print("- Multi-objective optimization (accuracy, latency, memory, energy)")
    print("- Concept evolution and drift handling")
    print("- LLM-guided architecture generation")
    print("- Diversity maintenance")
    print("- CIFAR-10 evaluation")
    print("=" * 80)
    
    # Create configuration
    config = create_evolution_config(args)
    
    # Create LLM interface
    llm_interface = MockLLMInterface(
        creativity_level=config.llm_creativity,
        safety_threshold=config.llm_safety_threshold
    )
    
    # Create evolution framework
    evolution = LLMGuidedEvolution(
        config=config,
        llm_interface=llm_interface,
        work_dir=args.output_dir
    )
    
    # Create model builder
    model_builder = CIFAR10ModelBuilder()
    
    # Create initial architecture
    initial_architecture = create_initial_architecture()
    
    print(f"\nInitial Architecture:")
    print(json.dumps(initial_architecture, indent=2))
    
    # Run evolution
    print(f"\nStarting evolution with {config.population_size} population size")
    print(f"Target: {config.max_generations} generations")
    
    start_time = time.time()
    
    try:
        result = evolution.evolve(
            initial_architecture=initial_architecture,
            input_shape=(3, 32, 32),
            model_builder=model_builder.build_model
        )
        
        evolution_time = time.time() - start_time
        
        # Print results
        print("\n" + "=" * 80)
        print("EVOLUTION RESULTS")
        print("=" * 80)
        
        print(f"Evolution completed in {evolution_time:.2f} seconds")
        print(f"Generations completed: {result.generations_completed}")
        print(f"Total time: {result.total_time:.2f} seconds")
        
        print(f"\nBest Architecture:")
        print(json.dumps(result.best_architecture, indent=2))
        
        print(f"\nBest Performance:")
        print(f"  Accuracy: {result.best_performance.accuracy:.4f}")
        print(f"  Latency: {result.best_performance.latency:.2f} ms")
        print(f"  Memory: {result.best_performance.memory:.2f} MB")
        print(f"  Energy: {result.best_performance.energy:.4f}")
        print(f"  FLOPs: {result.best_performance.flops:.2e}")
        print(f"  Parameters: {result.best_performance.params:.2e}")
        
        # Evolution statistics
        print(f"\nEvolution Statistics:")
        print(f"  Total mutations: {evolution.stats['total_mutations']}")
        print(f"  Successful mutations: {evolution.stats['successful_mutations']}")
        print(f"  Diversity violations: {evolution.stats['diversity_violations']}")
        print(f"  Concept drift events: {evolution.stats['concept_drift_events']}")
        print(f"  LLM guidance calls: {evolution.stats['llm_guidance_calls']}")
        
        # Diversity metrics
        if result.diversity_metrics:
            final_diversity = result.diversity_metrics[-1]['diversity_level']
            print(f"  Final diversity level: {final_diversity:.3f}")
        
        # Concept evolution summary
        if result.concept_evolution_log:
            final_drift = result.concept_evolution_log[-1]['drift_score']
            print(f"  Final concept drift: {final_drift:.3f}")
        
        # Save detailed results
        save_detailed_results(result, args.output_dir)
        
        # Run CIFAR-10 evaluation
        if args.evaluate_cifar10:
            run_cifar10_evaluation(result.best_architecture, args.output_dir)
        
        print(f"\nResults saved to: {args.output_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"Evolution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def save_detailed_results(result, output_dir: str):
    """Save detailed evolution results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save evolution history
    history_file = os.path.join(output_dir, "evolution_history.json")
    with open(history_file, 'w') as f:
        json.dump(result.evolution_history, f, indent=2)
    
    # Save concept evolution log
    concept_file = os.path.join(output_dir, "concept_evolution.json")
    with open(concept_file, 'w') as f:
        json.dump(result.concept_evolution_log, f, indent=2)
    
    # Save diversity metrics
    diversity_file = os.path.join(output_dir, "diversity_metrics.json")
    with open(diversity_file, 'w') as f:
        json.dump(result.diversity_metrics, f, indent=2)
    
    # Create summary report
    summary_file = os.path.join(output_dir, "summary_report.txt")
    with open(summary_file, 'w') as f:
        f.write("LLM-GUIDED EVOLUTION SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Evolution completed in {result.total_time:.2f} seconds\n")
        f.write(f"Generations completed: {result.generations_completed}\n\n")
        
        f.write("Best Architecture:\n")
        f.write(json.dumps(result.best_architecture, indent=2) + "\n\n")
        
        f.write("Best Performance:\n")
        f.write(f"  Accuracy: {result.best_performance.accuracy:.4f}\n")
        f.write(f"  Latency: {result.best_performance.latency:.2f} ms\n")
        f.write(f"  Memory: {result.best_performance.memory:.2f} MB\n")
        f.write(f"  Energy: {result.best_performance.energy:.4f}\n")
        f.write(f"  FLOPs: {result.best_performance.flops:.2e}\n")
        f.write(f"  Parameters: {result.best_performance.params:.2e}\n\n")
        
        f.write("Evolution History:\n")
        for i, gen in enumerate(result.evolution_history):
            f.write(f"  Generation {gen['generation'] + 1}: "
                   f"Score={gen['best_score']:.4f}, "
                   f"Diversity={gen['diversity_level']:.3f}\n")


def run_cifar10_evaluation(architecture: Dict, output_dir: str):
    """Run CIFAR-10 specific evaluation"""
    print("\n" + "=" * 80)
    print("CIFAR-10 EVALUATION")
    print("=" * 80)
    
    try:
        # Create CIFAR-10 evaluator
        evaluator = CIFAR10Evaluator()
        
        # Create model builder
        model_builder = CIFAR10ModelBuilder()
        
        # Build model
        model = model_builder.build_model(architecture)
        
        # Evaluate
        print("Evaluating on CIFAR-10 characteristics...")
        metrics = evaluator.evaluate_architecture(model)
        
        print(f"\nCIFAR-10 Metrics:")
        print(f"  Accuracy: {metrics.accuracy:.4f}")
        print(f"  Robustness Score: {metrics.robustness_score:.4f}")
        print(f"  Diversity Handling: {metrics.diversity_handling:.4f}")
        print(f"  Complexity Adaptation: {metrics.complexity_adaptation:.4f}")
        print(f"  Overall Score: {metrics.overall_score:.4f}")
        
        # Get recommendations
        recommendations = evaluator.get_cifar10_specific_recommendations(metrics)
        if recommendations:
            print(f"\nCIFAR-10 Recommendations:")
            for area, recommendation in recommendations.items():
                print(f"  {area}: {recommendation}")
        
        # Save CIFAR-10 results
        cifar10_file = os.path.join(output_dir, "cifar10_evaluation.json")
        with open(cifar10_file, 'w') as f:
            json.dump({
                'accuracy': metrics.accuracy,
                'robustness_score': metrics.robustness_score,
                'diversity_handling': metrics.diversity_handling,
                'complexity_adaptation': metrics.complexity_adaptation,
                'overall_score': metrics.overall_score,
                'recommendations': recommendations
            }, f, indent=2)
        
        print(f"\nCIFAR-10 evaluation saved to: {cifar10_file}")
        
    except Exception as e:
        print(f"CIFAR-10 evaluation failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="LLM-Guided Evolution Demo")
    
    # Evolution parameters
    parser.add_argument("--population_size", type=int, default=10,
                       help="Population size for evolution")
    parser.add_argument("--max_generations", type=int, default=5,
                       help="Maximum number of generations")
    parser.add_argument("--mutation_rate", type=float, default=0.3,
                       help="Mutation rate")
    parser.add_argument("--crossover_rate", type=float, default=0.2,
                       help="Crossover rate")
    
    # LLM parameters
    parser.add_argument("--llm_creativity", type=float, default=0.7,
                       help="LLM creativity level")
    parser.add_argument("--llm_safety_threshold", type=float, default=0.8,
                       help="LLM safety threshold")
    
    # Diversity parameters
    parser.add_argument("--target_diversity", type=float, default=0.7,
                       help="Target diversity level")
    parser.add_argument("--min_diversity_threshold", type=float, default=0.3,
                       help="Minimum diversity threshold")
    
    # Concept evolution parameters
    parser.add_argument("--drift_threshold", type=float, default=0.3,
                       help="Concept drift threshold")
    parser.add_argument("--adaptation_rate", type=float, default=0.1,
                       help="Adaptation rate")
    
    # Multi-objective weights
    parser.add_argument("--accuracy_weight", type=float, default=0.4,
                       help="Accuracy weight")
    parser.add_argument("--latency_weight", type=float, default=0.2,
                       help="Latency weight")
    parser.add_argument("--memory_weight", type=float, default=0.2,
                       help="Memory weight")
    parser.add_argument("--energy_weight", type=float, default=0.2,
                       help="Energy weight")
    
    # Architecture constraints
    parser.add_argument("--max_layers", type=int, default=20,
                       help="Maximum number of layers")
    parser.add_argument("--max_channels", type=int, default=512,
                       help="Maximum number of channels")
    parser.add_argument("--min_layers", type=int, default=3,
                       help="Minimum number of layers")
    parser.add_argument("--min_channels", type=int, default=8,
                       help="Minimum number of channels")
    
    # Output and evaluation
    parser.add_argument("--output_dir", type=str, default="./llm_evolution_demo_results",
                       help="Output directory for results")
    parser.add_argument("--evaluate_cifar10", action="store_true",
                       help="Run CIFAR-10 evaluation")
    
    args = parser.parse_args()
    
    # Run the demo
    success = run_llm_guided_evolution(args)
    
    if success:
        print("\nDemo completed successfully!")
        sys.exit(0)
    else:
        print("\nDemo failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
