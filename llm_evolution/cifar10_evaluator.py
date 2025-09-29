"""
CIFAR-10 Evaluator - Handles CIFAR-10 dataset evaluation for LLM-guided evolution
Addresses the challenges: small resolution, high diversity, pose variations, scale changes, background clutter
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import time
import os


@dataclass
class CIFAR10Metrics:
    """CIFAR-10 specific evaluation metrics"""
    accuracy: float
    robustness_score: float
    diversity_handling: float
    complexity_adaptation: float
    overall_score: float


class CIFAR10Evaluator:
    """
    Evaluates architectures on CIFAR-10 dataset characteristics
    Focuses on robustness to small resolution, high diversity, and challenging conditions
    """
    
    def __init__(self, 
                 data_dir: str = "./data",
                 batch_size: int = 128,
                 num_workers: int = 4,
                 use_augmentation: bool = True):
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_augmentation = use_augmentation
        
        # CIFAR-10 specific parameters
        self.input_shape = (3, 32, 32)
        self.num_classes = 10
        
        # Load datasets
        self.train_loader, self.test_loader = self._load_datasets()
        
        # Evaluation metrics
        self.robustness_transforms = self._create_robustness_transforms()
        
    def _load_datasets(self) -> Tuple[DataLoader, DataLoader]:
        """Load CIFAR-10 datasets with appropriate transforms"""
        # Standard transforms
        if self.use_augmentation:
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        else:
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=test_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        
        return train_loader, test_loader
    
    def _create_robustness_transforms(self) -> List[transforms.Compose]:
        """Create transforms for robustness evaluation"""
        robustness_transforms = [
            # Brightness variations
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Lambda(lambda x: torch.clamp(x * 0.7, 0, 1))  # Darker
            ]),
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Lambda(lambda x: torch.clamp(x * 1.3, 0, 1))  # Brighter
            ]),
            
            # Contrast variations
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Lambda(lambda x: (x - 0.5) * 0.5 + 0.5)  # Low contrast
            ]),
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Lambda(lambda x: (x - 0.5) * 1.5 + 0.5)  # High contrast
            ]),
            
            # Noise addition
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)  # Gaussian noise
            ])
        ]
        
        return robustness_transforms
    
    def evaluate_architecture(self, 
                            model: nn.Module,
                            model_builder: Optional[Callable] = None) -> CIFAR10Metrics:
        """
        Evaluate architecture on CIFAR-10 characteristics
        
        Args:
            model: Neural network model
            model_builder: Optional function to rebuild model for evaluation
            
        Returns:
            CIFAR10Metrics: CIFAR-10 specific metrics
        """
        # Ensure model is compatible with CIFAR-10
        if model_builder:
            model = model_builder(self.input_shape, self.num_classes)
        
        # Set model to evaluation mode
        model.eval()
        
        # Evaluate on standard test set
        accuracy = self._evaluate_accuracy(model)
        
        # Evaluate robustness
        robustness_score = self._evaluate_robustness(model)
        
        # Evaluate diversity handling
        diversity_score = self._evaluate_diversity_handling(model)
        
        # Evaluate complexity adaptation
        complexity_score = self._evaluate_complexity_adaptation(model)
        
        # Calculate overall score
        overall_score = (
            0.4 * accuracy +
            0.3 * robustness_score +
            0.2 * diversity_score +
            0.1 * complexity_score
        )
        
        return CIFAR10Metrics(
            accuracy=accuracy,
            robustness_score=robustness_score,
            diversity_handling=diversity_score,
            complexity_adaptation=complexity_score,
            overall_score=overall_score
        )
    
    def _evaluate_accuracy(self, model: nn.Module) -> float:
        """Evaluate standard accuracy on CIFAR-10 test set"""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = model(images)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return accuracy
    
    def _evaluate_robustness(self, model: nn.Module) -> float:
        """Evaluate robustness to challenging conditions"""
        robustness_scores = []
        
        for transform in self.robustness_transforms:
            # Create transformed dataset
            transformed_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, download=False, transform=transform
            )
            transformed_loader = DataLoader(
                transformed_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
            )
            
            # Evaluate on transformed data
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in transformed_loader:
                    outputs = model(images)
                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            robustness_scores.append(correct / total)
        
        # Average robustness score
        avg_robustness = np.mean(robustness_scores)
        return avg_robustness
    
    def _evaluate_diversity_handling(self, model: nn.Module) -> float:
        """Evaluate how well model handles CIFAR-10's high diversity"""
        # Create diverse test scenarios
        diverse_transforms = [
            # Different scales (simulated by cropping)
            transforms.Compose([
                transforms.RandomCrop(24),  # Smaller
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
            transforms.Compose([
                transforms.CenterCrop(28),  # Medium
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
            
            # Different orientations
            transforms.Compose([
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
            transforms.Compose([
                transforms.RandomRotation(-15),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
            
            # Background clutter simulation
            transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        ]
        
        diversity_scores = []
        
        for transform in diverse_transforms:
            # Create transformed dataset
            transformed_dataset = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=False, download=False, transform=transform
            )
            transformed_loader = DataLoader(
                transformed_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
            )
            
            # Evaluate on transformed data
            correct = 0
            total = 0
            
            with torch.no_grad():
                for images, labels in transformed_loader:
                    outputs = model(images)
                    if isinstance(outputs, (list, tuple)):
                        outputs = outputs[0]
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            diversity_scores.append(correct / total)
        
        # Average diversity handling score
        avg_diversity = np.mean(diversity_scores)
        return avg_diversity
    
    def _evaluate_complexity_adaptation(self, model: nn.Module) -> float:
        """Evaluate how well model adapts to CIFAR-10's complexity"""
        # Measure feature extraction capability
        feature_quality = self._measure_feature_quality(model)
        
        # Measure computational efficiency
        efficiency = self._measure_efficiency(model)
        
        # Measure generalization capability
        generalization = self._measure_generalization(model)
        
        # Combine metrics
        complexity_score = (feature_quality + efficiency + generalization) / 3.0
        return complexity_score
    
    def _measure_feature_quality(self, model: nn.Module) -> float:
        """Measure quality of feature extraction"""
        # Extract features from a sample of test data
        features = []
        labels = []
        
        with torch.no_grad():
            for i, (images, batch_labels) in enumerate(self.test_loader):
                if i >= 10:  # Limit to first 10 batches
                    break
                
                # Get features (before final classification layer)
                if hasattr(model, 'features'):
                    batch_features = model.features(images)
                else:
                    # For models without explicit feature extraction
                    batch_features = images
                
                features.append(batch_features.cpu().numpy())
                labels.append(batch_labels.cpu().numpy())
        
        # Concatenate features
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        # Calculate feature quality metrics
        # 1. Feature variance (higher is better)
        feature_variance = np.var(features)
        
        # 2. Class separability (higher is better)
        class_separability = self._calculate_class_separability(features, labels)
        
        # 3. Feature diversity (higher is better)
        feature_diversity = self._calculate_feature_diversity(features)
        
        # Combine metrics
        quality_score = (
            0.4 * min(feature_variance / 1000, 1.0) +
            0.4 * class_separability +
            0.2 * feature_diversity
        )
        
        return min(quality_score, 1.0)
    
    def _calculate_class_separability(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate class separability in feature space"""
        try:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.metrics import accuracy_score
            
            # Reduce dimensionality for LDA
            if features.shape[1] > 100:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=100)
                features_reduced = pca.fit_transform(features)
            else:
                features_reduced = features
            
            # Apply LDA
            lda = LinearDiscriminantAnalysis()
            lda.fit(features_reduced, labels)
            predictions = lda.predict(features_reduced)
            
            # Calculate separability score
            separability = accuracy_score(labels, predictions)
            return separability
            
        except ImportError:
            # Fallback: simple variance-based separability
            class_means = []
            for class_id in range(self.num_classes):
                class_features = features[labels == class_id]
                if len(class_features) > 0:
                    class_means.append(np.mean(class_features, axis=0))
            
            if len(class_means) > 1:
                # Calculate inter-class variance
                overall_mean = np.mean(class_means, axis=0)
                inter_class_var = np.mean([np.var(mean - overall_mean) for mean in class_means])
                return min(inter_class_var / 100, 1.0)
            
            return 0.5
    
    def _calculate_feature_diversity(self, features: np.ndarray) -> float:
        """Calculate diversity of extracted features"""
        # Calculate feature correlation
        feature_corr = np.corrcoef(features.T)
        
        # Remove diagonal (self-correlation)
        mask = ~np.eye(feature_corr.shape[0], dtype=bool)
        correlations = feature_corr[mask]
        
        # Diversity is inverse of average correlation
        avg_correlation = np.mean(np.abs(correlations))
        diversity = 1.0 - avg_correlation
        
        return max(diversity, 0.0)
    
    def _measure_efficiency(self, model: nn.Module) -> float:
        """Measure computational efficiency"""
        # Measure inference time
        model.eval()
        dummy_input = torch.randn(1, *self.input_shape)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Measure time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(dummy_input)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        
        # Efficiency score (lower time is better)
        efficiency_score = max(0, 1 - avg_time / 0.1)  # Normalize to 0.1s baseline
        return efficiency_score
    
    def _measure_generalization(self, model: nn.Module) -> float:
        """Measure generalization capability"""
        # Compare training and test performance
        train_accuracy = self._evaluate_accuracy_on_split(model, self.train_loader, max_batches=50)
        test_accuracy = self._evaluate_accuracy(model)
        
        # Generalization gap (smaller is better)
        gap = abs(train_accuracy - test_accuracy)
        generalization_score = max(0, 1 - gap / 0.2)  # Normalize to 0.2 gap baseline
        
        return generalization_score
    
    def _evaluate_accuracy_on_split(self, model: nn.Module, data_loader: DataLoader, max_batches: int = None) -> float:
        """Evaluate accuracy on a data split"""
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                if max_batches and i >= max_batches:
                    break
                
                outputs = model(images)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def get_cifar10_specific_recommendations(self, metrics: CIFAR10Metrics) -> Dict[str, str]:
        """Get CIFAR-10 specific recommendations based on metrics"""
        recommendations = {}
        
        if metrics.accuracy < 0.7:
            recommendations['accuracy'] = 'increase_model_capacity_for_small_images'
        
        if metrics.robustness_score < 0.6:
            recommendations['robustness'] = 'add_data_augmentation_and_regularization'
        
        if metrics.diversity_handling < 0.65:
            recommendations['diversity'] = 'improve_feature_extraction_for_high_diversity'
        
        if metrics.complexity_adaptation < 0.6:
            recommendations['complexity'] = 'optimize_architecture_for_cifar10_complexity'
        
        return recommendations
