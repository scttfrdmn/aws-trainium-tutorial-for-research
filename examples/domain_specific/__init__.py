"""Domain-Specific Machine Learning Examples for AWS Trainium and Inferentia.

This package contains comprehensive examples for three major research domains:
- Climate Science: Weather prediction and climate modeling
- Biomedical Research: Protein analysis and drug discovery
- Social Sciences: Large-scale text analysis and social media research

Each domain demonstrates:
- Custom dataset handling optimized for Trainium/Inferentia
- Domain-specific model architectures
- Cost-efficient research workflows
- Production deployment patterns
- Comprehensive cost tracking for academic budgets

Import examples:
    from examples.domain_specific.climate_science import ClimateTransformer, train_climate_model
    from examples.domain_specific.biomedical import MolecularTransformer, train_biomedical_model
    from examples.domain_specific.social_sciences import MultiTaskSocialAnalyzer, train_social_analysis_model
"""

__version__ = "1.0.0"
__author__ = "Scott Friedman"

# Domain-specific imports
try:
    from .biomedical import (
        MolecularDataset,
        MolecularTransformer,
        train_biomedical_model,
    )
    from .climate_science import ClimateDataset, ClimateTransformer, train_climate_model
    from .social_sciences import (
        MultiTaskSocialAnalyzer,
        SocialMediaDataset,
        train_social_analysis_model,
    )
except ImportError:
    # Handle missing dependencies gracefully for educational environments
    pass

__all__ = [
    # Climate Science
    "ClimateDataset",
    "ClimateTransformer",
    "train_climate_model",
    # Biomedical Research
    "MolecularDataset",
    "MolecularTransformer",
    "train_biomedical_model",
    # Social Sciences
    "SocialMediaDataset",
    "MultiTaskSocialAnalyzer",
    "train_social_analysis_model",
]
