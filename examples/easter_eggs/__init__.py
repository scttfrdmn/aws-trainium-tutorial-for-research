"""🥚 Easter Eggs: Creative and Experimental Computing on AWS ML Chips.

This package contains experimental and creative applications that push the
boundaries of what's possible with AWS Trainium and Inferentia chips beyond
their intended machine learning use cases.

⚠️  IMPORTANT DISCLAIMERS:
    - Educational and experimental purposes only
    - Not officially supported by AWS
    - May not comply with AWS terms of service for production use
    - Always check AWS ToS before implementing
    - Performance characteristics may vary significantly
    - No warranties or guarantees provided

Creative Applications:
    - Massively parallel mathematical computations
    - Scientific simulation acceleration
    - Procedural art and creative coding
    - Cryptographic operation exploration (educational only)
    - Signal processing and analysis
    - Financial modeling and Monte Carlo methods

The Philosophy:
    These examples demonstrate that ML chips are essentially very powerful
    tensor processing units that can be creatively applied to various
    computational problems. While designed for machine learning, their
    parallel processing capabilities can accelerate many mathematical
    and scientific computing tasks.

Key Insights:
    1. Tensor operations are more versatile than initially apparent
    2. ML chip cost efficiency enables larger experimental computations
    3. Creative thinking can unlock hidden computational potential
    4. Educational exploration reveals new computing paradigms
    5. Parallel processing patterns apply broadly beyond ML

Import Example:
    from examples.easter_eggs.creative_computing import CreativeNeuronComputing

    computer = CreativeNeuronComputing(device_type='trainium')
    results = computer.run_complete_showcase()

Remember: Always respect AWS terms of service and use these examples
responsibly for educational and experimental purposes only!
"""

__version__ = "1.0.0"
__author__ = "Scott Friedman"

# Creative computing imports
try:
    from .creative_showcase import CreativeShowcase, run_creative_showcase
    from .matrix_operations import MatrixOperationEngine
    from .monte_carlo import MonteCarloEngine
    from .precision_emulation import PrecisionEmulationEngine
except ImportError:
    # Handle missing dependencies gracefully
    pass

__all__ = [
    "CreativeShowcase",
    "run_creative_showcase",
    "MatrixOperationEngine",
    "MonteCarloEngine",
    "PrecisionEmulationEngine",
]
