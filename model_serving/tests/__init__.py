"""
Test Package for High-Performance Model Serving

This package contains comprehensive test suites for:
- TensorRT model conversion and optimization
- FastAPI serving endpoints and middleware
- Intelligent routing strategies
- Distributed tracing with OpenTelemetry
- Dynamic batching and model loading
- Integration tests for complete workflows

Test Organization:
- test_tensorrt.py: TensorRT converter tests
- test_serving.py: FastAPI server and endpoint tests
- test_routing.py: Router and load balancing tests
- test_tracing.py: OpenTelemetry tracing tests
- integration/: End-to-end integration tests

Usage:
    # Run all tests
    pytest tests/

    # Run specific test file
    pytest tests/test_tensorrt.py

    # Run with coverage
    pytest tests/ --cov=src --cov-report=html

    # Run integration tests only
    pytest tests/integration/

Requirements:
    - pytest>=7.0.0
    - pytest-asyncio>=0.21.0
    - pytest-mock>=3.10.0
    - pytest-cov>=4.0.0
    - httpx>=0.24.0
"""

import sys
from pathlib import Path

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

__version__ = "1.0.0"
__all__ = []
