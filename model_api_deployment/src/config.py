"""
Configuration Management Module

This module handles all application configuration including environment
variables, defaults, and validation.

Author: AI Infrastructure Curriculum
License: MIT
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# TODO: Load environment variables from .env file
# HINT: Use load_dotenv() to read .env file in project root
# This allows you to set configuration values without hardcoding them


# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class Config:
    """
    Application configuration class.

    Loads configuration from environment variables with sensible defaults.
    All configuration values should be accessed through this class to
    ensure consistency and ease of testing.

    Example:
        >>> config = Config()
        >>> print(config.MODEL_NAME)
        'resnet50'
        >>> print(config.PORT)
        5000
    """

    # =========================================================================
    # Model Configuration
    # =========================================================================

    # TODO: Define MODEL_NAME configuration
    # - Should read from environment variable 'MODEL_NAME'
    # - Default to 'resnet50' if not set
    # - Valid values: 'resnet50', 'mobilenet_v2'
    # - Type: str
    MODEL_NAME: str = os.getenv('MODEL_NAME', 'resnet50')
    
    # TODO: Define MODEL_PATH configuration
    # - Should read from environment variable 'MODEL_PATH'
    # - Default to '~/.cache/torch/hub' (standard PyTorch cache location)
    # - This is where model weights will be cached
    # - Type: str
    MODEL_PATH: str = os.path.expanduser(
        os.getenv('MODEL_PATH', '~/.cache/torch/hub')
    )

    # TODO: Define DEVICE configuration
    # - Should read from environment variable 'DEVICE'
    # - Default to 'cpu' (GPU support comes later)
    # - Valid values: 'cpu', 'cuda', 'mps' (for Apple Silicon)
    # - Type: str
    DEVICE: str = os.getenv('DEVICE', 'cpu')
    
    # =========================================================================
    # API Configuration
    # =========================================================================

    # TODO: Define HOST configuration
    # - Should read from environment variable 'HOST'
    # - Default to '0.0.0.0' (listen on all network interfaces)
    # - For local development only, you might use '127.0.0.1'
    # - Type: str
    HOST: str = os.getenv('HOST', '0.0.0.0')

    # TODO: Define PORT configuration
    # - Should read from environment variable 'PORT'
    # - Default to 5000
    # - Must be converted to integer (env vars are strings!)
    # - Type: int
    # HINT: Use int(os.getenv('PORT', '5000')) for type conversion
    PORT: int = int(os.getenv('PORT', '5000'))
    
    # TODO: Define DEBUG configuration
    # - Should read from environment variable 'DEBUG'
    # - Default to False
    # - Must be converted to boolean
    # - Type: bool
    # HINT: Check if string value is 'true' or '1'
    DEBUG: bool = os.getenv('DEBUG', 'false').lower() in ('true', '1', 'yes', 'on')

    # TODO: Define API_VERSION configuration
    # - Should read from environment variable 'API_VERSION'
    # - Default to '1.0.0'
    # - Follows semantic versioning (MAJOR.MINOR.PATCH)
    # - Type: str
    API_VERSION: str = os.getenv('API_VERSION', '1.0.0')

    # =========================================================================
    # Request Limits
    # =========================================================================

    # TODO: Define MAX_FILE_SIZE configuration
    # - Should read from environment variable 'MAX_FILE_SIZE'
    # - Default to 10MB (10 * 1024 * 1024 bytes)
    # - Must be converted to integer
    # - This prevents DOS attacks via huge file uploads
    # - Type: int
    MAX_FILE_SIZE: int = int(os.getenv('MAX_FILE_SIZE', str(10 * 1024 * 1024)))
   
    # TODO: Define MAX_IMAGE_DIMENSION configuration
    # - Should read from environment variable 'MAX_IMAGE_DIMENSION'
    # - Default to 4096 (pixels)
    # - Must be converted to integer
    # - Prevents memory issues from extremely large images
    # - Type: int
    MAX_IMAGE_DIMENSION: int = int(os.getenv('MAX_IMAGE_DIMENSION', '4096'))
    
    # TODO: Define REQUEST_TIMEOUT configuration
    # - Should read from environment variable 'REQUEST_TIMEOUT'
    # - Default to 30 seconds
    # - Must be converted to integer
    # - Type: int
    REQUEST_TIMEOUT: int = int(os.getenv('REQUEST_TIMEOUT', '30'))
    
    # TODO: Define DEFAULT_TOP_K configuration
    # - Should read from environment variable 'DEFAULT_TOP_K'
    # - Default to 5 (return top-5 predictions)
    # - Must be converted to integer
    # - Type: int
    DEFAULT_TOP_K: int = int(os.getenv('DEFAULT_TOP_K', '5'))

    # TODO: Define MAX_TOP_K configuration
    # - Should read from environment variable 'MAX_TOP_K'
    # - Default to 10 (maximum predictions to return)
    # - Must be converted to integer
    # - Prevents abuse by limiting result size
    # - Type: int
    MAX_TOP_K: int = int(os.getenv('MAX_TOP_K', '10'))

    # =========================================================================
    # Logging Configuration
    # =========================================================================

    # TODO: Define LOG_LEVEL configuration
    # - Should read from environment variable 'LOG_LEVEL'
    # - Default to 'INFO'
    # - Valid values: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    # - Type: str
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')

    # TODO: Define LOG_FORMAT configuration
    # - Should read from environment variable 'LOG_FORMAT'
    # - Default to 'json'
    # - Valid values: 'json', 'text'
    # - JSON format is better for production log aggregation
    # - Type: str
    LOG_FORMAT: str = os.getenv('LOG_FORMAT', 'json')

    # =========================================================================
    # ImageNet Labels
    # =========================================================================

    # TODO: Define IMAGENET_LABELS_URL configuration
    # - URL to download ImageNet class labels
    # - Default to a reliable source (see resources in project docs)
    # - Type: str
    # Example: 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    IMAGENET_LABELS_URL: str = os.getenv(
        'IMAGENET_LABELS_URL',
        'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def __init__(self):
        """
        Initialize configuration.

        TODO: Implement initialization logic
        - Call load_dotenv() to load .env file
        - Optionally validate configuration values
        - Log configuration (excluding sensitive values)
        """
        # TODO: Load environment variables from .env file
        self.validate()
        logger.info(f"Configuration loaded: {self}")

    @classmethod
    def validate(cls) -> bool:
        """
        Validate configuration values.

        TODO: Implement validation logic
        - Check MODEL_NAME is valid ('resnet50' or 'mobilenet_v2')
        - Check PORT is in valid range (1024-65535 for non-root)
        - Check LOG_LEVEL is valid
        - Check numeric values are positive
        - Return True if valid, raise ValueError if not

        Returns:
            True if all configuration is valid

        Raises:
            ValueError: If any configuration value is invalid

        Example:
            >>> config = Config()
            >>> config.validate()
            True
        """
        # TODO: Implement validation
        # Example validation:
        # if cls.MODEL_NAME not in ['resnet50', 'mobilenet_v2']:
        #     raise ValueError(f"Invalid MODEL_NAME: {cls.MODEL_NAME}")
        # Validate model name

        valid_models = ['resnet50', 'mobilenet_v2']
        if cls.MODEL_NAME not in valid_models:
            raise ValueError(
                f"Invalid MODEL_NAME: {cls.MODEL_NAME}. "
                f"Must be one of {valid_models}"
            )
        
        # Validate port
        if not (1024 <= cls.PORT <= 65535):
            raise ValueError(
                f"Invalid PORT: {cls.PORT}. "
                f"Must be between 1024 and 65535"
            )
        
        # Validate log level
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if cls.LOG_LEVEL not in valid_levels:
            raise ValueError(
                f"Invalid LOG_LEVEL: {cls.LOG_LEVEL}. "
                f"Must be one of {valid_levels}"
            )
        
        # Validate numeric values are positive
        if cls.MAX_FILE_SIZE <= 0:
            raise ValueError("MAX_FILE_SIZE must be positive")
        if cls.MAX_IMAGE_DIMENSION <= 0:
            raise ValueError("MAX_IMAGE_DIMENSION must be positive")
        if cls.REQUEST_TIMEOUT <= 0:
            raise ValueError("REQUEST_TIMEOUT must be positive")
        if cls.DEFAULT_TOP_K <= 0 or cls.DEFAULT_TOP_K > cls.MAX_TOP_K:
            raise ValueError(
                f"DEFAULT_TOP_K must be between 1 and {cls.MAX_TOP_K}"
            )
        
        return True

    @classmethod
    def to_dict(cls) -> dict:
        """
        Convert configuration to dictionary.

        TODO: Implement conversion to dict
        - Return all configuration values as a dictionary
        - Useful for logging and debugging
        - Exclude sensitive values if any (API keys, secrets, etc.)

        Returns:
            Dictionary of configuration values

        Example:
            >>> config = Config()
            >>> config_dict = config.to_dict()
            >>> print(config_dict['MODEL_NAME'])
            'resnet50'
        """
        # TODO: Implement conversion
        # HINT: Use {k: v for k, v in vars(cls).items() if not k.startswith('_')}
        return {
            key: value
            for key, value in vars(cls).items()
            if not key.startswith('_') and key.isupper()
        }

    def __repr__(self) -> str:
        """
        String representation of configuration.

        TODO: Implement __repr__
        - Return a readable string showing key configuration values
        - Don't include all values (too verbose)
        - Include: MODEL_NAME, PORT, LOG_LEVEL

        Example:
            >>> config = Config()
            >>> print(config)
            Config(MODEL_NAME='resnet50', PORT=5000, LOG_LEVEL='INFO')
        """
        # TODO: Implement __repr__
        return (
            f"Config(MODEL_NAME='{self.MODEL_NAME}', "
            f"PORT={self.PORT}, LOG_LEVEL='{self.LOG_LEVEL}')"
        )


# =========================================================================
# Helper Functions
# =========================================================================

def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get boolean value from environment variable.

    TODO: Implement boolean conversion
    - Get environment variable value
    - Convert to boolean
    - Return default if not set
    - Consider 'true', '1', 'yes', 'on' as True (case-insensitive)
    - All other values as False

    Args:
        key: Environment variable name
        default: Default value if variable not set

    Returns:
        Boolean value

    Example:
        >>> os.environ['DEBUG'] = 'true'
        >>> get_env_bool('DEBUG', False)
        True
        >>> get_env_bool('NONEXISTENT', False)
        False
    """
    # TODO: Implement boolean conversion
    # HINT: value = os.getenv(key, '').lower()
    #       return value in ('true', '1', 'yes', 'on')
    pass


def get_env_int(key: str, default: int) -> int:
    """
    Get integer value from environment variable.

    TODO: Implement integer conversion
    - Get environment variable value
    - Convert to integer
    - Return default if not set or invalid
    - Handle ValueError for non-numeric values

    Args:
        key: Environment variable name
        default: Default value if variable not set or invalid

    Returns:
        Integer value

    Example:
        >>> os.environ['PORT'] = '8080'
        >>> get_env_int('PORT', 5000)
        8080
        >>> get_env_int('NONEXISTENT', 5000)
        5000
    """
    # TODO: Implement integer conversion
    # HINT: Use try/except to handle ValueError
    pass


# =========================================================================
# Module-level Configuration Instance
# =========================================================================

# TODO: Create a global configuration instance
# - Uncomment the line below after implementing Config class
# - This allows importing as: from config import config
config = Config()


# =========================================================================
# Usage Example (for testing during development)
# =========================================================================

if __name__ == "__main__":
    """
    Test configuration loading.

    Run this file directly to test your implementation:
    $ python config.py
    """
    # TODO: Test your implementation
    # Example:
    # config = Config()
    # print(f"Model: {config.MODEL_NAME}")
    # print(f"Port: {config.PORT}")
    # print(f"Debug: {config.DEBUG}")
    # print(f"Configuration: {config}")
    # config.validate()
    # print("Configuration is valid!")
    print("Testing Configuration...")
    print(f"Config: {config}")
    print(f"\nAll settings:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    print("\n✓ Configuration is valid!")
