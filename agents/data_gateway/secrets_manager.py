"""
Secrets Manager for Data Gateway Agents

This module provides secure secrets management for external API authentication
in Terra Constellata data gateway agents.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import json

logger = logging.getLogger(__name__)


class SecretsManager(ABC):
    """Abstract base class for secrets management."""

    @abstractmethod
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Retrieve a secret by key."""
        pass

    @abstractmethod
    def set_secret(self, key: str, value: str) -> None:
        """Store a secret."""
        pass

    @abstractmethod
    def has_secret(self, key: str) -> bool:
        """Check if a secret exists."""
        pass


class EnvironmentSecretsManager(SecretsManager):
    """Secrets manager using environment variables."""

    def __init__(self, prefix: str = "TC_"):
        """
        Initialize environment secrets manager.

        Args:
            prefix: Prefix for environment variable names
        """
        self.prefix = prefix

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret from environment variable."""
        env_key = f"{self.prefix}{key.upper()}"
        value = os.getenv(env_key, default)
        if value:
            logger.debug(f"Retrieved secret for key: {key}")
        else:
            logger.warning(f"Secret not found for key: {key}")
        return value

    def set_secret(self, key: str, value: str) -> None:
        """Set environment variable (not recommended for production)."""
        env_key = f"{self.prefix}{key.upper()}"
        os.environ[env_key] = value
        logger.info(f"Set secret for key: {key}")

    def has_secret(self, key: str) -> bool:
        """Check if environment variable exists."""
        env_key = f"{self.prefix}{key.upper()}"
        return env_key in os.environ


class FileSecretsManager(SecretsManager):
    """Secrets manager using encrypted file storage."""

    def __init__(self, secrets_file: str = ".terra_constellata_secrets.json"):
        """
        Initialize file-based secrets manager.

        Args:
            secrets_file: Path to secrets file
        """
        self.secrets_file = secrets_file
        self._secrets: Dict[str, str] = {}
        self._load_secrets()

    def _load_secrets(self):
        """Load secrets from file."""
        try:
            if os.path.exists(self.secrets_file):
                with open(self.secrets_file, 'r') as f:
                    self._secrets = json.load(f)
                logger.info(f"Loaded secrets from {self.secrets_file}")
        except Exception as e:
            logger.error(f"Failed to load secrets file: {e}")
            self._secrets = {}

    def _save_secrets(self):
        """Save secrets to file."""
        try:
            with open(self.secrets_file, 'w') as f:
                json.dump(self._secrets, f, indent=2)
            logger.info(f"Saved secrets to {self.secrets_file}")
        except Exception as e:
            logger.error(f"Failed to save secrets file: {e}")

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret from file storage."""
        value = self._secrets.get(key, default)
        if value:
            logger.debug(f"Retrieved secret for key: {key}")
        else:
            logger.warning(f"Secret not found for key: {key}")
        return value

    def set_secret(self, key: str, value: str) -> None:
        """Store secret in file."""
        self._secrets[key] = value
        self._save_secrets()
        logger.info(f"Set secret for key: {key}")

    def has_secret(self, key: str) -> bool:
        """Check if secret exists in file."""
        return key in self._secrets


class CompositeSecretsManager(SecretsManager):
    """Composite secrets manager that tries multiple sources."""

    def __init__(self, managers: list[SecretsManager]):
        """
        Initialize composite secrets manager.

        Args:
            managers: List of secrets managers to try in order
        """
        self.managers = managers

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Try to get secret from each manager in order."""
        for manager in self.managers:
            try:
                value = manager.get_secret(key)
                if value is not None:
                    return value
            except Exception as e:
                logger.debug(f"Failed to get secret from {manager.__class__.__name__}: {e}")
                continue
        return default

    def set_secret(self, key: str, value: str) -> None:
        """Set secret in the first manager that supports it."""
        for manager in self.managers:
            try:
                manager.set_secret(key, value)
                return
            except Exception as e:
                logger.debug(f"Failed to set secret in {manager.__class__.__name__}: {e}")
                continue
        raise RuntimeError("Failed to set secret in any manager")

    def has_secret(self, key: str) -> bool:
        """Check if secret exists in any manager."""
        return any(manager.has_secret(key) for manager in self.managers)


# Global secrets manager instance
def get_secrets_manager() -> SecretsManager:
    """Get the configured secrets manager."""
    # Try environment variables first, then file-based
    managers = [
        EnvironmentSecretsManager(),
        FileSecretsManager(),
    ]
    return CompositeSecretsManager(managers)


# Default instance
secrets_manager = get_secrets_manager()


def resolve_secret_placeholder(placeholder: str) -> Optional[str]:
    """
    Resolve a secret placeholder like {{SECRETS.API_KEY}}.

    Args:
        placeholder: Placeholder string

    Returns:
        Resolved secret value or None
    """
    if not placeholder.startswith("{{SECRETS.") or not placeholder.endswith("}}"):
        return placeholder

    # Extract key from {{SECRETS.KEY}}
    key = placeholder[11:-2]  # Remove {{SECRETS. and }}
    return secrets_manager.get_secret(key)


def resolve_agent_secrets(agent_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve all secret placeholders in agent configuration.

    Args:
        agent_config: Agent configuration dictionary

    Returns:
        Configuration with secrets resolved
    """
    resolved_config = {}

    for key, value in agent_config.items():
        if isinstance(value, str):
            resolved_config[key] = resolve_secret_placeholder(value)
        elif isinstance(value, dict):
            resolved_config[key] = resolve_agent_secrets(value)
        elif isinstance(value, list):
            resolved_config[key] = [
                resolve_secret_placeholder(item) if isinstance(item, str) else item
                for item in value
            ]
        else:
            resolved_config[key] = value

    return resolved_config