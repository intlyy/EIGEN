"""Public exception hierarchy for EIGEN."""


class EIGENError(Exception):
    """Base class for all expected EIGEN failures."""


class ConfigurationError(EIGENError):
    """Raised when a project configuration or engine profile is invalid."""


class CandidateError(EIGENError):
    """Raised when a candidate violates its conditional search space."""


class PluginError(EIGENError):
    """Raised when a requested plugin is missing or incompatible."""


class RunnerError(EIGENError):
    """Raised when an evaluation runner cannot complete a benchmark."""


class ResultError(EIGENError):
    """Raised when benchmark output does not satisfy the result contract."""
