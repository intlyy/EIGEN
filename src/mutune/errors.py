"""Public exception hierarchy for muTune."""


class MuTuneError(Exception):
    """Base class for all expected muTune failures."""


class ConfigurationError(MuTuneError):
    """Raised when a project configuration or engine profile is invalid."""


class CandidateError(MuTuneError):
    """Raised when a candidate violates its conditional search space."""


class PluginError(MuTuneError):
    """Raised when a requested plugin is missing or incompatible."""


class RunnerError(MuTuneError):
    """Raised when an evaluation runner cannot complete a benchmark."""


class ResultError(MuTuneError):
    """Raised when benchmark output does not satisfy the result contract."""
