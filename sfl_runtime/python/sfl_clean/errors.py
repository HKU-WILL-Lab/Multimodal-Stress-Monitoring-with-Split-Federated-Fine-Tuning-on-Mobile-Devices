"""Domain errors translated to actionable RPC status codes."""


class SflError(Exception):
    """Base class for expected, user-actionable failures."""


class InvalidRequest(SflError, ValueError):
    """A request or tensor violates the wire contract."""


class StateConflict(SflError):
    """The operation is valid in general, but not in the current state."""


class DuplicateUpdate(SflError):
    """A client attempted a second, conflicting operation for one round."""


class NotBootstrapped(SflError):
    """No initial prefix state has been installed yet."""

