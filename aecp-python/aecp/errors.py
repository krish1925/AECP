"""
AECP Error Handling & Resilience

Comprehensive error hierarchy and resilience patterns:
- Custom exception types for all failure modes
- Circuit breaker for failing connections
- Retry policies with exponential backoff
- Graceful degradation to English text
"""

import time
import threading
import logging
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from dataclasses import dataclass, field
from functools import wraps

logger = logging.getLogger("aecp.errors")

T = TypeVar("T")


# ── Exception Hierarchy ──────────────────────────────────────────────

class AECPError(Exception):
    """Base exception for all AECP errors."""

    def __init__(self, message: str, recoverable: bool = True, context: Optional[Dict] = None):
        self.message = message
        self.recoverable = recoverable
        self.context = context or {}
        super().__init__(message)


class CalibrationError(AECPError):
    """Raised when calibration fails."""
    pass


class TransferError(AECPError):
    """Raised when embedding transfer fails."""
    pass


class NegotiationError(AECPError):
    """Raised when protocol negotiation fails."""
    pass


class AdapterError(AECPError):
    """Raised when an embedding adapter fails."""
    pass


class MatrixExpiredError(TransferError):
    """Raised when a transfer matrix has expired."""

    def __init__(self, agent_pair: str, expired_at: str):
        super().__init__(
            f"Transfer matrix for {agent_pair} expired at {expired_at}. "
            "Recalibration required.",
            recoverable=True,
            context={"agent_pair": agent_pair, "expired_at": expired_at},
        )


class QualityBelowThresholdError(CalibrationError):
    """Raised when quality drops below threshold."""

    def __init__(self, quality: float, threshold: float):
        super().__init__(
            f"Quality {quality:.4f} is below threshold {threshold:.4f}. "
            "Consider recalibrating with a larger vocabulary.",
            recoverable=True,
            context={"quality": quality, "threshold": threshold},
        )


class AgentNotCalibratedError(TransferError):
    """Raised when trying to transfer without calibration."""

    def __init__(self, source: str, target: str):
        super().__init__(
            f"No calibration found between '{source}' and '{target}'. "
            "Call calibrate() or calibrate_with() first.",
            recoverable=True,
            context={"source": source, "target": target},
        )


class CircuitOpenError(AECPError):
    """Raised when circuit breaker is open."""

    def __init__(self, agent_id: str, failures: int, reset_at: float):
        import datetime
        reset_str = datetime.datetime.fromtimestamp(reset_at).strftime("%H:%M:%S")
        super().__init__(
            f"Circuit breaker OPEN for agent '{agent_id}' after {failures} failures. "
            f"Will retry at {reset_str}. Falling back to English text.",
            recoverable=True,
            context={"agent_id": agent_id, "failures": failures, "reset_at": reset_at},
        )


# ── Circuit Breaker ─────────────────────────────────────────────────

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for AECP connections.

    Prevents cascading failures by temporarily stopping requests
    to a failing service and falling back to English text.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests are rejected (fallback to text)
    - HALF_OPEN: Testing recovery, one request allowed through

    Example:
        >>> cb = CircuitBreaker(failure_threshold=3, reset_timeout=30)
        >>> try:
        ...     result = cb.call(risky_function, arg1, arg2)
        ... except CircuitOpenError:
        ...     # Fall back to English text
        ...     result = send_as_text(message)
    """
    failure_threshold: int = 5
    reset_timeout: float = 60.0  # seconds
    half_open_max_calls: int = 1

    # Internal state
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float = field(default=0.0, init=False)
    _half_open_calls: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _agent_id: str = field(default="unknown", init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for timeout transitions."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.reset_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info(
                        f"Circuit breaker for '{self._agent_id}' transitioning to HALF_OPEN"
                    )
            return self._state

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            CircuitOpenError: If circuit is open
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitOpenError(
                self._agent_id,
                self._failure_count,
                self._last_failure_time + self.reset_timeout,
            )

        if current_state == CircuitState.HALF_OPEN:
            with self._lock:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitOpenError(
                        self._agent_id,
                        self._failure_count,
                        self._last_failure_time + self.reset_timeout,
                    )
                self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._failure_count = 0
            self._success_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.info(
                    f"Circuit breaker for '{self._agent_id}' recovered (CLOSED)"
                )

    def _on_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker for '{self._agent_id}' re-opened after half-open failure"
                )
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    f"Circuit breaker for '{self._agent_id}' OPENED "
                    f"after {self._failure_count} failures"
                )

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "agent_id": self._agent_id,
            "failure_threshold": self.failure_threshold,
            "reset_timeout": self.reset_timeout,
        }


# ── Retry Policy ─────────────────────────────────────────────────────

@dataclass
class RetryPolicy:
    """
    Configurable retry policy with exponential backoff.

    Example:
        >>> policy = RetryPolicy(max_retries=3, base_delay=1.0)
        >>> result = policy.execute(risky_function, arg1, arg2)
    """
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            The last exception if all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except self.retryable_exceptions as e:
                last_error = e

                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay,
                    )

                    if self.jitter:
                        import random
                        delay *= (0.5 + random.random())

                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries} after {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)

        raise last_error  # type: ignore


# ── Graceful Degradation ─────────────────────────────────────────────

class GracefulDegradation:
    """
    Handles graceful degradation from AECP to English text communication.

    When AECP is unavailable (circuit open, calibration failed, agent doesn't
    support AECP), this class provides seamless fallback to plain English
    text communication.

    Example:
        >>> degradation = GracefulDegradation()
        >>> result = degradation.send(
        ...     sender=agent1,
        ...     receiver=agent2,
        ...     message="Process this data",
        ...     aecp_func=lambda: agent1.transfer_to(agent2.agent_id, "Process this data"),
        ... )
        >>> # If AECP fails, result will be plain English text
        >>> print(result)
        {'method': 'text', 'message': 'Process this data', 'language': 'en'}
    """

    def __init__(
        self,
        default_language: str = "en",
        log_fallbacks: bool = True,
    ):
        self.default_language = default_language
        self.log_fallbacks = log_fallbacks
        self._fallback_count = 0
        self._aecp_count = 0

    def send(
        self,
        message: str,
        aecp_func: Optional[Callable] = None,
        fallback_metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Try AECP first, fall back to English text if unavailable.

        Args:
            message: Message to send
            aecp_func: Optional AECP transfer function
            fallback_metadata: Additional metadata for fallback

        Returns:
            Dictionary with either AECP transfer or plain text result
        """
        if aecp_func is not None:
            try:
                result = aecp_func()
                self._aecp_count += 1
                return {
                    "method": "aecp",
                    "data": result,
                    "language": None,  # Embeddings are language-agnostic
                }
            except (AECPError, Exception) as e:
                if self.log_fallbacks:
                    logger.info(
                        f"AECP unavailable ({type(e).__name__}: {e}), "
                        f"falling back to English text"
                    )

        # Fallback to English text
        self._fallback_count += 1
        return self._create_text_fallback(message, fallback_metadata)

    def _create_text_fallback(
        self,
        message: str,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Create a text fallback response."""
        result = {
            "method": "text",
            "message": message,
            "language": self.default_language,
            "encoding": "utf-8",
            "fallback": True,
            "note": (
                "AECP embedding transfer unavailable. "
                "Message sent as plain English text. "
                "Both agents can understand this format natively."
            ),
        }
        if metadata:
            result["metadata"] = metadata
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get fallback statistics."""
        total = self._aecp_count + self._fallback_count
        return {
            "aecp_count": self._aecp_count,
            "fallback_count": self._fallback_count,
            "total": total,
            "aecp_percentage": (
                self._aecp_count / total * 100 if total > 0 else 0
            ),
            "fallback_percentage": (
                self._fallback_count / total * 100 if total > 0 else 0
            ),
        }


# ── Decorator for Error Handling ─────────────────────────────────────

def handle_aecp_errors(
    fallback_to_text: bool = True,
    default_message: str = "",
    log_errors: bool = True,
):
    """
    Decorator that wraps AECP operations with error handling.

    If the operation fails and fallback_to_text is True, returns
    a plain English text response instead of raising an exception.

    Args:
        fallback_to_text: Whether to fall back to text on error
        default_message: Default message for text fallback
        log_errors: Whether to log errors

    Example:
        >>> @handle_aecp_errors(fallback_to_text=True)
        ... def transfer_data(agent, target, message):
        ...     return agent.transfer_to(target.agent_id, message)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except CircuitOpenError as e:
                if log_errors:
                    logger.warning(f"Circuit open: {e}")
                if fallback_to_text:
                    msg = kwargs.get("message", default_message) or str(args[2] if len(args) > 2 else "")
                    return {
                        "method": "text",
                        "message": msg,
                        "language": "en",
                        "fallback": True,
                        "reason": str(e),
                    }
                raise
            except AECPError as e:
                if log_errors:
                    logger.error(f"AECP error: {e}")
                if fallback_to_text and e.recoverable:
                    msg = kwargs.get("message", default_message) or str(args[2] if len(args) > 2 else "")
                    return {
                        "method": "text",
                        "message": msg,
                        "language": "en",
                        "fallback": True,
                        "reason": str(e),
                    }
                raise
            except Exception as e:
                if log_errors:
                    logger.error(f"Unexpected error in AECP operation: {e}")
                if fallback_to_text:
                    msg = kwargs.get("message", default_message) or str(args[2] if len(args) > 2 else "")
                    return {
                        "method": "text",
                        "message": msg,
                        "language": "en",
                        "fallback": True,
                        "reason": f"Unexpected error: {e}",
                    }
                raise
        return wrapper
    return decorator


__all__ = [
    # Exceptions
    "AECPError",
    "CalibrationError",
    "TransferError",
    "NegotiationError",
    "AdapterError",
    "MatrixExpiredError",
    "QualityBelowThresholdError",
    "AgentNotCalibratedError",
    "CircuitOpenError",
    # Resilience
    "CircuitBreaker",
    "CircuitState",
    "RetryPolicy",
    "GracefulDegradation",
    # Utilities
    "handle_aecp_errors",
]
