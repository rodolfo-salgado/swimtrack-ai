class ServiceError(Exception):
    status_code = 500
    code = "service_error"

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class NotReadyError(ServiceError):
    status_code = 503
    code = "detector_not_ready"


class SessionNotFoundError(ServiceError):
    status_code = 404
    code = "session_not_found"


class SessionCapacityError(ServiceError):
    status_code = 503
    code = "session_capacity_reached"


class ConflictError(ServiceError):
    status_code = 409
    code = "batch_conflict"


class PayloadTooLargeError(ServiceError):
    status_code = 413
    code = "payload_too_large"


class InvalidFrameError(ServiceError):
    status_code = 422
    code = "invalid_frame"
