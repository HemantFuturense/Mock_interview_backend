from fastapi import HTTPException


class NotFoundException(HTTPException):
    def __init__(self, detail: str = "Resource not found."):
        super().__init__(status_code=404, detail=detail)


class BadRequestException(HTTPException):
    def __init__(self, detail: str = "Bad request."):
        super().__init__(status_code=400, detail=detail)


class UnauthorizedException(HTTPException):
    def __init__(self, detail: str = "Unauthorized access."):
        super().__init__(status_code=401, detail=detail)


class ForbiddenException(HTTPException):
    def __init__(self, detail: str = "Access forbidden."):
        super().__init__(status_code=403, detail=detail)


class DatabaseException(HTTPException):
    def __init__(self, detail: str = "Database operation failed."):
        super().__init__(status_code=500, detail=detail)


class AIExecutionException(HTTPException):
    def __init__(self, detail: str = "AI service generation failed."):
        super().__init__(status_code=503, detail=detail)
