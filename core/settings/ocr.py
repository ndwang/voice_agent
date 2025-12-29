from pydantic import BaseModel


class OCRSettings(BaseModel):
    """OCR service configuration"""
    host: str = "0.0.0.0"
    port: int = 8004
    language: str = "ch"
    interval_ms: int = 1000
    texts_storage_file_prefix: str = "ocr_detected_texts"

    @classmethod
    def from_dict(cls, data: dict) -> "OCRSettings":
        """Create OCRSettings from config dict"""
        return cls(**data)
