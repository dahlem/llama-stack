# llama_stack/providers/remote/eval/lm_harness/config.py
from pydantic import BaseModel, Field
from typing import Optional

class LMHarnessEvalConfig(BaseModel):
    """Configuration for LM Evaluation Harness remote provider."""
    api_endpoint: str = Field(
        default="http://localhost:8000",
        description="LM Evaluation Harness API endpoint",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication if required",
    )
    timeout: int = Field(
        default=300,
        description="Timeout in seconds for API calls",
    )