# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional
from pydantic import BaseModel, Field
from llama_models.schema_utils import json_schema_type


@json_schema_type
class GuardrailsConfig(BaseModel):
    """Configuration for the Guardrails.ai safety provider."""
    api_endpoint: str = Field(
        default="https://api.guardrails.ai",
        description="Guardrails.ai API endpoint",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Guardrails.ai API key. If not provided, will use GUARDRAILS_API_KEY environment variable.",
    )
    timeout: int = Field(
        default=30,
        description="Timeout in seconds for API calls",
    )