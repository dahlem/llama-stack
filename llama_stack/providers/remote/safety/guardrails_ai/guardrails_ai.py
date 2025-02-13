# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
from typing import Any, Dict, List

from llama_stack.apis.inference import Message
from llama_stack.apis.safety import (
    RunShieldResponse,
    Safety,
    SafetyViolation,
    ViolationLevel,
)
from llama_stack.apis.shields import Shield
from llama_stack.providers.datatypes import ShieldsProtocolPrivate

from .config import GuardrailsConfig


logger = logging.getLogger(__name__)


class GuardrailsAdapter(Safety, ShieldsProtocolPrivate):
    def __init__(self, config: GuardrailsConfig) -> None:
        self.config = config
        self.registered_shields = []
        self.client = None

    async def initialize(self) -> None:
        try:
            from guardrails.client import Client
            self.client = Client(
                api_key=self.config.api_key,
                api_endpoint=self.config.api_endpoint,
            )
        except Exception as e:
            raise RuntimeError("Error initializing GuardrailsAdapter") from e

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        """Register a shield with the guardrails provider."""
        # Store the shield configuration for later use
        self.registered_shields.append(shield)

    async def run_shield(
        self, shield_id: str, messages: List[Message], params: Dict[str, Any] = None
    ) -> RunShieldResponse:
        shield = await self.shield_store.get_shield(shield_id)
        if not shield:
            raise ValueError(f"Shield {shield_id} not found")

        # Combine all messages into a single string for validation
        text = "\n".join(msg.content for msg in messages if msg.content)
        
        try:
            # Validate the text using guardrails remote API
            response = await self.client.validate_async(
                rail_string=shield.content,
                value=text,
                metadata=shield.metadata,
                timeout=self.config.timeout,
            )
            
            if response.valid:
                return RunShieldResponse()
            
            return RunShieldResponse(
                violation=SafetyViolation(
                    violation_level=ViolationLevel.ERROR,
                    user_message=response.error_message,
                    metadata={
                        "validation_results": response.validation_results,
                        "error_type": "guardrails_validation_error",
                    },
                )
            )
        except Exception as e:
            logger.error(f"Error running guardrails shield: {e}")
            return RunShieldResponse(
                violation=SafetyViolation(
                    violation_level=ViolationLevel.ERROR,
                    user_message=str(e),
                    metadata={"error_type": type(e).__name__},
                )
            )