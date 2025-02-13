# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import pytest
import subprocess
import time
from pathlib import Path

from llama_stack.apis.inference import UserMessage
from llama_stack.apis.safety import ViolationLevel
from llama_stack.apis.shields import Shield
from llama_stack.providers.remote.safety.guardrails_ai.config import GuardrailsConfig
from llama_stack.providers.remote.safety.guardrails_ai.guardrails_ai import GuardrailsAdapter


@pytest.fixture(scope="session")
def guardrails_config():
    """Create a temporary config file and start the guardrails server."""
    # Create a temporary directory for guardrails config
    tmp_dir = Path("test_guardrails_config")
    tmp_dir.mkdir(exist_ok=True)
    
    # Create config.py with a test guard
    config_path = tmp_dir / "config.py"
    config_content = """
from guardrails import Guard
from guardrails.hub import RegexMatch

# Create a guard that checks for sensitive information like credit card numbers
sensitive_info_guard = Guard(
    name="sensitive-info",
    description="Checks for sensitive information like credit card numbers"
).use(
    RegexMatch(
        regex=r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card number pattern
        on_fail="Text contains sensitive information (credit card number)"
    )
)
"""
    config_path.write_text(config_content)
    
    # Start the guardrails server
    server_process = subprocess.Popen(
        ["guardrails", "start", "--config", str(config_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(5)
    
    yield GuardrailsConfig(
        api_endpoint="http://localhost:8000",
    )
    
    # Cleanup
    server_process.terminate()
    server_process.wait()


@pytest.fixture
def shield():
    return Shield(
        identifier="test-sensitive-info",
        content="""
        <rail>
        <output>
            <regex_match>
                <pattern>\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b</pattern>
                <on_fail>Text contains sensitive information (credit card number)</on_fail>
            </regex_match>
        </output>
        </rail>
        """,
        metadata={"name": "sensitive-info"},
    )


@pytest.mark.guardrails
class TestGuardrailsAI:
    @pytest.mark.asyncio
    async def test_initialization(self, guardrails_config):
        """Test that the adapter initializes correctly with the test server."""
        adapter = GuardrailsAdapter(guardrails_config)
        await adapter.initialize()
        assert adapter.client is not None

    @pytest.mark.asyncio
    async def test_safe_content(self, guardrails_config, shield):
        """Test content without sensitive information."""
        adapter = GuardrailsAdapter(guardrails_config)
        await adapter.initialize()
        
        # Mock the shield store
        adapter.shield_store = AsyncMock()
        adapter.shield_store.get_shield.return_value = shield
        
        response = await adapter.run_shield(
            shield_id="test-sensitive-info",
            messages=[
                UserMessage(content="This is a safe message without sensitive information."),
            ],
        )
        assert response.violation is None

    @pytest.mark.asyncio
    async def test_unsafe_content(self, guardrails_config, shield):
        """Test content with sensitive information (credit card number)."""
        adapter = GuardrailsAdapter(guardrails_config)
        await adapter.initialize()
        
        # Mock the shield store
        adapter.shield_store = AsyncMock()
        adapter.shield_store.get_shield.return_value = shield
        
        response = await adapter.run_shield(
            shield_id="test-sensitive-info",
            messages=[
                UserMessage(content="My credit card number is 4532 1234 5678 9012"),
            ],
        )
        
        assert response.violation is not None
        assert response.violation.violation_level == ViolationLevel.ERROR
        assert "sensitive information" in response.violation.user_message.lower()

    @pytest.mark.asyncio
    async def test_multiple_messages(self, guardrails_config, shield):
        """Test multiple messages, including one with sensitive information."""
        adapter = GuardrailsAdapter(guardrails_config)
        await adapter.initialize()
        
        # Mock the shield store
        adapter.shield_store = AsyncMock()
        adapter.shield_store.get_shield.return_value = shield
        
        response = await adapter.run_shield(
            shield_id="test-sensitive-info",
            messages=[
                UserMessage(content="First message is safe"),
                UserMessage(content="Second message with card 4532-1234-5678-9012"),
            ],
        )
        
        assert response.violation is not None
        assert response.violation.violation_level == ViolationLevel.ERROR
        assert "sensitive information" in response.violation.user_message.lower()

    @pytest.mark.asyncio
    async def test_invalid_shield(self, guardrails_config):
        """Test handling of invalid shield ID."""
        adapter = GuardrailsAdapter(guardrails_config)
        await adapter.initialize()
        
        # Mock the shield store to return None
        adapter.shield_store = AsyncMock()
        adapter.shield_store.get_shield.return_value = None

        with pytest.raises(ValueError) as exc_info:
            await adapter.run_shield(
                shield_id="invalid_shield",
                messages=[
                    UserMessage(content="Test content"),
                ],
            )
        assert "Shield invalid_shield not found" in str(exc_info.value)


def test_guardrails_server_available(guardrails_config):
    """Test that the guardrails server is running and accessible."""
    import requests
    try:
        response = requests.get(f"{guardrails_config.api_endpoint}/health")
        assert response.status_code == 200
    except requests.exceptions.ConnectionError:
        pytest.fail("Guardrails server is not running")