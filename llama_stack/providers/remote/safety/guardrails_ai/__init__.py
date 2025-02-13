# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from .config import GuardrailsConfig


async def get_adapter_impl(config: GuardrailsConfig, _deps) -> Any:
    from .guardrails import GuardrailsAdapter

    impl = GuardrailsAdapter(config)
    await impl.initialize()
    return impl