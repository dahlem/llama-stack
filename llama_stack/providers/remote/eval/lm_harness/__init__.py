# llama_stack/providers/remote/eval/lm_harness/__init__.py
from typing import Any

from .config import LMHarnessEvalConfig

async def get_adapter_impl(config: LMHarnessEvalConfig, _deps) -> Any:
    from .lm_harness import LMHarnessEvalAdapter

    impl = LMHarnessEvalAdapter(config)
    await impl.initialize()
    return impl