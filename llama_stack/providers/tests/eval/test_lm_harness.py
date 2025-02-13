# llama_stack/providers/tests/eval/test_lm_harness.py
import pytest
from unittest.mock import AsyncMock

from llama_stack.apis.inference import UserMessage
from llama_stack.apis.eval_tasks import EvalTask
from llama_stack.apis.eval import EvalTaskConfig, JobStatus
from llama_stack.providers.remote.eval.lm_harness.config import LMHarnessEvalConfig
from llama_stack.providers.remote.eval.lm_harness.lm_harness import LMHarnessEvalAdapter


class MockInferenceAPI:
    async def generate(self, *args, **kwargs):
        return "mock response"


class TestLMHarness:
    @pytest.mark.asyncio
    async def test_task_registration(self):
        """Test registering an evaluation task."""
        config = LMHarnessEvalConfig()
        impl = LMHarnessEvalAdapter(config)
        await impl.initialize()

        task = EvalTask(
            identifier="test_task",
            dataset_id="hellaswag",  # A task from lm-evaluation-harness
            scoring_functions=["accuracy"],
        )
        await impl.register_eval_task(task)
        assert task.identifier in impl.eval_tasks

    @pytest.mark.asyncio
    async def test_task_evaluation(self):
        """Test running an evaluation task."""
        config = LMHarnessEvalConfig(
            api_endpoint="http://localhost:8000",
            timeout=30,
        )
        impl = LMHarnessEvalAdapter(config)
        await impl.initialize()

        # Register and run a task
        task = EvalTask(
            identifier="test_eval",
            dataset_id="hellaswag",
            scoring_functions=["accuracy"],
        )
        await impl.register_eval_task(task)
        
        job = await impl.run_eval(
            task_id="test_eval",
            task_config=EvalTaskConfig(
                eval_candidate="test-model",
            ),
        )
        
        # Get results
        results = await impl.get_eval_results(job.job_id)
        assert results is not None
        assert results.status == JobStatus.COMPLETED