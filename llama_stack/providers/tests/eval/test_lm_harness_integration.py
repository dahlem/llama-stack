import pytest
import asyncio
from llama_stack.apis.eval_tasks import EvalTask
from llama_stack.apis.eval import EvalTaskConfig, JobStatus
from llama_stack.providers.remote.eval.lm_harness.config import LMHarnessEvalConfig
from llama_stack.providers.remote.eval.lm_harness.lm_harness import LMHarnessEvalAdapter

@pytest.mark.integration
class TestLMHarnessIntegration:
    @pytest.mark.asyncio
    async def test_real_server(self):
        """Test with a real LM Evaluation Harness server."""
        config = LMHarnessEvalConfig(
            api_endpoint="http://localhost:8000",
            timeout=300,  # Longer timeout for real evaluation
        )
        impl = LMHarnessEvalAdapter(config)
        await impl.initialize()

        # Register a task
        task = EvalTask(
            identifier="hellaswag_test",
            dataset_id="hellaswag",
            scoring_functions=["accuracy"],
        )
        await impl.register_eval_task(task)

        # Run evaluation
        job = await impl.run_eval(
            task_id="hellaswag_test",
            task_config=EvalTaskConfig(
                eval_candidate="gpt2",  # Using GPT-2 as an example
                metadata={
                    "num_fewshot": 0,
                    "batch_size": 1,
                    "max_examples": 10,  # Limit examples for testing
                }
            ),
        )

        # Poll for results
        while True:
            results = await impl.get_eval_results(job.job_id)
            if results.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                break
            await asyncio.sleep(5)  # Wait 5 seconds between checks

        assert results.status == JobStatus.COMPLETED
        assert "accuracy" in results.results