# llama_stack/providers/remote/eval/lm_harness/lm_harness.py
import json
import logging
import aiohttp
from typing import Any, Dict, List, Optional

from llama_stack.apis.eval_tasks import EvalTask
from llama_stack.apis.eval import (
    Eval,
    EvalTaskConfig,
    EvaluateResponse,
    JobStatus,
)
from llama_stack.apis.common.job_types import Job
from llama_stack.providers.datatypes import EvalTasksProtocolPrivate
from .config import LMHarnessEvalConfig

logger = logging.getLogger(__name__)

class LMHarnessEvalAdapter(Eval, EvalTasksProtocolPrivate):
    def __init__(self, config: LMHarnessEvalConfig) -> None:
        self.config = config
        self.eval_tasks = {}
        self.jobs = {}
        self.session = None

    async def initialize(self) -> None:
        """Initialize the adapter and verify connection to the LM Harness server."""
        try:
            self.session = aiohttp.ClientSession(
                base_url=self.config.api_endpoint,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            )
            # Verify connection and get available tasks
            async with self.session.get("/tasks") as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to connect to LM Harness server: {response.status}")
                self.available_tasks = await response.json()
        except Exception as e:
            if self.session:
                await self.session.close()
            raise RuntimeError(f"Error initializing LMHarnessEvalAdapter: {e}") from e

    async def shutdown(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()

    async def register_eval_task(self, task_def: EvalTask) -> None:
        """Register an evaluation task with the remote LM Harness server."""
        try:
            # Verify task exists on the server
            async with self.session.get(f"/tasks/{task_def.dataset_id}") as response:
                if response.status != 200:
                    raise ValueError(f"Task {task_def.dataset_id} not found on LM Harness server")
                task_info = await response.json()
                
            # Store task with its metadata
            self.eval_tasks[task_def.identifier] = {
                "task_def": task_def,
                "task_info": task_info,
            }
        except Exception as e:
            logger.error(f"Error registering task {task_def.identifier}: {e}")
            raise

    async def run_eval(
        self,
        task_id: str,
        task_config: EvalTaskConfig,
    ) -> Job:
        """Start an evaluation task on the remote server."""
        if task_id not in self.eval_tasks:
            raise ValueError(f"Task {task_id} not found")

        task_info = self.eval_tasks[task_id]
        task_def = task_info["task_def"]

        try:
            # Start evaluation on remote server
            payload = {
                "task": task_def.dataset_id,
                "model": task_config.eval_candidate,
                "num_fewshot": task_config.metadata.get("num_fewshot", 0),
                "batch_size": task_config.metadata.get("batch_size", 1),
                "max_examples": task_config.metadata.get("max_examples", None),
            }

            async with self.session.post("/evaluate", json=payload) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to start evaluation: {response.status}")
                job_data = await response.json()
                remote_job_id = job_data["job_id"]

            # Store job information
            job_id = f"job_{task_id}_{len(self.jobs)}"
            self.jobs[job_id] = {
                "remote_job_id": remote_job_id,
                "status": JobStatus.RUNNING,
                "results": None,
            }

            return Job(job_id=job_id)

        except Exception as e:
            logger.error(f"Error starting evaluation for task {task_id}: {e}")
            raise

    async def get_eval_results(self, job_id: str) -> Optional[EvaluateResponse]:
        """Get the results of an evaluation job from the remote server."""
        if job_id not in self.jobs:
            return None

        job = self.jobs[job_id]
        if job["results"] is not None:
            return EvaluateResponse(
                status=job["status"],
                results=job["results"],
            )

        try:
            # Check job status on remote server
            async with self.session.get(f"/jobs/{job['remote_job_id']}") as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to get job status: {response.status}")
                status_data = await response.json()

            if status_data["status"] == "completed":
                # Get full results
                async with self.session.get(f"/results/{job['remote_job_id']}") as response:
                    if response.status != 200:
                        raise RuntimeError(f"Failed to get results: {response.status}")
                    results = await response.json()

                job["status"] = JobStatus.COMPLETED
                job["results"] = results
                return EvaluateResponse(
                    status=JobStatus.COMPLETED,
                    results=results,
                )
            elif status_data["status"] == "failed":
                job["status"] = JobStatus.FAILED
                return EvaluateResponse(
                    status=JobStatus.FAILED,
                    error=status_data.get("error", "Unknown error"),
                )
            else:
                return EvaluateResponse(status=JobStatus.RUNNING)

        except Exception as e:
            logger.error(f"Error getting results for job {job_id}: {e}")
            raise