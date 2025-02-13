from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import lm_eval
from lm_eval import tasks, evaluator, utils
import asyncio
import uuid

app = FastAPI()

# Store running jobs
jobs = {}

class EvalRequest(BaseModel):
    task: str
    model: str
    num_fewshot: int = 0
    batch_size: int = 1
    max_examples: Optional[int] = None

@app.get("/tasks")
async def list_tasks():
    """List all available tasks."""
    return {"tasks": list(tasks.ALL_TASKS.keys())}

@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get information about a specific task."""
    if task_id not in tasks.ALL_TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task": task_id,
        "metrics": ["accuracy"]  # You can expand this based on task
    }

@app.post("/evaluate")
async def start_evaluation(request: EvalRequest):
    """Start an evaluation job."""
    if request.task not in tasks.ALL_TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    
    job_id = str(uuid.uuid4())
    
    # Start evaluation in background
    asyncio.create_task(run_evaluation(job_id, request))
    
    return {"job_id": job_id}

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

async def run_evaluation(job_id: str, request: EvalRequest):
    """Run the evaluation in background."""
    try:
        jobs[job_id] = {"status": "running"}
        
        # Configure the task
        task_dict = {
            request.task: {
                "num_fewshot": request.num_fewshot,
                "batch_size": request.batch_size,
            }
        }
        
        if request.max_examples:
            task_dict[request.task]["max_examples"] = request.max_examples
        
        # Run evaluation
        results = evaluator.simple_evaluate(
            model=request.model,
            tasks=task_dict,
            num_fewshot=request.num_fewshot,
            batch_size=request.batch_size,
        )
        
        jobs[job_id] = {
            "status": "completed",
            "results": results
        }
    except Exception as e:
        jobs[job_id] = {
            "status": "failed",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)