"""Container-only execution tool for the Implementer (Stage 7).

The authoring agent gets this MCP tool as its ONLY way to run code — never a raw host
shell. Commands execute in a Docker container (via JobRunner) with the agent's code dir
mounted, networking disabled, as the host user. Returns stdout/exit for the build-test-fix
loop."""
from __future__ import annotations

from pathlib import Path

from claude_agent_sdk import create_sdk_mcp_server, tool

from ..job_runner import JobRunner, JobSpec
from ..util import ensure_dir


def make_sandbox_server(job_runner: JobRunner, *, code_dir: str | Path, image: str | None,
                        scratch_dir: str | Path, data_dir: str | None = None,
                        server_name: str = "sandbox"):
    """Returns (mcp_server_config, tool_fullname). The tool runs a command in the container
    with /code (the agent's code) and optionally /data (train split) mounted ro, /artifacts
    a scratch dir, and --network none."""
    code_dir = str(Path(code_dir).resolve())
    scratch = str(ensure_dir(scratch_dir))

    @tool(
        "run",
        "Run a shell command inside the isolated GPU container (NETWORK DISABLED). Your "
        "code is mounted read-only at /code (also $LAB_CODE); a writable scratch dir is at "
        "/artifacts ($LAB_ARTIFACTS); the training data (if any) is read-only at /data "
        "($LAB_DATA). Use this to test the code you write. Returns exit code + output.",
        {"command": str},
    )
    async def run(args):
        log_path = Path(scratch) / "sandbox.log"
        result = job_runner.run(JobSpec(
            exp_id="author-sandbox", command=args["command"], workdir=code_dir,
            artifacts_dir=scratch, code_dir=code_dir, data_dir=data_dir,
            log_path=str(log_path), image=image, mode="docker", network="none",
        ))
        out = log_path.read_text(errors="replace")[-4000:] if log_path.exists() else ""
        return {"content": [{"type": "text", "text": f"exit_code={result.exit_code}\n{out}"}]}

    server = create_sdk_mcp_server(server_name, "1.0.0", tools=[run])
    return server, f"mcp__{server_name}__run"
