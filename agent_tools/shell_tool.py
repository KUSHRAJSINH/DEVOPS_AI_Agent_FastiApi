# agent_tools/shell_tool.py
import asyncio
import shlex
from typing import Dict

# Whitelisted commands (prefix match allowed). Expand as needed.
WHITELIST = [
    "ls", "pwd", "cat", "echo", "git status", "git rev-parse", "git log", "python", "pip", "npm", "pytest"
]

async def run_shell(command: str, cwd: str = ".", timeout: int = 20) -> Dict:
    # Normalize
    cmd_trim = command.strip()
    # basic safety: check whitelist (prefix match)
    allowed = any(cmd_trim.startswith(w) for w in WHITELIST)
    if not allowed:
        return {"ok": False, "error": "command_not_whitelisted", "stdout": "", "stderr": "Command not allowed"}

    # Build args
    args = shlex.split(cmd_trim)
    try:
        proc = await asyncio.create_subprocess_exec(
            *args, cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return {"ok": False, "error": "timeout", "stdout": "", "stderr": "Command timed out"}
        return {"ok": True, "stdout": stdout.decode(errors="ignore"), "stderr": stderr.decode(errors="ignore")}
    except Exception as e:
        return {"ok": False, "error": "exec_error", "stdout": "", "stderr": str(e)}
