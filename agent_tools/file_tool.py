# agent_tools/file_tool.py
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # adjust if needed; ensures safe base

def safe_path(path: str) -> Path:
    p = (PROJECT_ROOT / path).resolve()
    if PROJECT_ROOT not in p.parents and p != PROJECT_ROOT:
        raise PermissionError("Attempt to access outside project root")
    return p

def read_file(path: str) -> Dict:
    try:
        p = safe_path(path)
        if not p.exists():
            return {"ok": False, "error": "not_found", "content": ""}
        content = p.read_text(encoding="utf-8")
        return {"ok": True, "content": content}
    except Exception as e:
        return {"ok": False, "error": str(e), "content": ""}

def write_file(path: str, content: str, create_dirs: bool = False) -> Dict:
    try:
        p = safe_path(path)
        if create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
