# simplechat_agent.py — Clean production-ready agent
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
import json, re, os, time
from dotenv import load_dotenv
import requests

# Tools (must exist in agent_tools package)
from agent_tools.file_tool import read_file, write_file
from agent_tools.shell_tool import run_shell
from agent_tools.github_tool import (
    HEADERS,
    create_or_update_file,
    create_pull_request,
    create_repository,
    get_default_branch,
    list_branches,
    list_repos,
)

load_dotenv()

# Initialize LLM
llm = ChatGroq(model=os.getenv("LLM_MODEL", "llama-3.1-8b-instant"))

# -------------------------
# Normalizer
# -------------------------
def normalize_message(m) -> BaseMessage:
    if isinstance(m, BaseMessage):
        return m
    if isinstance(m, dict):
        role = (m.get("role") or m.get("type") or "").lower()
        content = m.get("content") if "content" in m else m.get("text", "")
        if role in ("user", "human"):
            return HumanMessage(content=content)
        if role in ("assistant", "ai"):
            return AIMessage(content=content)
        return SystemMessage(content=content or json.dumps(m))
    if isinstance(m, str):
        return HumanMessage(content=m)
    return SystemMessage(content=str(m))

# -------------------------
# State
# -------------------------
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], ...]
    tool_call: dict | None
    tool_result: dict | None

# -------------------------
# Router (strict single-action)
# -------------------------
def router_node(state: ChatState):
    """
    Strict natural-language router:
    - Tries specific intents in order.
    - Returns a single tool_call dict or empty dict.
    """

    last = state["messages"][-1] if state["messages"] else None
    if last is None:
        return {}

    try:
        msg = str(last.content).strip()
    except Exception:
        msg = str(last).strip()

    lm = msg.lower()

    # helper
    def match(regex_list):
        for r in regex_list:
            if re.search(r, lm):
                return True
        return False

    # 1) FILE: read/write/update/create
    if match([r"\bread file\b", r"\breadfile\b", r"\bshow file\b", r"\bopen file\b"]):
        # read file path
        m = re.search(r"read (?:file )?(.+)$", msg, re.I)
        path = m.group(1).strip() if m else None
        return {"tool_call": {"tool": "file", "args": {"action": "read", "path": path, "text": msg}}}

    if match([r"\bwrite file\b", r"\bcreate file\b", r"\badd file\b", r"\bupdate file\b", r"\bedit file\b"]):
        # format: write file <path>: <content>
        m = re.search(r"(?:write|create|update|edit) (?:file )?([^\:]+):\s*(.+)$", msg, re.I)
        if m:
            path = m.group(1).strip()
            content = m.group(2).strip()
        else:
            # fallback: path only
            mm = re.search(r"(?:write|create|update|edit) (?:file )?(.+)$", msg, re.I)
            path = mm.group(1).strip() if mm else None
            content = None
        return {"tool_call": {"tool": "file", "args": {"action": "write", "path": path, "content": content, "text": msg}}}

    # 2) GITHUB: create repo
    if match([r"\bcreate repo\b", r"\bcreate repository\b", r"\bmake (?:a )?repo\b", r"\bnew repo\b"]):
        m = re.search(r"(?:repo|repository|repo called|repo named)\s+([A-Za-z0-9._-]+)", msg, re.I)
        name = m.group(1) if m else None
        return {"tool_call": {"tool": "github", "args": {"action": "create_repo", "name": name, "text": msg}}}

    # 3) GITHUB: list repos
    if match([r"\blist repos\b", r"\bshow repos\b", r"\bmy github repos\b", r"\brepositories\b"]):
        return {"tool_call": {"tool": "github", "args": {"action": "list_repos", "text": msg}}}

    # 4) GITHUB: update readme
    if match([r"\bupdate readme\b", r"\bupdate README\b", r"\bedit README.md\b"]):
        # try to capture repo and content
        m = re.search(r"update readme (?:for\s+([A-Za-z0-9_-]+/[A-Za-z0-9_-]+))?(?: with|:)?\s*(.+)?", msg, re.I)
        repo_spec = m.group(1) if m and m.group(1) else None
        content = m.group(2).strip() if m and m.group(2) else None
        owner, repo = None, None
        if repo_spec:
            parts = repo_spec.split("/")
            owner, repo = parts[0], parts[1]
        return {"tool_call": {"tool": "github", "args": {"action": "update_file", "owner": owner, "repo": repo, "path": "README.md", "content": content, "text": msg}}}

    # 5) GITHUB: create pull request (phrases like "create pr from X to Y in owner/repo")
    if match([r"\bcreate pr\b", r"\bcreate pull request\b", r"\bopen pr\b", r"\bopen pull request\b", r"\bmake a pull request\b"]):
        owner_repo = None
        m = re.search(r"([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)", msg)
        if m:
            owner_repo = (m.group(1), m.group(2))
        mm = re.search(r"from\s+([A-Za-z0-9_\-\/]+)\s+to\s+([A-Za-z0-9_\-\/]+)", msg)
        head = mm.group(1) if mm else None
        base = mm.group(2) if mm else None
        owner = owner_repo[0] if owner_repo else (os.getenv("GITHUB_OWNER") or None)
        repo = owner_repo[1] if owner_repo else (os.getenv("GITHUB_REPO") or None)
        return {"tool_call": {"tool": "github", "args": {"action": "create_pr", "owner": owner, "repo": repo, "head": head, "base": base, "text": msg}}}

    # 6) GITHUB: list branches
    if match([r"\blist branches\b", r"\bshow branches\b", r"\bbranches\b"]):
        m = re.search(r"([A-Za-z0-9_-]+)\/([A-Za-z0-9_-]+)", msg)
        owner_repo = (m.group(1), m.group(2)) if m else (os.getenv("GITHUB_OWNER"), os.getenv("GITHUB_REPO"))
        owner = owner_repo[0]
        repo = owner_repo[1]
        return {"tool_call": {"tool": "github", "args": {"action": "list_branches", "owner": owner, "repo": repo, "text": msg}}}

    # 7) PUSH intent (safe)
    if match([r"\bpush my code\b", r"\bpush code\b", r"\bpush repo\b"]):
        return {"tool_call": {"tool": "shell", "args": {"action": "push_intent", "command": "git status -b", "text": msg}}}

    # 8) REPO-WIDE fix
    if match([r"\bfix my repo\b", r"\bfix repo\b", r"\bmake tests pass\b", r"\brun tests\b"]):
        return {"tool_call": {"tool": "repo", "args": {"action": "fix_repo", "text": msg}}}

    # 9) Generic shell execution (explicit)
    if match([r"^\s*run\:", r"\bexecute\b", r"\bshell\b", r"^\s*ls\b", r"^\s*git\b", r"pytest", r"npm"]):
        # prefer explicit "run: <cmd>" syntax
        m = re.search(r"run:\s*(.+)$", msg, re.I)
        cmd = m.group(1).strip() if m else msg
        return {"tool_call": {"tool": "shell", "args": {"action": "exec", "command": cmd, "text": msg}}}

    # Default: no tool
    return {}

# -------------------------
# Tool Node (single tool execution)
# -------------------------
async def tool_node(state: ChatState):
    tc = state.get("tool_call")
    if not tc:
        return {"tool_result": None}

    tool = tc.get("tool")
    args = tc.get("args") or {}

    # SHELL
    if tool == "shell":
        action = args.get("action")
        cmd = args.get("command", "")
        # For push_intent run a safe command and return status
        if action == "push_intent":
            res = await run_shell(cmd, cwd=".")
            # Do NOT run git push on server automatically
            return {"tool_result": {"tool": "shell", "result": {"ok": True, "type": "push_intent_checked", "status": res}}}
        # exec any other command (danger: user responsibility)
        res = await run_shell(cmd, cwd=".")
        return {"tool_result": {"tool": "shell", "result": res}}

    # FILE TOOL
    if tool == "file":
        action = args.get("action")
        if action == "read":
            path = args.get("path") or args.get("text") or ""
            try:
                res = read_file(path)
            except Exception as e:
                res = {"ok": False, "error": str(e)}
            return {"tool_result": {"tool": "file", "result": res}}
        if action == "write":
            path = args.get("path")
            content = args.get("content") or ""
            if not path:
                return {"tool_result": {"tool": "file", "result": {"ok": False, "error": "missing_path"}}}
            try:
                res = write_file(path, content, create_dirs=True)
            except Exception as e:
                res = {"tool_result": {"tool": "file", "result": {"ok": False, "error": str(e)}}}
            return {"tool_result": {"tool": "file", "result": res}}

    # GITHUB TOOL
    if tool == "github":
        action = args.get("action")
        # LIST_REPOS
        if action == "list_repos":
            username = args.get("username") or os.getenv("GITHUB_OWNER")
            res = list_repos(username)
            return {"tool_result": {"tool": "github", "result": res}}

        # CREATE_REPO
        if action == "create_repo":
            name = args.get("name")
            if not name:
                return {"tool_result": {"tool": "github", "result": {"ok": False, "error": "missing_name"}}}
            res = create_repository(name=name, description="Created by AI Agent", private=False)
            return {"tool_result": {"tool": "github", "result": res}}

        # UPDATE FILE (README)
        if action == "update_file":
            owner = args.get("owner") or os.getenv("GITHUB_OWNER")
            repo = args.get("repo") or os.getenv("GITHUB_REPO")
            path = args.get("path") or "README.md"
            content = args.get("content") or args.get("text") or ""
            if not repo:
                return {"tool_result": {"tool": "github", "result": {"ok": False, "error": "missing_repo"}}}
            res = create_or_update_file(owner, repo, path, content, "Agent update")
            return {"tool_result": {"tool": "github", "result": res}}

        # CREATE PR
        if action == "create_pr":
            owner = args.get("owner") or os.getenv("GITHUB_OWNER")
            repo = args.get("repo") or os.getenv("GITHUB_REPO")
            head = args.get("head")
            base = args.get("base")
            if not repo:
                return {"tool_result": {"tool": "github", "result": {"ok": False, "error": "missing_repo"}}}
            # attempt to auto-create head branch if absent (best-effort)
            if not head:
                head = f"agent/auto-{int(time.time())}"
                try:
                    br_url = f"https://api.github.com/repos/{owner}/{repo}/git/ref/heads/{base or 'main'}"
                    r = requests.get(br_url, headers=HEADERS)
                    if r.status_code == 200:
                        sha = r.json()["object"]["sha"]
                        create_ref_url = f"https://api.github.com/repos/{owner}/{repo}/git/refs"
                        payload = {"ref": f"refs/heads/{head}", "sha": sha}
                        requests.post(create_ref_url, headers=HEADERS, json=payload)
                except Exception:
                    pass
            title = args.get("title") or f"Automated PR by Agent: {datetime.utcnow().isoformat()}"
            body = args.get("body") or args.get("text") or ""
            res = create_pull_request(owner, repo, head, base or get_default_branch(owner, repo), title, body)
            return {"tool_result": {"tool": "github", "result": res}}

        # LIST_BRANCHES
        if action == "list_branches":
            owner = args.get("owner") or os.getenv("GITHUB_OWNER")
            repo = args.get("repo") or os.getenv("GITHUB_REPO")
            if not repo:
                return {"tool_result": {"tool": "github", "result": {"ok": False, "error": "missing_repo"}}}
            res = list_branches(owner, repo)
            return {"tool_result": {"tool": "github", "result": res}}

        return {"tool_result": {"tool": "github", "result": {"ok": False, "error": "unknown_action"}}}

    # REPO coordinator
    if tool == "repo":
        action = args.get("action")
        if action == "fix_repo":
            tests = await run_shell("pytest -q", cwd=".")
            # flake8 may not be installed — handle gracefully
            try:
                lint = await run_shell("flake8 .", cwd=".")
            except Exception:
                lint = {"ok": False, "error": "flake8_not_available"}
            return {"tool_result": {"tool": "repo", "result": {"tests": tests, "lint": lint}}}
        return {"tool_result": {"tool": "repo", "result": {"ok": False, "error": "unknown_repo_action"}}}

    return {"tool_result": None}

# -------------------------
# LLM Node (single final summarizer)
# -------------------------
async def llm_node(state: ChatState):
    """
    Build prompt:
    - Include short system instruction (summarize only the LAST tool_result).
    - Include limited history.
    - Append the last tool_result as TOOL_OUTPUT (stringified but small).
    """
    raw_messages = state.get("messages", []) or []
    tool_result = state.get("tool_result")

    MAX_HISTORY = 12
    history = raw_messages[-MAX_HISTORY:]

    prompt_msgs = [SystemMessage(content=(
        "You are an AI DevOps Assistant. Be concise and factual.\n"
        "If a TOOL OUTPUT is provided, summarize the final result in one short line for the user.\n"
        "Do NOT invent actions or repeat unrelated previous tool outputs.\n"
        "If tool_result.ok is false, produce a short error message explaining why."
    ))]

    for m in history:
        prompt_msgs.append(normalize_message(m))

    if tool_result:
        # include only the last tool_result summary (not entire history)
        truncated = json.dumps(tool_result, default=str)
        if len(truncated) > 2000:
            truncated = truncated[:2000] + "..."
        prompt_msgs.append(SystemMessage(content=f"TOOL_OUTPUT: {truncated}"))

    try:
        result = await llm.ainvoke(input=prompt_msgs)
    except Exception as e:
        return {"messages": [SystemMessage(content=f"LLM call failed: {str(e)}")]}

    # Normalize returned shape
    if hasattr(result, "messages"):
        returned = result.messages
    elif isinstance(result, dict) and "messages" in result:
        returned = result["messages"]
    else:
        returned = result

    if not isinstance(returned, list):
        returned = [returned]

    cleaned = [normalize_message(m) for m in returned]
    return {"messages": cleaned}

# -------------------------
# Build Graph
# -------------------------
graph = StateGraph(ChatState)
graph.add_node("router_node", router_node)
graph.add_node("tool_node", tool_node)
graph.add_node("llm_node", llm_node)

graph.add_edge(START, "router_node")
graph.add_edge("router_node", "tool_node")
graph.add_edge("tool_node", "llm_node")
graph.add_edge("llm_node", END)

agent = graph.compile()
