# agent_tools/github_tool.py — Clean, production-ready GitHub wrapper
import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------------------
# TOKEN + CONSTANTS
# -------------------------------------------------------------------
GITHUB_TOKEN = (
    os.getenv("GITHUB_MCP_TOKEN")
    or os.getenv("GITHUB_TOKEN")
)

if not GITHUB_TOKEN:
    raise ValueError("❌ Missing GITHUB_MCP_TOKEN or GITHUB_TOKEN in .env")

GITHUB_API = "https://api.github.com"

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


# -------------------------------------------------------------------
# SAFE REQUEST WRAPPER
# -------------------------------------------------------------------
def _safe_request(method, url, **kwargs):
    """Small wrapper to avoid exceptions and unify return format."""
    try:
        r = requests.request(method, url, headers=HEADERS, timeout=15, **kwargs)

        try:
            data = r.json()
        except Exception:
            data = r.text

        return {
            "ok": r.status_code in (200, 201, 204),
            "status": r.status_code,
            "data": data,
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "status": None}


# -------------------------------------------------------------------
# DEFAULT BRANCH
# -------------------------------------------------------------------
def get_default_branch(owner: str, repo: str) -> str:
    url = f"{GITHUB_API}/repos/{owner}/{repo}"
    r = _safe_request("GET", url)
    if r["ok"]:
        return r["data"].get("default_branch", "main")
    return "main"


# -------------------------------------------------------------------
# CREATE REPOSITORY
# -------------------------------------------------------------------
def create_repository(name: str, description: str = "", private: bool = False):
    url = f"{GITHUB_API}/user/repos"

    payload = {
        "name": name,
        "description": description,
        "private": private,
    }

    r = _safe_request("POST", url, json=payload)

    if r["ok"]:
        return {"ok": True, "data": r["data"]}

    return {"ok": False, "status": r["status"], "error": r["data"]}


# -------------------------------------------------------------------
# CREATE / UPDATE FILE (README, etc.)
# -------------------------------------------------------------------
def create_or_update_file(owner: str, repo: str, path: str, content: str, message: str):
    if not owner or not repo or not path:
        return {"ok": False, "error": "Missing owner/repo/path"}

    branch = get_default_branch(owner, repo)

    # Fetch existing file SHA
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"
    r = _safe_request("GET", url, params={"ref": branch})

    body = {
        "message": message,
        "content": base64.b64encode(content.encode()).decode(),
        "branch": branch,
    }

    if r["ok"] and isinstance(r["data"], dict):
        sha = r["data"].get("sha")
        if sha:
            body["sha"] = sha

    # PUT create/update
    resp = requests.put(url, headers=HEADERS, json=body)

    try:
        data = resp.json()
    except Exception:
        data = resp.text

    ok = resp.status_code in (200, 201)

    return {"ok": ok, "status": resp.status_code, "data": data}


# -------------------------------------------------------------------
# AUTHENTICATED LIST OF ALL REPOS (private + public)
# PAGINATION INCLUDED
# -------------------------------------------------------------------
def list_repos(username: str | None = None):
    """
    If token exists → use authenticated /user/repos (private + public)
    Paginated by 100 per page.
    """
    repos = []
    per_page = 100
    page = 1

    # Use authenticated endpoint ALWAYS if token is present
    url = f"{GITHUB_API}/user/repos"

    while True:
        params = {"per_page": per_page, "page": page}
        r = _safe_request("GET", url, params=params)

        if not r["ok"]:
            return {"ok": False, "status": r.get("status"), "error": r.get("data")}

        data = r["data"]
        if not isinstance(data, list):
            break

        # Collect repo info
        for repo in data:
            repos.append({
                "name": repo.get("name"),
                "full_name": repo.get("full_name"),
                "private": repo.get("private", False),
                "html_url": repo.get("html_url"),
            })

        # Pagination exit condition
        if len(data) < per_page:
            break

        page += 1

    return {
        "ok": True,
        "total": len(repos),
        "repositories": repos,
    }


# -------------------------------------------------------------------
# LIST BRANCHES
# -------------------------------------------------------------------
def list_branches(owner: str, repo: str):
    url = f"{GITHUB_API}/repos/{owner}/{repo}/branches"
    r = _safe_request("GET", url)
    if r["ok"]:
        branches = [b.get("name") for b in r["data"]]
        return {"ok": True, "branches": branches}

    return {"ok": False, "status": r["status"], "error": r["data"]}


# -------------------------------------------------------------------
# CREATE PULL REQUEST
# -------------------------------------------------------------------
def create_pull_request(owner: str, repo: str, head: str, base: str, title: str, body: str = ""):
    if not owner or not repo or not head or not base:
        return {"ok": False, "error": "Missing owner/repo/head/base"}

    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls"

    payload = {
        "title": title,
        "head": head,
        "base": base,
        "body": body,
    }

    r = _safe_request("POST", url, json=payload)

    if r["ok"]:
        return {"ok": True, "data": r["data"]}

    return {"ok": False, "status": r["status"], "error": r["data"]}
