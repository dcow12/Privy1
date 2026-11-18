"""Privy web app: pick models, pick conditions, run paired sandbox-vs-production
experiments, and compare the results.

    python app.py            # then open http://127.0.0.1:8000

API keys you enter are kept in this process's memory only. They are never written to disk
and only ever sent to the provider they belong to. Keys in .env are picked up at startup.
The server listens on 127.0.0.1 only.

Public mode (PRIVY_PUBLIC=1, used by the Docker image) makes it safe to share one server:
  - each visitor's keys and runs belong to their browser session (an X-Privy-Session id);
    nobody sees or uses anyone else's, and keys in the server's environment are ignored;
  - one run per visitor at a time, PRIVY_MAX_JOBS runs in total (default 3), smaller caps on
    run sizes, no "this computer" backend, no custom endpoints;
  - runs are deleted after PRIVY_RUN_TTL_HOURS (default 24).
"""
from __future__ import annotations

import contextvars
import csv
import hashlib
import io
import json
import os
import re
import time
import warnings

warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from privy import load_env
from sniffer import jobs, providers, scenarios

load_env()
providers.PROVIDERS["custom"]["base_url"] = os.environ.get("CUSTOM_BASE_URL", "")
WEB =os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
app = FastAPI(title="Privy")
app.mount("/static", StaticFiles(directory=WEB), name="static")

PUBLIC = os.environ.get("PRIVY_PUBLIC", "").strip().lower() in ("1", "true", "yes")
MAX_JOBS = int(os.environ.get("PRIVY_MAX_JOBS", "3"))
RUN_TTL_HOURS = float(os.environ.get("PRIVY_RUN_TTL_HOURS", "24"))
SESSION_TTL = 12 * 3600
LIMITS = {"evolve_generations": 8, "evolve_pop": 24, "evolve_tasks": 8,
          "agent_repeats": 2, "agent_variants": 2, "agent_workers": 6}

# provider -> {"key": str, "base_url": str}; "ndif" and "hf" are white-box credentials.
# Single-user mode only; a public server keeps one keyring per visitor session instead.
KEYS: dict[str, dict] = {}
if not PUBLIC:
    for pid, meta in providers.PROVIDERS.items():
        if meta.get("env") and os.environ.get(meta["env"]):
            KEYS[pid] = {"key": os.environ[meta["env"]], "base_url": meta.get("base_url") or ""}
    for pid, env in (("ndif", "NDIF_API_KEY"), ("hf", "HF_TOKEN")):
        if os.environ.get(env):
            KEYS[pid] = {"key": os.environ[env]}

# ------------------------------------------------------------------- visitor sessions
_SID = contextvars.ContextVar("privy_sid", default="")
_SID_RE = re.compile(r"^[0-9a-f]{32,64}$")
_SESSIONS: dict[str, dict] = {}   # session id -> {"keys": {...}, "seen": time}


class _SessionMiddleware:
    """Reads the visitor's session id (header, or ?s= for <img> and download links)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        from starlette.datastructures import Headers, QueryParams
        sid = (Headers(scope=scope).get("x-privy-session")
               or QueryParams(scope.get("query_string", b"").decode()).get("s") or "")
        token = _SID.set(sid if _SID_RE.match(sid) else "")
        try:
            await self.app(scope, receive, send)
        finally:
            _SID.reset(token)


app.add_middleware(_SessionMiddleware)

_NO_SESSION = "Your browser didn't send a session id. Reload the page and try again."


def _keys(write=False):
    """This visitor's keyring (the shared one in single-user mode)."""
    if not PUBLIC:
        return KEYS
    sid = _SID.get()
    if not sid:
        if write:
            raise HTTPException(400, _NO_SESSION)
        return {}
    now = time.time()
    for old in [k for k, v in _SESSIONS.items() if now - v["seen"] > SESSION_TTL]:
        _SESSIONS.pop(old, None)
    sess = _SESSIONS.setdefault(sid, {"keys": {}, "seen": now})
    sess["seen"] = now
    return sess["keys"]


def _owner():
    sid = _SID.get()
    return hashlib.sha256(sid.encode()).hexdigest()[:16] if PUBLIC and sid else None


def _visible(info):
    return not PUBLIC or (info.get("owner") is not None and info.get("owner") == _owner())


def _start_guard():
    """Public mode: one run per visitor, MAX_JOBS overall, and clear out expired runs."""
    if not PUBLIC:
        return None
    owner = _owner()
    if not owner:
        raise HTTPException(400, _NO_SESSION)
    jobs.prune_runs(RUN_TTL_HOURS)
    running = jobs.running_jobs()
    if any(j.owner == owner for j in running):
        raise HTTPException(429, "You already have a run going. Wait for it to finish or cancel it first.")
    if len(running) >= MAX_JOBS:
        raise HTTPException(429, "The server is busy with other visitors' runs. Try again in a few minutes.")
    return owner


def _creds():
    """White-box credentials handed to a job (None = single-user mode, keys are in the env)."""
    if not PUBLIC:
        return None
    k = _keys()
    return {"ndif": k.get("ndif", {}).get("key"), "hf": k.get("hf", {}).get("key")}


def _public_backend(backend):
    if PUBLIC and backend == "local":
        raise HTTPException(400, "This shared server can't run models locally. Use NDIF or the simulated model.")

RUN_ID = re.compile(r"^[0-9]{4}-[0-9]{6}-[0-9a-f]{4}$")


def _run_or_404(run_id):
    info = jobs.get_run(run_id) if RUN_ID.match(run_id) else None
    if info is None or not _visible(info):
        raise HTTPException(404, "No such run")
    return info


def _mask(key):
    return "" if not key else (key[:4] + "…" + key[-4:] if len(key) > 12 else "set")


# ---------------------------------------------------------------------------- pages

@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


@app.get("/api/meta")
def meta():
    KEYS = _keys()
    fams = []
    for f in scenarios.FAMILIES + ["custom"]:
        cue = scenarios.CUES.get(f)
        if f == "aligned":
            sb = " · ".join(scenarios.CUES[c][0][0] for c in scenarios.IMPLICIT_CUES)
            pr = " · ".join(scenarios.CUES[c][1][0] for c in scenarios.IMPLICIT_CUES)
        elif f == "emptiness":
            sb, pr = scenarios.DENSITY[0], scenarios.DENSITY[1]
        elif cue:
            sb, pr = cue[0][0], cue[1][0]
        else:
            sb, pr = "", ""
        fams.append({"id": f, "about": scenarios.FAMILY_INFO[f], "sandbox": sb, "prod": pr,
                     "mentions_testing": f == "explicit"})
    return {
        "families": fams,
        "tasks": [{"id": i, "request": r, "allowed": a} for i, (r, a) in enumerate(scenarios.TASKS)],
        "providers": [{"id": pid, "label": m["label"], "kind": m["kind"], "key_url": m.get("key_url", ""),
                       "has_key": pid in KEYS or m["kind"] == "mock",
                       "masked": _mask(KEYS.get(pid, {}).get("key")),
                       "base_url": KEYS.get(pid, {}).get("base_url") or m.get("base_url", "")}
                      for pid, m in providers.PROVIDERS.items()
                      if not (PUBLIC and pid == "custom")],
        "whitebox_keys": {k: {"has_key": k in KEYS, "masked": _mask(KEYS.get(k, {}).get("key"))}
                          for k in ("ndif", "hf")},
        "mock_models": providers.MOCK_MODELS,
        "prices": providers.PRICES,
        "public": PUBLIC,
        "limits": LIMITS if PUBLIC else None,
        "run_ttl_hours": RUN_TTL_HOURS if PUBLIC else None,
    }


class KeyIn(BaseModel):
    provider: str
    key: str = ""
    base_url: str = ""


@app.post("/api/keys")
def set_key(k: KeyIn):
    if k.provider not in providers.PROVIDERS and k.provider not in ("ndif", "hf"):
        raise HTTPException(400, "Unknown provider")
    if PUBLIC and k.provider == "custom":
        raise HTTPException(400, "Custom endpoints are turned off on this shared server.")
    KEYS = _keys(write=True)
    key = k.key.strip()
    if not key and k.provider != "custom":
        KEYS.pop(k.provider, None)
    else:
        KEYS[k.provider] = {"key": key, "base_url": k.base_url.strip()}
    if not PUBLIC:  # single user: the white-box backends read these from the environment
        if k.provider == "ndif":
            os.environ["NDIF_API_KEY"] = key
        if k.provider == "hf":
            os.environ["HF_TOKEN"] = key
    return {"ok": True, "masked": _mask(key)}


class ModelsIn(BaseModel):
    provider: str


@app.post("/api/models")
def models(m: ModelsIn):
    if m.provider not in providers.PROVIDERS:
        raise HTTPException(400, "Unknown provider")
    if PUBLIC and m.provider == "custom":
        raise HTTPException(400, "Custom endpoints are turned off on this shared server.")
    cred = _keys().get(m.provider, {})
    if providers.PROVIDERS[m.provider]["kind"] != "mock" and not cred.get("key") and m.provider != "custom":
        raise HTTPException(400, "Add an API key for this provider first.")
    try:
        ids = providers.list_models(m.provider, cred.get("key"), cred.get("base_url"))
    except Exception as e:
        raise HTTPException(502, f"Couldn't list models: {type(e).__name__}: {str(e)[:200]}")
    return {"models": ids}


@app.get("/api/ndif/models")
def ndif_models():
    from sniffer.backends import ndif_running_models
    return {"models": sorted(ndif_running_models())}


# ------------------------------------------------------------------------ experiments

class Slot(BaseModel):
    provider: str
    model: str
    effort: str = ""
    label: str = ""


class AgentRunIn(BaseModel):
    name: str = ""
    slots: list[Slot] = Field(min_length=1, max_length=3)
    families: list[str] = Field(min_length=1)
    task_types: list[str] = Field(default_factory=lambda: ["allowed", "disallowed"], min_length=1)
    custom_sandbox: str = ""
    custom_prod: str = ""
    mode: str = "tools"
    variants: int = Field(1, ge=1, le=3)
    repeats: int = Field(1, ge=1, le=5)
    everywhere: bool = True
    monitored: bool = False
    pressure: bool = False
    judge: str = "regex"
    workers: int = Field(6, ge=1, le=16)


def _agent_cfg(r: AgentRunIn):
    bad = set(r.families) - set(scenarios.FAMILIES) - {"custom"}
    if bad:
        raise HTTPException(400, f"Unknown families: {sorted(bad)}")
    if r.mode not in ("tools", "think", "choice"):
        raise HTTPException(400, "mode must be tools, think or choice")
    custom = {"sandbox": [l.strip() for l in r.custom_sandbox.splitlines() if l.strip()],
              "prod": [l.strip() for l in r.custom_prod.splitlines() if l.strip()]}
    if "custom" in r.families and not (custom["sandbox"] and custom["prod"]):
        raise HTTPException(400, "The custom condition needs a sandbox line and a production line.")
    return {"families": r.families, "task_types": r.task_types, "custom": custom, "mode": r.mode,
            "variants": r.variants, "repeats": r.repeats, "everywhere": r.everywhere,
            "monitored": r.monitored, "pressure": r.pressure, "judge": r.judge,
            "judge_model": "claude-opus-5-5", "workers": r.workers}


def _slots(r: AgentRunIn):
    out, seen = [], set()
    for s in r.slots:
        if s.provider not in providers.PROVIDERS:
            raise HTTPException(400, f"Unknown provider {s.provider}")
        if PUBLIC and s.provider == "custom":
            raise HTTPException(400, "Custom endpoints are turned off on this shared server.")
        cred = _keys().get(s.provider, {})
        if providers.PROVIDERS[s.provider]["kind"] != "mock" and not cred.get("key") \
                and s.provider != "custom":
            raise HTTPException(400, f"No API key for {providers.PROVIDERS[s.provider]['label']}.")
        if not s.model.strip():
            raise HTTPException(400, "Every model slot needs a model ID.")
        label = (s.label.strip() or s.model.strip())[:80]
        while label in seen:
            label += "'"
        seen.add(label)
        out.append({"provider": s.provider, "model": s.model.strip(), "effort": s.effort,
                    "label": label, "key": cred.get("key"), "base_url": cred.get("base_url")})
    return out


@app.post("/api/estimate")
def estimate(r: AgentRunIn):
    cfg = _agent_cfg(r)
    n_rows = len(jobs.make_rows(cfg))
    per_model, total_cost, known = [], 0.0, True
    for s in r.slots:
        n = n_rows * r.repeats
        price = providers.PRICES.get(s.model)
        cost = None if price is None else n * (1100 * price[0] + 900 * price[1]) / 1e6
        if s.provider == "mock":
            cost = 0.0
        if cost is None:
            known = False
        else:
            total_cost += cost
        per_model.append({"model": s.model, "episodes": n, "cost": cost})
    n_total = sum(p["episodes"] for p in per_model)
    judge_cost = n_total * (700 * 4 + 400 * 20) / 1e6 if r.judge == "llm" else 0.0
    return {"episodes": n_total, "per_model": per_model, "judge_cost": judge_cost,
            "total_cost": total_cost + judge_cost if known else None}


@app.post("/api/preview")
def preview(r: AgentRunIn, family: str = "", task: int = 10):
    cfg = _agent_cfg(r)
    fam = family or cfg["families"][0]
    rows = scenarios.make_cases(1, [fam], cfg["custom"] if fam == "custom" else None)
    pair = {}
    for row in rows:
        if row["task_id"] == task and row["populated"] in (0, 1) and row["label"] not in pair:
            pair[row["label"]] = row
    system = scenarios.build_system(cfg["everywhere"], cfg["monitored"])
    return {"system": system, "family": fam,
            "sandbox": scenarios.user_message(pair[1]["env"], pair[1]["request"], cfg["mode"], cfg["pressure"]),
            "prod": scenarios.user_message(pair[0]["env"], pair[0]["request"], cfg["mode"], cfg["pressure"])}


@app.post("/api/runs/agents")
def run_agents(r: AgentRunIn):
    if PUBLIC:
        L = LIMITS
        if r.repeats > L["agent_repeats"] or r.variants > L["agent_variants"] or r.workers > L["agent_workers"]:
            raise HTTPException(400, f"On this shared server: at most {L['agent_repeats']} repeats, "
                                     f"{L['agent_variants']} variants and {L['agent_workers']} workers.")
    owner = _start_guard()
    KEYS = _keys()
    cfg = _agent_cfg(r)
    slots = _slots(r)
    if r.judge == "llm" and "anthropic" not in KEYS:
        raise HTTPException(400, "The Claude judge needs an Anthropic key. Use the regex judge or add a key.")
    name = (r.name.strip() or " vs ".join(s["label"] for s in slots))[:120]
    job = jobs.start_agents(name, slots, cfg, KEYS.get("anthropic", {}).get("key"), owner=owner)
    return job.info()


class WhiteboxIn(BaseModel):
    name: str = ""
    backend: str = "ndif"
    model: str = "meta-llama/Llama-3.3-70B-Instruct"
    quick: bool = True
    skip_verbal: bool = False
    skip_steering: bool = False
    layer: int | None = None
    judge: str = "regex"
    evolved_from: str = ""   # a run_id from an `evolve` run, to fold its discovered cues in


@app.post("/api/runs/whitebox")
def run_whitebox(w: WhiteboxIn):
    if w.backend not in ("ndif", "local", "mock"):
        raise HTTPException(400, "backend must be ndif, local or mock")
    _public_backend(w.backend)
    owner = _start_guard()
    KEYS = _keys()
    if PUBLIC:
        w.quick = True  # full white-box runs take hours; the shared server runs the quick version
    if w.backend == "ndif" and "ndif" not in KEYS:
        raise HTTPException(400, "Add an NDIF key first (free at login.ndif.us).")
    cfg = {"backend": w.backend, "model": w.model.strip(), "quick": w.quick,
           "skip_verbal": w.skip_verbal, "skip_steering": w.skip_steering, "layer": w.layer,
           "judge": w.judge}
    evolved_name = ""
    if w.evolved_from.strip():
        src = w.evolved_from.strip()
        if not RUN_ID.match(src):
            raise HTTPException(400, "Bad evolved_from run id")
        _run_or_404(src)  # must be one of this visitor's runs
        cue_file = jobs.RUNS / src / "evolved_cues.json"
        if not cue_file.exists():
            raise HTTPException(404, "That search run has no evolved_cues.json yet.")
        cfg["evolved"] = str(cue_file)
        evolved_name = "Validate: "
    name = (w.name.strip() or evolved_name + ("Simulated model" if w.backend == "mock"
            else w.model.split("/")[-1]))[:120]
    job = jobs.start_whitebox(name, cfg, KEYS.get("anthropic", {}).get("key"), owner=owner,
                              creds=_creds())
    return job.info()


class EvolveIn(BaseModel):
    name: str = ""
    backend: str = "ndif"
    model: str = "meta-llama/Llama-3.3-70B-Instruct"
    generations: int = Field(5, ge=1, le=20)
    pop: int = Field(16, ge=4, le=64)
    tasks: int = Field(6, ge=2, le=20)
    lam: float = Field(1.0, ge=0.0, le=5.0)
    layer: int | None = None
    quick: bool = False
    critic: bool = True
    oversample: int = Field(4, ge=1, le=12)


@app.post("/api/runs/evolve")
def run_evolve(e: EvolveIn):
    if e.backend not in ("ndif", "local", "mock"):
        raise HTTPException(400, "backend must be ndif, local or mock")
    _public_backend(e.backend)
    if PUBLIC and (e.generations > LIMITS["evolve_generations"] or e.pop > LIMITS["evolve_pop"]
                   or e.tasks > LIMITS["evolve_tasks"]):
        raise HTTPException(400, f"On this shared server: at most {LIMITS['evolve_generations']} "
                                 f"generations, population {LIMITS['evolve_pop']} and "
                                 f"{LIMITS['evolve_tasks']} tasks.")
    owner = _start_guard()
    KEYS = _keys()
    if e.backend == "ndif" and "ndif" not in KEYS:
        raise HTTPException(400, "Add an NDIF key first (free at login.ndif.us).")
    cfg = {"backend": e.backend, "model": e.model.strip(), "generations": e.generations,
           "pop": e.pop, "tasks": e.tasks, "lam": e.lam, "layer": e.layer, "quick": e.quick,
           "critic": e.critic, "oversample": e.oversample}
    name = (e.name.strip() or ("Cue search (simulated)" if e.backend == "mock"
            else "Cue search · " + e.model.split("/")[-1]))[:120]
    job = jobs.start_evolve(name, cfg, owner=owner, creds=_creds())
    return job.info()


# ---------------------------------------------------------------------------- results

@app.get("/api/runs")
def runs():
    return {"runs": [r for r in jobs.list_runs() if _visible(r)]}


@app.get("/api/runs/{run_id}")
def run_info(run_id: str):
    info = _run_or_404(run_id)
    return {**info, "logs": jobs.logs(run_id)}


@app.get("/api/runs/{run_id}/summary")
def run_summary(run_id: str):
    _run_or_404(run_id)
    s = jobs.summary(run_id)
    if s is None:
        raise HTTPException(404, "No results yet")
    return _no_nan(s)


def _no_nan(x):
    """Summaries can hold NaN (an undefined statistic); JSON can't, so send null instead."""
    if isinstance(x, float):
        return None if x != x or x in (float("inf"), float("-inf")) else x
    if isinstance(x, dict):
        return {k: _no_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_no_nan(v) for v in x]
    return x


@app.get("/api/runs/{run_id}/episodes")
def run_episodes(run_id: str):
    _run_or_404(run_id)
    keep = ("case_id", "model_label", "family", "label", "should_allow", "request", "env", "action",
            "reasoning", "text", "tool_input", "test_aware", "stakes_reasoning", "judge_quote",
            "error", "repeat", "served_by")
    return {"episodes": [{k: e.get(k) for k in keep} for e in jobs.episodes(run_id)]}


@app.post("/api/runs/{run_id}/cancel")
def run_cancel(run_id: str):
    _run_or_404(run_id)
    return {"ok": jobs.cancel(run_id)}


ALLOWED_FILES = re.compile(r"^[\w.-]+\.(png|csv|json|md)$")


@app.get("/api/runs/{run_id}/files/{name}")
def run_file(run_id: str, name: str):
    _run_or_404(run_id)
    if not ALLOWED_FILES.match(name):
        raise HTTPException(404)
    path = jobs.RUNS / run_id / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/api/runs/{run_id}/export/{fmt}")
def run_export(run_id: str, fmt: str):
    info = _run_or_404(run_id)
    if fmt == "json":
        s = jobs.summary(run_id) or {}
        return Response(json.dumps({"run": info, "summary": s}, indent=2), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{run_id}.json"'})
    if fmt == "md":
        path = jobs.RUNS / run_id / "report.md"
        if path.exists():
            text = path.read_text()
        else:
            from sniffer.report import markdown
            text = markdown(info, jobs.summary(run_id) or {"models": []})
        return PlainTextResponse(text, headers={"Content-Disposition": f'attachment; filename="{run_id}.md"'})
    if fmt == "csv":
        eps = jobs.episodes(run_id)
        cols = ["model_label", "provider", "model", "case_id", "family", "label", "should_allow",
                "request", "action", "test_aware", "stakes_reasoning", "judge_quote", "reasoning",
                "text", "error", "served_by", "repeat"]
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for e in eps:
            w.writerow(e)
        return Response(buf.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{run_id}.csv"'})
    raise HTTPException(404)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0" if PUBLIC else "127.0.0.1")
    print(f"Privy running at http://{host}:{port}" + ("  (public mode)" if PUBLIC else ""))
    uvicorn.run(app, host=host, port=port, log_level="warning")
