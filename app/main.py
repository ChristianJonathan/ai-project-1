"""No-JS AI Assistant – simple polling version (no streaming)."""
from __future__ import annotations
import asyncio, csv, datetime as dt, io, json, logging, os, re, secrets, sqlite3, time, uuid

from app.settings import settings  

import bleach, httpx, redis.asyncio as redis
from httpx import HTTPStatusError
from markdown_it import MarkdownIt

from fastapi import (
    BackgroundTasks, Depends, FastAPI, Form, HTTPException, Request
)
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

security   = HTTPBasic()
templates  = Jinja2Templates(directory="app/templates")
md         = MarkdownIt("commonmark")
ALLOWED    = set(bleach.sanitizer.ALLOWED_TAGS) | {"p", "pre", "code"}
THINK_RE   = re.compile(r"<think>(.*?)</think>", re.S | re.I)

# ───────────────────────── DB bootstrap / migration ───────────────────
def init_db() -> sqlite3.Connection:
    db = sqlite3.connect(settings.DB_PATH, check_same_thread=False)
    db.executescript("""
        pragma journal_mode=WAL;
        pragma synchronous=NORMAL;
        create table if not exists sessions(
            id text primary key,
            started_at real
        );
    """)
    cols = {c for _, c, *_ in db.execute("pragma table_info(messages)")}
    if not cols:
        db.execute("""
            create table messages(
                id       integer primary key,
                session_id text not null references sessions(id) on delete cascade,
                ts       real,
                role     text,
                content  text,
                thought  text,
                model    text,
                params   text
            );
        """)
    else:
        for col in ("session_id", "thought", "model", "params"):
            if col not in cols:
                db.execute(f"alter table messages add column {col} text;")
    db.execute("create index if not exists idx_messages_session on messages(session_id);")
    db.commit()
    return db

DB       = init_db()
db_lock  = asyncio.Lock()

# ───────────────────────── Helper funcs ───────────────────────────────
def md_to_html(text: str) -> str:
    return bleach.clean(md.render(text), tags=ALLOWED, strip=True)

def get_session(req: Request) -> str:
    cookie = req.cookies.get("sid", "")
    if "|" in cookie:
        return cookie.split("|", 1)[0]
    return "default"

def new_session_response() -> RedirectResponse:
    sid = uuid.uuid4().hex
    cookie = f"{sid}|{int(time.time())}|sig"
    resp = RedirectResponse("/", 303)
    resp.set_cookie("sid", cookie, httponly=True, samesite="lax")
    return resp

def _export_rows(sid: str) -> tuple[str, io.StringIO]:
    rows = DB.execute("""
        select ts,role,content,model,params
          from messages where session_id=? order by id
    """, (sid,)).fetchall()
    if not rows:
        raise HTTPException(404, "Session not found")

    txt = "\n".join(f"{r.upper()}: {c}" for _, r, c, _, _ in rows)

    buf = io.StringIO()
    csv.writer(buf).writerows(
        [("timestamp","role","content","model","params"), *rows]
    )
    buf.seek(0)
    return txt, buf

# ───────────────────────── Rate-limit helper ───────────────────────────
def _rl(spec: str) -> RateLimiter:
    """
    Convert a string like '10/60' → RateLimiter(times=10, seconds=60).
    Keeps decorator lines short and readable.
    """
    n, secs = map(int, spec.split("/", 1))
    return RateLimiter(times=n, seconds=secs)

# ───────────────────────── Chat-history helpers ─────────────────────────
def rough_tokens(txt: str) -> int:
    """
    Very cheap token estimate: 1 token ≈ 4 characters.
    Good enough to keep us under the LLM context window.
    """
    return max(1, len(txt) // 4)

async def build_context(
    sid: str,
    new_user_prompt: str,
    token_budget: int = 4_096,
    keep_last_n: int = 8,
) -> list[dict]:
    """
    Return a list of {"role": "...", "content": "..."} messages containing
    the *last* `keep_last_n` user-assistant pairs for this session, while
    never exceeding `token_budget` (rough estimate).

    The newest messages are kept; older ones are discarded first.
    """
    # fetch the newest (DESC) rows, then reverse → oldest-first
    rows = DB.execute(
        """
        SELECT role, content
          FROM messages
         WHERE session_id = ?
           AND role IN ('user','assistant')
      ORDER BY id DESC
         LIMIT ?
        """,
        (sid, keep_last_n * 2),
    ).fetchall()[::-1]  # chronological order

    context: list[dict] = []
    total_tokens = 0
    for role, content in rows:
        est = rough_tokens(content)
        if total_tokens + est > token_budget:
            # stop adding older turns once we would exceed budget
            break
        context.append({"role": role, "content": content})
        total_tokens += est

    # finally add the NEW user prompt
    context.append({"role": "user", "content": new_user_prompt})
    return context

# ───────────────────────── Auth & startup ─────────────────────────────
async def check_auth(creds: HTTPBasicCredentials = Depends(security)):
    if not (secrets.compare_digest(creds.username, settings.AUTH_USER) and
            secrets.compare_digest(creds.password, settings.AUTH_PASS)):
        raise HTTPException(
            401, detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"}
        )

async def _startup():
    r = redis.from_url("redis://redis:6379", encoding="utf-8",
                       decode_responses=True)
    await FastAPILimiter.init(r, prefix="rl")

app = FastAPI(on_startup=[_startup])
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ───────────────────────── Error message ────────────────────────
def _render_error(request: Request, code: int, detail: str):
    return templates.TemplateResponse(
        "error.html",
        {"request": request, "code": code, "detail": detail},
        status_code=code,
    )

RATE_LIMIT   = _rl(settings.RATE_LIMIT)     # e.g. "10/60"
STREAM_LIMIT = _rl(settings.STREAM_LIMIT)   # e.g. "20/60"

# ───────────────────────── Background LLM call ────────────────────────
async def generate_reply(prompt: str, model: str, temp: float, max_tok: int,
                         row_id: int, sid: str):
    
    msgs = [
        {"role": "system", "content": settings.SYSTEM_PROMPT},
        *await build_context(sid, prompt,
                             token_budget=4_096,
                             keep_last_n=8),
    ]

    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": temp, "num_predict": max_tok},
        "messages": msgs 
    }

    try:
        async with httpx.AsyncClient(timeout=None) as cli:
            r = await cli.post(settings.LLM_URL, json=payload)
            r.raise_for_status()
            data = r.json()               # ← MUST be before we log it
            logging.warning("RAW JSON from LLM:\n%s",
                            json.dumps(data, indent=2)[:800])

    except HTTPStatusError as exc:
        thought = ""
        answer  = (f"💥 Model {model!r} unavailable "
                   f"({exc.response.status_code}). "
                   f"Run `ollama pull {model}` in another terminal.")
    else:
        def _g(d: dict, k: str) -> str:   # tiny helper
            return d.get(k, "") if isinstance(d, dict) else ""
        
        raw = (
            _g(data, "response")                                     # /completion
            or _g(_g(data, "message"), "content")                    # /chat
            or _g(_g(_g(data, "choices")[0], "message"), "content")  # openai-ish
            or _g(_g(data, "choices")[0], "text")                    # some GGML
        ).strip()

        m = THINK_RE.search(raw)
        thought = m.group(1).strip() if m else ""
        answer  = raw[m.end():].lstrip() if m else raw

    async with db_lock:
        DB.execute("""
          insert into messages(id,session_id,ts,role,
                               content,thought,model,params)
               values (?,?,?,?,?,?,?,?)
        """,(row_id+1, sid, time.time(), "assistant",
             answer, thought,
             model, json.dumps({"temperature": temp, "max_tok": max_tok})))
        DB.commit()

# ───────────────────────── Routes – sessions / history ────────────────
@app.get("/new-session", response_class=HTMLResponse,
         dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])
def new_session_prompt(request: Request):
    sid   = get_session(request)
    count = DB.execute("select count(*) from messages where session_id=?",
                       (sid,)).fetchone()[0]
    if count == 0:
        return new_session_response()
    return templates.TemplateResponse("confirm_new_session.html",
        {"request": request, "msg_count": count})

@app.post("/new-session/confirm",
          dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])
def new_session_confirm():
    return new_session_response()

@app.post("/clear", dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])
async def clear(request: Request):
    sid = get_session(request)
    async with db_lock:
        DB.execute("delete from messages where session_id=?", (sid,))
        DB.commit()
    return RedirectResponse("/", 303)

@app.get("/export/{sid}", dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])
def export_txt(sid: str):
    txt, _ = _export_rows(sid)
    name = f"chat_{sid[:8]}_{dt.datetime.utcnow():%Y%m%dT%H%M%SZ}.txt"
    return HTMLResponse(txt, headers={
        "Content-Type":"text/plain; charset=utf-8",
        "Content-Disposition":f'attachment; filename="{name}"'
    })

@app.get("/export.csv/{sid}", dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])
def export_csv(sid: str):
    _, buf = _export_rows(sid)
    name = f"chat_{sid[:8]}_{dt.datetime.utcnow():%Y%m%dT%H%M%SZ}.csv"
    return StreamingResponse(buf, media_type="text/csv",
        headers={"Content-Disposition":f'attachment; filename="{name}"'})

# ───────────────────────── Chat history (list & thread) ───────────────

@app.get(
    "/history",
    response_class=HTMLResponse,
    dependencies=[Depends(check_auth), Depends(RATE_LIMIT)],
)
def history(request: Request):
    """
    Show a list of previous sessions with first-message preview and turn count.
    """
    sessions = DB.execute(
        """
        SELECT
            s.id,
            datetime(s.started_at,'unixepoch','localtime')  AS started,
            COALESCE((
                SELECT content FROM messages
                 WHERE session_id = s.id AND role = 'user'
                 ORDER BY id LIMIT 1
            ), '')                                          AS title,
            COUNT(m.id)                                     AS turns
        FROM sessions AS s
        LEFT JOIN messages AS m ON m.session_id = s.id
        GROUP BY s.id
        ORDER BY s.started_at DESC
        """
    ).fetchall()

    return templates.TemplateResponse(
        "history.html",
        {"request": request, "sessions": sessions},
    )

@app.get(
    "/history/{sid}",
    response_class=HTMLResponse,
    dependencies=[Depends(check_auth), Depends(RATE_LIMIT)],
)
def history_thread(request: Request, sid: str):
    """
    Show a single session transcript (chronological).
    """
    rows = DB.execute(
        """
        SELECT role,
               datetime(ts,'unixepoch','localtime') AS ts,
               content,
               COALESCE(thought,'')
          FROM messages
         WHERE session_id = ?
      ORDER BY id
        """,
        (sid,),
    ).fetchall()

    # get start-time & first user message for header
    started, title = DB.execute(
        """
        SELECT
          datetime(started_at,'unixepoch','localtime'),
          COALESCE((
             SELECT content FROM messages
              WHERE session_id = ? AND role = 'user'
              ORDER BY id LIMIT 1
          ), '')
        FROM sessions WHERE id = ?
        """,
        (sid, sid),
    ).fetchone()

    return templates.TemplateResponse(
        "history_thread.html",
        {
            "request": request,
            "rows": rows,
            "sid": sid,
            "started": started,
            "title": title,
            "md_to_html": md_to_html,  # if you want markdown in template
        },
    )

# ───────────────────────── Chat (polling) flow ────────────────────────
@app.post("/chat", dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])
async def chat(
    request: Request,
    background_tasks: BackgroundTasks,
    prompt: str  = Form(...),
    model:  str  = Form(settings.MODEL_DEFAULT),
    temp:   float= Form(0.7),
    max_tok:int  = Form(1000),
):
    sid = get_session(request)
    async with db_lock:
        DB.execute("insert or ignore into sessions(id,started_at) values(?,?)",
                   (sid, time.time()))
        row_id = DB.execute("""
            insert into messages(session_id,ts,role,content,model,params)
                 values (?,?,?,?,?,?)
        """,(sid, time.time(), "user", prompt, model,
             json.dumps({"temperature": temp, "max_tok": max_tok}))
        ).lastrowid
        DB.commit()

    background_tasks.add_task(generate_reply,
                              prompt, model, temp, max_tok, row_id, sid)
    return RedirectResponse(f"/wait/{row_id}", 303)

@app.get("/wait/{msg_id}", response_class=HTMLResponse,
         dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])
async def wait(request: Request, msg_id: int):
    sid = get_session(request)
    done = DB.execute("""
        select 1 from messages
         where id=? and role='assistant' and session_id=?
    """,(msg_id+1, sid)).fetchone()
    if done is None:
        return templates.TemplateResponse("waiting.html",
            {"request": request, "refresh": True, "msg_id": msg_id})
    return RedirectResponse("/", 303)

# ───────────────────────── Home page ──────────────────────────────────
@app.get("/", response_class=HTMLResponse,
         dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])
def home(request: Request):
    sid = get_session(request)
    rows_raw = DB.execute("""
        select role,
               datetime(ts,'unixepoch','localtime'),
               content,
               coalesce(thought,'')
          from messages
         where session_id=?
      order by id
    """,(sid,)).fetchall()

    rows = [(r, ts, md_to_html(c), md_to_html(t)) for r, ts, c, t in rows_raw]
    return templates.TemplateResponse("index.html",
        {"request": request,
         "rows": rows,
         "models": [settings.MODEL_DEFAULT],
         "sid": sid,
         }
    )  
