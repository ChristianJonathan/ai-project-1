from fastapi import (
    FastAPI, Form, Request, BackgroundTasks,
    Depends, HTTPException
)
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles 
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

from httpx import HTTPStatusError

from markdown_it import MarkdownIt
import bleach, httpx, os, sqlite3, time, asyncio, secrets, redis.asyncio as redis
import uuid, datetime, csv, io, json, logging, re

# ───────────────────────── Configuration ────────────────────────────────
DB_PATH  = "/data/chat.db"
LLM_URL  = os.getenv("LLM_URL", "http://host.docker.internal:11434/api/chat")
USER     = os.getenv("AUTH_USER", "admin")
PASS     = os.getenv("AUTH_PASS", "admin")
SYSTEM_PROMPT = (
    You are a helpful assistant.
    Think step-by-step **in ≤ 100 tokens**, enclosed in <think> … </think>.
    Then answer the user clearly.
)

logging.basicConfig(level=logging.INFO)

RATE_LIMIT = RateLimiter(times=10, seconds=60)
security   = HTTPBasic()
templates  = Jinja2Templates(directory="app/templates")
md         = MarkdownIt("commonmark")
ALLOWED    = set(bleach.sanitizer.ALLOWED_TAGS).union({"p", "pre", "code"})

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)

# ───────────────────────── DB bootstrap / migration ─────────────────────
def init_db() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL;")
    db.execute("PRAGMA synchronous=NORMAL;")

    db.execute("""CREATE TABLE IF NOT EXISTS sessions(
                    id TEXT PRIMARY KEY,
                    started_at REAL
                  )""")

    cols = {row[1] for row in db.execute("PRAGMA table_info(messages)")}
    if not cols:
        db.execute("""CREATE TABLE messages(
                        id INTEGER PRIMARY KEY,
                        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                        ts REAL,
                        role TEXT,
                        content TEXT,
                        thought TEXT,                     -- ❶ NEW
                        model TEXT,
                        params TEXT
                      )""")
    else:
        for col in ("session_id","model","params","thought"):           # ❷ ensure all cols
            if col not in cols:
                db.execute(f"ALTER TABLE messages ADD COLUMN {col} TEXT")
    db.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
    db.commit()
    return db

DB       = init_db()
db_lock  = asyncio.Lock()

# ───────────────────────── Helpers ──────────────────────────────────────
def md_to_html(text: str) -> str:
    raw = md.render(text)
    return bleach.clean(raw, tags=ALLOWED, strip=True)

def get_session(request: Request) -> str:
    cookie = request.cookies.get("sid")
    if cookie and "|" in cookie:
        return cookie.split("|")[0].strip()
    return "default"

def new_session_response() -> RedirectResponse:
    sid = uuid.uuid4().hex
    stamp = int(time.time())
    cookie = f"{sid}|{stamp}|sig"
    resp = RedirectResponse("/", 303)
    resp.set_cookie("sid", cookie, httponly=True, samesite="lax")
    return resp

# helper for exporting ---------------------------------------------------
def _session_rows_txt_csv(sid: str):
    rows = DB.execute("""
        SELECT ts, role, content, model, params FROM messages
        WHERE session_id=? ORDER BY id
    """, (sid,)).fetchall()
    if not rows:
        raise HTTPException(404, "Session not found")

    txt = "\n".join(f"{r.upper()}: {c}" for _, r, c, _, _ in rows)
    buf = io.StringIO()
    w = csv.writer(buf); w.writerow(["timestamp","role","content","model","params"])
    w.writerows(rows); buf.seek(0)
    return txt, buf

# ───────────────────────── Auth & startup ───────────────────────────────
async def check_auth(creds: HTTPBasicCredentials = Depends(security)):
    if not (secrets.compare_digest(creds.username, USER) and
            secrets.compare_digest(creds.password, PASS)):
        raise HTTPException(
            401, detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"}
        )

async def _startup():
    r = redis.from_url("redis://redis:6379", encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(r, prefix="rl")

app = FastAPI(on_startup=[_startup])
app.mount("/static", StaticFiles(directory="app/static"), name="static")   # ❷ serve CSS/JS


# ───────────────────────── Background LLM call ──────────────────────────
import re
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S | re.I)

async def generate_reply(prompt, model, temp, max_tok, row_id, session):
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": temp, "num_predict": max_tok},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    }

    # ────────── 1. talk to Ollama, catch “model not found”  ───────────
    try:
        async with httpx.AsyncClient(timeout=None) as cli:
            r = await cli.post(LLM_URL, json=payload)
            r.raise_for_status()                           # ← NEW
    except HTTPStatusError as exc:
        # 404 or 400 typically means the model isn't downloaded
        raw = (f"💥 Model {model!r} unavailable "
               f"({exc.response.status_code}). "
               f"Run `ollama pull {model}` in another terminal.")
        thought = ""                                       # no CoT
        answer  = raw
    else:
        # ────────── 2. normal successful reply  ──────────────────────
        data = r.json()                                    # ← full dict
        if "choices" in data:
            raw = data["choices"][0]["message"]["content"].strip()
        elif "message" in data:                            # LocalAI style
            raw = data["message"]["content"].strip()
        else:                                              # /completion style
            raw = data.get("response", "").strip()

        # split <think> … </think>
        m = THINK_RE.search(raw)
        if m:
            thought = m.group(1).strip()
            answer  = raw[m.end():].lstrip()
        else:
            thought = ""
            answer  = raw

    # ────────── 3. write to SQLite  ───────────────────────────────────
    async with db_lock:
        DB.execute("""
          INSERT INTO messages(id, session_id, ts, role, content, thought, model, params)
          VALUES (?,?,?,?,?,?,?,?)
        """, (
            row_id + 1, session, time.time(), "assistant",
            answer, thought,
            model,
            json.dumps({"temperature": temp, "max_tok": max_tok}),
        ))
        DB.commit()

# ───────────────────────── Routes (unchanged except queries) ────────────
# ... new-session, clear, export routes stay as in previous version ...

# ── create / confirm new session ───────────────────────────────────────
@app.get("/new-session", response_class=HTMLResponse,
         dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])
def new_session_prompt(request: Request):
    sid = get_session(request)
    count = DB.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=?", (sid,)
    ).fetchone()[0]
    # no messages → start fresh immediately
    if count == 0:
        return new_session_response()

    # ask for confirmation
    return templates.TemplateResponse(
        "confirm_new_session.html",
        {"request": request, "msg_count": count}
    )

@app.post("/new-session/confirm",
          dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])
def new_session_confirm():
    return new_session_response()

# ---------- history routes ---------------------------------------------
@app.get("/history", response_class=HTMLResponse,
         dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])
def history(request: Request):
    rows = DB.execute("""
        SELECT s.id,
               datetime(s.started_at,'unixepoch','localtime') AS started,
               COALESCE((
                 SELECT content FROM messages
                  WHERE session_id=s.id AND role='user'
                  ORDER BY id LIMIT 1
               ), '')                                         AS title,
               COUNT(m.id)                                    AS turns
        FROM sessions AS s
        LEFT JOIN messages m ON m.session_id = s.id
        GROUP BY s.id
        ORDER BY s.started_at DESC;
    """).fetchall()
    return templates.TemplateResponse("history.html",
        {"request": request, "sessions": rows})

@app.get("/history/{sid}", response_class=HTMLResponse,
         dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])
def history_thread(request: Request, sid: str):
    rows = DB.execute("""
        SELECT role,
               datetime(ts,'unixepoch','localtime') AS tstr,
               content,
               COALESCE(thought,'')
        FROM messages
        WHERE session_id = ?
        ORDER BY id
    """, (sid,)).fetchall()

    started, title = DB.execute("""
        SELECT datetime(started_at,'unixepoch','localtime'),
               COALESCE((
                 SELECT content FROM messages
                  WHERE session_id=? AND role='user'
                  ORDER BY id LIMIT 1
               ), '')
        FROM sessions WHERE id=?;
    """,(sid,sid)).fetchone()

    return templates.TemplateResponse("history_thread.html",
        {"request": request, "rows": rows, "sid": sid,
         "started": started, "title": title})

# ---------- chat & home -------------------------------------------------
@app.post("/chat", dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])
async def chat(
    request: Request,
    background_tasks: BackgroundTasks,
    prompt: str  = Form(...),
    model: str   = Form("qwen3:1.7b"),
    temp:  float = Form(0.7),
    max_tok: int = Form(400),
):
    sid = get_session(request)
    async with db_lock:
        DB.execute("INSERT OR IGNORE INTO sessions(id, started_at) VALUES(?,?)",
                   (sid, time.time()))
        row_id = DB.execute("""
            INSERT INTO messages(session_id, ts, role, content, model, params)
            VALUES (?,?,?,?,?,?)
        """,(sid, time.time(), "user", prompt, model,
             json.dumps({"temperature": temp, "max_tok": max_tok}))
        ).lastrowid
        DB.commit()

    background_tasks.add_task(generate_reply, prompt, model, temp, max_tok, row_id, sid)
    return RedirectResponse(f"/wait/{row_id}", 303)

@app.get("/wait/{msg_id}", response_class=HTMLResponse,
         dependencies=[Depends(check_auth),
                        Depends(RateLimiter(times=120, seconds=60))])  # relaxed
async def wait(request: Request, msg_id: int):
    sid = get_session(request)
    row = DB.execute("""
        SELECT 1 FROM messages
         WHERE id=? AND role='assistant' AND session_id=?
    """,(msg_id+1, sid)).fetchone()
    if row is None:
        return templates.TemplateResponse("waiting.html",
            {"request": request, "refresh": True, "msg_id": msg_id})
    return RedirectResponse("/", 303)

@app.get("/", response_class=HTMLResponse,
         dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])

# ── Home page query (4 columns) ────────────────────────────────────────
@app.get("/", response_class=HTMLResponse,
         dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])
def home(request: Request):
    sid = get_session(request)
    rows_raw = DB.execute("""
        SELECT
          role,
          datetime(ts,'unixepoch','localtime') AS ts,
          content,
          COALESCE(thought,'')
        FROM messages
        WHERE session_id=? ORDER BY id
    """,(sid,)).fetchall()

    rows = [(r, ts, md_to_html(c), md_to_html(t)) for r,ts,c,t in rows_raw]

    return templates.TemplateResponse(
        "index.html",
        {"request": request,
         "rows": rows,
         "models": ["qwen3:1.7b"]}
    )


