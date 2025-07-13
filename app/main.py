"""No-JS AI Assistant – simple polling version (no streaming)."""
from __future__ import annotations
import asyncio, csv, datetime as dt, io, json, logging, os, re, secrets, sqlite3, time, uuid, tempfile, pathlib, shutil

from app.settings import settings  

import bleach, httpx, redis.asyncio as redis
from httpx import HTTPStatusError
from markdown_it import MarkdownIt

from fastapi import (
    BackgroundTasks, Depends, FastAPI, Form, HTTPException, Request, UploadFile, File
)
import base64, imghdr, mimetypes
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.concurrency import run_until_first_complete
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

from starlette.exceptions import HTTPException as StarletteHTTPException

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

security   = HTTPBasic()
templates  = Jinja2Templates(directory="app/templates")
md         = MarkdownIt("commonmark")
ALLOWED    = set(bleach.sanitizer.ALLOWED_TAGS) | {"p", "pre", "code"}
THINK_RE   = re.compile(r"<think>(.*?)</think>", re.S | re.I)

TMP_IMG_DIR = "/tmp/ollama_images"
os.makedirs(TMP_IMG_DIR, exist_ok=True)

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
        return _render_error(request, 404, "Session not found")

    def _fmt(ts, role, content, model, _params):
        if role == "assistant":
            return f"{role.upper()}({model}): {content}"
        return f"{role.upper()}: {content}"
    txt = "\n".join(_fmt(*row) for row in rows)

    buf = io.StringIO()
    csv.writer(buf).writerows(
        [("timestamp","role","content","model","params"), *rows]
    )
    buf.seek(0)
    return txt, buf

def _model_guard(name: str) -> str:
    if name not in settings.MODEL_CHOICES:
        return _render_error(request, 400, f"Unknown model {name!r}")
    return name

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
        return _render_error(
            request, 401, detail="Invalid credentials",
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
async def generate_reply(
        prompt:   str,
        model:    str,
        temp:     float,
        max_tok:  int,
        row_id:   int,
        sid:      str,
        img_b64: str | None = None,    # <- this is now a plain string
):
    
    # ─── optional image handling ─────────────────────────
    images_field: list[str] = []

    msgs = [
        {"role": "system", "content": settings.SYSTEM_PROMPT},
        *await build_context(sid, prompt,
                             token_budget=4_096,
                             keep_last_n=8),
    ]

    if img_b64:                      # add image to the user turn
        msgs.append({
            "role": "user",
            "content": "",           # Gemma vision ignores this
            "images": [img_b64],     # Ollama /chat accepts list[str]
        })

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

# ─────────────── helper that just inserts the assistant row ───────────────
async def _store_assistant_row(
        sid: str, answer: str, thought: str,
        model: str, temp: float, max_tok: int
):
    async with db_lock:
        DB.execute("""insert into messages(session_id,ts,role,
                                           content,thought,model,params)
                         values (?,?,?,?,?,?,?)""",
                   (sid, time.time(), "assistant",
                    answer, thought, model,
                    json.dumps({"temperature": temp, "max_tok": max_tok})))
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

@app.get("/history", response_class=HTMLResponse,
         dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])
def history(request: Request):
    """
    Show all sessions, newest first.
    """
    rows = DB.execute(
        """
        SELECT  s.id,
                datetime(s.started_at,'unixepoch','localtime')          AS started,
                COALESCE((
                    SELECT content
                      FROM messages
                     WHERE session_id = s.id AND role = 'user'
                  ORDER BY id LIMIT 1
                ), '')                                                  AS title,
                COUNT(m.id)                                             AS turns
          FROM sessions AS s
     LEFT JOIN messages  AS m  ON m.session_id = s.id
      GROUP BY s.id
      ORDER BY s.started_at DESC
        """
    ).fetchall()

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "sessions": rows,   # each row: (sid, started, title, turns)
        },
    )
@app.get("/history/{sid}", response_class=HTMLResponse,
         dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])
def history_thread(request: Request, sid: str):
    """Single session transcript (chronological)."""
    rows_raw = DB.execute(
        """
        SELECT role,
               datetime(ts,'unixepoch','localtime'),
               content,
               COALESCE(thought,'') AS thought,
               COALESCE(model,'')   AS model
          FROM messages
         WHERE session_id = ?
      ORDER BY id
        """,
        (sid,),
    ).fetchall()

    # 🖼️ render Markdown here (same pattern as home())
    rows = [
        (role, ts, md_to_html(content), md_to_html(thought), model)
        for role, ts, content, thought, model in rows_raw
    ]

    # header info
    started, title = DB.execute(
        """
        SELECT datetime(started_at,'unixepoch','localtime'),
               COALESCE((
                   SELECT content FROM messages
                    WHERE session_id = ? AND role = 'user'
                 ORDER BY id LIMIT 1
               ), '')
          FROM sessions
         WHERE id = ?
        """,
        (sid, sid),
    ).fetchone()

    return templates.TemplateResponse(
        "history_thread.html",
        {
            "request": request,
            "rows": rows,          # now contains HTML-ready chunks
            "sid": sid,
            "started": started,
            "title": title,
        },
    )

# ───────────────────────── Chat (streaming) via hidden <iframe> ───────────
@app.post("/chat-stream", dependencies=[Depends(check_auth), Depends(STREAM_LIMIT)])
async def chat_stream(
    request: Request,
    prompt:  str  = Form(...),
    model:   str  = Form(settings.MODEL_DEFAULT),
    temp:    float = Form(0.7),
    max_tok: int  = Form(2000),
    image: UploadFile | None = File(None), 
):
    model = _model_guard(model)
    sid = get_session(request)

    # ── 0️⃣  persist the uploaded image (if any) ───────────────────────
    extra_images: list[str] = []                         
    if image and image.filename:                         
        head = await image.read(32)
        await image.seek(0)
        if imghdr.what(None, head) is None:
            return _render_error(request, 400, "Uploaded file is not a valid image")
        fd, tmp_name = tempfile.mkstemp(
            dir=TMP_IMG_DIR,
            suffix=pathlib.Path(image.filename).suffix or ".png",
        )
        with open(fd, "wb") as f:
            shutil.copyfileobj(image.file, f)
        extra_images.append(tmp_name)                    

    # 1) save the USER turn immediately
    async with db_lock:
        DB.execute("insert or ignore into sessions(id,started_at) values(?,?)",
                   (sid, time.time()))
        DB.execute("""insert into messages(session_id,ts,role,content,model,params)
                         values (?,?,?,?,?,?)""",
                   (sid, time.time(), "user", prompt, model,
                    json.dumps({"temperature": temp, "max_tok": max_tok})))
        DB.commit()

    # 2) build context and fire Ollama *streaming*
    msgs = [
        {"role": "system", "content": settings.SYSTEM_PROMPT},
        *await build_context(sid, prompt, token_budget=4096, keep_last_n=8),
    ]
    if extra_images:                                     # ⬅️ NEW
        msgs.append({
            "role": "user",
            "content": "",
            "images": extra_images,      # Ollama passes file paths
        })

    # # ─── optional image handling ───────────────────────────────
    # if image and image.filename:
    #     raw_bytes = await image.read()
    #     # tiny sanity-check – reject non-images
    #     if imghdr.what(None, raw_bytes) is None:
    #         raise HTTPException(400, "Uploaded file is not a valid image")
    #     b64 = base64.b64encode(raw_bytes).decode()
    #     mime = (
    #         mimetypes.guess_type(image.filename)[0]
    #         or "image/png"
    #     )
    # # Gemma vision uses OpenAI style: {"type":"image_url","image_url":{"url":"data:<mime>;base64,<data>"}}
    #     msgs.append(
    #     {
    #         "role": "user",
    #         "content": "",
    #         "images": [  # 🤖 Ollama /chat supports this field
    #         f"data:{mime};base64,{b64}"
    #         ],
    #     }
    #     )

    payload = {
        "model": model,
        "stream": True,
        "options": {"temperature": temp, "num_predict": max_tok},
        "messages": msgs,
    }

    async def _gen():
        collected: list[str] = []
        yield (
        "<!doctype html><meta charset=utf-8>"
        "<body style='font-family:system-ui'>"
        )
        # 1️⃣  Stream tokens from Ollama → browser
        async with httpx.AsyncClient(timeout=None) as cli:
            async with cli.stream("POST", settings.LLM_URL, json=payload) as r:
                async for line in r.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = (
                        # ① OpenAI-style delta
                        data.get("choices", [{}])[0].get("delta", {}).get("content")
                        # ② Ollama /completion streaming
                        or data.get("response")
                        # ③ Ollama /chat streaming  ← NEW
                        or (data.get("message") or {}).get("content")
                        # ④ Fallbacks (GGML etc.)
                        or data.get("text")
                        or ""
                    )
                    if chunk:
                        collected.append(chunk)
                        yield bleach.clean(chunk).replace("\n", "<br>")
        yield "</body></html>"     # 2️⃣  close the tiny HTML doc

        # 3️⃣  Build answer + schedule DB write **in background**
        raw = "".join(collected)
        m   = THINK_RE.search(raw)
        thought = m.group(1).strip() if m else ""
        answer  = raw[m.end():].lstrip() if m else raw

        async def _store_and_clean():                    
            await _store_assistant_row(
                sid, answer, thought, model, temp, max_tok
            )
            for path in extra_images:
                try: pathlib.Path(path).unlink()
                except Exception: pass

        asyncio.create_task(_store_and_clean())

    return StreamingResponse(_gen(), media_type="text/html")
    
# ───────────────────────── Chat (polling) flow ────────────────────────
@app.post("/chat", dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])
async def chat(
    request: Request,
    background_tasks: BackgroundTasks,
    prompt: str  = Form(...),
    model:  str  = Form(settings.MODEL_DEFAULT),
    temp:   float= Form(0.7),
    max_tok:int  = Form(2000),
    image: UploadFile | None = File(None)
):
    
    if image and image.filename:                       
        return _render_error(request, 400, "Image uploads require /chat-stream")
    
    model = _model_guard(model)
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

# ───────────────────────── Exception handling ─────────────────────────────
@app.exception_handler(StarletteHTTPException)
async def http_exc(request: Request, exc: StarletteHTTPException):
    # FastAPI already turned it into StarletteHTTPException
    # -> re-use our nicer template
    return _render_error(request, exc.status_code, exc.detail)

@app.exception_handler(Exception)
async def unhandled_exc(request: Request, exc: Exception):
    logging.exception("UNCAUGHT ERROR")
    return _render_error(request, 500, "Internal server error")

# ───────────────────────── Home page ──────────────────────────────────
@app.get("/", response_class=HTMLResponse,
         dependencies=[Depends(check_auth), Depends(RATE_LIMIT)])
def home(request: Request):
    sid = get_session(request)

    # 1️⃣  fetch RAW ts (epoch), do NOT call datetime(...) in SQL
    rows_raw = DB.execute(
        """
        SELECT role,
               ts,                       -- ← raw float
               content,
               COALESCE(thought,'') AS thought,
               COALESCE(model ,'')  AS model
          FROM messages
         WHERE session_id = ?
      ORDER BY id
        """,
        (sid,),
    ).fetchall()

    # 2️⃣  convert ts → local string & build tuples for the template
    def _fmt_epoch(epoch: float) -> str:
        return dt.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")

    rows = [
        (role,
         _fmt_epoch(ts),                # formatted timestamp
         md_to_html(content),
         md_to_html(thought),
         model)
        for role, ts, content, thought, model in rows_raw
    ]

    # 3️⃣  model picker
    models_list = [settings.MODEL_DEFAULT, "gemma3:4b"]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "rows": rows,
            "models": models_list,
            "sid": sid,
        },
    )

