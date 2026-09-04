# Project Mental Model — AI Mock Interview & Readiness Platform

> **This is a living document, not a permanent snapshot.** Original pass read directly
> from source on 2026-08-31. **Updated 2026-09-01** after a security-focused audit that
> fixed all 14 originally-listed known issues plus several more found in the same pass —
> see Step 5, which is the section most affected. Steps 1-4 (architecture, flows,
> load-bearing files, external boundaries) were spot-corrected wherever the fixes changed
> what a code block actually shows, but the flows and architecture themselves are
> unchanged. Codebases drift: re-verify against the real source before trusting a pasted
> block or line number, and update this file whenever a significant architectural change
> lands.
>
> Backend root: `job_readiness_backend-main/job_readiness_backend-main/job_readiness_backend_vercel/`
> Frontend root: `ai_mock_interview_v5-main/ai_mock_interview_v5-main/`
> (both relative to the repo root; all file paths below are relative to these roots
> unless stated otherwise)

---

## STEP 1 — System boundaries

### What this system does

A student practices for job interviews by taking AI-generated mock interviews (technical,
behavioral, or HR-style; text, coding, speech, or system-design questions) and receiving
AI-scored feedback afterward. Questions can be generic (from a question bank), tailored to
a specific company using a RAG pipeline built from company-specific "playbook" PDFs, or
generated from the student's own resume and a target job description. Coding questions run
in a real sandboxed code execution backend. Speech and video-based answers get sentiment/
demeanor analysis. Institutions (via an "admin" role) manage the student roster, the
question bank, company playbooks, and view aggregate analytics.

### Actors

- **Student** — the primary end user.
- **Mentor** — not a real authenticated role; the only "mentor" surface,
  `POST /mentors/students/import`, is gated by the **admin** JWT as of 2026-09-01 (see
  Flow E) — before that it had no auth at all, and there is still no self-service mentor
  login, so a mentor visiting `/register` currently can't actually use this endpoint
  themselves.
- **Admin** — institution staff, separate JWT namespace from students.
- **Gemini** — question generation, grading, feedback, video sentiment, embeddings.
- **Judge0 (public `ce.judge0.com`) / Piston (public `emkc.org/api/v2/piston`)** —
  external code execution.
- **SMTP server** — password-reset and CSV-import credential emails.

### App entry point — `app/main.py` (full file)

```python
import logging
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import db_pool
from app.modules.auth.router import router as auth_router
from app.modules.students.router import router as students_router
from app.modules.students.dashboard_extras_router import router as dashboard_extras_router
from app.modules.sandbox.router import router as sandbox_router
from app.modules.interview.router import router as interview_router
from app.modules.admin.router import router as admin_router

app = FastAPI(
    title="AI Mock Interview & Readiness Platform (Modular Monolith)",
    version="2.0.0-modular",
    description="Enterprise-grade modular monolith backend combining student readiness, hybrid code execution, AI grading, and institutional oversight.",
)

app.add_middleware(
    CORSMiddleware,
    # Dev-only: no deployed frontend URL yet. Add the production origin(s) here
    # before deploying - allow_origins=["*"] with allow_credentials=True makes
    # the browser honor authenticated cross-origin requests from ANY site.
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include feature module routers
app.include_router(auth_router)
app.include_router(students_router)
app.include_router(dashboard_extras_router)
app.include_router(sandbox_router)
app.include_router(interview_router)
app.include_router(admin_router)


@app.get("/", tags=["Root"])
async def root():
    return {
        "app": "AI Mock Interview & Readiness Platform",
        "architecture": "Modular Monolith",
        "version": "2.0.0-modular",
        "status": "online",
        "modules": {
            "auth": "Student authentication and password recovery",
            "students": "Student profiles, resume parsing, and job description processing",
            "sandbox": "Hybrid code execution (Local, Judge0, Piston)",
            "interview": "Live AI interview sessions, audio/video analysis, and grading",
            "admin": "Institutional analytics, dashboard oversight, and question management",
        },
    }


@app.get("/health", tags=["Root"])
async def health_check():
    db_ok = False
    try:
        with db_pool.get_connection() as conn:
            if conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    db_ok = True
    except Exception as exc:
        logger.error("Health check DB failure: %s", exc)

    return {
        "status": "healthy" if db_ok else "degraded",
        "database": "connected" if db_ok else "disconnected",
        "timestamp": datetime.now().isoformat(),
    }
```

No `@app.on_event("startup")` exists anywhere — no table creation runs at boot. All
schema is ensured **lazily, per-call**, by `ensure_*` helpers in `app/core/database.py`
(see Step 3). CORS was wide open (`allow_origins=["*"]` with `allow_credentials=True`,
letting any website's browser make authenticated cross-origin requests) until
2026-09-01, when it was restricted to the local dev frontend origins shown above — the
real production origin still needs to be added before deploying.

### Major zones and what each owns

| Backend module | Owns |
|---|---|
| `app/modules/auth/` | Student login/register, password reset, student JWT, password hashing (reused by admin) |
| `app/modules/students/` | Resume/JD upload+parsing, resume-based (RAG) question generation, mentor CSV import |
| `app/modules/interview/` | Session lifecycle: start, submit answer, video, feedback, scores, trending companies |
| `app/modules/sandbox/` | Hybrid code execution: local Python/SQL, Judge0, Piston fallback |
| `app/modules/admin/` | Admin login, analytics, student oversight, question bank, company + playbook management |
| `app/modules/rag/` | `company_playbook_chunks` pgvector table, similarity retrieval, PDF extraction/chunking |
| `app/modules/ai/` | `client.py` (Gemini fallback+retry, embeddings), `grading.py`, `video.py`, `prompts.py` |
| `app/core/` | `database.py` (connection pool + lazy-migration helpers), `logger.py` |
| `app/config/` | `settings.py` (env config singleton), `constants.py` (model list, weights, rubrics) |
| `data/company_playbooks/` | Offline CLI tool (`ingest_kb.py`), not called by the live app |

| Frontend area | Owns |
|---|---|
| `src/App.js` | Routing, top-level auth state, anti-refresh guard on `/interview` |
| `src/api.js` | Every backend call — three axios instances (see Step 3) |
| `src/Dashboard.js` | Student home: interview setup, trending companies, history |
| `src/InterviewScreen.js` | Live interview UI: timer, proctoring, speech-to-text, video, answer submission |
| `src/CodingWorkspace.js` | Monaco-editor coding UI, talks to the Piston proxy |
| `src/SystemDesignCanvas.js` | `reactflow` diagram editor |
| `src/AdminPage.js` | The entire admin panel (all tabs) in one component |
| `src/MentorRegister.js` | Unauthenticated CSV bulk-import page |

### Where data enters and leaves

**Enters:** HTTP requests; file uploads (resumes, JDs, videos, admin logos/playbook PDFs,
mentor CSVs); responses from Gemini/Judge0/Piston.
**Leaves:** HTTP responses; outbound Gemini/Judge0/Piston/SMTP calls; local disk writes
under `MEDIA_ROOT` (videos, deleted after analysis), `uploads/` (resumes/JDs, retained),
and the **frontend's own** `public/logos/` folder (see Step 5, issue 8).

---

## STEP 2 — Core request flows, end to end, with real code at every hop

### Flow A — Student starts an interview

**1. Frontend call site** — `ai_mock_interview_v5-main/src/Dashboard.js:1242-1277`:

```javascript
const params = new URLSearchParams();
const headers = {};

if (isFromCompanyCard) {
  headers['X-Request-Source'] = 'company-card';
}

params.append('student_name', student ? student.name : 'Anonymous User');
params.append('job_role', effectiveJobRole);
params.append('industry_type', effectiveIndustryType);
params.append('company_name', effectiveCompanyName);
params.append('interview_type', effectiveInterviewType);
params.append('work_experience', effectiveWorkExperience);
if (effectiveJobDescription?.job_description_id || effectiveJobDescription?.id) {
  params.append('job_description_id', effectiveJobDescription.job_description_id || effectiveJobDescription.id);
}
if (effectiveJobDescription?.job_desc) {
  params.append('job_description_text', effectiveJobDescription.job_desc);
  params.append('job_description_raw_text', effectiveJobDescription.job_desc);
}

if (effectiveUseResumeQuestions && effectiveResumeData) {
  params.append('use_resume_questions', 'true');
  params.append('resume_id', effectiveResumeData.resume_id);
  if (effectiveResumeData.generatedQuestions && effectiveResumeData.generatedQuestions.length > 0) {
    params.append('pre_generated_questions', JSON.stringify(effectiveResumeData.generatedQuestions));
  }
}

if (force) {
  params.append('force_reattempt', 'true');
}

const response = await interviewApi.post('/interview/start', params, { headers });
```

As of 2026-09-01, `interviewApi` carries an `Authorization` header via the
`attachStudentToken` interceptor (Step 3) — the `student_email` form field shown in the
original version of this snippet was removed from the request entirely; the backend now
derives the email from the verified token instead (see the router snippet below).

**2. Router** — `app/modules/interview/router.py`:

```python
@router.post("/interview/start", tags=["Interview Session"])
async def start_interview(
    request: Request,
    student_name: str = Form(...),
    job_role: str = Form(...),
    industry_type: str = Form(...),
    company_name: str = Form(...),
    interview_type: Optional[str] = Form(None),
    work_experience: Optional[str] = Form(None),
    job_description_id: Optional[int] = Form(None),
    job_description_text: Optional[str] = Form(None),
    job_description_raw_text: Optional[str] = Form(None),
    force_reattempt: bool = Form(False),
    use_resume_questions: bool = Form(False),
    resume_id: Optional[int] = Form(None),
    pre_generated_questions: Optional[str] = Form(None),
    student: Dict[str, Any] = Depends(verify_student_token),
):
    """Start new interview with student tracking and reattempt detection."""
    try:
        return await start_interview_service(
            request=request,
            student_name=student_name,
            student_email=student["email"],
            job_role=job_role,
            industry_type=industry_type,
            company_name=company_name,
            interview_type=interview_type,
            work_experience=work_experience,
            job_description_id=job_description_id,
            job_description_text=job_description_text,
            job_description_raw_text=job_description_raw_text,
            force_reattempt=force_reattempt,
            use_resume_questions=use_resume_questions,
            resume_id=resume_id,
            pre_generated_questions=pre_generated_questions,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error starting interview: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
```

**3. Service — reattempt check** — `app/modules/interview/service.py:688-714`:

```python
    if not force_reattempt and not resume_generation_enabled:
        is_company_card = False
        if hasattr(request, "headers") and hasattr(request, "query_params"):
            is_company_card = (
                request.headers.get("X-Request-Source") == "company-card"
                or request.query_params.get("source") == "company-card"
                or request.query_params.get("is_company_card", "").lower() == "true"
            )
        try:
            existing_sessions = InterviewRepository.find_existing_sessions(
                student_name=student_name,
                student_email=student_email,
                job_role=job_role,
                industry_type=industry_type,
                company_name=company_name,
                interview_type=normalized_interview_type,
                work_experience=normalized_work_experience,
                is_company_card=is_company_card,
            )
            if existing_sessions:
                return {
                    "requires_confirmation": True,
                    "existing_sessions": existing_sessions,
                    "message": "Existing interview attempts found. Confirm to start a reattempt.",
                }
        except Exception as exc:
            logger.error(f"Error finding existing sessions: {exc}", exc_info=True)
```

**4. Session creation** — `service.py:716-724` calling into
`app/modules/interview/repository.py:397-433`:

```python
# service.py:716-724
    session_id = InterviewRepository.create_enhanced_session(
        student_name,
        student_email,
        job_role,
        industry_type,
        company_name,
        normalized_interview_type,
        normalized_work_experience,
    )
    if not session_id:
        raise HTTPException(status_code=500, detail="Failed to create session")
```

```python
# repository.py:397-433
    @classmethod
    def create_enhanced_session(
        cls,
        student_name: str,
        student_email: Optional[str],
        job_role: str,
        industry_type: str,
        company_name: str,
        interview_type: Optional[str] = None,
        work_experience: Optional[str] = None,
    ) -> Optional[str]:
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return None
                with conn.cursor() as cur:
                    student_id = cls.find_student_id(student_name, student_email)
                    if not student_id:
                        cur.execute(
                            "INSERT INTO students (name, email) VALUES (%s, %s) RETURNING student_id",
                            (student_name, student_email)
                        )
                        student_id = cur.fetchone()[0]

                    session_id = str(uuid.uuid4())
                    cur.execute(
                        """
                        INSERT INTO session_metadata (session_id, student_id, student_name, status, interview_type, work_experience)
                        VALUES (%s, %s, %s, 'active', %s, %s)
                        """,
                        (session_id, student_id, student_name, interview_type, work_experience)
                    )
                    conn.commit()
                    return session_id
        except Exception as e:
            logger.error(f"Error creating enhanced session: {e}")
            return None
```

Note: `find_student_id(student_name, student_email)` matches by **name and/or email
text**, not a stable numeric identity from a login token — the "anonymous@example.com"
string from step 1 would collide across every anonymous user searched by that same email.

**5. First-question resolution cascade** — `service.py:589-643`:

```python
async def _resolve_next_question(
    job_role: str,
    industry_type: Optional[str],
    company_name: str,
    question_number: int,
    session_id: str,
    interview_type: Optional[str],
    work_experience: Optional[str],
    question_type_filter: Optional[str] = None,
    preferred_difficulty: Optional[str] = None,
) -> Dict[str, Any]:
    """Question resolution order: company-specific DB match -> RAG+Gemini generation
    (triggered as soon as the company-specific DB lookup misses) -> company-agnostic
    DB relaxation chain / hardcoded placeholder as the final safety net."""
    company_question = InterviewRepository.find_company_specific_question(
        job_role, industry_type, company_name, session_id,
        interview_type=interview_type, work_experience=work_experience,
        question_type_filter=question_type_filter, preferred_difficulty=preferred_difficulty,
    )
    if company_question:
        return company_question

    try:
        rag_question = await _generate_company_question_via_rag(
            job_role, industry_type, company_name, interview_type, work_experience,
            question_type_filter, preferred_difficulty, session_id=session_id,
        )
    except Exception as exc:
        logger.warning(f"RAG+Gemini generation raised for company '{company_name}'/{job_role}: {exc}")
        rag_question = None

    if rag_question:
        return rag_question

    return InterviewRepository.find_generic_fallback_question(
        job_role, industry_type, company_name, question_number, session_id,
        interview_type=interview_type, work_experience=work_experience,
        question_type_filter=question_type_filter,
    )
```

Called from `service.py:790-807` for a live (non-resume-driven) start:

```python
        first_question_data = await _resolve_next_question(
            job_role, industry_type, company_name, 1, session_id,
            interview_type=normalized_interview_type,
            work_experience=normalized_work_experience,
            preferred_difficulty="medium",
        )
        first_question = first_question_data["question"]
        mandatory_skills = first_question_data["mandatory_skills"]
        first_question_difficulty = first_question_data.get("difficulty", "medium") or "medium"
        first_question_type = first_question_data.get("question_type", "standard")
        first_question_context = first_question_data.get("generation_context") or {}
        first_difficulty_label = (first_question_difficulty or "medium").lower()
        first_weight = DIFFICULTY_WEIGHTS.get(first_difficulty_label, 2)
        current_max_questions = _determine_max_questions([first_weight], 1)
```

`_determine_max_questions` — `service.py:99-112`:

```python
def _determine_max_questions(difficulty_weights: List[int], answered_count: int = 0) -> int:
    try:
        average_weight = sum(difficulty_weights) / len(difficulty_weights) if difficulty_weights else 2.0
    except ZeroDivisionError:
        average_weight = 2.0

    if average_weight < 1.4:
        calculated_max = 5
    elif average_weight < 2.3:
        calculated_max = 6
    else:
        calculated_max = 7

    return max(calculated_max, answered_count)
```

The first question is then persisted via `InterviewRepository.save_first_question`
(`repository.py:558+`) into `interview_data`, and the response payload (`service.py:839-861`)
returns `session_id`, `first_question`, and `current_max_questions` to `Dashboard.js`.

---

### Flow B — Student submits an answer, including code execution

Code execution is a **separate prior call**. When the student clicks "Run" in
`CodingWorkspace.js`:

**1. Frontend runs code** — `ai_mock_interview_v5-main/src/CodingWorkspace.js:460-554`:

```javascript
  const runCode = async () => {
    if (!hasMeaningfulCode(code)) {
      notify('Please write your solution before running.');
      return { stdout: '', stderr: '', success: false, internalError: 'No code to execute', language: selectedRuntime?.language || normalizedDefaultLanguage };
    }
    if (!selectedRuntime) {
      notify('Please select a language runtime before running.');
      return { stdout: '', stderr: '', success: false, internalError: 'No runtime selected', language: normalizedDefaultLanguage };
    }
    ...
    setIsRunning(true);
    try {
      const runtimeLanguage = selectedRuntime.language;
      const fileName = /* switch on language: main.py, main.js, Main.java, main.cpp, ... */;
      const payload = {
        language: selectedRuntime.language,
        version: selectedRuntime.version,
        stdin: stdinText,
        files: [{ name: fileName, content: code }],
      };
      const response = await executeWithPiston(payload);
      const data = response?.data || {};
      const runResult = data.run || {};
      const compileResult = data.compile || {};
      const stdoutValue = runResult.output ?? runResult.stdout ?? '';
      ...
```

`executeWithPiston` — `src/api.js` (comment corrected 2026-09-01; see Step 3 for the
full accurate picture of the proxy chain):

```javascript
export const fetchPistonRuntimes = () => backendApi.get('/piston/runtimes');

export const executeWithPiston = (payload) => backendApi.post('/piston/execute', payload);
```

**2. Backend router** — `app/modules/sandbox/router.py` (full file):

```python
from typing import Any
from fastapi import APIRouter, HTTPException, Request
from app.modules.sandbox.service import execute_hybrid_code, fetch_runtimes

router = APIRouter(prefix="/piston", tags=["Code Sandbox"])


@router.post("/execute")
async def execute_code_endpoint(request: Request) -> Any:
    """Execute code reliably using local runtime, Judge0 CE, or Piston."""
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    return await execute_hybrid_code(payload)


@router.get("/runtimes")
async def get_runtimes_endpoint() -> Any:
    """Fetch runtimes or return robust default built-in runtimes fallback."""
    return await fetch_runtimes()
```

**3. Three-tier execution cascade** — `app/modules/sandbox/service.py:77-107`:

```python
async def execute_hybrid_code(payload: Dict[str, Any]) -> Any:
    """Execute code across local, Judge0, and Piston tiers with automatic fallback."""
    payload = inject_python_visualization_bootstrap(payload)
    language = str(payload.get("language", "")).lower()
    files = payload.get("files") or []
    code = ""
    if isinstance(files, list) and files and isinstance(files[0], dict):
        code = str(files[0].get("content") or "")
    stdin_val = str(payload.get("stdin") or "")

    # Execute locally for Python and SQL for instant reliability & zero network dependencies
    if language in {"python", "py", "py3", "python3"}:
        return await asyncio.to_thread(run_local_python_sync, code, stdin_val)
    elif language in {"sql", "sqlite", "sqlite3", "mysql", "postgres", "postgresql"}:
        return await asyncio.to_thread(run_local_sql, code)

    # Try external Judge0 CE for other languages
    try:
        return await run_via_judge0(language, code, stdin_val)
    except Exception as judge0_err:
        logger.warning("Judge0 execution failed, attempting Piston fallback: %s", judge0_err)

    # Final fallback: proxy to Piston instance
    try:
        return await execute_via_piston(payload)
    except urllib.error.HTTPError as http_err:
        logger.error("Piston execute HTTP error: %s", http_err)
        raise HTTPException(status_code=http_err.code, detail="Failed to execute code in runner")
    except Exception as exc:
        logger.error("Piston execute error: %s", exc)
        raise HTTPException(status_code=502, detail="Error contacting code runner service")
```

The three actual executors — `app/modules/sandbox/runners.py`:

```python
def run_local_python_sync(code: str, stdin_val: str) -> Dict[str, Any]:
    """Execute Python code locally in a subprocess with a 15-second timeout."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(code)
        tf_path = tf.name
    try:
        proc = subprocess.run(
            [sys.executable, tf_path],
            input=stdin_val.encode("utf-8") if stdin_val else b"",
            capture_output=True,
            timeout=15.0,
        )
        stdout_str = proc.stdout.decode("utf-8", errors="replace")
        stderr_str = proc.stderr.decode("utf-8", errors="replace")
        return {
            "run": {"stdout": stdout_str, "stderr": stderr_str,
                     "output": stdout_str + ("\n" + stderr_str if stderr_str else ""),
                     "code": proc.returncode or 0},
            "compile": {"output": "", "stderr": "", "code": 0},
        }
    except subprocess.TimeoutExpired:
        return {"run": {"stdout": "", "stderr": "Execution timed out (exceeded 15 seconds limit).",
                          "output": "Execution timed out (exceeded 15 seconds limit).", "code": 124},
                "compile": {"output": "", "stderr": "", "code": 0}}
    finally:
        try:
            os.remove(tf_path)
        except Exception:
            pass
```

```python
async def run_via_judge0(language: str, code: str, stdin_val: str) -> Dict[str, Any]:
    """Execute code using public Judge0 CE API when local runner is not available."""
    lang_id = JUDGE0_LANG_MAP.get(language, 71)

    def _do_judge0() -> Dict[str, Any]:
        body = json.dumps({"source_code": code, "language_id": lang_id, "stdin": stdin_val}).encode("utf-8")
        req = urllib.request.Request(
            "https://ce.judge0.com/submissions?base64_encoded=false&wait=true",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    res = await asyncio.to_thread(_do_judge0)
    # ... maps res["stdout"]/["stderr"]/["compile_output"]/["status"]["id"] into the
    # same {"run": {...}, "compile": {...}} shape as the local runners
```

```python
async def execute_via_piston(payload: Dict[str, Any]) -> Any:
    """Proxy execution to remote Piston instance."""
    base_url = config.PISTON_BASE_URL.rstrip("/")   # defaults to public emkc.org — see Step 5
    url = f"{base_url}/execute"

    def _post() -> Any:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    return await asyncio.to_thread(_post)
```

**4. Frontend submits the answer** — with the already-computed stdout/stderr from step 1
folded in — via `InterviewScreen.js`'s `postAnswer`, `POST /interview/{session_id}/answer`.

**5. Router** — `app/modules/interview/router.py:177-210`:

```python
@router.post("/interview/{session_id}/answer", tags=["Answer Evaluation"])
async def submit_answer(
    background_tasks: BackgroundTasks,
    session_id: str,
    answer: str = Form(...),
    question_type: Optional[str] = Form(None),
    code: Optional[str] = Form(None),
    stdin: Optional[str] = Form(None),
    stdout: Optional[str] = Form(None),
    stderr: Optional[str] = Form(None),
    runtime_error: Optional[str] = Form(None),
    execution_success: Optional[str] = Form(None),
    has_run: Optional[str] = Form(None),
    is_final: Optional[str] = Form(None),
    response_video: Optional[UploadFile] = File(None),
    system_design_diagram: Optional[str] = Form(None),
    student: Dict[str, Any] = Depends(verify_student_token),
):
    """Process candidate response and determine the next question dynamically."""
    try:
        return await submit_answer_service(
            background_tasks=background_tasks, session_id=session_id,
            student_id=student["student_id"], answer=answer,
            question_type=question_type, code=code, stdin=stdin, stdout=stdout, stderr=stderr,
            runtime_error=runtime_error, execution_success=execution_success, has_run=has_run,
            is_final=is_final, response_video=response_video, system_design_diagram=system_design_diagram,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error submitting answer for session {session_id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to submit answer")
```

As of 2026-09-01: requires the student token, 403s if `session_id` doesn't belong to that
student (`submit_answer_service` calls `InterviewRepository.verify_session_belongs_to_student`
before doing anything else), and the router now has a try/except (it had none before —
see Step 5).

**6. Service, Phase 1 — save the answer** — `service.py:969-991`, writing into
`app/modules/interview/repository.py:1350-1462`:

```python
# service.py:969-991
    try:
        context = InterviewRepository.save_answer_and_get_context(
            session_id=session_id, answer=stored_answer_value, question_type=question_type or "standard",
            code=code, stdin=stdin, stdout=stdout, stderr=stderr, runtime_error=runtime_error,
            execution_success=bool(execution_success_bool), has_run=bool(has_run_bool),
            is_final=bool(is_final_flag), saved_video_path=None,
            is_system_design=is_system_design, system_design_diagram=system_design_diagram,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error(f"Database error saving answer for {session_id}: {exc}")
        raise HTTPException(status_code=500, detail="Database operation failed")
```

```python
# repository.py:1400-1434 — the actual UPDATE
                cur.execute(
                    """
                    UPDATE interview_data
                    SET answer = %s,
                        analysis_status = %s,
                        code_submission = %s,
                        stdin_input = %s,
                        stdout_output = %s,
                        stderr_output = %s,
                        runtime_error = %s,
                        execution_success = %s,
                        manual_run = %s,
                        video_clip_path = COALESCE(%s, video_clip_path),
                        is_system_design = %s,
                        system_design_diagram = %s
                    WHERE session_id = %s AND question_number = %s
                    """,
                    (answer, "processing" if not is_final else "completed", code, stdin, stdout, stderr,
                     runtime_error, execution_success, has_run, saved_video_path, is_system_design,
                     system_design_diagram, session_id, current_q),
                )
                conn.commit()
```

**7. Phase 2 — AI evaluation with a hard timeout** — `service.py:1023-1041`:

```python
    try:
        ai_future = analyze_answer_and_generate_response(
            question=current_question, answer=effective_answer,
            mandatory_skills=mandatory_skills, job_role=job_role,
        )
        sentiment_score, acknowledgment, next_difficulty_raw = await asyncio.wait_for(ai_future, timeout=12.0)
    except asyncio.TimeoutError:
        logger.warning(f"AI evaluation timed out for {session_id} Q#{current_q}; using default fallback values.")
        sentiment_score = 5.0
        acknowledgment = "Thank you for sharing your experience. Let's move on to the next topic."
        next_difficulty_raw = "medium"
    except Exception as exc:
        logger.error(f"Error during AI evaluation for {session_id} Q#{current_q}: {exc}")
        sentiment_score = 5.0
        acknowledgment = "Got it. Let's continue with our discussion."
        next_difficulty_raw = "medium"
```

**8. Phase 3 — persist sentiment, flip status if final** — `service.py:1047-1053` calling
`repository.py:1464-1507`:

```python
# repository.py:1464-1507
    @classmethod
    def save_sentiment_and_check_final(cls, session_id: str, question_number: int,
                                        sentiment_score: float, is_final_question: bool) -> None:
        with db_pool.get_connection() as conn:
            if not conn:
                return
            with conn.cursor() as cur:
                cls._ensure_interview_data_columns(cur)
                cls._ensure_session_metadata_columns(cur)
                if is_final_question:
                    cur.execute(
                        "UPDATE interview_data SET sentiment_score = %s, analysis_status = 'completed' "
                        "WHERE session_id = %s AND question_number = %s",
                        (sentiment_score, session_id, question_number),
                    )
                    cur.execute(
                        "UPDATE session_metadata SET status = 'completed', completed_at = CURRENT_TIMESTAMP "
                        "WHERE session_id = %s",
                        (session_id,),
                    )
                else:
                    cur.execute(
                        "UPDATE interview_data SET sentiment_score = %s "
                        "WHERE session_id = %s AND question_number = %s",
                        (sentiment_score, session_id, question_number),
                    )
                conn.commit()
```

**9. Completion branch triggers Flow C** — `service.py:1055-1057` (everything from here
to the end of the function has **no exception handling** — Step 5, issue 6):

```python
    if is_final_question:
        InterviewRepository.mark_feedback_pending(session_id)
        background_tasks.add_task(generate_and_process_feedback_background, session_id)
        return { "session_id": session_id, "response": "Interview completed! Generating comprehensive feedback report...", ... }
```

**Video sub-flow** (speech questions, same request) — `service.py:1010-1015`:

```python
    is_speech_question = "speech" in (db_question_type or "").lower()
    saved_video_path = _save_uploaded_video(session_id, current_q, response_video) if is_speech_question else None
    if saved_video_path:
        InterviewRepository.update_video_upload_path(session_id, current_q, saved_video_path)
        background_tasks.add_task(process_video_analysis_background, session_id, current_q, saved_video_path)
```

`_save_uploaded_video` — `service.py:250-269`:

```python
def _save_uploaded_video(session_id: str, question_number: int, upload: UploadFile | None) -> Optional[str]:
    if not upload or not upload.filename:
        return None
    try:
        upload_dir = settings.MEDIA_ROOT / session_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_ext = Path(upload.filename).suffix or ".webm"
        file_path = upload_dir / f"q_{question_number}{file_ext}"
        if hasattr(upload, "file") and upload.file:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(upload.file, buffer)
        else:
            content = upload.file.read() if hasattr(upload.file, "read") else None
            if content:
                with open(file_path, "wb") as buffer:
                    buffer.write(content)
        return str(file_path.relative_to(settings.MEDIA_ROOT.parent)) if settings.MEDIA_ROOT.parent in file_path.parents else str(file_path)
    except Exception as exc:
        logger.error(f"Failed saving video for session {session_id} Q#{question_number}: {exc}")
        return None
```

The dedicated upload-video endpoint (`router.py:213-227`, used outside the answer form
too) is the same path:

```python
@router.post("/interview/{session_id}/upload-video", tags=["Video Analysis"])
async def upload_video(
    background_tasks: BackgroundTasks, session_id: str,
    question_number: int = Form(...), video: UploadFile = Form(...),
):
    saved_video_path = _save_uploaded_video(session_id, question_number, video)
    if not saved_video_path:
        raise HTTPException(status_code=500, detail="Failed to save video file")

    InterviewRepository.update_video_upload_path(session_id, question_number, saved_video_path)
    background_tasks.add_task(process_video_analysis_background, session_id, question_number, saved_video_path)
    return {"status": "success", "message": "Video uploaded and queued for analysis", "file_path": saved_video_path}
```

**Video analysis background job** — `app/modules/ai/video.py:36-113` (Gemini call) and
`116-159` (DB write + cleanup):

```python
def analyze_video_sentiment_sync(video_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not video_path or not os.path.exists(video_path):
        return None

    def _run_gemini_video_call() -> Dict[str, Any]:
        model = genai.GenerativeModel(config.GEMINI_VIDEO_MODEL)   # NOT the fallback wrapper — see Step 5
        with open(video_path, "rb") as clip_file:
            response = model.generate_content(
                [
                    "... Return STRICT JSON only with: {sentiment, engagement, dominant_expression, verbal_strength} ...",
                    {"mime_type": "video/webm", "data": clip_file.read()},
                ],
                generation_config={"response_mime_type": "application/json"},
            )
        if not response or not response.text:
            raise ValueError("Empty response from Gemini video model")
        try:
            raw = json.loads(response.text)
        except json.JSONDecodeError as decode_error:
            logger.warning(f"Gemini video response not JSON, falling back to heuristic parsing: {decode_error}")
            raw_score = extract_first_number(response.text)
            return {"sentiment": raw_score if raw_score is not None else 5.0, ...}
        return {"sentiment": float(raw.get("sentiment", 5.0)), ...}

    try:
        analysis = _run_gemini_video_call()
        ...
        return analysis
    except Exception as exc:
        logger.error(f"Synchronous video analysis failed for clip {video_path}: {exc}")
        return None


def process_video_analysis_background(session_id: str, question_number: int, video_path: str) -> None:
    try:
        analysis = analyze_video_sentiment_sync(video_path)
        with db_pool.get_connection() as conn:
            if not conn:
                logger.error(f"Failed to get DB connection for video analysis update (session {session_id})")
                return
            with conn.cursor() as cur:
                if analysis:
                    cur.execute(
                        "UPDATE interview_data SET video_analysis = %s, video_analysis_status = 'completed' "
                        "WHERE session_id = %s AND question_number = %s",
                        (json.dumps(analysis), session_id, question_number),
                    )
                else:
                    cur.execute(
                        "UPDATE interview_data SET video_analysis_status = 'failed' "
                        "WHERE session_id = %s AND question_number = %s",
                        (session_id, question_number),
                    )
                conn.commit()

        # Clean up video file after analysis
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except OSError as cleanup_error:
                logger.warning(f"Failed to remove video clip {video_path}: {cleanup_error}")
    except Exception as exc:
        logger.error(f"Background video analysis failed for session {session_id}, question {question_number}: {exc}")
        # ... falls into a second try/except that marks video_analysis_status='failed' in DB
```

**10. Frontend receives the response** — `InterviewScreen.js:1093-1149`:

```javascript
    const postAnswer = async (formData) => {
        const response = await interviewApi.post(`/interview/${sessionId}/answer`, formData);
        const {
            next_question, next_question_meta, acknowledgment, completed, is_completed,
            question_number, current_max_questions, response: nextResponseText, response_meta,
        } = response.data;

        const effectiveCompleted = completed || is_completed;
        ...
        if (effectiveCompleted) {
            clearToastTimer();
            toastTimerRef.current = setTimeout(() => { beginFeedbackPolling(); }, 1100);
            return;
        }
        const normalizedNext = effectiveNextMeta ? normalizeQuestion(effectiveNextMeta) : normalizeQuestion(effectiveNextQuestion);
        if (normalizedNext) {
            setQuestion(normalizedNext);
            const nextNum = typeof question_number === 'number' ? question_number : questionNumber + 1;
            setQuestionNumber(nextNum);
            ...
        }
        ...
    };
```

---

### Flow C — Feedback generation

Triggered from Flow B's completion branch, or manually via
`POST /interview/{session_id}/generate-feedback` (`router.py:230-237`, which just
`await`s the same background function synchronously).

**The full function** — `app/modules/interview/service.py:1143-1353`:

```python
async def generate_and_process_feedback_background(session_id: str) -> None:
    try:
        max_wait_time = 300
        check_interval = 2
        elapsed_time = 0
        while elapsed_time < max_wait_time:
            pending_count, processing_count = InterviewRepository.get_pending_video_count(session_id)
            if pending_count == 0 and processing_count == 0:
                break
            await asyncio.sleep(check_interval)
            elapsed_time += check_interval

        InterviewRepository.mark_feedback_processing(session_id)
        qa_history, job_role, industry_type, company_name, interview_type = (
            InterviewRepository.get_qa_history_for_feedback(session_id)
        )

        if not qa_history:
            InterviewRepository.mark_feedback_failed(session_id, "No answered questions available to grade.")
            return

        formatted_exchanges = []
        unique_skills_set = set()
        answer_lookup = {}
        question_type_lookup = {}
        rag_lookup_pairs = []

        for row in qa_history:
            (q_num, q_text, ans_text, sent_score, mand_skills, vid_sent, vid_dem,
             code_sub, code_succ, code_run, q_type) = row
            prompt_answer_text = ans_text
            if _is_system_design_question_type(q_type):
                prompt_answer_text = _describe_system_design(ans_text)
            exchange = f"Q{q_num}: {q_text}\nA: {prompt_answer_text}"
            if code_sub:
                exchange += f"\n[Code Submitted ({'Success' if code_succ else 'Failed/Unchecked'})]: {code_sub[:300]}"
            formatted_exchanges.append(exchange)
            if mand_skills:
                for sk in str(mand_skills).split(","):
                    sk_clean = sk.strip()
                    if sk_clean:
                        unique_skills_set.add(sk_clean)

            candidate_answer = ans_text or ""
            normalized_qt = str(q_type or "").lower()
            if normalized_qt == "coding" or normalized_qt.startswith("coding "):
                code_snippet = code_sub or ""
                if code_snippet and code_snippet.strip():
                    candidate_answer = code_snippet.strip()

            answer_lookup[int(q_num)] = candidate_answer
            question_type_lookup[int(q_num)] = q_type
            rag_lookup_pairs.append((int(q_num), q_text))

        unique_skills = sorted(list(unique_skills_set))
        conversation_excerpt = _compose_conversation_excerpt(formatted_exchanges, char_limit=18000)
        template_info = _resolve_feedback_template(interview_type)
        resolved_template = template_info["question_template"]

        reference_answers = []
        if company_name:
            try:
                rag_results = await asyncio.gather(
                    *[retrieve_company_context(company_name, q_text, top_k=1) for _, q_text in rag_lookup_pairs]
                )
                for (q_num, _), chunks in zip(rag_lookup_pairs, rag_results):
                    if chunks:
                        reference_answers.append({"number": q_num, "answer": chunks[0]})
            except Exception as exc:
                logger.warning("RAG reference-answer lookup failed for session %s: %s", session_id, exc)
                reference_answers = []

        rendered_question_prompt = render_template(
            resolved_template, job_role=job_role, industry_type=industry_type,
            company_name=company_name, interview_type=interview_type or "standard",
            conversation_excerpt=conversation_excerpt, conversation_text=conversation_excerpt,
            unique_skills=unique_skills, reference_answers=reference_answers, scoring_guide=SCORING_GUIDE,
        )
        rendered_competency_prompt = render_template(
            template_info["competency_template"], job_role=job_role, industry_type=industry_type,
            company_name=company_name, interview_type=interview_type or "standard",
            conversation_excerpt=conversation_excerpt, conversation_text=conversation_excerpt,
            unique_skills=unique_skills, core_competencies=template_info["competencies"],
        )

        # Question-quality feedback and competency scoring are independent prompts over the
        # same transcript, so they run concurrently rather than paying two sequential Gemini
        # round-trips (same pattern as the RAG reference-answer lookups above).
        question_response, competency_response = await asyncio.gather(
            generate_content_with_fallback(rendered_question_prompt, retry_label=f"Feedback generation for {session_id}"),
            generate_content_with_fallback(rendered_competency_prompt, retry_label=f"Competency scoring for {session_id}"),
            return_exceptions=True,
        )

        if isinstance(question_response, Exception):
            raise question_response

        raw_feedback_text = question_response.text if question_response else ""
        if not raw_feedback_text or not raw_feedback_text.strip():
            raise RuntimeError("Gemini returned empty feedback report.")

        parsed_json = _parse_feedback_json(raw_feedback_text)

        if parsed_json and isinstance(parsed_json, dict):
            questions_list = parsed_json.get("questions")
            if isinstance(questions_list, list):
                for entry in questions_list:
                    if not isinstance(entry, dict):
                        continue
                    num = entry.get("number")
                    if num is None:
                        continue
                    try:
                        idx = int(num)
                    except (ValueError, TypeError):
                        continue
                    original_answer = answer_lookup.get(idx, "")
                    if original_answer:
                        entry["answer"] = original_answer
                        entry["original_answer"] = original_answer
                    qt_value = question_type_lookup.get(idx)
                    if qt_value:
                        entry["question_type"] = qt_value
                        normalized_qt = str(qt_value).lower()
                        if normalized_qt == "coding" or normalized_qt.startswith("coding "):
                            entry["is_coding"] = True

        if not parsed_json:
            # Storing unparseable raw text and marking the session "completed" anyway made the
            # frontend poll forever: FeedbackScreen sees status=completed, tries to fetch
            # structured feedback, gets nothing back (same broken text fails to parse every
            # time), and re-polls -- forever, at whatever cadence the retry fires, hammering
            # this endpoint. Treat an unparseable AI response as a real failure instead, so the
            # frontend surfaces a "regenerate" error state rather than looping indefinitely.
            raise RuntimeError("Gemini returned malformed feedback JSON that could not be parsed.")

        # Competency scoring is best-effort: a failed/malformed response still lets the
        # question-level feedback save, just without a weighted overall_score.
        if isinstance(competency_response, Exception):
            logger.warning("Competency scoring Gemini call failed for session %s: %s", session_id, competency_response)
        else:
            raw_competency_text = competency_response.text if competency_response else ""
            competency_parsed = _parse_feedback_json(raw_competency_text) if raw_competency_text.strip() else None
            if isinstance(competency_parsed, dict) and competency_parsed.get("core_competencies"):
                parsed_json["core_competencies"] = competency_parsed["core_competencies"]
                for summary_key in ("technical_summary", "communication_summary", "attitude_summary"):
                    if summary_key in competency_parsed:
                        parsed_json[summary_key] = competency_parsed[summary_key]
            else:
                logger.warning("Competency scoring response for session %s had no usable core_competencies", session_id)

        stored_payload = json.dumps(parsed_json, ensure_ascii=False)
        saved = InterviewRepository.save_detailed_feedback(session_id, stored_payload)
        if not saved:
            raise RuntimeError("Database error persisting detailed feedback.")

        update_scores_from_feedback(session_id, stored_payload)
        InterviewRepository.mark_feedback_completed(session_id)

    except Exception as exc:
        logger.error(f"Background feedback generation failed for session {session_id}: {exc}", exc_info=True)
        InterviewRepository.mark_feedback_failed(session_id, str(exc))
```

**Score extraction** — `app/modules/ai/grading.py:166-259` (`extract_scores_from_feedback`,
called by `update_scores_from_feedback` at line 317-367), the weighted-average path:

```python
def extract_scores_from_feedback(feedback_text: str) -> dict:
    from app.utils.text_utils import _parse_feedback_json
    try:
        parsed = _parse_feedback_json(feedback_text)
        if parsed and "core_competencies" in parsed:
            competencies = parsed.get("core_competencies", [])
            if not competencies:
                return {"overall_score": 0.0}
            total_weighted_score = 0.0
            total_weight = 0.0
            for comp in competencies:
                if isinstance(comp, dict):
                    score = float(comp.get("score", 0.0) or 0.0)
                    weight = float(comp.get("weight", 0.0) or 0.0)
                    total_weighted_score += score * weight
                    total_weight += weight
            if total_weight > 0:
                overall_score = total_weighted_score / total_weight
                rubric_entries = [{"name": c.get("name"), "score": c.get("score")} for c in competencies if isinstance(c, dict)]
                interview_type = parsed.get("metadata", {}).get("interview_type") or "unknown"
                return {"overall_score": overall_score, "rubric_scores": {"interview_type": interview_type, "rubric": rubric_entries}}
            return {"overall_score": 0.0}
        # ... falls back to legacy technical_summary/communication_summary/attitude_summary
        # averaging, then to regex-matched markdown "### Score: X/5" sections, then to 0.0
    except Exception as e:
        logger.error(f"Error extracting scores from feedback: {e}")
        return {"overall_score": 0.0}
```

**Frontend polling** — `GET /feedback-status/{session_id}` (`router.py:258-274`) until
`status: "completed"`, then `GET /feedback/{session_id}` (`router.py:277-287`), which
reads via `repository.py:1792-1844`:

```python
    @classmethod
    def get_feedback_payload(cls, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            with db_pool.get_connection() as conn:
                if not conn:
                    return None
                with conn.cursor() as cur:
                    cls._ensure_interview_data_columns(cur)
                    cur.execute(
                        """
                        SELECT detailed_feedback
                        FROM interview_data
                        WHERE session_id = %s AND detailed_feedback IS NOT NULL
                        ORDER BY question_number DESC
                        LIMIT 1
                        """,
                        (session_id,)
                    )
                    result = cur.fetchone()
                    if not result or not result[0]:
                        return None
                    payload_text = result[0]
                    parsed = _parse_feedback_json(payload_text)
                    # The Gemini feedback JSON schema does not include question_type per
                    # question ... merge in the ground-truth question_type already stored
                    # per question_number so the UI can render diagrams correctly.
                    if isinstance(parsed, dict) and isinstance(parsed.get("questions"), list):
                        cur.execute(
                            "SELECT question_number, question_type FROM interview_data WHERE session_id = %s",
                            (session_id,),
                        )
                        question_type_by_number = {row[0]: row[1] for row in cur.fetchall() if row[1]}
                        for item in parsed["questions"]:
                            if not isinstance(item, dict):
                                continue
                            question_number = item.get("number")
                            if question_number in question_type_by_number and not item.get("question_type"):
                                item["question_type"] = question_type_by_number[question_number]
                    return {"structured": parsed, "raw": payload_text}
        except Exception as exc:
            logger.error(f"Error retrieving feedback payload for {session_id}: {exc}")
        return None
```

---

### Flow D — Admin creates a company and uploads its playbook (RAG ingestion)

**1. Frontend** — `AdminPage.js:1747-1784` (`handleCreateCompanySubmit`):

```javascript
  const handleCreateCompanySubmit = async (event) => {
    event.preventDefault();
    if (newCompanyLoading) return;
    if (!newCompanyForm.name.trim()) {
      setNewCompanyErrors({ name: 'Company name is required' });
      return;
    }
    setNewCompanyLoading(true);
    try {
      const response = await createCompany(newCompanyForm);
      const created = response?.data;
      showToast(`${created?.name || newCompanyForm.name} added successfully`, 'success');
      setNewCompanyForm({ name: '', industry: '', difficulty_tag: '', work_experience_tag: '', logo: null });
      setNewCompanyErrors({});
      await loadCompanies();
      if (created?.name) {
        setSelectedPlaybookCompanyName(created.name);
      }
    } catch (err) {
      setNewCompanyErrors({ name: getErrorMessage(err, 'Failed to create company') });
      showToast(getErrorMessage(err, 'Failed to create company'), 'error');
    } finally {
      setNewCompanyLoading(false);
    }
  };
```

**2. Router → service** — `app/modules/admin/router.py:94-118`:

```python
@router.post("/admin/companies", dependencies=[Depends(verify_admin_token)], tags=["Admin Companies"])
async def create_company(
    name: str = Form(...), industry: Optional[str] = Form(None),
    difficulty_tag: Optional[str] = Form(None), work_experience_tag: Optional[str] = Form(None),
    logo: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    return await create_company_service(name, industry, difficulty_tag, work_experience_tag, logo)

@router.post("/admin/companies/{company_id}/upload-playbook", dependencies=[Depends(verify_admin_token)], tags=["Admin Companies"])
async def upload_company_playbook(company_id: int, playbook_file: UploadFile = File(...)) -> Dict[str, Any]:
    return await upload_company_playbook_service(company_id, playbook_file)
```

**3. Company creation, logo-after-insert ordering** — `admin/service.py:168-184`
calling `admin/repository.py:1789-1836`:

```python
# admin/service.py:168-184
async def create_company_service(name, industry, difficulty_tag, work_experience_tag, logo) -> Dict[str, Any]:
    record = _normalize_company_payload(name, industry, difficulty_tag, work_experience_tag)
    record["logo_url"] = None
    created = AdminRepository.insert_company(record)   # 409 raised here on duplicate name

    if logo is not None:
        logo_url = await _save_company_logo_temp_local(created["name"], logo)
        if logo_url:
            created = AdminRepository.update_company_logo(created["id"], logo_url)
    return created
```

```python
# admin/repository.py:1789-1816
    @staticmethod
    def insert_company(record: Dict[str, Any]) -> Dict[str, Any]:
        try:
            with db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    ensure_companies_table(cur)
                    cur.execute(
                        """
                        INSERT INTO companies (name, logo_url, industry, difficulty_tag, work_experience_tag)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id, name, logo_url, industry, difficulty_tag, work_experience_tag, is_active, created_at
                        """,
                        (record["name"], record.get("logo_url"), record.get("industry"),
                         record.get("difficulty_tag"), record.get("work_experience_tag")),
                    )
                    row = cur.fetchone()
                    conn.commit()
                    return AdminRepository._company_row_to_dict(row)
        except UniqueViolation as exc:
            raise HTTPException(status_code=409, detail="A company with this name already exists") from exc
        except Exception as exc:
            logger.error("Error inserting company: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to create company") from exc
```

**4. Playbook upload, PDF → chunks → embeddings → DB** — `admin/service.py:187-216`:

```python
async def upload_company_playbook_service(company_id: int, playbook_file: UploadFile) -> Dict[str, Any]:
    company = AdminRepository.get_company_by_id(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    is_pdf_extension = (playbook_file.filename or "").lower().endswith(".pdf")
    if playbook_file.content_type not in {"application/pdf", "application/octet-stream"} and not is_pdf_extension:
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    try:
        text = extract_pdf_text(playbook_file.file)
    except Exception as exc:
        logger.error("Failed to read uploaded playbook PDF: %s", exc)
        raise HTTPException(status_code=400, detail="Unable to read uploaded PDF") from exc

    chunks = chunk_playbook_section(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="No text could be extracted from this PDF (it may be scanned/image-only, or contain no recognized section headings). Nothing was ingested.")

    chunks_ingested = await AdminRepository.replace_company_playbook_chunks(company["name"], chunks)
    return {"message": "Playbook uploaded successfully", "company_id": company["id"], "company_name": company["name"], "chunks_ingested": chunks_ingested}
```

Shared extraction/chunking — `app/modules/rag/ingestion.py` (full file):

```python
"""Shared PDF-extraction and chunking logic for company playbook ingestion.

Used by both the offline `data/company_playbooks/ingest_kb.py` script and the
admin "upload playbook" endpoint, so there is one implementation of each step.
"""
import re
from typing import BinaryIO, Dict, List, Tuple, Union
import pdfplumber

SECTION_TITLES = {
    "company overview", "interview process", "technical questions", "behavioral questions",
    "resume questions", "common mistakes", "hiring signals", "faq", "ai mock interview dataset",
}

def extract_pdf_text(pdf_source: Union[str, BinaryIO]) -> str:
    text = ""
    with pdfplumber.open(pdf_source) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text:
                text += page_text + "\n"
    return text

def chunk_playbook_section(section_text: str, max_chars: int = 1500) -> List[Dict[str, str]]:
    lines = section_text.splitlines()
    subsections: List[Tuple[str, str]] = []
    current_title = None
    current_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in SECTION_TITLES:
            if current_lines:
                subsections.append((current_title, "\n".join(current_lines).strip()))
            current_title = stripped
            current_lines = []
            continue
        current_lines.append(line)
    if current_lines:
        subsections.append((current_title, "\n".join(current_lines).strip()))

    chunks: List[Dict[str, str]] = []
    for section_title, text in subsections:
        text = text.strip()
        if not text:
            continue
        if len(text) <= max_chars:
            chunks.append({"text": text, "section": section_title})
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        buffer = ""
        for paragraph in paragraphs:
            candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
            if len(candidate) > max_chars and buffer:
                chunks.append({"text": buffer, "section": section_title})
                buffer = paragraph
            else:
                buffer = candidate
        if buffer:
            chunks.append({"text": buffer, "section": section_title})
    return chunks
```

Embed-then-write — `admin/repository.py:1868-1898`:

```python
    @staticmethod
    async def replace_company_playbook_chunks(company_name: str, chunks: List[Dict[str, str]]) -> int:
        if not chunks:
            return 0
        try:
            embedded_chunks = []
            for chunk in chunks:
                text = str(chunk["text"])
                embedding = await embed_text(text, task_type="retrieval_document")
                embedded_chunks.append((chunk.get("section"), text, embedding))

            with db_pool.get_connection() as conn:
                with conn.cursor() as cur:
                    ensure_company_playbook_table(cur)
                    cur.execute(
                        "DELETE FROM company_playbook_chunks WHERE LOWER(TRIM(company_name)) = LOWER(TRIM(%s))",
                        (company_name,),
                    )
                    for section, text, embedding in embedded_chunks:
                        cur.execute(
                            """
                            INSERT INTO company_playbook_chunks (company_name, source_section, chunk_text, embedding, metadata)
                            VALUES (%s, %s, %s, %s::vector, %s)
                            """,
                            (company_name, section, text, embedding, Json({})),
                        )
                conn.commit()
            return len(chunks)
        except Exception as exc:
            logger.error("Error replacing playbook chunks for %s: %s", company_name, exc)
            raise HTTPException(status_code=500, detail="Failed to ingest company playbook") from exc
```

**5. RAG retrieval that later reads this data** — `app/modules/rag/service.py` (full
file, the `MATERIALIZED` CTE query):

```python
"""PostgreSQL/pgvector-backed company interview-context retrieval."""
from typing import List
from app.core.database import db_pool
from app.core.logger import logger
from app.modules.ai.client import embed_text


def ensure_company_playbook_table(cur) -> None:
    """Create the RAG store when pgvector is available in PostgreSQL."""
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS company_playbook_chunks (
            id SERIAL PRIMARY KEY,
            company_name TEXT NOT NULL,
            source_section TEXT,
            chunk_text TEXT NOT NULL,
            embedding vector(768) NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_company_playbook_company "
        "ON company_playbook_chunks (LOWER(TRIM(company_name)))"
    )


async def retrieve_company_context(company_name: str, query: str, top_k: int = 5) -> List[str]:
    """RAG is deliberately best-effort: a missing knowledge base must not prevent
    standard resume/JD question generation."""
    if not company_name or not query:
        return []
    try:
        embedding = await embed_text(query)
        vector_literal = "[" + ",".join(str(value) for value in embedding) + "]"
        with db_pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH candidates AS MATERIALIZED (
                        SELECT chunk_text, embedding
                        FROM company_playbook_chunks
                        WHERE LOWER(TRIM(company_name)) = LOWER(TRIM(%s))
                           OR POSITION(LOWER(TRIM(%s)) IN LOWER(company_name)) > 0
                    )
                    SELECT chunk_text, embedding <=> %s::vector AS distance
                    FROM candidates
                    ORDER BY distance
                    LIMIT %s
                    """,
                    (company_name, company_name, vector_literal, max(1, min(top_k, 10))),
                )
                rows = cur.fetchall()
                return [row[0] for row in rows]
    except Exception as exc:
        logger.warning("Company-playbook RAG retrieval skipped for %s: %s", company_name, exc)
        return []
```

---

### Flow E — Admin bulk-imports students via CSV (mentor tool)

**1. Frontend** — `MentorRegister.js` → `importStudentsCsv` (`api.js`, now posting via
`adminApi` instead of `backendApi` so the admin token is attached) →
`POST /mentors/students/import` (admin-JWT-gated as of 2026-09-01, `students/router.py`):

```python
@router.post(
    "/mentors/students/import",
    response_model=StudentImportResult,
    dependencies=[Depends(verify_admin_token)],
)
async def import_students_from_csv(
    file: UploadFile = File(...), program_id: Optional[int] = Form(None), ubp_id: Optional[int] = Form(None),
    university_name: Optional[str] = Form(None), program_name: Optional[str] = Form(None), batch_label: Optional[str] = Form(None),
) -> Any:
    """Bulk register students using a CSV upload, attaching them to a program context."""
    return await import_students_from_csv_service(file, program_id, ubp_id, university_name, program_name, batch_label)
```

Before 2026-09-01 this had **no** `dependencies` line at all - despite being described as
admin-gated in earlier notes, it was actually open to anyone. Gated now as an interim
measure; there is still no real mentor-auth system, so `MentorRegister.js`'s self-service
CSV upload only works if an admin happens to be logged into `/admin` in the same browser.

**2. Service — `program_id` resolution fallback chain** —
`students/service.py:751-761`:

```python
                    resolved_program_id = ubp_id or program_id
                    if not resolved_program_id and (university_name and program_name and batch_label):
                        resolved_program_id = resolve_ubp_id(
                            (university_name or "").strip(), (program_name or "").strip(), (batch_label or "").strip(),
                        )
                    if not resolved_program_id and program_from_csv:
                        resolved_program_id = resolve_program_id_by_name(program_from_csv)
                    if not resolved_program_id:
                        raise ValueError("Program context not provided or not found (select University/Program/Batch)")
```

**3. Insert + decoupled email send** — `students/service.py:780-807`:

```python
                    cur.execute(
                        """
                        INSERT INTO students (name, email, program_id, password, last_active)
                        VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                        RETURNING student_id
                        """,
                        (name, email, resolved_program_id, hashed_password),
                    )

                    imported += 1
                    try:
                        send_credentials_email(name, email, temp_password)
                        email_sent += 1
                    except Exception as email_exc:
                        logger.warning(
                            "Student %s (%s) created but credential email failed to send: %s",
                            name, email, email_exc,
                        )
                        email_warnings.append(
                            StudentEmailWarning(row=row_index, email=email,
                                note="Student created, credential email could not be sent - check SMTP config")
                        )
                except Exception as exc:
                    errors.append(StudentImportError(row=row_index, email=email or None, error=str(exc)))
```

`imported += 1` runs before the email attempt, and the email failure is caught in its
**own** nested try/except — so an SMTP failure never demotes a created student into an
`errors` entry. Compare this to Flow C/Step 5's password-reset email, which has no such
wrapping.

---

## STEP 3 — Load-bearing files, with actual code

### `app/core/database.py` — the global connection pool (full `get_connection`)

```python
class DatabasePool:
    """Singleton thread-safe connection pool for PostgreSQL using psycopg2."""
    _instance = None
    _pool = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._pool is None:
            try:
                self._pool = psycopg2.pool.ThreadedConnectionPool(
                    config.MIN_CONNECTIONS, config.MAX_CONNECTIONS, **config.DB_CONFIG
                )
                logger.info("Database pool initialized")
            except Exception as e:
                logger.error(f"Failed to create database pool: {e}")

    @contextmanager
    def get_connection(self):
        """Context manager to lease and return database connections safely."""
        conn = None
        try:
            conn = self._pool.getconn()
            if conn.closed != 0:
                self._pool.putconn(conn, close=True)
                conn = self._pool.getconn()

            yield conn

            if conn and not conn.closed:
                self._pool.putconn(conn)
                conn = None
        except psycopg2.OperationalError as e:
            logger.error(f"Database operational error (connection possibly dropped): {e}")
            if conn and not conn.closed:
                try:
                    conn.rollback()
                except Exception:
                    pass
                self._pool.putconn(conn, close=True)
                conn = None
            raise
        except Exception as e:
            logger.error(f"Database error: {e}")
            if conn and not conn.closed:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        finally:
            if conn and not conn.closed:
                self._pool.putconn(conn)


db_pool = DatabasePool()
```

Note the `finally` block re-runs the same `putconn` as the clean-exit path — on the
happy path, `conn` was already set to `None` right after the first `putconn`, so the
`finally`'s `if conn and not conn.closed` is a no-op there; it only actually matters on
the `OperationalError` path (where `conn` was also already set to `None`) or the generic
`Exception` path (where `conn` is *not* reset to `None`, so `finally` puts it back a
second time — redundant but harmless since `putconn` is idempotent on an already-pooled
connection). **This function never has a code path that yields `None`** — see Step 5,
issue 7, for callers that defensively check for it anyway.

Every `ensure_*` lazy-migration helper lives in this same file and is called by
essentially every repository method before it queries — e.g. the one added most
recently, `ensure_companies_table` (lines 217-232):

```python
def ensure_companies_table(cur) -> None:
    """Ensure companies table exists, matching its current live schema exactly."""
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS companies (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            logo_url TEXT,
            industry VARCHAR(255),
            difficulty_tag VARCHAR(50),
            work_experience_tag VARCHAR(100),
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
```

### `app/modules/ai/client.py` — the shared Gemini channel (full file)

```python
import asyncio
import inspect
from typing import Any, Dict, Optional
import google.generativeai as genai

from app.config.constants import FALLBACK_GEMINI_MODELS
from app.config.settings import config
from app.core.logger import logger

genai.configure(api_key=config.GEMINI_API_KEY)


async def execute_with_retries(
    func: Any, *args: Any, max_retries: int = 3, base_delay: float = 1.0,
    backoff_factor: float = 2.0, retry_label: str = "Gemini call", **kwargs: Any,
) -> Any:
    """Execute a callable with retry support and exponential backoff."""
    for attempt in range(max_retries):
        try:
            if inspect.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.to_thread(func, *args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception as exc:
            logger.warning(f"{retry_label} attempt {attempt + 1} failed: {exc}")
            if attempt >= max_retries - 1:
                raise
            await asyncio.sleep(base_delay * (backoff_factor ** attempt))


async def generate_content_with_fallback(
    prompt: Any, request_options: Optional[Dict[str, Any]] = None,
    generation_config: Optional[Dict[str, Any]] = None, retry_label: str = "Gemini call",
) -> Any:
    """Try primary and fallback Gemini models automatically when 429 quota or rate limit occurs."""
    if request_options is None:
        request_options = {"timeout": 120}
    last_exc = None
    for model_name in FALLBACK_GEMINI_MODELS:
        try:
            return await execute_with_retries(
                lambda m=model_name: genai.GenerativeModel(m).generate_content(
                    prompt, request_options=request_options, generation_config=generation_config,
                ),
                retry_label=f"{retry_label} ({model_name})", max_retries=2, base_delay=0.5,
            )
        except Exception as exc:
            last_exc = exc
            logger.warning(f"Model {model_name} quota/rate limit exceeded or failed during {retry_label}: {exc}. Trying next fallback model...")
            continue
    raise last_exc or RuntimeError("All fallback Gemini models failed")


_generate_content_with_fallback = generate_content_with_fallback


async def embed_text(text: str, task_type: str = "retrieval_query") -> list[float]:
    """Create a Gemini embedding for RAG retrieval."""
    result = await asyncio.to_thread(
        genai.embed_content, model=config.GEMINI_EMBEDDING_MODEL, content=text,
        task_type=task_type, output_dimensionality=768,
    )
    return result["embedding"]
```

Worst case: up to 3 models × 2 retries = 6 attempts with exponential backoff before
`generate_content_with_fallback` raises. `app/modules/ai/video.py` deliberately does
**not** use this function (see Flow B and Step 5).

### `app/config/settings.py` — the config singleton (full file)

```python
import os
from pathlib import Path
from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_ROOT / ".env")


class Config:
    DB_CONFIG = {
        "dbname": os.getenv("DB_NAME", "ai_mock_interviews"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
    }

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    if not (GEMINI_API_KEY.startswith("AIza") or GEMINI_API_KEY.startswith("AQ.")):
        raise ValueError(
            "GEMINI_API_KEY does not look like a Google AI Studio key. "
            "Use a standard key (AIza...) or authorization key (AQ...) from Google AI Studio."
        )

    SMTP_SETTINGS = {
        "host": os.getenv("SMTP_HOST"), "port": int(os.getenv("SMTP_PORT", "587")),
        "username": os.getenv("SMTP_USERNAME"), "password": os.getenv("SMTP_PASSWORD"),
        "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() == "true",
    }
    EMAIL_SENDER = os.getenv("EMAIL_SENDER")
    PASSWORD_RESET_URL_BASE = os.getenv("PASSWORD_RESET_URL_BASE")

    CACHE_TTL = int(os.getenv("CACHE_TTL", "900"))
    MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "128"))
    MIN_CONNECTIONS = int(os.getenv("MIN_DB_CONNECTIONS", "5"))
    MAX_CONNECTIONS = int(os.getenv("MAX_DB_CONNECTIONS", "50"))

    PISTON_BASE_URL = os.getenv("PISTON_BASE_URL", "https://emkc.org/api/v2/piston")

    MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", Path(__file__).resolve().parent.parent.parent / "media"))
    INTERVIEW_VIDEO_DIR = MEDIA_ROOT / "interview_videos"

    GEMINI_VIDEO_MODEL = os.getenv("GEMINI_VIDEO_MODEL", "gemini-2.5-flash-lite")
    GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")

    ADMIN_JWT_SECRET = os.getenv("ADMIN_JWT_SECRET")
    if not ADMIN_JWT_SECRET:
        raise ValueError("ADMIN_JWT_SECRET environment variable not set")
    ADMIN_JWT_ALGORITHM = os.getenv("ADMIN_JWT_ALGORITHM", "HS256")
    ADMIN_JWT_EXPIRES_MINUTES = int(os.getenv("ADMIN_JWT_EXPIRES", "1440"))

    STUDENT_JWT_SECRET = os.getenv("STUDENT_JWT_SECRET")
    if not STUDENT_JWT_SECRET:
        raise ValueError("STUDENT_JWT_SECRET environment variable not set")
    STUDENT_JWT_ALGORITHM = os.getenv("STUDENT_JWT_ALGORITHM", "HS256")
    STUDENT_JWT_EXPIRES_MINUTES = int(os.getenv("STUDENT_JWT_EXPIRES", "1440"))

    # TEMPORARY LOCAL-ONLY: path to the frontend's public/logos folder, used by
    # the admin "add company" flow to save uploaded logos as static frontend
    # assets. Only viable pre-deploy, when both apps run from source on the
    # same machine. See app/modules/admin/service.py for the full warning.
    FRONTEND_LOGOS_DIR = Path(os.getenv("FRONTEND_LOGOS_DIR", str(
        BACKEND_ROOT.parents[2] / "ai_mock_interview_v5-main" / "ai_mock_interview_v5-main" / "public" / "logos"
    )))


config = Config()
settings = config
# Media directories are created lazily by the code paths that write into them
# (see interview/service.py:_save_uploaded_video) rather than here at import
# time, so a read-only MEDIA_ROOT doesn't crash app startup.
```

`raise ValueError(...)` at class-body evaluation time (lines inside `class Config:`)
means importing this module — which nearly every other module does, transitively —
crashes the whole app if `GEMINI_API_KEY`, `ADMIN_JWT_SECRET`, or `STUDENT_JWT_SECRET`
is unset. That's now deliberate for all three (as of 2026-09-01): the two JWT secrets
used to silently fall back to a hardcoded, guessable string, and `STUDENT_JWT_SECRET`
was in fact **missing entirely** from `.env` — every student JWT was signed with that
fallback until this was caught and fixed. The `.mkdir()` calls that used to run at
**import time** (crashing startup outright on a read-only filesystem) were also removed
the same day; both media directories are now created lazily at write time instead.

### `app/config/constants.py` — the Gemini fallback model list

```python
# Gemini Fallback Models for rate limiting / quota protection.
# NOTE: gemini-2.5-flash, gemini-2.0-flash, gemini-2.0-flash-lite, and gemini-2.5-flash-lite
# all 404 ("no longer available to new users") against this project's API key as of 2026-08-14 --
# verified directly against the Gemini API, not just assumed. Keeping them in this list wastes
# a full retry+backoff cycle per dead model on every single Gemini call in the app before it
# ever reaches a working model. Only list models confirmed to actually respond.
FALLBACK_GEMINI_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
]
```

This list is read by `generate_content_with_fallback` (above) on every single Gemini
text call in the entire app — question generation, grading, feedback, competency
scoring. The comment documents that this has already broken once in production.

### `app/modules/auth/service.py` — shared password hashing (reused by admin + students)

```python
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    iterations = 200_000
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    salt_b64 = base64.b64encode(salt).decode("ascii")
    derived_b64 = base64.b64encode(derived).decode("ascii")
    return f"pbkdf2_sha256${iterations}${salt_b64}${derived_b64}"


def verify_password(password: str, stored_password: Optional[str]) -> bool:
    if not password or not stored_password:
        return False
    if stored_password.startswith("pbkdf2_sha256$"):
        try:
            _, iterations_str, salt_b64, derived_b64 = stored_password.split("$", 3)
            iterations = int(iterations_str)
            salt = base64.b64decode(salt_b64.encode("ascii"))
            expected = base64.b64decode(derived_b64.encode("ascii"))
            derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
            return hmac.compare_digest(derived, expected)
        except Exception:
            return False
    return hmac.compare_digest(password, stored_password)
```

Imported directly by `admin/service.py` (admin login) and `students/service.py`
(CSV-imported student passwords) — one implementation, every credential in the system.

### `ai_mock_interview_v5-main/src/api.js` — the three axios instances (lines 1-20)

```javascript
import axios from 'axios';

// Attach the logged-in student's JWT, when present, to a request. Used on both
// interviewApi and backendApi since student-scoped endpoints live on both.
const attachStudentToken = (config) => {
  try {
    const stored = localStorage.getItem('student');
    const token = stored ? JSON.parse(stored).access_token : null;
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  } catch (e) {
    // Malformed/missing student data in storage; proceed unauthenticated.
  }
  return config;
};

export const interviewApi = axios.create({
  baseURL: 'http://localhost:8000',
});
interviewApi.interceptors.request.use(attachStudentToken);

export const adminApi = axios.create({
  baseURL: 'http://localhost:8000',
});

export const backendApi = axios.create({
  baseURL: 'http://localhost:8000',
});
backendApi.interceptors.request.use(attachStudentToken);

// The frontend never talks to Piston directly - this hop is proxied through the
// FastAPI backend (avoids CORS). The backend, in turn, only reaches Piston as its
// last-resort fallback tier behind local execution and Judge0 CE, and by default
// forwards to the public https://emkc.org/api/v2/piston endpoint, not a self-hosted
// instance - see PISTON_BASE_URL in app/config/settings.py to override that.
export const fetchPistonRuntimes = () => backendApi.get('/piston/runtimes');

export const executeWithPiston = (payload) => backendApi.post('/piston/execute', payload);
```

As of 2026-09-01, both `interviewApi` and `backendApi` carry `attachStudentToken`, and
`adminApi` still carries its own separate admin-token interceptor shown below — `api.js`:

```javascript
// Admin authentication helpers
let adminAuthToken = null;

export const setAdminAuthToken = (token) => {
  adminAuthToken = token || null;
};

export const adminLogin = (payload) => adminApi.post('/admin/auth/login', payload);
export const adminLogout = () => adminApi.post('/admin/auth/logout');
export const fetchAdminProfile = () => adminApi.get('/admin/auth/me');

// Apply auth token to outbound admin requests when available
adminApi.interceptors.request.use((config) => {
  if (adminAuthToken) {
    config.headers.Authorization = `Bearer ${adminAuthToken}`;
  }
  return config;
});
```

Before 2026-09-01, `interviewApi` and `backendApi` had **no** equivalent interceptor and
carried no `Authorization` header at all on any student-facing call — see Step 5 for the
full account of what that let through and how it was fixed.

---

## STEP 4 — External boundaries: the actual error-handling code

### Gemini — text generation

Covered fully in Step 3 (`generate_content_with_fallback`). Caller-side handling varies —
Flow B's grading call has a hard timeout with hardcoded fallbacks:

```python
# interview/service.py:1023-1041
    try:
        ai_future = analyze_answer_and_generate_response(...)
        sentiment_score, acknowledgment, next_difficulty_raw = await asyncio.wait_for(ai_future, timeout=12.0)
    except asyncio.TimeoutError:
        sentiment_score = 5.0
        acknowledgment = "Thank you for sharing your experience. Let's move on to the next topic."
        next_difficulty_raw = "medium"
    except Exception as exc:
        sentiment_score = 5.0
        acknowledgment = "Got it. Let's continue with our discussion."
        next_difficulty_raw = "medium"
```

Feedback generation (Flow C) instead treats the question-feedback call as required and
lets it raise (`if isinstance(question_response, Exception): raise question_response`),
while treating competency scoring as best-effort (`logger.warning(...)`, no raise) — both
shown in full in Step 2, Flow C.

### Gemini — embeddings / RAG

`retrieve_company_context` (Step 2, Flow D) wraps everything in one `try/except` and
returns `[]` on any failure — shown in full above. Playbook **ingestion**
(`replace_company_playbook_chunks`, also shown in full above) has no equivalent softness:
an embedding failure raises `HTTPException(500)` before any DB write happens.

### Gemini — video analysis

`analyze_video_sentiment_sync` (Step 2, Flow B) now goes through
`generate_content_with_fallback` (as of 2026-09-01, bridged via `asyncio.run()` since
this stays a synchronous function) instead of calling Gemini directly — same
retry/fallback protection as every other Gemini call, with `GEMINI_VIDEO_MODEL` tried
first. Failure still returns `None`, which the caller
(`process_video_analysis_background`) turns into `video_analysis_status = 'failed'` in
the DB, then deletes the source file regardless:

```python
# ai/video.py:151-157
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
                logger.debug(f"Removed video clip after analysis: {video_path}")
            except OSError as cleanup_error:
                logger.warning(f"Failed to remove video clip {video_path}: {cleanup_error}")
```

### Judge0 / Piston

Full cascade shown in Step 2, Flow B (`execute_hybrid_code`). Judge0 failure is caught
and logged as a warning with a silent fallthrough to Piston; Piston failure raises a real
HTTP error back to the caller (`HTTPException(502)` or the mapped status code).

### SMTP — two code paths, now symmetric (fixed 2026-09-01; was an asymmetry before)

**CSV import (fixed)** — `students/service.py:790-804` (shown in full in Step 2, Flow E):
email failure caught in its own try/except, logged, appended to `email_warnings`, student
still counts as imported.

**Password reset (fixed 2026-09-01, now matches the CSV-import pattern)** —
`auth/service.py`:

```python
def create_password_reset_token(email: str) -> Dict[str, Any]:
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=1)

    with db_pool.get_connection() as conn:
        if not conn:
            raise HTTPException(status_code=500, detail="Database connection unavailable")
        with conn.cursor() as cur:
            cur.execute(
                "SELECT student_id FROM students WHERE LOWER(TRIM(email)) = LOWER(TRIM(%s))",
                (email.strip(),),
            )
            student_row = cur.fetchone()
            if not student_row:
                raise HTTPException(status_code=404, detail="No account found for this email")

            cur.execute(
                """
                INSERT INTO password_reset_tokens (email, token, expires_at)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (email.strip(), token, expires_at),
            )
            conn.commit()

    email_sent = True
    try:
        send_password_reset_email(email.strip(), token, expires_at)
    except Exception as email_exc:
        email_sent = False
        logger.warning(
            "Password reset token created for %s but reset email failed to send: %s",
            email.strip(), email_exc,
        )

    return {"token": token, "expires_at": expires_at.isoformat(), "email_sent": email_sent}
```

And `send_password_reset_email` itself, `auth/service.py:307-322`, does raise on SMTP
failure — it just has no caller ready to catch it:

```python
    try:
        if smtp_cfg.get("use_tls", True):
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"]) as server:
                server.starttls(context=context)
                if smtp_cfg.get("username") and smtp_cfg.get("password"):
                    server.login(smtp_cfg["username"], smtp_cfg["password"])
                server.send_message(message)
        else:
            with smtplib.SMTP(smtp_cfg["host"], smtp_cfg["port"]) as server:
                if smtp_cfg.get("username") and smtp_cfg.get("password"):
                    server.login(smtp_cfg["username"], smtp_cfg["password"])
                server.send_message(message)
    except Exception as exc:
        logger.error(f"Failed to send password reset email to {email}: {exc}")
        raise HTTPException(status_code=500, detail="Unable to send password reset email") from exc
```

The token row was already `conn.commit()`-ed by the time this raises — the student sees
a 500, but a valid, usable token now exists in `password_reset_tokens`.

### Database connection pool

Full `get_connection()` shown in Step 3 — rolls back and re-raises on failure, never
yields `None`.

### Local filesystem

`MEDIA_ROOT`/`INTERVIEW_VIDEO_DIR` creation was an import-time side effect in
`settings.py` until 2026-09-01 (see Step 3) — both are now created lazily by the code
that writes into them, so a read-only filesystem no longer crashes app startup. The
admin logo save (writes into the **frontend's** source tree, still an open item, not
fixed — confirmed still clearly marked as temporary) —
`admin/service.py:112-147`:

```python
async def _save_company_logo_temp_local(company_name: str, logo: Optional[UploadFile]) -> Optional[str]:
    """TEMPORARY LOCAL-ONLY SOLUTION.

    Saves an admin-uploaded company logo directly into the frontend's
    public/logos/ folder so it's servable as a static asset immediately,
    without adding a static-file route to this backend. This only works
    because the frontend and backend both run from source on the same
    machine during local development.

    Local disk writes do not survive on serverless hosting. BEFORE
    DEPLOYING to Vercel, replace this with real cloud storage (Vercel
    Blob / S3 / Cloudinary) and point logo_url at that instead.
    """
    if logo is None or not logo.filename:
        return None
    if logo.content_type not in {"image/png", "application/octet-stream"}:
        logger.warning("Rejected non-PNG logo upload for %s (content_type=%s)", company_name, logo.content_type)
        return None
    try:
        slug = _slugify_company_name(company_name)
        if not slug:
            return None
        logos_dir = config.FRONTEND_LOGOS_DIR
        os.makedirs(logos_dir, exist_ok=True)
        file_path = logos_dir / f"{slug}.png"
        content = await logo.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        return f"/logos/{slug}.png"
    except Exception as exc:
        logger.warning("Failed to save company logo for %s, falling back to no logo: %s", company_name, exc)
        return None
```

---

## STEP 5 — Known issues, with the actual buggy/risky code

**Updated 2026-09-01.** All 14 issues originally listed here were fixed the same day,
except #13 (React Context) which was left alone deliberately — it's an architectural
style choice, not a bug. Fixing the rest, endpoint by endpoint, surfaced a second, larger
wave of the same root problem (#4 below), which is why this section is now organized
around that instead of a flat numbered list.

### The core problem: unverified identity, found in three separate audit passes

The single biggest issue in this codebase was **routes trusting a client-supplied
`student_id`/`student_email`/`session_id` instead of verifying who's actually logged
in**. It showed up in three waves as the audit got more thorough:

**Wave 1 — the original #4.** `verify_student_token` (`auth/service.py`) was wired to
exactly one route, `GET /students/me`. `start_interview` and `submit_answer` both
trusted a plain `student_email` form field with no verification. Fixed by requiring
`Depends(verify_student_token)` on both, using the verified token's email/student_id
instead of the client-supplied one, and adding a session-ownership check to
`submit_answer` (a new `InterviewRepository.verify_session_belongs_to_student` helper).

**Wave 2 — a full re-sweep of every router turned up ~10 more endpoints with the exact
same shape**: `GET /students/profile/{email}`, `GET/POST /students/{id}/resume`,
`GET/POST /students/{id}/job-description`, `GET /students/{id}/resume-interview-sessions`,
`GET /students/{id}/performance-summary`, `GET /students/{id}/interview-history`,
`POST /api/generate-resume-questions`, and `GET /students/sessions/by_email/{email}`
(this last one was mislabeled `tags=["Admin Student Oversight"]` in `admin/router.py`
next to ~35 correctly-guarded admin endpoints, despite actually being called by the
student dashboard, not the admin panel). All fixed the same way: require the token,
403 if the verified identity doesn't match what was requested. Every frontend call site
was checked first to confirm the "self only" policy didn't break a legitimate
admin-looks-up-any-student flow — none of these were ever called from the admin UI.

**Wave 3 — a third pass over `interview/router.py` specifically** (triggered by
diagnosing the `adminApi`/`backendApi` frontend bug below, which prompted a systematic
recheck) found six more: `GET /interview/active-session`, `POST /interview/reattempt/check`,
`GET/POST /students/sessions/{id}/rating` (email-trusted), and
`POST /interview/{id}/terminate` / `POST /interview/{id}/upload-video` (session-ID-only,
no identity check of any kind, not even an unverified email). Same fix pattern.

**The frontend needed two matching changes.** `interviewApi` and `backendApi` (`api.js`)
had no `Authorization`-header interceptor at all before this — only `adminApi` did. A
shared `attachStudentToken` interceptor was added to both (see Step 3's `api.js` block).
Separately, `Dashboard.js`'s fetch of `/students/sessions/by_email/{email}` was found to
be using **`adminApi`** instead of `backendApi` — a pre-existing mistake invisible for as
long as that endpoint had no auth check to fail against. Once Wave 2 locked the endpoint
down, every student's "Mock Interview Records" panel went silently, permanently empty
until this was traced and fixed. A full audit of all 60+ functions in `api.js` against
their axios instance found this was the only occurrence of that specific mistake.

### Other issues from the original list, now fixed

- **Password-reset email failure was unhandled**, unlike the identical CSV-import case.
  `create_password_reset_token` now wraps `send_password_reset_email(...)` in its own
  try/except, matching the CSV-import pattern — logs a warning and returns an
  `email_sent: False` flag instead of raising a 500.
- **Video analysis bypassed the shared Gemini fallback wrapper.** `generate_content_with_fallback`
  gained an optional `models` param so a caller can put its own preferred model first
  while still getting the shared retry/fallback behavior; `ai/video.py` now uses it with
  `[GEMINI_VIDEO_MODEL, *FALLBACK_GEMINI_MODELS]`.
- **Dead duplicate function** `save_uploaded_video` in `ai/video.py` — deleted, along with
  its now-unused imports.
- **`MEDIA_ROOT.mkdir()` at config-import time** — removed; both media directories are now
  created lazily by the code that actually writes into them.
- **`submit_answer` had no exception handling from Phase 3 onward, and its router had
  none at all** — both now wrapped, matching the existing Phase 1 pattern.
- **`db_pool.get_connection()` never yields `None`** — confirmed correct by re-reading the
  implementation. Deliberately left as-is: the `if not conn:` checks are genuinely dead
  code (unreachable, since a failure there raises before yielding), but removing them
  would be a wide, mechanical, zero-functional-benefit sweep across dozens of call sites.
  Not worth the churn; noted here so nobody "fixes" it by half-measures later.
- **Company logo upload writes into the frontend's own source tree** — confirmed still
  clearly marked `TEMPORARY LOCAL-ONLY` in both `admin/service.py` and `settings.py`,
  with the real fix (cloud storage) documented in the docstring. Not fixed — deliberately
  out of scope pre-deploy work, tracked separately.
- **Piston "self-hosted" comment was misleading** — corrected in `api.js` to describe the
  actual chain: frontend→backend is proxied (avoids CORS), backend→Piston goes to the
  public `emkc.org` endpoint by default and is only the last-resort fallback tier behind
  local execution and Judge0 CE.
- **Stray bare `6` in `interview/router.py`** and **dead `visibleCompanies` variable in
  `TrendingCompanies.js`** — both turned out to be uncommitted leftovers never part of any
  commit; removing them restored the files to exactly match git HEAD.
- **Two overlapping auth-guard mechanisms in `App.js`** — consolidated to declarative-only.
  The imperative `useEffect` redirect duplicated every declarative per-route guard except
  one case (redirecting an authenticated student away from `/login`, which no declarative
  route handled). That case moved onto the `/login` route itself; the effect was deleted.
  Safe because `<Routes>` is already gated behind `isHydrated`, which flips true in the
  same effect that populates `student`, so there's no render before `student` is accurate.

### New issues found the same day, not on the original list

- **`ADMIN_JWT_SECRET`/`STUDENT_JWT_SECRET` defaulted to a hardcoded, guessable string**
  (`"admin_jwt_secret"` / `"student_jwt_secret"`) when unset. `STUDENT_JWT_SECRET` was in
  fact **completely absent** from `.env` — every student JWT issued before this was
  caught was signed with that hardcoded fallback and forgeable by anyone who read the
  source. Both now fail loudly at startup if unset (matching the existing `GEMINI_API_KEY`
  pattern), and a real `STUDENT_JWT_SECRET` was generated. Consequence: every
  previously-issued student token became invalid the moment the secret changed —
  expected, not a bug, but it did cause a round of confusing 401s on every student-data
  endpoint until each affected browser session re-logged in.
- **CORS allowed `allow_origins=["*"]` with `allow_credentials=True`** — restricted to
  local dev origins; production origin still needs adding before deploy (see Step 1).
- **`POST /mentors/students/import` had no auth at all** — gated behind
  `Depends(verify_admin_token)` as an interim measure (see Flow E). No real mentor-auth
  system exists, so this is a known incomplete state, not a finished fix.

### Still open — needs a product/deployment decision, not just a code fix

- **`.env` files with live `DB_PASSWORD`/`GEMINI_API_KEY` are tracked in git history** —
  both `job_readiness_backend-main/.env` and the nested `.../job_readiness_backend_vercel/.env`.
  Rotation + untracking commands were documented but not executed as of this writing.
- **Four `/debug/*` endpoints in `interview/router.py` have no auth**, including one
  (`force-score-update`) that mutates data. Development leftovers, never removed.
- **`/piston/execute` and `/piston/runtimes` have no auth** — may be intentional for a
  public practice tool, needs a deliberate product call rather than a default.
- **No real mentor-auth system** (see above) — `/mentors/students/import` is currently
  admin-only in practice, with no path for an actual mentor to self-serve.

---

*End of document. See also (narrower, more granular references this document draws on):
[`scripts/changes_reference.md`](../scripts/changes_reference.md) and
[`scripts/full_session_documentation.md`](../scripts/full_session_documentation.md) —
both predate the 2026-09-01 fixes and are historical, not current.*
