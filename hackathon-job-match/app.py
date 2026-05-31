from __future__ import annotations

import json
import mimetypes
from base64 import b64decode
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from everos_context import (
    remember_analysis_result,
    remember_job_feedback,
    retrieve_user_preferences,
)
from llm_analyzer import LLMUnavailable, analyze_with_llm, llm_is_configured
from matcher import analyze_fit
from pdf_text import extract_pdf_text


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
HOST = "127.0.0.1"
PORT = 5050


class BoleHandler(BaseHTTPRequestHandler):
    server_version = "Bole/0.1"

    def do_GET(self) -> None:
        if self.path == "/" or self.path.startswith("/?"):
            self._send_file(STATIC_DIR / "index.html")
            return

        requested = (STATIC_DIR / self.path.lstrip("/")).resolve()
        if STATIC_DIR not in requested.parents and requested != STATIC_DIR:
            self._send_json({"error": "Not found"}, status=404)
            return

        if requested.is_file():
            self._send_file(requested)
            return

        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        if self.path == "/api/extract-resume":
            self._handle_extract_resume()
            return

        if self.path == "/api/job-feedback":
            self._handle_job_feedback()
            return

        if self.path != "/api/analyze":
            self._send_json({"error": "Not found"}, status=404)
            return

        try:
            payload = self._read_json()
            user_id = normalize_user_id(str(payload.get("user_id", "")))
            resume_text = str(payload.get("resume_text", "")).strip()
            career_url = str(payload.get("career_url", "")).strip()
            jobs_text = str(payload.get("jobs_text", "")).strip()
            use_llm = bool(payload.get("use_llm", False))

            if not user_id:
                self._send_json(
                    {"error": "Please sign in with an email or demo username first."},
                    status=400,
                )
                return

            if not resume_text:
                self._send_json({"error": "Resume text is required."}, status=400)
                return

            memory_lookup = retrieve_user_preferences(
                user_id,
                resume_text=resume_text,
                career_url=career_url,
            )

            page_text = jobs_text
            fetch_error = ""
            if career_url and not page_text:
                page_text, fetch_error = fetch_career_page(career_url)

            if not page_text:
                message = fetch_error or "A career page URL or pasted job text is required."
                self._send_json({"error": message}, status=400)
                return

            result = analyze_fit(resume_text, page_text, source_url=career_url)
            result["user_id"] = user_id
            result["memory"] = {
                "enabled": memory_lookup.enabled,
                "context": memory_lookup.text,
                "status": memory_lookup.status,
            }
            if memory_lookup.warning:
                result["warnings"].append(memory_lookup.warning)
            if fetch_error:
                result["warnings"].append(fetch_error)
            result["llm_enabled"] = llm_is_configured()

            if use_llm:
                try:
                    llm_result = analyze_with_llm(
                        resume_text,
                        result["recommendations"],
                        memory_context=memory_lookup.text,
                    )
                    apply_llm_result(result, llm_result)
                except LLMUnavailable as exc:
                    result["warnings"].append(str(exc))

            memory_write = remember_analysis_result(
                user_id,
                resume_profile=result.get("candidate_profile", {}),
                top_jobs=result.get("recommendations", []),
                warnings=result.get("warnings", []),
                user_intent=build_user_intent(
                    career_url=career_url,
                    jobs_text=jobs_text,
                    resume_text=resume_text,
                ),
                career_url=career_url,
            )
            result["memory"]["write"] = {
                "enabled": memory_write.enabled,
                "stored": memory_write.stored,
                "status": memory_write.status,
            }
            if memory_write.warning:
                result["warnings"].append(memory_write.warning)

            self._send_json(result)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body."}, status=400)
        except Exception as exc:  # pragma: no cover - keeps hackathon server resilient.
            self._send_json({"error": f"Analysis failed: {exc}"}, status=500)

    def _handle_extract_resume(self) -> None:
        try:
            payload = self._read_json()
            filename = str(payload.get("filename", "resume")).lower()
            content_type = str(payload.get("content_type", ""))
            encoded = str(payload.get("data", ""))
            if not encoded:
                self._send_json({"error": "File data is required."}, status=400)
                return

            raw = b64decode(encoded, validate=True)
            if filename.endswith(".pdf") or content_type == "application/pdf":
                text = extract_pdf_text(raw)
            else:
                text = raw.decode("utf-8", errors="ignore").strip()

            if not text:
                self._send_json({"error": "No readable text was found in this file."}, status=400)
                return

            self._send_json({"text": text})
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body."}, status=400)
        except Exception as exc:  # pragma: no cover - keeps hackathon server resilient.
            self._send_json({"error": f"Resume extraction failed: {exc}"}, status=500)

    def _handle_job_feedback(self) -> None:
        try:
            payload = self._read_json()
            user_id = normalize_user_id(str(payload.get("user_id", "")))
            feedback = str(payload.get("feedback", "")).strip()
            job = payload.get("job", {})

            if not user_id:
                self._send_json(
                    {"error": "Please sign in before saving job feedback."},
                    status=400,
                )
                return
            if feedback not in {"save", "not_interested", "applied"}:
                self._send_json({"error": "Invalid feedback type."}, status=400)
                return
            if not isinstance(job, dict) or not str(job.get("title", "")).strip():
                self._send_json({"error": "Job title is required."}, status=400)
                return

            write = remember_job_feedback(user_id, feedback=feedback, job=job)
            payload = {
                "feedback": feedback,
                "memory": {
                    "enabled": write.enabled,
                    "stored": write.stored,
                    "status": write.status,
                },
            }
            if write.warning:
                payload["warning"] = write.warning
            self._send_json(payload)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body."}, status=400)
        except Exception as exc:  # pragma: no cover - keeps hackathon server resilient.
            self._send_json({"error": f"Feedback failed: {exc}"}, status=500)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def fetch_career_page(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "", "Enter a valid http or https career page URL."

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urlopen(request, timeout=12) as response:
            content_type = response.headers.get("Content-Type", "")
            raw = response.read(2_500_000)
    except HTTPError as exc:
        return "", f"The career page returned HTTP {exc.code}. Paste job text instead."
    except URLError as exc:
        return "", f"Could not fetch the career page: {exc.reason}."
    except TimeoutError:
        return "", "The career page request timed out. Paste job text instead."

    if "html" not in content_type and "text" not in content_type and content_type:
        return "", "The career page did not return readable text or HTML."

    return raw.decode("utf-8", errors="ignore"), ""


def normalize_user_id(value: str) -> str:
    value = value.strip().lower()
    if not value:
        return ""
    safe = []
    for char in value:
        if char.isalnum() or char in {"@", ".", "_", "-"}:
            safe.append(char)
    return "".join(safe)[:120]


def build_user_intent(*, career_url: str, jobs_text: str, resume_text: str) -> str:
    source = career_url or "pasted job descriptions"
    resume_hint = " ".join(resume_text.split())[:220]
    if jobs_text:
        return (
            f"Find roles from {source} that fit my resume and career preferences. "
            f"Resume hint: {resume_hint}"
        )
    return f"Screen the career page {source} for roles that fit my resume. Resume hint: {resume_hint}"


def apply_llm_result(result: dict[str, Any], llm_result: dict[str, Any]) -> None:
    result["candidate_profile"] = llm_result.get("candidate_profile", {})
    updates = {
        update.get("title", "").lower(): update
        for update in llm_result.get("job_updates", [])
        if update.get("title")
    }

    for job in result.get("recommendations", []):
        update = updates.get(str(job.get("title", "")).lower())
        if not update:
            continue
        job["llm_score"] = update.get("llm_score", 0)
        job["llm_reason"] = update.get("llm_reason", "")
        job["interview_pitch"] = update.get("interview_pitch", "")
        job["risk_flags"] = update.get("risk_flags", [])

    result["recommendations"].sort(
        key=lambda item: (
            item.get("llm_score", item.get("score", 0)),
            item.get("score", 0),
        ),
        reverse=True,
    )


def run() -> None:
    server = ThreadingHTTPServer((HOST, PORT), BoleHandler)
    print(f"Bole running at http://{HOST}:{PORT}/")
    server.serve_forever()


if __name__ == "__main__":
    run()
