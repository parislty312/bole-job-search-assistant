from __future__ import annotations

import json
import mimetypes
import os
import re
import uuid
from base64 import b64decode
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

from everos_context import (
    remember_analysis_result,
    remember_job_feedback,
    retrieve_user_preferences,
)
from llm_analyzer import (
    LLMUnavailable,
    apply_multi_agent_result,
    llm_is_configured,
    run_multi_agent_analysis,
)
from matcher import analyze_fit
from pdf_text import extract_pdf_text


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
HOST = "127.0.0.1"
PORT = 5050
OPENAI_AUDIO_TRANSCRIPTIONS_URL = "https://api.openai.com/v1/audio/transcriptions"
DEFAULT_TRANSCRIPTION_MODEL = "whisper-1"
EIGHTFOLD_FETCH_LIMIT = 40


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

        if self.path == "/api/transcribe-voice":
            self._handle_transcribe_voice()
            return

        if self.path != "/api/analyze":
            self._send_json({"error": "Not found"}, status=404)
            return

        try:
            payload = self._read_json()
            user_id = normalize_user_id(str(payload.get("user_id", "")))
            resume_text = str(payload.get("resume_text", "")).strip()
            career_url = str(payload.get("career_url", "")).strip()
            target_intent = str(payload.get("target_intent", "")).strip()
            jobs_text = str(payload.get("jobs_text", "")).strip()
            # The UI control is retained for compatibility, but configured AI review now
            # runs automatically as a coordinated career team.
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
                target_intent=target_intent,
            )

            page_text = jobs_text
            fetch_error = ""
            if career_url and not page_text:
                page_text, fetch_error = fetch_career_page(career_url)

            if not page_text:
                message = fetch_error or "A career page URL or pasted job text is required."
                self._send_json({"error": message}, status=400)
                return

            result = analyze_fit(
                resume_text,
                page_text,
                source_url=career_url,
                target_intent=target_intent,
            )
            result["target_intent"] = target_intent
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

            result["review_status"] = "not_configured"
            if llm_is_configured():
                try:
                    agent_result = run_multi_agent_analysis(
                        resume_text,
                        result["recommendations"],
                        memory_context=memory_lookup.text,
                        target_intent=target_intent,
                    )
                    apply_multi_agent_result(result, agent_result)
                except LLMUnavailable as exc:
                    result["review_status"] = "unavailable"
                    result["agent_analysis"] = {
                        "status": "unavailable",
                        "summary": "AI career-team review was unavailable; Bole kept the baseline matching results.",
                        "agents": {},
                    }
                    result["warnings"].append(str(exc))
            elif use_llm:
                result["warnings"].append(
                    "AI career-team review is unavailable because OPENAI_API_KEY is not configured."
                )

            memory_write = remember_analysis_result(
                user_id,
                resume_profile=result.get("candidate_profile", {}),
                top_jobs=result.get("recommendations", []),
                warnings=result.get("warnings", []),
                user_intent=build_user_intent(
                    career_url=career_url,
                    jobs_text=jobs_text,
                    target_intent=target_intent,
                    resume_text=resume_text,
                ),
                career_url=career_url,
                memory_updates=result.get("memory_updates", {}),
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

    def _handle_transcribe_voice(self) -> None:
        try:
            if not os.getenv("OPENAI_API_KEY"):
                self._send_json(
                    {"error": "Voice transcription needs OPENAI_API_KEY on the server."},
                    status=400,
                )
                return

            payload = self._read_json()
            encoded = str(payload.get("data", ""))
            content_type = str(payload.get("content_type", "audio/webm")) or "audio/webm"
            if not encoded:
                self._send_json({"error": "Audio data is required."}, status=400)
                return

            audio = b64decode(encoded, validate=True)
            if len(audio) > 8 * 1024 * 1024:
                self._send_json({"error": "Voice recording is too large. Try a shorter clip."}, status=400)
                return

            text = transcribe_voice_audio(audio, content_type=content_type)
            if not text:
                self._send_json({"error": "No speech was detected. Try again or type your target role."}, status=400)
                return

            self._send_json({"text": text})
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON body."}, status=400)
        except Exception as exc:  # pragma: no cover - keeps hackathon server resilient.
            self._send_json({"error": f"Voice transcription failed: {exc}"}, status=500)

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

    text = raw.decode("utf-8", errors="ignore")
    if is_eightfold_career_page(url, text):
        jobs_html, warning = fetch_eightfold_jobs(url, text)
        if jobs_html:
            return jobs_html, warning

    return text, ""


def is_eightfold_career_page(url: str, html: str) -> bool:
    host = urlparse(url).netloc.lower()
    return (
        "jobs.nvidia.com" in host
        or "/api/pcsx/search" in html
        or 'window._EF_PRODUCT = "PCS"' in html
        or "get_html_smartapply_matches_v2" in html
    )


def fetch_eightfold_jobs(url: str, html: str) -> tuple[str, str]:
    parsed = urlparse(url)
    domain = infer_eightfold_domain(url, html)
    if not domain:
        return "", ""

    detail_position_id = extract_eightfold_position_id(url)
    if detail_position_id:
        try:
            details = fetch_eightfold_position_details(
                parsed,
                domain,
                detail_position_id,
                referer=url,
            )
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
            return "", ""
        if not details:
            return "", ""

        position = {
            "id": detail_position_id,
            "positionUrl": parsed.path,
            **details,
        }
        warning = (
            "Bole detected a single Eightfold-powered job page and matched only "
            "that job description."
        )
        return eightfold_positions_to_html([position], parsed), warning

    try:
        positions = fetch_eightfold_positions(parsed, domain)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
        return "", ""
    if not positions:
        return "", ""

    detailed_positions = []
    for position in positions[:EIGHTFOLD_FETCH_LIMIT]:
        position_id = str(position.get("id", "")).strip()
        try:
            details = fetch_eightfold_position_details(parsed, domain, position_id, referer=url)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
            details = {}
        if details:
            detailed_positions.append({**position, **details})
        else:
            detailed_positions.append(position)

    warning = (
        f"Bole detected an Eightfold-powered career page and loaded "
        f"{len(detailed_positions)} recent jobs from its jobs API for matching."
    )
    return eightfold_positions_to_html(detailed_positions, parsed), warning


def extract_eightfold_position_id(url: str) -> str:
    parsed = urlparse(url)
    match = re.search(r"/careers/job/(\d+)", parsed.path)
    if match:
        return match.group(1)
    return ""


def infer_eightfold_domain(url: str, html: str) -> str:
    query_domain = parse_qs(urlparse(url).query).get("domain", [""])[0].strip()
    if query_domain:
        return query_domain

    match = re.search(r'window\._EF_GROUP_ID\s*=\s*"([^"]+)"', html)
    if match:
        return match.group(1)

    host = urlparse(url).netloc.lower()
    if host == "jobs.nvidia.com":
        return "nvidia.com"
    return ""


def fetch_eightfold_positions(parsed_url: Any, domain: str) -> list[dict[str, Any]]:
    query = parse_qs(parsed_url.query)
    start = int(query.get("start", ["0"])[0] or "0")
    page_size = 10
    collected: list[dict[str, Any]] = []
    seen: set[str] = set()

    while len(collected) < EIGHTFOLD_FETCH_LIMIT:
        params = {
            "domain": domain,
            "start": str(start + len(collected)),
            "num": str(page_size),
            "sort_by": query.get("sort_by", ["timestamp"])[0] or "timestamp",
        }
        if query.get("query", [""])[0]:
            params["query"] = query["query"][0]
        if query.get("location", [""])[0]:
            params["location"] = query["location"][0]

        api_url = f"{parsed_url.scheme}://{parsed_url.netloc}/api/pcsx/search?{urlencode(params)}"
        payload = fetch_json(api_url, referer=parsed_url.geturl())
        positions = payload.get("data", {}).get("positions", [])
        if not isinstance(positions, list) or not positions:
            break

        for position in positions:
            key = str(position.get("id") or position.get("positionUrl") or position.get("name", ""))
            if key in seen:
                continue
            seen.add(key)
            collected.append(position)
            if len(collected) == EIGHTFOLD_FETCH_LIMIT:
                break

        if len(positions) < page_size:
            break

    return collected


def fetch_eightfold_position_details(
    parsed_url: Any,
    domain: str,
    position_id: str,
    *,
    referer: str,
) -> dict[str, Any]:
    if not position_id:
        return {}
    params = urlencode({"position_id": position_id, "domain": domain})
    api_url = f"{parsed_url.scheme}://{parsed_url.netloc}/api/pcsx/position_details?{params}"
    payload = fetch_json(api_url, referer=referer)
    data = payload.get("data", {})
    return data if isinstance(data, dict) else {}


def fetch_json(url: str, *, referer: str = "") -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
            "Referer": referer,
            "X-Requested-With": "XMLHttpRequest",
        },
    )
    with urlopen(request, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def eightfold_positions_to_html(positions: list[dict[str, Any]], parsed_url: Any) -> str:
    parts = ["<html><body>"]
    for position in positions:
        title = str(position.get("name", "")).strip()
        if not title:
            continue
        relative_url = str(position.get("positionUrl") or f"/careers/job/{position.get('id', '')}")
        job_url = urljoin(f"{parsed_url.scheme}://{parsed_url.netloc}", relative_url)
        description = strip_html_to_text(str(position.get("jobDescription") or ""))
        locations = ", ".join(str(item) for item in position.get("locations", [])[:6])
        work_location = str(position.get("workLocationOption") or "")
        metadata = " ".join(
            item
            for item in [
                f"Locations: {locations}." if locations else "",
                f"Work location: {work_location}." if work_location else "",
            ]
            if item
        )

        parts.extend(
            [
                f'<h2><a href="{escape(job_url)}">{escape(title)}</a></h2>',
                f"<p>{escape(metadata)}</p>",
                f"<p>{escape(description)}</p>",
            ]
        )
    parts.append("</body></html>")
    return "\n".join(parts)


def strip_html_to_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_user_id(value: str) -> str:
    value = value.strip().lower()
    if not value:
        return ""
    safe = []
    for char in value:
        if char.isalnum() or char in {"@", ".", "_", "-"}:
            safe.append(char)
    return "".join(safe)[:120]


def transcribe_voice_audio(audio: bytes, *, content_type: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    boundary = f"----BoleVoice{uuid.uuid4().hex}"
    model = os.getenv("OPENAI_TRANSCRIBE_MODEL") or DEFAULT_TRANSCRIPTION_MODEL
    body = build_multipart_form(
        boundary=boundary,
        fields={"model": model},
        files={
            "file": {
                "filename": guess_audio_filename(content_type),
                "content_type": content_type,
                "data": audio,
            }
        },
    )
    request = Request(
        OPENAI_AUDIO_TRANSCRIPTIONS_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=45) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise ValueError(f"Transcription request failed with HTTP {exc.code}: {detail[:240]}")
    except URLError as exc:
        raise ValueError(f"Could not reach the transcription API: {exc.reason}")
    except TimeoutError:
        raise ValueError("The transcription request timed out. Try a shorter clip.")

    return str(payload.get("text", "")).strip()


def build_multipart_form(
    *,
    boundary: str,
    fields: dict[str, str],
    files: dict[str, dict[str, Any]],
) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    for name, file_info in files.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{file_info["filename"]}"\r\n'
                ).encode("utf-8"),
                f'Content-Type: {file_info["content_type"]}\r\n\r\n'.encode("utf-8"),
                file_info["data"],
                b"\r\n",
            ]
        )

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks)


def guess_audio_filename(content_type: str) -> str:
    if "mp4" in content_type:
        return "voice-input.mp4"
    if "mpeg" in content_type or "mp3" in content_type:
        return "voice-input.mp3"
    if "wav" in content_type:
        return "voice-input.wav"
    if "ogg" in content_type:
        return "voice-input.ogg"
    return "voice-input.webm"


def build_user_intent(
    *,
    career_url: str,
    jobs_text: str,
    target_intent: str = "",
    resume_text: str,
) -> str:
    source = career_url or "pasted job descriptions"
    resume_hint = " ".join(resume_text.split())[:220]
    target_note = f" Target role preference: {target_intent}" if target_intent else ""
    if jobs_text:
        return (
            f"Find roles from {source} that fit my resume and career preferences. "
            f"Resume hint: {resume_hint}.{target_note}"
        )
    return (
        f"Screen the career page {source} for roles that fit my resume. "
        f"Resume hint: {resume_hint}.{target_note}"
    )


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
