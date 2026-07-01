from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class MemoryLookup:
    enabled: bool
    text: str
    status: str
    warning: str = ""


@dataclass(frozen=True)
class MemoryWrite:
    enabled: bool
    stored: bool
    status: str
    warning: str = ""


def retrieve_user_preferences(
    user_id: str,
    *,
    resume_text: str,
    career_url: str = "",
    target_intent: str = "",
) -> MemoryLookup:
    """Retrieve long-term job-search context from EverOS before analysis."""

    if not os.getenv("EVEROS_API_KEY"):
        return MemoryLookup(
            enabled=False,
            text="",
            status="EverOS is not configured. Set EVEROS_API_KEY on the server to enable memory recall.",
        )

    try:
        from integrations.everos_memory import EverOSMemory
    except ImportError as exc:
        return MemoryLookup(
            enabled=False,
            text="",
            status="EverOS helper could not be imported.",
            warning=str(exc),
        )

    query = build_memory_query(
        resume_text=resume_text,
        career_url=career_url,
        target_intent=target_intent,
    )

    try:
        memory = EverOSMemory()
        context = memory.retrieve_context(
            user_id=user_id,
            query=query,
            method="hybrid",
            top_k=5,
            memory_types=("episodic_memory", "profile"),
        )
    except Exception as exc:
        return MemoryLookup(
            enabled=True,
            text="",
            status="EverOS memory recall failed; continuing with current resume and jobs.",
            warning=str(exc),
        )

    if not context.text:
        return MemoryLookup(
            enabled=True,
            text="",
            status="EverOS searched this user, but no prior preferences were found yet.",
        )

    return MemoryLookup(
        enabled=True,
        text=context.text,
        status="EverOS recalled prior job-search context for this user.",
    )


def remember_analysis_result(
    user_id: str,
    *,
    resume_profile: dict[str, Any],
    top_jobs: list[dict[str, Any]],
    warnings: list[str],
    user_intent: str,
    career_url: str = "",
) -> MemoryWrite:
    """Write the completed job-search analysis back to EverOS."""

    if not os.getenv("EVEROS_API_KEY"):
        return MemoryWrite(
            enabled=False,
            stored=False,
            status="EverOS is not configured, so this analysis was not saved to long-term memory.",
        )

    try:
        from integrations.everos_memory import EverOSMemory
    except ImportError as exc:
        return MemoryWrite(
            enabled=False,
            stored=False,
            status="EverOS helper could not be imported, so this analysis was not saved.",
            warning=str(exc),
        )

    summary = build_analysis_memory(
        resume_profile=resume_profile,
        top_jobs=top_jobs,
        warnings=warnings,
        user_intent=user_intent,
        career_url=career_url,
    )
    session_id = f"bole-analysis-{int(time.time())}"

    try:
        memory = EverOSMemory()
        memory.remember_agent_messages(
            user_id=user_id,
            session_id=session_id,
            async_mode=True,
            messages=[
                {
                    "role": "user",
                    "timestamp": now_ms(),
                    "content": user_intent,
                },
                {
                    "role": "assistant",
                    "timestamp": now_ms() + 1,
                    "content": summary,
                },
            ],
        )
        memory.flush_agent_memory(user_id=user_id, session_id=session_id)
    except Exception as exc:
        return MemoryWrite(
            enabled=True,
            stored=False,
            status="EverOS write failed; the current analysis still completed.",
            warning=str(exc),
        )

    return MemoryWrite(
        enabled=True,
        stored=True,
        status="Bole saved this resume analysis and top job recommendations to EverOS.",
    )


def remember_job_feedback(
    user_id: str,
    *,
    feedback: str,
    job: dict[str, Any],
) -> MemoryWrite:
    """Store explicit user feedback on one recommended job."""

    if feedback not in {"save", "not_interested", "applied"}:
        return MemoryWrite(
            enabled=False,
            stored=False,
            status="Invalid feedback type.",
        )

    if not os.getenv("EVEROS_API_KEY"):
        return MemoryWrite(
            enabled=False,
            stored=False,
            status="EverOS is not configured, so this job feedback was not saved.",
        )

    try:
        from integrations.everos_memory import EverOSMemory
    except ImportError as exc:
        return MemoryWrite(
            enabled=False,
            stored=False,
            status="EverOS helper could not be imported, so this feedback was not saved.",
            warning=str(exc),
        )

    summary = build_feedback_memory(feedback=feedback, job=job)
    session_id = f"bole-feedback-{int(time.time())}"

    try:
        memory = EverOSMemory()
        memory.remember_agent_messages(
            user_id=user_id,
            session_id=session_id,
            async_mode=True,
            messages=[
                {
                    "role": "user",
                    "timestamp": now_ms(),
                    "content": f"Job feedback: {feedback}",
                },
                {
                    "role": "assistant",
                    "timestamp": now_ms() + 1,
                    "content": summary,
                },
            ],
        )
        memory.flush_agent_memory(user_id=user_id, session_id=session_id)
    except Exception as exc:
        return MemoryWrite(
            enabled=True,
            stored=False,
            status="EverOS feedback write failed.",
            warning=str(exc),
        )

    return MemoryWrite(
        enabled=True,
        stored=True,
        status=f"Bole saved '{feedback_label(feedback)}' feedback to long-term memory.",
    )


def build_memory_query(
    *,
    resume_text: str,
    career_url: str = "",
    target_intent: str = "",
) -> str:
    resume_hint = " ".join(resume_text.split())[:900]
    target_hint = " ".join(target_intent.split())[:500]
    parts = [
        "Recall this user's long-term job search preferences, target companies,",
        "preferred roles, rejected role patterns, location or remote preferences,",
        "visa constraints, seniority preferences, and recurring skill gaps.",
    ]
    if career_url:
        parts.append(f"Current company career page: {career_url}.")
    if target_hint:
        parts.append(f"Current voice or typed target role preference: {target_hint}.")
    if resume_hint:
        parts.append(f"Current resume summary text: {resume_hint}")
    return " ".join(parts)


def build_analysis_memory(
    *,
    resume_profile: dict[str, Any],
    top_jobs: list[dict[str, Any]],
    warnings: list[str],
    user_intent: str,
    career_url: str = "",
) -> str:
    profile = resume_profile or {}
    compact_jobs = []
    for job in top_jobs[:5]:
        compact_jobs.append(
            {
                "title": job.get("title", ""),
                "score": job.get("llm_score") or job.get("score", 0),
                "matched_skills": job.get("matched_skills", []),
                "missing_skills": job.get("missing_skills", []),
                "reason": job.get("llm_reason") or job.get("why", ""),
                "risk_flags": job.get("risk_flags", []),
            }
        )

    memory_payload = {
        "memory_type": "bole_job_search_analysis",
        "user_intent": user_intent,
        "career_url": career_url,
        "resume_profile": {
            "headline": profile.get("headline", ""),
            "core_strengths": profile.get("core_strengths", []),
            "domains": profile.get("domains", []),
            "seniority_signal": profile.get("seniority_signal", ""),
            "search_strategy": profile.get("search_strategy", ""),
        },
        "top_jobs": compact_jobs,
        "warnings": warnings[:8],
    }
    return (
        "Bole completed a job-search analysis. Store this as long-term user "
        f"preference and job-search history:\n{memory_payload}"
    )


def build_feedback_memory(*, feedback: str, job: dict[str, Any]) -> str:
    memory_payload = {
        "memory_type": "bole_job_feedback",
        "feedback": feedback,
        "feedback_label": feedback_label(feedback),
        "job": {
            "title": job.get("title", ""),
            "url": job.get("url", ""),
            "score": job.get("llm_score") or job.get("score", 0),
            "matched_skills": job.get("matched_skills", []),
            "missing_skills": job.get("missing_skills", []),
            "reason": job.get("llm_reason") or job.get("why", ""),
            "risk_flags": job.get("risk_flags", []),
        },
    }
    return (
        "The user gave explicit feedback on a recommended job. Store this as "
        f"long-term job-search preference signal:\n{memory_payload}"
    )


def feedback_label(feedback: str) -> str:
    return {
        "save": "Save",
        "not_interested": "Not interested",
        "applied": "Applied",
    }.get(feedback, feedback)


def now_ms() -> int:
    return int(time.time() * 1000)
