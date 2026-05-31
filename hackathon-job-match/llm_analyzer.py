from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.2"


class LLMUnavailable(RuntimeError):
    pass


def llm_is_configured() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def analyze_with_llm(
    resume_text: str,
    recommendations: list[dict[str, Any]],
    *,
    memory_context: str = "",
    model: str | None = None,
) -> dict[str, Any]:
    """Use an LLM to understand resume context and refine job recommendations."""

    resolved_api_key = os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise LLMUnavailable(
            "Set OPENAI_API_KEY to enable Deep LLM analysis. Baseline matching still works."
        )

    payload = {
        "model": model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL,
        "instructions": (
            "You are a career matching analyst. Read the resume and job recommendation "
            "candidates. Return only valid JSON. Be specific and evidence-based. "
            "Do not invent skills that are not supported by the resume."
        ),
        "input": build_prompt(
            resume_text,
            recommendations,
            memory_context=memory_context,
        ),
    }

    response = call_openai_responses_api(payload, resolved_api_key)
    text = extract_output_text(response)
    parsed = parse_json_object(text)
    return normalize_llm_result(parsed)


def build_prompt(
    resume_text: str,
    recommendations: list[dict[str, Any]],
    *,
    memory_context: str = "",
) -> str:
    compact_jobs = []
    for job in recommendations[:12]:
        compact_jobs.append(
            {
                "title": job.get("title", ""),
                "baseline_score": job.get("score", 0),
                "matched_skills": job.get("matched_skills", []),
                "missing_skills": job.get("missing_skills", []),
                "evidence": job.get("evidence", []),
            }
        )

    return json.dumps(
        {
            "task": (
                "Create a deeper candidate profile from the resume, then refine the "
                "job ranking beyond keywords. Consider project depth, domain fit, "
                "seniority, transferable skills, missing-but-learnable gaps, and "
                "the user's long-term job-search preferences when available."
            ),
            "long_term_memory": {
                "source": "EverOS",
                "usage_policy": (
                    "Use this as user preference history, not as proof of resume skills. "
                    "It can influence ranking, risk flags, search strategy, and fit reasons. "
                    "If it conflicts with the current resume or job description, mention the conflict."
                ),
                "context": memory_context[:6_000],
            },
            "resume_text": resume_text[:18_000],
            "candidate_jobs": compact_jobs,
            "required_json_shape": {
                "candidate_profile": {
                    "headline": "one sentence summary",
                    "core_strengths": ["strength"],
                    "domains": ["domain"],
                    "seniority_signal": "brief assessment",
                    "search_strategy": "what job families to prioritize",
                },
                "job_updates": [
                    {
                        "title": "same title as candidate_jobs",
                        "llm_score": 0,
                        "llm_reason": "specific reason",
                        "interview_pitch": "one sentence pitch for this role",
                        "risk_flags": ["gap or concern"],
                    }
                ],
            },
        },
        ensure_ascii=False,
    )


def call_openai_responses_api(payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    request = Request(
        OPENAI_RESPONSES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise LLMUnavailable(f"LLM request failed with HTTP {exc.code}: {detail[:240]}")
    except URLError as exc:
        raise LLMUnavailable(f"Could not reach the LLM API: {exc.reason}")
    except TimeoutError:
        raise LLMUnavailable("The LLM request timed out. Try again or use baseline matching.")


def extract_output_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]

    parts: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                parts.append(content["text"])
    return "\n".join(parts)


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise LLMUnavailable("The LLM returned text that was not valid JSON.")


def normalize_llm_result(result: dict[str, Any]) -> dict[str, Any]:
    profile = result.get("candidate_profile") or {}
    updates = result.get("job_updates") or []
    return {
        "candidate_profile": {
            "headline": str(profile.get("headline", "")).strip(),
            "core_strengths": list_of_strings(profile.get("core_strengths", []))[:8],
            "domains": list_of_strings(profile.get("domains", []))[:6],
            "seniority_signal": str(profile.get("seniority_signal", "")).strip(),
            "search_strategy": str(profile.get("search_strategy", "")).strip(),
        },
        "job_updates": [
            {
                "title": str(item.get("title", "")).strip(),
                "llm_score": clamp_int(item.get("llm_score", 0)),
                "llm_reason": str(item.get("llm_reason", "")).strip(),
                "interview_pitch": str(item.get("interview_pitch", "")).strip(),
                "risk_flags": list_of_strings(item.get("risk_flags", []))[:5],
            }
            for item in updates
            if isinstance(item, dict)
        ],
    }


def list_of_strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def clamp_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, number))
