from __future__ import annotations

import html
import hashlib
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Iterable


SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "Python": ("python", "pandas", "numpy", "scikit-learn", "sklearn", "pytest"),
    "JavaScript": ("javascript", "typescript", "node.js", "nodejs", "react", "next.js"),
    "SQL": ("sql", "postgres", "postgresql", "mysql", "snowflake", "bigquery"),
    "Machine Learning": ("machine learning", "ml", "classification", "regression"),
    "Deep Learning": ("deep learning", "pytorch", "tensorflow", "keras"),
    "LLM": ("llm", "large language model", "rag", "prompt engineering", "agents"),
    "NLP": ("nlp", "natural language processing", "text mining"),
    "Data Engineering": ("etl", "airflow", "spark", "databricks", "data pipeline"),
    "Cloud": ("aws", "gcp", "azure", "cloud"),
    "Docker": ("docker", "container", "kubernetes", "k8s"),
    "Backend": ("api", "rest", "graphql", "fastapi", "flask", "django"),
    "Frontend": ("frontend", "ui", "ux", "react", "html", "css"),
    "Product": ("product", "roadmap", "user research", "stakeholder"),
    "Analytics": ("analytics", "dashboard", "metrics", "experimentation", "a/b"),
    "Statistics": ("statistics", "probability", "causal", "hypothesis"),
    "Optimization": ("optimization", "linear programming", "operations research"),
    "Robotics": ("robotics", "ros", "control", "planning"),
    "Reinforcement Learning": ("reinforcement learning", "rl", "policy gradient"),
    "Security": ("security", "privacy", "compliance", "soc2"),
    "Leadership": ("leadership", "mentoring", "cross-functional", "collaboration"),
}

JOB_TITLE_HINTS = (
    "engineer",
    "scientist",
    "analyst",
    "manager",
    "designer",
    "researcher",
    "architect",
    "developer",
    "intern",
    "specialist",
    "consultant",
    "director",
)

ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "Technical Program Manager": (
        "technical program manager",
        "tpm",
    ),
    "Product Manager": (
        "product manager",
        "group product manager",
    ),
    "Engineering Manager": (
        "engineering manager",
        "software engineering manager",
    ),
    "Software Engineer": (
        "software engineer",
        "software developer",
    ),
    "Research Engineer": (
        "research engineer",
        "research scientist",
    ),
    "Data Scientist": (
        "data scientist",
        "machine learning scientist",
    ),
}

ROLE_MISMATCH_TITLE_HINTS = (
    "accountant",
    "analyst",
    "designer",
    "engineer",
    "manager",
    "researcher",
    "scientist",
    "specialist",
)

INTENT_STOPWORDS = {
    "about",
    "and",
    "are",
    "for",
    "from",
    "job",
    "jobs",
    "like",
    "looking",
    "role",
    "roles",
    "search",
    "that",
    "the",
    "this",
    "want",
    "with",
    "work",
}


@dataclass(frozen=True)
class JobPosting:
    title: str
    description: str
    url: str = ""


class TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._skip_depth = 0
        self._current_href = ""
        self._current_link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "svg", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in {"h1", "h2", "h3", "h4", "li", "p", "section", "article", "div", "br"}:
            self.parts.append("\n")
        if tag == "a":
            self._current_href = ""
            self._current_link_text = []
            for key, value in attrs:
                if key == "href" and value:
                    self._current_href = value

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "svg", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == "a" and self._current_href:
            text = normalize_space(" ".join(self._current_link_text))
            if text:
                self.links.append((text, self._current_href))
            self._current_href = ""
            self._current_link_text = []
        if tag in {"h1", "h2", "h3", "h4", "li", "p", "section", "article", "div"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = html.unescape(data)
        self.parts.append(text)
        if self._current_href:
            self._current_link_text.append(text)

    def text(self) -> str:
        return normalize_lines("\n".join(self.parts))


def analyze_fit(
    resume_text: str,
    page_text: str,
    source_url: str = "",
    target_intent: str = "",
) -> dict[str, Any]:
    resume_skills = extract_skills(resume_text)
    resume_roles = extract_role_terms(resume_text)
    target_skills = extract_skills(target_intent)
    target_terms = extract_intent_terms(target_intent)
    jobs = extract_jobs(page_text, source_url=source_url)
    recommendations = [
        score_job(
            job,
            resume_skills,
            resume_roles=resume_roles,
            target_skills=target_skills,
            target_terms=target_terms,
        )
        for job in jobs
    ]
    recommendations.sort(
        key=lambda item: (
            item["score"],
            len(item["role_matches"]),
            len(item["intent_matches"]),
            len(item["matched_skills"]),
        ),
        reverse=True,
    )

    warnings: list[str] = []
    if not jobs:
        warnings.append("No job postings were detected. Paste full job descriptions for better results.")
    elif len(jobs) == 1:
        warnings.append("Only one job-like block was detected. A richer career page improves ranking.")

    return {
        "resume_skills": resume_skills,
        "source_url": source_url,
        "job_count": len(jobs),
        "recommendations": select_recommendations(recommendations),
        "warnings": warnings,
    }


def select_recommendations(
    recommendations: list[dict[str, Any]],
    *,
    limit: int = 60,
) -> list[dict[str, Any]]:
    if len(recommendations) <= limit:
        return recommendations

    selected = recommendations[:limit]
    if any(int(item.get("score", 0)) < 100 for item in selected):
        return selected

    diverse: list[dict[str, Any]] = []
    selected_keys = {
        (str(item.get("title", "")), str(item.get("url", "")))
        for item in selected
    }

    score_bands = [
        lambda score: 70 <= score < 100,
        lambda score: 50 <= score < 70,
        lambda score: score < 50,
    ]
    for matches_band in score_bands:
        for item in recommendations:
            key = (str(item.get("title", "")), str(item.get("url", "")))
            if key in selected_keys:
                continue
            if matches_band(int(item.get("score", 0))):
                diverse.append(item)
                selected_keys.add(key)
                break

    if not diverse:
        return selected

    return [*selected[: limit - len(diverse)], *diverse]


def extract_skills(text: str) -> list[str]:
    normalized = normalize_for_match(text)
    found = []
    for skill, aliases in SKILL_ALIASES.items():
        if any(contains_phrase(normalized, alias) for alias in aliases):
            found.append(skill)
    return found


def extract_role_terms(text: str) -> list[str]:
    normalized = normalize_for_match(text[:1200])
    roles: list[str] = []
    for role, aliases in ROLE_ALIASES.items():
        if any(contains_phrase(normalized, alias) for alias in aliases):
            roles.append(role)
    return roles


def extract_jobs(page_text: str, source_url: str = "") -> list[JobPosting]:
    text, links = extract_readable_text_and_links(page_text)
    linked_jobs = jobs_from_links(links, source_url)
    block_jobs = jobs_from_text_blocks(text, source_url, links)

    jobs: list[JobPosting] = []
    seen: dict[str, int] = {}
    for job in [*linked_jobs, *block_jobs]:
        key = normalize_for_match(f"{job.title} {job.url}")
        if key in seen:
            existing_index = seen[key]
            if len(job.description) > len(jobs[existing_index].description):
                jobs[existing_index] = job
            continue
        seen[key] = len(jobs)
        jobs.append(job)
    return jobs


def extract_readable_text_and_links(page_text: str) -> tuple[str, list[tuple[str, str]]]:
    if "<html" in page_text.lower() or "</" in page_text:
        parser = TextExtractor()
        parser.feed(page_text)
        return parser.text(), parser.links
    return normalize_lines(page_text), []


def jobs_from_links(links: list[tuple[str, str]], source_url: str) -> list[JobPosting]:
    jobs = []
    for text, href in links:
        title = clean_title(text)
        if is_probable_job_title(title):
            jobs.append(JobPosting(title=title, description=title, url=absolute_url(source_url, href)))
    return jobs


def jobs_from_text_blocks(
    text: str,
    source_url: str,
    links: list[tuple[str, str]] | None = None,
) -> list[JobPosting]:
    links = links or []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    starts = [index for index, line in enumerate(lines) if is_probable_job_title(line)]
    jobs = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else min(len(lines), start + 28)
        block = lines[start:end]
        description = normalize_space(" ".join(block))
        if len(description) < 40 and position + 1 < len(starts):
            continue
        title = clean_title(lines[start])
        jobs.append(
            JobPosting(
                title=title,
                description=description,
                url=find_matching_job_url(title, links, source_url) or source_url,
            )
        )
    if jobs:
        return jobs

    chunks = split_long_text(text)
    return [
        JobPosting(title=infer_title(chunk, fallback=f"Job Match {index + 1}"), description=chunk, url=source_url)
        for index, chunk in enumerate(chunks)
    ]


def score_job(
    job: JobPosting,
    resume_skills: list[str],
    *,
    resume_roles: list[str] | None = None,
    target_skills: list[str] | None = None,
    target_terms: set[str] | None = None,
) -> dict[str, Any]:
    job_skills = extract_skills(job.description)
    matched = sorted(set(resume_skills).intersection(job_skills))
    missing = sorted(set(job_skills).difference(resume_skills))
    resume_set = set(resume_skills)
    job_set = set(job_skills)
    intent_matches, intent_bonus = calculate_intent_boost(
        job,
        job_skills=job_skills,
        target_skills=target_skills or [],
        target_terms=target_terms or set(),
    )
    role_matches, role_adjustment = calculate_role_alignment(job, resume_roles or [])

    if not job_set:
        overlap_score = 20 if any(token in normalize_for_match(job.description) for token in resume_set) else 8
    else:
        overlap_score = round(100 * len(resume_set.intersection(job_set)) / max(len(job_set), 1))

    if role_matches:
        overlap_score = max(overlap_score, 60)

    seniority_penalty = seniority_mismatch_penalty(job.description, resume_skills)
    score = max(0, min(100, overlap_score - seniority_penalty + intent_bonus + role_adjustment))

    return {
        "job_id": build_job_id(job),
        "title": job.title,
        "url": job.url,
        "score": score,
        "matched_skills": matched,
        "missing_skills": missing[:6],
        "job_skills": job_skills,
        "intent_matches": intent_matches,
        "role_matches": role_matches,
        "why": build_reason(score, matched, missing),
        "recommendation": build_action_recommendation(
            score=score,
            matched=matched,
            missing=missing,
            title=job.title,
        ),
        "evidence": evidence_snippets(job.description, matched or job_skills),
    }


def build_job_id(job: JobPosting) -> str:
    """Create a stable identifier so agents never join duplicate titles by name."""

    identity = "|".join(
        (
            normalize_for_match(job.url),
            normalize_for_match(job.title),
            normalize_for_match(job.description)[:240],
        )
    )
    return f"job_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"


def extract_intent_terms(text: str) -> set[str]:
    normalized = normalize_for_match(text)
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) >= 2 and token not in INTENT_STOPWORDS
    }


def calculate_intent_boost(
    job: JobPosting,
    *,
    job_skills: list[str],
    target_skills: list[str],
    target_terms: set[str],
) -> tuple[list[str], int]:
    if not target_skills and not target_terms:
        return [], 0

    title = normalize_for_match(job.title)
    description = normalize_for_match(job.description)
    skill_hits = sorted(set(target_skills).intersection(job_skills))
    title_hits = sorted(term for term in target_terms if contains_phrase(title, term))
    description_hits = sorted(
        term
        for term in target_terms
        if term not in title_hits and contains_phrase(description, term)
    )

    bonus = min(12, len(skill_hits) * 4)
    bonus += min(24, len(title_hits) * 7)
    bonus += min(6, len(description_hits) * 1)

    matches = [*skill_hits, *title_hits[:4], *description_hits[:3]]
    return matches[:8], bonus


def calculate_role_alignment(job: JobPosting, resume_roles: list[str]) -> tuple[list[str], int]:
    if not resume_roles:
        return [], 0

    title = normalize_for_match(job.title)
    description = normalize_for_match(job.description)
    matches: list[str] = []
    best_bonus = 0

    for role in resume_roles:
        aliases = ROLE_ALIASES.get(role, (role.lower(),))
        title_hits = [alias for alias in aliases if contains_phrase(title, alias)]
        description_hits = [
            alias
            for alias in aliases
            if alias not in title_hits and contains_phrase(description, alias)
        ]
        if title_hits:
            matches.append(role)
            best_bonus = max(best_bonus, 32 if "technical program manager" in title_hits else 24)
        elif description_hits:
            matches.append(role)
            best_bonus = max(best_bonus, 10)

    if matches:
        return matches[:3], best_bonus

    if any(hint in title for hint in ROLE_MISMATCH_TITLE_HINTS):
        return [], -24
    return [], -12


def build_reason(score: int, matched: list[str], missing: list[str]) -> str:
    if score >= 75:
        if not matched:
            return "Strong fit: the role title aligns closely with your resume headline."
        return f"Strong fit: your profile directly matches {', '.join(matched[:4])}."
    if score >= 45:
        return f"Possible fit: clear overlap in {', '.join(matched[:3])}, with a few gaps."
    if matched:
        return f"Stretch role: some overlap in {', '.join(matched[:2])}, but key requirements differ."
    if missing:
        return "Low fit: the job description emphasizes skills not found in the resume."
    return "Needs review: the job description did not expose enough skill signals."


def build_action_recommendation(
    *,
    score: int,
    matched: list[str],
    missing: list[str],
    title: str,
) -> dict[str, Any] | None:
    if score <= 60:
        return None

    matched_text = ", ".join(matched[:4]) if matched else "your most relevant experience"
    missing_text = ", ".join(missing[:4]) if missing else "role-specific outcomes"
    actions = []

    if matched:
        actions.append(
            f"Resume update: add a targeted bullet for {title} that quantifies your work with {matched_text}."
        )
        actions.append(
            "Move the most relevant project into the top half of your resume and mirror the job description's language."
        )
    else:
        actions.append(
            f"Resume update: add a short summary line that explains why your background transfers to {title}."
        )

    if missing:
        actions.extend(skill_gap_actions(missing[:3]))
    else:
        actions.append(
            f"Portfolio proof: publish a short technical blog or project note showing measurable impact in {missing_text}."
        )

    actions.append(
        "Application strategy: write a one-paragraph cover note connecting your strongest evidence to this team's mission."
    )

    return {
        "headline": "Recommended next steps",
        "resume_update": actions[0],
        "improvement_plan": actions[1:5],
    }


def skill_gap_actions(missing: list[str]) -> list[str]:
    actions = []
    for skill in missing:
        if skill in {"LLM", "Machine Learning", "Deep Learning", "NLP"}:
            actions.append(
                f"Build proof for {skill}: join a hackathon or ship a small public project, then document the architecture and results."
            )
        elif skill in {"Product", "Analytics", "Statistics"}:
            actions.append(
                f"Close the {skill} gap: take a focused course and write a case-study style blog post using a real product metric."
            )
        elif skill in {"Cloud", "Docker", "Backend", "Data Engineering"}:
            actions.append(
                f"Demonstrate {skill}: deploy a personal project, include a README with system design and tradeoffs, and link it on your resume."
            )
        else:
            actions.append(
                f"Strengthen {skill}: complete a small project or course and add one concrete outcome to your resume."
            )
    return actions


def evidence_snippets(description: str, skills: Iterable[str]) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", description)
    selected = []
    normalized_skills = [(skill, SKILL_ALIASES.get(skill, (skill.lower(),))) for skill in skills]
    for sentence in sentences:
        normalized_sentence = normalize_for_match(sentence)
        if any(contains_phrase(normalized_sentence, alias) for _, aliases in normalized_skills for alias in aliases):
            selected.append(trim_text(normalize_space(sentence), 180))
        if len(selected) == 3:
            break
    return selected or [trim_text(description, 180)]


def split_long_text(text: str) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n{2,}", text) if len(paragraph.strip()) > 120]
    if paragraphs:
        return paragraphs[:12]
    if len(text) > 160:
        return [text[:2500]]
    return []


def infer_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if is_probable_job_title(line):
            return clean_title(line)
    first = normalize_space(text).split(".")[0]
    return clean_title(first[:80]) or fallback


def find_matching_job_url(
    title: str,
    links: list[tuple[str, str]],
    source_url: str,
) -> str:
    title_key = normalize_title_for_link_match(title)
    if not title_key:
        return ""

    best_href = ""
    best_score = 0
    title_tokens = set(title_key.split())
    for link_text, href in links:
        link_key = normalize_title_for_link_match(link_text)
        if not link_key:
            continue

        score = 0
        if link_key == title_key:
            score = 100
        elif link_key.startswith(f"{title_key} "):
            score = 95
        elif title_key in link_key:
            score = 80
        else:
            link_tokens = set(link_key.split())
            overlap = len(title_tokens.intersection(link_tokens))
            if overlap >= max(3, len(title_tokens) - 1):
                score = 60 + overlap

        if score > best_score:
            best_score = score
            best_href = href

    if best_score < 60:
        return ""
    return absolute_url(source_url, best_href)


def is_probable_job_title(text: str) -> bool:
    raw_title = normalize_space(text)
    if len(raw_title) > 140:
        return False
    title = clean_title(text)
    lower = title.lower()
    if len(title) < 4 or len(title) > 95:
        return False
    if not any(hint in lower for hint in JOB_TITLE_HINTS):
        return False
    blocked = ("cookie", "privacy", "benefit", "equal opportunity", "sign in", "alert")
    return not any(item in lower for item in blocked)


def clean_title(text: str) -> str:
    text = normalize_space(re.sub(r"Apply\s+Now|View\s+Job|Learn\s+More", "", text, flags=re.I))
    text = re.sub(r"\s+\bApply\b$", "", text, flags=re.I)
    text = re.sub(
        r"\s+(Remote-Friendly|San Francisco|New York City|Seattle|London|Ontario|Boston|Washington, DC)\b.*$",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"^[^\w+]+|[^\w)]+$", "", text)
    return text[:95].strip()


def normalize_title_for_link_match(text: str) -> str:
    text = clean_title(text).lower()
    text = re.sub(r"[,;]", " ", text)
    text = re.sub(r"\b(remote|hybrid|onsite)\b", " ", text)
    text = re.sub(r"\b(united states|us|uk|ca|california|new york|seattle|london)\b", " ", text)
    text = re.sub(r"\b(mountain view|san francisco|new york city)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def seniority_mismatch_penalty(description: str, resume_skills: list[str]) -> int:
    normalized = normalize_for_match(description)
    if "principal" in normalized or "staff" in normalized:
        return 8 if "Leadership" not in resume_skills else 0
    if "director" in normalized or "head of" in normalized:
        return 18 if "Leadership" not in resume_skills else 5
    return 0


def absolute_url(base: str, href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if not base or href.startswith("#") or href.startswith("mailto:"):
        return href
    parsed = re.match(r"^(https?://[^/]+)", base)
    if not parsed:
        return href
    root = parsed.group(1)
    if href.startswith("/"):
        return f"{root}{href}"
    base_path = base.rsplit("/", 1)[0]
    return f"{base_path}/{href}"


def contains_phrase(text: str, phrase: str) -> bool:
    normalized = normalize_for_match(phrase)
    if re.search(r"[+#.]", normalized):
        return normalized in text
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", text))


def normalize_for_match(text: str) -> str:
    return normalize_space(text).lower()


def normalize_lines(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def trim_text(text: str, limit: int) -> str:
    text = normalize_space(text)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rsplit(" ", 1)[0] + "..."
