import unittest
import zlib

from matcher import analyze_fit, extract_jobs, extract_skills
from app import build_user_intent, normalize_user_id
from everos_context import build_analysis_memory, build_feedback_memory, build_memory_query
from llm_analyzer import build_prompt, extract_output_text, normalize_llm_result
from pdf_text import extract_pdf_text


class MatcherTests(unittest.TestCase):
    def test_extract_skills_maps_aliases(self):
        skills = extract_skills("Built RAG agents with Python, PyTorch, SQL, and AWS.")

        self.assertIn("Python", skills)
        self.assertIn("LLM", skills)
        self.assertIn("Deep Learning", skills)
        self.assertIn("Cloud", skills)

    def test_extract_jobs_from_html_links_and_blocks(self):
        html = """
        <html><body>
          <h2>Machine Learning Engineer</h2>
          <p>Build LLM and Python systems with cloud deployment.</p>
          <a href="/jobs/123">Data Scientist</a>
        </body></html>
        """

        jobs = extract_jobs(html, "https://example.com/careers")

        self.assertTrue(any(job.title == "Machine Learning Engineer" for job in jobs))
        self.assertTrue(any(job.title == "Data Scientist" for job in jobs))

    def test_analyze_fit_ranks_strong_skill_overlap(self):
        resume = "Python, SQL, machine learning, LLM agents, dashboards"
        jobs = """
        Machine Learning Engineer
        You will build Python services, RAG workflows, SQL datasets, and LLM systems.

        Product Designer
        You will create design systems and visual prototypes.
        """

        result = analyze_fit(resume, jobs)

        self.assertEqual(result["recommendations"][0]["title"], "Machine Learning Engineer")
        self.assertGreaterEqual(result["recommendations"][0]["score"], 70)
        self.assertIsNotNone(result["recommendations"][0]["recommendation"])

    def test_analyze_fit_keeps_source_url_for_open_fallback(self):
        result = analyze_fit(
            "Python SQL",
            "Data Engineer\nBuild SQL data pipelines with Python.",
            source_url="https://example.com/careers",
        )

        self.assertEqual(result["source_url"], "https://example.com/careers")
        self.assertEqual(result["recommendations"][0]["url"], "https://example.com/careers")

    def test_extract_jobs_matches_greenhouse_title_to_apply_link(self):
        html = """
        <html><body>
          <h2>Group Product Manager, Agent Security</h2>
          <p>Lead product strategy for agent security, AI safety, and cross-functional execution.</p>
          <a href="https://job-boards.greenhouse.io/deepmind/jobs/7651614">
            Group Product Manager, Agent Security Mountain View, California, US;
            New York City, New York, US; San Francisco, California, US
          </a>
        </body></html>
        """

        jobs = extract_jobs(html, "https://job-boards.greenhouse.io/deepmind")
        product_job = next(job for job in jobs if job.title == "Group Product Manager, Agent Security")

        self.assertEqual(
            product_job.url,
            "https://job-boards.greenhouse.io/deepmind/jobs/7651614",
        )

    def test_low_score_jobs_do_not_get_action_plan(self):
        result = analyze_fit(
            "Python",
            "Product Designer\nCreate visual design systems and user research prototypes.",
        )

        self.assertIsNone(result["recommendations"][0]["recommendation"])

    def test_extract_pdf_text_from_compressed_stream(self):
        stream = zlib.compress(
            b"BT (Experience with Python SQL Machine Learning projects and skills) Tj ET"
        )
        pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj << /Length 99 /Filter /FlateDecode >> stream\n"
            + stream
            + b"\nendstream endobj\n%%EOF"
        )

        text = extract_pdf_text(pdf)

        self.assertIn("Python SQL Machine Learning", text)

    def test_extract_output_text_from_responses_shape(self):
        response = {
            "output": [
                {
                    "content": [
                        {"type": "output_text", "text": "{\"candidate_profile\": {}}"}
                    ]
                }
            ]
        }

        self.assertEqual(extract_output_text(response), "{\"candidate_profile\": {}}")

    def test_normalize_llm_result_clamps_scores(self):
        result = normalize_llm_result(
            {
                "candidate_profile": {"headline": "ML systems builder"},
                "job_updates": [{"title": "ML Engineer", "llm_score": 130}],
            }
        )

        self.assertEqual(result["candidate_profile"]["headline"], "ML systems builder")
        self.assertEqual(result["job_updates"][0]["llm_score"], 100)

    def test_normalize_user_id(self):
        self.assertEqual(normalize_user_id(" Paris+Demo@Example.COM "), "parisdemo@example.com")
        self.assertEqual(normalize_user_id("demo_user-001"), "demo_user-001")

    def test_build_memory_query_includes_career_context(self):
        query = build_memory_query(
            resume_text="Python LLM product engineer",
            career_url="https://example.com/careers",
        )

        self.assertIn("job search preferences", query)
        self.assertIn("https://example.com/careers", query)
        self.assertIn("Python LLM product engineer", query)

    def test_llm_prompt_includes_long_term_memory(self):
        prompt = build_prompt(
            "Python engineer",
            [{"title": "AI Engineer", "score": 80}],
            memory_context="- User prefers remote AI product roles.",
        )

        self.assertIn("long_term_memory", prompt)
        self.assertIn("User prefers remote AI product roles", prompt)
        self.assertIn("not as proof of resume skills", prompt)

    def test_build_analysis_memory_includes_top_jobs_and_intent(self):
        memory = build_analysis_memory(
            resume_profile={"headline": "LLM product engineer"},
            top_jobs=[{"title": "AI Engineer", "score": 92, "matched_skills": ["LLM"]}],
            warnings=["Set OPENAI_API_KEY"],
            user_intent="Find AI product roles",
            career_url="https://example.com/careers",
        )

        self.assertIn("bole_job_search_analysis", memory)
        self.assertIn("Find AI product roles", memory)
        self.assertIn("AI Engineer", memory)

    def test_build_user_intent_summarizes_source(self):
        intent = build_user_intent(
            career_url="https://example.com/careers",
            jobs_text="",
            resume_text="Python LLM engineer with data background",
        )

        self.assertIn("Screen the career page https://example.com/careers", intent)
        self.assertIn("Python LLM engineer", intent)

    def test_build_feedback_memory_includes_job_and_feedback(self):
        memory = build_feedback_memory(
            feedback="not_interested",
            job={"title": "Data Analyst", "score": 84, "missing_skills": ["Product"]},
        )

        self.assertIn("bole_job_feedback", memory)
        self.assertIn("not_interested", memory)
        self.assertIn("Data Analyst", memory)


if __name__ == "__main__":
    unittest.main()
