# Bole Job Matcher

Bole is a hackathon MVP that matches a candidate's real skill profile
against actual job descriptions from a company career page.

The pain point: job search tools over-index on titles, but identical titles can
hide very different required skill sets. Bole ranks roles by fit,
surfaces evidence from job descriptions, and highlights missing skills.

`Bole` is the pinyin for `伯乐`: someone who recognizes exceptional talent.

## Run Locally

Install dependencies:

```bash
cd /Users/parisli/Documents/GitHub/tingyu-paris-li.github.io
python3 -m pip install -r requirements.txt
```

Configure server-side secrets in your shell:

```bash
export EVEROS_API_KEY="your_everos_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

Then start Bole:

```bash
cd /Users/parisli/Documents/GitHub/tingyu-paris-li.github.io/hackathon-job-match
python3 app.py
```

Open:

```text
http://127.0.0.1:5050/
```

## MVP Flow

1. Sign in with an email or register a demo username.
2. Paste resume text or upload a `.pdf`, `.txt`, or `.md` resume.
3. Provide a company career page URL, or paste job descriptions directly.
4. Review ranked recommendations with matched skills, missing skills, and
   evidence snippets.
5. Turn on Smarter recommendations for a richer candidate profile and semantic
   ranking.
6. When `EVEROS_API_KEY` is set on the server, Bole recalls the signed-in
   user's prior job-search preferences before analysis.
7. After analysis, Bole stores the resume profile, top jobs, warnings, and user
   intent back into EverOS.
8. Mark individual recommendations as Save, Not interested, or Applied so Bole
   can remember feedback for future searches.

## Notes

- The app uses only Python standard-library modules for hackathon portability.
- Smarter recommendations use the OpenAI Responses API when `OPENAI_API_KEY` is set
  on the server. API keys are never collected in the browser UI. You can
  override the default model with `OPENAI_MODEL`.
- Long-term memory recall uses EverOS when `EVEROS_API_KEY` is set on the
  server.
- Some career sites block server-side fetching or render jobs only in
  JavaScript. In that case, paste the job text into the fallback field.
- PDF resume parsing is best effort. Text-based PDFs work best; scanned image
  PDFs should be OCR'd or pasted as text.
- The matching engine is intentionally transparent and deterministic so judges
  can understand why each role was recommended.
