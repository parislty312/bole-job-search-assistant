# Bole

Your personal job search assistant with long-term memory.

Bole reads a resume, scans a company career page or pasted job descriptions, and recommends roles based on the actual job description fit instead of job title matching. It combines skill extraction, job-description analysis, action-plan recommendations, and EverOS long-term memory so user feedback such as saved roles, ignored roles, and applications can improve future matches.

## Features

- Resume upload and preview, including PDF extraction
- Career page/job description matching
- Large match score and distance-to-JD explanations
- Skill gap cards with severity labels
- Resume evidence vs JD requirement comparison
- Action plans for stronger-fit jobs
- Optional OpenAI-powered deeper analysis from the server
- EverOS memory recall and feedback storage
- Per-job feedback: Save, Not interested, Applied

## Run Locally

```bash
python3 -m pip install -r requirements.txt

export EVEROS_API_KEY="your_everos_api_key"
export OPENAI_API_KEY="your_openai_api_key"

cd hackathon-job-match
python3 app.py
```

Then open:

```text
http://127.0.0.1:5050/
```

`OPENAI_API_KEY` is only required for smarter recommendations. `EVEROS_API_KEY` is required for long-term memory recall and feedback storage.

## Tests

```bash
cd hackathon-job-match
python3 -m unittest test_matcher.py

cd ..
python3 -m unittest tests/test_everos_memory.py
```

## Deployment

The hackathon frontend demo is deployed on Butterbase:

```text
https://bole-ai-career-memory.butterbase.dev
```

The current production demo deploys the static frontend. The full local app uses the Python backend for resume extraction, matching, LLM analysis, and EverOS memory writes.
