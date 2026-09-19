# PR Diff Summarizer

A CLI tool that reads a `git diff` and uses Gemini to generate a
structured pull-request summary: title, risk level, key changes, and
what a reviewer should specifically test — instead of a developer
writing the PR description by hand.

## How it works

```
git diff -> Gemini (structured output mode + Pydantic schema)
         -> validated PRSummary object
              - title, summary, risk_level (low/medium/high)
              - key_changes (bullet list)
              - testing_notes (what a reviewer should check)
```

Like the ticket triage agent, this uses Gemini's native structured
output mode (`response_mime_type="application/json"` with a Pydantic
`response_schema`), so the result is validated, typed data — not text
that has to be parsed and hoped to be well-formed.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then paste in your Gemini API key
```

## Usage

On a real diff from any git repository:
```bash
git diff | python summarize_pr.py
```

Or try it on the bundled example diff (a login-hardening change):
```bash
python summarize_pr.py sample_diff.txt
```

## Stack

Python, Google Gemini API (`google-genai`), Pydantic (schema-validated
structured output).
