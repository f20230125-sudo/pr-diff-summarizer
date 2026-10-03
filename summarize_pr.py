"""
summarize_pr.py
----------------
Reads a git diff (from stdin, or a file) and asks Gemini to generate a
structured pull-request summary: what changed, why it matters, risk
areas, and what should be tested before merging.

This mirrors the "developer infrastructure" pattern of AI-assisted
spec-to-PR / code-review tooling: instead of a developer manually writing
a PR description, the diff itself is turned directly into one.

Usage:
    git diff | python summarize_pr.py
    python summarize_pr.py sample_diff.txt        # run on the bundled example
    git diff main...feature-branch | python summarize_pr.py
"""

import os
import sys
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel
from typing import List

load_dotenv()

MODEL = "gemini-3.6-flash"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5


class PRSummary(BaseModel):
    title: str
    summary: str
    risk_level: str  # "low", "medium", or "high"
    key_changes: List[str]
    testing_notes: str


def summarize_diff(client: genai.Client, diff_text: str) -> PRSummary:
    prompt = f"""You are a senior engineer reviewing a code diff before it
becomes a pull request. Read the diff below and produce a PR summary.

Diff:
{diff_text}

Provide:
- title: a concise PR title (imperative mood, e.g. "Add input validation to signup form")
- summary: 2-3 sentences explaining what changed and why it likely matters
- risk_level: "low", "medium", or "high" based on the blast radius of this change
- key_changes: a short bullet list of the concrete changes made
- testing_notes: what a reviewer should specifically test or check before approving
"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PRSummary,
                ),
            )
            if response.parsed is None:
                # The reply was blocked or cut off, so there was no complete
                # JSON to validate against the schema.
                raise SystemExit("Gemini returned no usable summary. Try again, or pass a smaller diff.")
            return response.parsed
        except genai_errors.ServerError as e:
            # Gemini's servers occasionally return a temporary 503 under
            # high demand -- this is not a bug in our request, so back off
            # and retry a few times before giving up.
            if attempt == MAX_RETRIES:
                raise
            # stderr, so the notice stays out of a summary piped to a file.
            print(f"  (model temporarily unavailable, retrying in {RETRY_DELAY_SECONDS}s... "
                  f"attempt {attempt}/{MAX_RETRIES})", file=sys.stderr)
            time.sleep(RETRY_DELAY_SECONDS)


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    client = genai.Client(api_key=api_key)

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            diff_text = f.read()
    else:
        diff_text = sys.stdin.read()

    if not diff_text.strip():
        raise SystemExit("No diff provided. Pipe a `git diff` in, or pass a file path.")

    result = summarize_diff(client, diff_text)

    print("=" * 70)
    print(f"Title:       {result.title}")
    print(f"Risk level:  {result.risk_level}")
    print(f"Summary:     {result.summary}")
    print("Key changes:")
    for change in result.key_changes:
        print(f"  - {change}")
    print(f"Testing:     {result.testing_notes}")
    print("=" * 70)


if __name__ == "__main__":
    main()
