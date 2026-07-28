"""Creative review of a rendered reel: viewer simulation first, then expert
critique as timestamped comments. Critical comments are machine-readable and
feed the revise loop. Borrowed shape: Palo creative-review; trimmed to what a
20s auto-cut reel needs.

Usage: python scripts/review_reel.py reel.mp4 [--json out.json]
"""
import json, os, sys, time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
load_dotenv()

PROMPT = """You review a short-form reel draft frame by frame, the way a viewer sees it.

STAGE 1 (internal): simulate a scrolling viewer whose default state is leaving. React in first person every 2-4 seconds. If you would scroll, say when and why.
STAGE 2 (internal): become the retention analyst. For each significant reaction: what in the video caused it, and what specific edit fixes it.

Then output ONLY a JSON object:
{
  "title": "3-7 word descriptive title",
  "would_post": true/false,
  "comments": [
    {"start_sec": 0.0, "end_sec": 2.0,
     "category": "hook|pacing|retention|clarity|structure|audio|visual|general",
     "severity": "critical|suggested|quick_fix",
     "text": "1-3 sentences: what is happening, why it costs attention, the concrete fix. Viewer-experience language, no structural jargon, no em dashes."}
  ]
}

RULES:
- 3-8 comments, tight timestamps (a 2-second issue gets a 2-second range), total seconds as floats (1:07 -> 67.0).
- Ground every claim in something visible or audible at that timestamp. No fabricated durations.
- Comments are markers, not coverage. Gaps are fine. Comment on the strongest moment too if it is load-bearing.
- critical = fix before posting. Use sparingly and honestly."""

def main():
    path = sys.argv[1]
    out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    f = client.files.upload(file=path)
    while f.state == "PROCESSING":
        time.sleep(2); f = client.files.get(name=f.name)
    r = client.models.generate_content(
        model="gemini-2.5-pro", contents=[f, PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.2))
    review = json.loads(r.text)
    print(f"# {review.get('title')}  |  would_post: {review.get('would_post')}")
    for c in review.get("comments", []):
        sev = c.get("severity", "-")
        print(f"[{c['start_sec']:.1f}-{c.get('end_sec', c['start_sec']):.1f}] "
              f"({c.get('category','general')}/{sev}) {c['text']}")
    if out_json:
        Path(out_json).write_text(json.dumps(review, indent=2))

if __name__ == "__main__":
    main()
