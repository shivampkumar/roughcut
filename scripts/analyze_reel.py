"""Send a finished reel back to Gemini for quality critique."""
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

PROMPT = """Analyze this short reel about Kesha singing at a karaoke bar.

Focus on AUDIO across the whole timeline. The vocals are reportedly being muffled at random places.

For every second of the reel, listen carefully. Identify:
1. Time ranges (in seconds) where the singer's voice is muffled, distorted, sudden volume drops, or hard to hear.
2. What's causing each issue: SFX overlap, clip-boundary artifact, source-recording quality, encoder artifact?
3. Audio clicks/pops at clip transitions — list exact timestamps.
4. Overall audio quality verdict.

Also visual edit critique:
5. Is the close (last 2-3 seconds) effective? It seems to be a dark blurry forehead.
6. Hook strength (first 3 seconds)?
7. Pacing — too slow, too fast, just right?
8. Anything else weak about the edit?

Be specific with timestamps. Return plain text."""


def main(video_path: str) -> None:
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    print(f"uploading {video_path}...")
    f = client.files.upload(file=video_path)
    while f.state == "PROCESSING":
        time.sleep(2)
        f = client.files.get(name=f.name)
    print(f"state: {f.state}")
    resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[f, PROMPT],
        config=types.GenerateContentConfig(temperature=0.2),
    )
    print("\n=== GEMINI VERDICT ===\n")
    print(resp.text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output/kesha_reel.mp4")
