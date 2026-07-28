"""Reel quality benchmark. Scores videos with a fixed blind-judge rubric,
parses scores, appends to bench/history.jsonl with git rev + label.
Usage: python scripts/benchmark.py LABEL file1.mp4 [file2 ...]"""
import json, os, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
load_dotenv()

RUBRIC = """You are a brutal Instagram Reels editor judging whether a concert reel is POST-WORTHY.
Score 1-10 on each: HOOK (first 2s stop scroll?), TENSION (does it build?), PAYOFF (does it land?),
AUDIO (is the music continuous and pleasant, or chopped/jarring?), CAPTIONS (help or hurt?),
ENDING (does it resolve or just stop?), OVERALL postability. Then one sentence: biggest flaw. Format:
HOOK: n | TENSION: n | PAYOFF: n | AUDIO: n | CAPTIONS: n | ENDING: n | OVERALL: n
FLAW: ..."""

def judge(client, path):
    f = client.files.upload(file=path)
    while f.state == "PROCESSING":
        time.sleep(2); f = client.files.get(name=f.name)
    r = client.models.generate_content(model="gemini-2.5-pro", contents=[f, RUBRIC],
        config=types.GenerateContentConfig(temperature=0.1))
    text = r.text.strip()
    scores = dict(re.findall(r"(HOOK|TENSION|PAYOFF|AUDIO|CAPTIONS|ENDING|OVERALL):\s*(\d+)", text))
    flaw = (re.search(r"FLAW:\s*(.+)", text) or [None, ""])[1]
    return {k.lower(): int(v) for k, v in scores.items()}, flaw.strip()

def main():
    label, paths = sys.argv[1], sys.argv[2:]
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    out = Path("bench/history.jsonl"); out.parent.mkdir(exist_ok=True)
    for p in paths:
        scores, flaw = judge(client, p)
        row = {"ts": datetime.now(timezone.utc).isoformat(), "label": label, "rev": rev,
               "file": Path(p).name, "scores": scores, "flaw": flaw}
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{Path(p).name}: {scores} | {flaw[:80]}")

if __name__ == "__main__":
    main()
