import os, sys, time
from dotenv import load_dotenv
from google import genai
from google.genai import types
load_dotenv()

RUBRIC = """You are a brutal Instagram Reels editor judging whether a concert reel is POST-WORTHY.
Score 1-10 on each: HOOK (first 2s stop scroll?), TENSION (does it build?), PAYOFF (does it land?),
AUDIO (is the music continuous and pleasant, or chopped/jarring?), CAPTIONS (help or hurt?),
OVERALL postability. Then one sentence: the single biggest flaw. Format:
HOOK: n | TENSION: n | PAYOFF: n | AUDIO: n | CAPTIONS: n | OVERALL: n
FLAW: ..."""

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
for path in sys.argv[1:]:
    f = client.files.upload(file=path)
    while f.state == "PROCESSING":
        time.sleep(2); f = client.files.get(name=f.name)
    r = client.models.generate_content(model="gemini-2.5-pro", contents=[f, RUBRIC],
        config=types.GenerateContentConfig(temperature=0.1))
    name = path.split("/")[-1]
    print(f"=== {name} ===")
    print(r.text.strip()[:400])
