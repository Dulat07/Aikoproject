import time
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ------------------------------
# ПРАВИЛЬНЫЕ ПУТИ (Pathlib ❤️)
# ------------------------------

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
SAVE_DIR = BASE_DIR / "saved"
SAVE_DIR.mkdir(exist_ok=True)

# ------------------------------
# FastAPI setup
# ------------------------------

app = FastAPI()

print("📂 BASE_DIR =", BASE_DIR)
print("📂 SAVE_DIR =", SAVE_DIR)
print("📂 STATIC_DIR =", STATIC_DIR)
print("📂 TEMPLATES_DIR =", TEMPLATES_DIR)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ------------------------------
# GET /
# ------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("collector.html", {"request": request})

# ------------------------------
# POST /submit
# ------------------------------
@app.post("/submit")
async def submit_data(request: Request):
    data = await request.json()

    if not isinstance(data, list):
        return {"status": "error", "msg": "Expected list"}

    if len(data) == 0:
        return {"status": "error", "msg": "Empty keystroke list"}

    timestamp = int(time.time())
    filename = SAVE_DIR / f"participant_{timestamp}.csv"

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

    print(f"💾 Saved {filename}")

    return {"status": "ok", "saved": str(filename)}
