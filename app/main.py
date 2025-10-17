from fastapi import FastAPI, Request, UploadFile, WebSocket, Form, Depends, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from app.db import get_db, Message
from app.emotion_model import analyze_text
import pandas as pd
import io
from sqlalchemy.orm import Session

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze/")
def analyze(request: Request, text: str = Form(...), db: Session = Depends(get_db)):
    result = analyze_text(text)
    if "error" not in result:
        msg = Message(text=text, emotion=result["label"])
        db.add(msg)
        db.commit()
    return templates.TemplateResponse("index.html", {"request": request, "result": result, "input_text": text})

@app.post("/upload_csv/")
async def upload_csv(file: UploadFile, db: Session = Depends(get_db)):
    df = pd.read_csv(io.BytesIO(await file.read()))
    df["Emotion"] = df["text"].apply(lambda x: analyze_text(x)["label"])
    output_path = "output.csv"
    df.to_csv(output_path, index=False)
    return FileResponse(output_path, filename="output.csv")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_text("WebSocket connection established.")
    await ws.close()

@app.get("/history", response_class=HTMLResponse)
def history(request: Request, db: Session = Depends(get_db)):
    messages = db.query(Message).order_by(Message.timestamp.desc()).all()
    return templates.TemplateResponse("history.html", {"request": request, "messages": messages})