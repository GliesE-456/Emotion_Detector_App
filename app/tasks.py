import io
import csv
from .emotion_model import analyze_text
from flask_socketio import SocketIO

def process_csv_and_stream(file_stream, sid: str, socketio: SocketIO):
    """
    Reads CSV (expects a column named 'text' or first column), runs predictions row-by-row,
    and emits websocket messages with progress to client with session id (sid).
    Returns bytes of resulting CSV with additional columns.
    """
    text_io = io.StringIO(file_stream.read().decode("utf-8"))
    reader = csv.DictReader(text_io)
    fieldnames = reader.fieldnames or []
    # Ensure we have 'text' column; otherwise take first column
    if 'text' not in fieldnames:
        text_col = fieldnames[0] if fieldnames else 'text'
    else:
        text_col = 'text'

    out_io = io.StringIO()
    out_fields = fieldnames + ["pred_top_label", "pred_top_score"]
    writer = csv.DictWriter(out_io, fieldnames=out_fields)
    writer.writeheader()

    rows = list(reader)
    total = len(rows)
    for i, row in enumerate(rows, start=1):
        text = row.get(text_col, "")
        result = analyze_text(text)
        row["pred_top_label"] = result.get("label", "neutral")
        row["pred_top_score"] = result.get("score", 1.0)
        writer.writerow(row)
        socketio.emit("progress", {"progress": i / total}, room=sid)

    out_io.seek(0)
    return out_io.read().encode("utf-8")