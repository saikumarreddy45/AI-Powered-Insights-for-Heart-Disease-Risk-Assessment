# db.py
import sqlite3, os, json, time
DB_PATH = os.environ.get("HEART_DB", "heart_app.db")

DDL = """
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at INTEGER NOT NULL,
  patient_name TEXT NOT NULL,
  attrs_json TEXT NOT NULL,
  prob REAL NOT NULL,
  risk_level INTEGER NOT NULL,
  risk_label TEXT NOT NULL,
  model_name TEXT NOT NULL,
  app_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_patient_time ON predictions(patient_name, created_at DESC);
"""

def _conn():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL;")
    return con

def init_db():
    con = _conn()
    for stmt in DDL.strip().split(";"):
        s = stmt.strip()
        if s:
            con.execute(s)
    con.commit(); con.close()

def save_prediction(patient_name: str, attrs: dict, prob: float, risk_level: int, risk_label: str,
                    model_name: str, app_version: str="v1"):
    con = _conn()
    con.execute(
        "INSERT INTO predictions (created_at, patient_name, attrs_json, prob, risk_level, risk_label, model_name, app_version) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (int(time.time()), patient_name, json.dumps(attrs), float(prob), int(risk_level), risk_label, model_name, app_version)
    )
    con.commit(); con.close()

def list_predictions(search: str=None, min_level: int=None, max_level: int=None, limit:int=200):
    con = _conn()
    q = "SELECT id, created_at, patient_name, prob, risk_level, risk_label, model_name, attrs_json FROM predictions WHERE 1=1"
    params = []
    if search:
        q += " AND patient_name LIKE ?"; params.append(f"%{search}%")
    if min_level is not None:
        q += " AND risk_level >= ?"; params.append(min_level)
    if max_level is not None:
        q += " AND risk_level <= ?"; params.append(max_level)
    q += " ORDER BY created_at DESC LIMIT ?"; params.append(limit)
    rows = _to_dicts(con.execute(q, params).fetchall())
    con.close(); return rows

def patient_history(name: str, limit:int=50):
    con = _conn()
    rows = _to_dicts(con.execute(
        "SELECT created_at, prob, risk_level, risk_label FROM predictions WHERE patient_name=? ORDER BY created_at DESC LIMIT ?",
        (name, limit)
    ).fetchall())
    con.close(); return rows

def _to_dicts(rows):
    out=[]
    for r in rows:
        d={"id":r[0], "created_at":r[1], "patient_name":r[2], "prob":r[3],
           "risk_level":r[4], "risk_label":r[5], "model_name":r[6]}
        if len(r)>7 and r[7]: d["attrs"]=json.loads(r[7])
        out.append(d)
    return out
