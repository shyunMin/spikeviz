"""spikeviz - 음원 파형 스파이크 간격 분석 웹 UI."""
from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from analysis import AUDIO_EXT, Params, analyze, probe_duration, window_of
from longform import LongParams, analyze_long
import report

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "output"
STATE = BASE / "state.json"

app = Flask(__name__)
app.json.ensure_ascii = False


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"files": [], "params": Params().to_dict()}


def save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=1, ensure_ascii=False))


def expand(raw: str) -> list[Path]:
    """파일 경로면 그 파일, 폴더 경로면 그 안의 음원 전부."""
    p = Path(os.path.expanduser(raw.strip().strip("'\"")))
    if not p.exists():
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {p}")
    if p.is_dir():
        found = sorted(f for f in p.iterdir()
                       if f.is_file() and f.suffix.lower() in AUDIO_EXT)
        if not found:
            raise FileNotFoundError(f"폴더 안에 음원 파일이 없습니다: {p}")
        return found
    if p.suffix.lower() not in AUDIO_EXT:
        raise ValueError(f"지원하지 않는 확장자입니다: {p.suffix}")
    return [p]


@app.get("/")
def index():
    return render_template("index.html")


def mark_missing(files: list[dict]) -> list[dict]:
    for f in files:
        f["missing"] = not Path(f["path"]).exists()
    return files


@app.get("/api/state")
def api_state():
    s = load_state()
    mark_missing(s["files"])
    s["runs"] = [d.name for d in sorted(OUTPUT.glob("run_*"), reverse=True)] if OUTPUT.exists() else []
    s["defaults"] = Params().to_dict()
    s["long_defaults"] = LongParams().to_dict()
    return jsonify(s)


@app.post("/api/files")
def api_add():
    s = load_state()
    known = {f["path"] for f in s["files"]}
    added, skipped = [], []
    try:
        paths = expand(request.json.get("path", ""))
    except (FileNotFoundError, ValueError) as e:
        return jsonify(error=str(e)), 400
    for p in paths:
        key = str(p.resolve())
        if key in known:
            skipped.append(p.name)
            continue
        try:
            dur = probe_duration(key)
        except Exception as e:
            skipped.append(f"{p.name} ({e})")
            continue
        s["files"].append({"path": key, "name": p.stem, "filename": p.name,
                           "duration": dur, "size": p.stat().st_size,
                           "added": datetime.now().isoformat(timespec="seconds")})
        known.add(key)
        added.append(p.name)
    save_state(s)
    return jsonify(files=mark_missing(s["files"]), added=added, skipped=skipped)


@app.post("/api/files/remove")
def api_remove():
    s = load_state()
    target = request.json.get("path")
    s["files"] = [f for f in s["files"] if f["path"] != target]
    save_state(s)
    return jsonify(files=mark_missing(s["files"]))


@app.post("/api/run")
def api_run():
    s = load_state()
    if not s["files"]:
        return jsonify(error="분석할 파일이 없습니다. 먼저 경로를 추가하세요."), 400
    incoming = request.json.get("params") or {}
    fields = Params().to_dict()
    params = Params(**{k: type(v)(incoming.get(k, v)) for k, v in fields.items()})
    s["params"] = params.to_dict()
    save_state(s)

    results, failed = [], []
    for f in s["files"]:
        if not Path(f["path"]).exists():
            failed.append({"file": f["filename"],
                           "error": "파일을 찾을 수 없습니다 (옮겼거나 지운 것 같습니다). 목록에서 지우고 다시 추가하세요."})
            continue
        try:
            results.append(analyze(f["path"], params))
        except Exception as e:
            failed.append({"file": f["filename"], "error": str(e)})
            traceback.print_exc()
    if not results:
        reasons = " / ".join(f"{x['file']}: {x['error']}" for x in failed[:3])
        return jsonify(error=f"분석에 성공한 파일이 없습니다. {reasons}", failed=failed), 500

    run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    d = OUTPUT / run_id
    d.mkdir(parents=True, exist_ok=True)
    g1 = report.graph1(results, d / "graph1_waveforms.png", baseline=params.baseline)
    g2 = report.graph2(results, d / "graph2_overlay.png", baseline=params.baseline)
    report.write_csv(results, d / "spike_intervals.csv")
    report.write_json(results, params.to_dict(), d / "spike_report.json")
    html_path = report.write_html(results, params.to_dict(), run_id, g1, g2, d / "report.html")
    pdf = report.write_pdf(html_path, d / "report.pdf")

    latest = OUTPUT / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(d.name)

    cols = report.colors_for(results, params.baseline)
    summary = [{"name": r["name"], "filename": r["filename"], "color": cols[i],
                "window": [r["t0"], r["t1"]], "thr": r["thr"], "max_ratio": r["max_ratio"],
                "spikes": len(r["times"]), "anchor": r["abs_anchor"], "source": r["source"],
                "stats": r["stats"], "dominant": r["dominant"], "regularity": r["regularity"],
                "intervals": [round(v, 2) for v in r["intervals"]]}
               for i, r in enumerate(results)]
    return jsonify(run_id=run_id, dir=str(d), files=summary, failed=failed,
                   pdf=bool(pdf), notes=report.observations(results, params.baseline))


@app.post("/api/run_long")
def api_run_long():
    """기능 2: 파일 하나를 잡음 제거 + 빈도 변화로 분석한다."""
    target = (request.json or {}).get("path", "")
    if not Path(target).exists():
        return jsonify(error="파일을 찾을 수 없습니다."), 400
    incoming = (request.json or {}).get("params") or {}
    fields = LongParams().to_dict()
    params = LongParams(**{k: type(v)(incoming.get(k, v)) for k, v in fields.items()})
    try:
        r = analyze_long(target, params)
    except Exception as e:
        traceback.print_exc()
        return jsonify(error=f"분석 실패: {e}"), 500

    run_id = "long_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + Path(target).stem
    d = OUTPUT / run_id
    d.mkdir(parents=True, exist_ok=True)
    g1 = report.graph_long_timeline(r, d / "timeline.png")
    g2 = report.graph_long_trend(r, d / "trend.png")
    report.write_long_csv(r, d / "spikes_and_segments.csv")
    report.write_long_json(r, params.to_dict(), d / "long_report.json")
    html_path = report.write_long_html(r, params.to_dict(), run_id, g1, g2, d / "report.html")
    pdf = report.write_pdf(html_path, d / "report.pdf")
    return jsonify(run_id=run_id, mode="long", pdf=bool(pdf), name=r["name"],
                   span=r["span"], spikes=len(r["times"]), rejected=r["rejected"],
                   trend=r["trend"], regularity=r["regularity"],
                   segments=[{k: v for k, v in s.items()} for s in r["segments"]],
                   notes=report.long_observations(r))


@app.get("/runs/<run_id>/<path:filename>")
def runs(run_id, filename):
    return send_from_directory(OUTPUT / run_id, filename)


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    port = int(os.environ.get("PORT", "8765"))
    print(f"\n  spikeviz  →  http://localhost:{port}\n")
    app.run(host="127.0.0.1", port=port, threaded=True)
