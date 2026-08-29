"""UI 없이 폴더 하나를 통째로 분석한다.  사용법: ./analyze.sh [경로 ...] [옵션]"""
from __future__ import annotations

import argparse
import html
import re
import sys
from datetime import datetime
from pathlib import Path

from analysis import AUDIO_EXT, Params, analyze
import report

BASE = Path(__file__).resolve().parent
INPUT = BASE / "input"
OUTPUT = BASE / "output"


def collect(targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for t in targets:
        p = Path(t).expanduser()
        if not p.exists():
            sys.exit(f"경로를 찾을 수 없습니다: {p}")
        if p.is_dir():
            files += sorted(f for f in p.rglob("*")
                            if f.is_file() and f.suffix.lower() in AUDIO_EXT)
        elif p.suffix.lower() in AUDIO_EXT:
            files.append(p)
    seen, out = set(), []
    for f in files:
        k = str(f.resolve())
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


def main() -> None:
    d = Params()
    ap = argparse.ArgumentParser(description="폴더 안의 음원을 모두 분석해 그래프와 리포트를 만든다.")
    ap.add_argument("paths", nargs="*", default=[str(INPUT)],
                    help=f"음원 파일 또는 폴더 (기본: {INPUT})")
    ap.add_argument("--skip", type=float, default=d.skip_s, help="앞부분 건너뛸 초 (기본 %(default)s)")
    ap.add_argument("--max", type=float, default=d.max_s, help="읽을 최대 초 (기본 %(default)s)")
    ap.add_argument("--min-keep", type=float, default=d.min_keep_s, help="이보다 짧게 남으면 파일 전체 사용")
    ap.add_argument("--k-abs", type=float, default=d.k_abs, help="잡음 바닥 대비 최소 배율")
    ap.add_argument("--k-rel", type=float, default=d.k_rel, help="후보 p90 대비 계수")
    ap.add_argument("--refractory", type=float, default=d.refractory_s, help="이 안에 붙은 피크는 병합")
    ap.add_argument("--hop-ms", type=int, default=d.hop_ms, help="엔벌로프 프레임 ms")
    ap.add_argument("--sr", type=int, default=d.sr, help="분석 샘플레이트")
    ap.add_argument("--no-pdf", action="store_true", help="PDF 생성 건너뛰기")
    a = ap.parse_args()

    files = collect(a.paths or [str(INPUT)])
    if not files:
        sys.exit(f"음원 파일이 없습니다: {', '.join(a.paths)}\n"
                 f"→ {INPUT} 안에 파일을 넣거나, 폴더 경로를 인자로 주세요.")

    p = Params(skip_s=a.skip, max_s=a.max, min_keep_s=a.min_keep, sr=a.sr, hop_ms=a.hop_ms,
               k_abs=a.k_abs, k_rel=a.k_rel, refractory_s=a.refractory)
    print(f"[spikeviz] {len(files)}개 파일 분석 시작")
    results = []
    for f in files:
        try:
            r = analyze(str(f), p)
        except Exception as e:
            print(f"  ! {f.name}: {e}")
            continue
        results.append(r)
        print(f"  · {f.name}: 구간 {report.mmss(r['t0'])}–{report.mmss(r['t1'])}, "
              f"임계 {r['thr']:.0f}×, 스파이크 {len(r['times'])}개")
    if not results:
        sys.exit("분석에 성공한 파일이 없습니다.")

    run_id = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT / run_id
    out.mkdir(parents=True, exist_ok=True)
    g1 = report.graph1(results, out / "graph1_waveforms.png")
    g2 = report.graph2(results, out / "graph2_overlay.png")
    report.write_csv(results, out / "spike_intervals.csv")
    report.write_json(results, p.to_dict(), out / "spike_report.json")
    report_html = report.write_html(results, p.to_dict(), run_id, g1, g2, out / "report.html")
    pdf = None if a.no_pdf else report.write_pdf(report_html, out / "report.pdf")

    latest = OUTPUT / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(out.name)

    w = max(len(r["name"]) for r in results)
    print(f"\n  {'파일'.ljust(w)}  스파이크  임계   판정      p값     CV      R(필요)     주기   간격중앙값")
    for r in results:
        s, g = r["stats"], r["regularity"]
        num = lambda v, f="{:6.2f}": f.format(v) if v is not None else "     –"
        rr = f"{num(g['R'])}({num(g['r_needed'], '{:.2f}')})" if g["R"] is not None else "      –      "
        per = f"{g['period']:6.1f}s" if g["period"] else "      –"
        print(f"  {r['name'].ljust(w)}  {len(r['times']):>7}  {r['thr']:>4.0f}×  {g['verdict']:8}"
              f"{num(g['p'], '{:6.3f}')} {num(g['cv'])}  {rr} {per}"
              f"{num(s['median'], '{:9.1f}') if s else '        –'}s")
    print()
    for n in report.observations(results):
        print("  - " + html.unescape(re.sub(r"<[^>]+>", "", n)))
    print(f"\n[spikeviz] 완료 → {out}")
    pdf_line = f"\n           PDF:    {pdf}" if pdf else (
        "" if a.no_pdf else "\n           PDF:    (Chrome을 찾지 못해 건너뜀)")
    print(f"           리포트: {report_html}{pdf_line}")


if __name__ == "__main__":
    main()
