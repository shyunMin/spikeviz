"""UI 없이 폴더 하나를 통째로 분석한다.  사용법: ./analyze.sh [경로 ...] [옵션]"""
from __future__ import annotations

import argparse
import html
import re
import sys
from datetime import datetime
from pathlib import Path

from analysis import AUDIO_EXT, Params, analyze
from longform import LongParams, analyze_long
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
    ap.add_argument("--k-cap", type=float, default=d.k_cap, help="임계 상한 (잡음 바닥 대비 배율)")
    ap.add_argument("--baseline", default=d.baseline,
                    help="그래프 2 위 패널의 기준 파일 이름 (예: 260829_035407)")
    ap.add_argument("--refractory", type=float, default=d.refractory_s, help="이 안에 붙은 피크는 병합")
    ap.add_argument("--hop-ms", type=int, default=d.hop_ms, help="엔벌로프 프레임 ms")
    ap.add_argument("--sr", type=int, default=d.sr, help="분석 샘플레이트")
    ap.add_argument("--k-local", type=float, default=LongParams().k_local,
                    help="[기능 2] 국소 잡음 바닥 대비 최소 배율")
    ap.add_argument("--max-sustain", type=float, default=LongParams().max_sustain_s,
                    help="[기능 2] 스파이크로 인정하는 최대 지속폭(초)")
    ap.add_argument("--long", action="store_true",
                    help="기능 2: 긴 파일 하나하나를 잡음 제거 + 빈도 변화로 분석 (기본은 기능 1: 파일 비교)")
    ap.add_argument("--no-pdf", action="store_true", help="PDF 생성 건너뛰기")
    a = ap.parse_args()

    files = collect(a.paths or [str(INPUT)])
    if not files:
        sys.exit(f"음원 파일이 없습니다: {', '.join(a.paths)}\n"
                 f"→ {INPUT} 안에 파일을 넣거나, 폴더 경로를 인자로 주세요.")

    if a.long:
        run_long(files, a)
        return

    p = Params(skip_s=a.skip, max_s=a.max, min_keep_s=a.min_keep, sr=a.sr, hop_ms=a.hop_ms,
               k_abs=a.k_abs, k_rel=a.k_rel, k_cap=a.k_cap, refractory_s=a.refractory,
               baseline=a.baseline)
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
    g1 = report.graph1(results, out / "graph1_waveforms.png", baseline=p.baseline)
    g2 = report.graph2(results, out / "graph2_overlay.png", baseline=p.baseline)
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
        how = "  수동" if r.get("source") == "manual" else f"{r['thr']:>4.0f}×"
        print(f"  {r['name'].ljust(w)}  {len(r['times']):>7}  {how}  {g['verdict']:8}"
              f"{num(g['p'], '{:6.3f}')} {num(g['cv'])}  {rr} {per}"
              f"{num(s['median'], '{:9.1f}') if s else '        –'}s")
    print()
    for n in report.observations(results, p.baseline):
        print("  - " + html.unescape(re.sub(r"<[^>]+>", "", n)))
    print(f"\n[spikeviz] 완료 → {out}")
    pdf_line = f"\n           PDF:    {pdf}" if pdf else (
        "" if a.no_pdf else "\n           PDF:    (Chrome을 찾지 못해 건너뜀)")
    print(f"           리포트: {report_html}{pdf_line}")


def run_long(files: list[Path], a) -> None:
    """기능 2: 파일마다 잡음을 걸러내고 빈도 변화를 본다."""
    p = LongParams(skip_s=a.skip, sr=a.sr, hop_ms=a.hop_ms,
                   k_local=a.k_local, max_sustain_s=a.max_sustain)
    print(f"[spikeviz] 긴 파일 분석 {len(files)}개 (파일마다 개별 리포트)")
    for f in files:
        try:
            r = analyze_long(str(f), p)
        except Exception as e:
            print(f"  ! {f.name}: {e}")
            continue
        run_id = "long_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + f.stem
        out = OUTPUT / run_id
        out.mkdir(parents=True, exist_ok=True)
        g1 = report.graph_long_timeline(r, out / "timeline.png")
        g2 = report.graph_long_trend(r, out / "trend.png")
        report.write_long_csv(r, out / "spikes_and_segments.csv")
        report.write_long_json(r, p.to_dict(), out / "long_report.json")
        html_path = report.write_long_html(r, p.to_dict(), run_id, g1, g2, out / "report.html")
        pdf = None if a.no_pdf else report.write_pdf(html_path, out / "report.pdf")

        tr, g = r["trend"], r["regularity"]
        print(f"\n  {f.name}: {r['span'] / 3600:.2f}시간 분석, 스파이크 {len(r['times'])}개 "
              f"(잡음 제외 {sum(r['rejected'].values())}건)")
        if tr["slope_per_hour"] is not None:
            print(f"    빈도 {tr['verdict']}: {tr['slope_per_hour']:+.2f}초/시간, p={tr['p']:.2g} "
                  f"({tr['first']:.1f}초 → {tr['last']:.1f}초, {tr['change_pct']:+.0f}%)")
        print(f"    전체 규칙성: {g['verdict']}" +
              (f" (주기 {g['period']:.1f}초)" if g["period"] else ""))
        reg = sum(1 for s in r["segments"] if s["verdict"] in ("규칙적", "약한 규칙성"))
        print(f"    구간 {len(r['segments'])}개 중 {reg}개가 규칙적")
        for n in report.long_observations(r):
            print("      - " + html.unescape(re.sub(r"<[^>]+>", "", n)))
        print(f"    → {out}" + (f"\n      PDF: {pdf}" if pdf else ""))


if __name__ == "__main__":
    main()
