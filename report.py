"""그래프 2장 + HTML 리포트 + PDF 생성."""
from __future__ import annotations

import base64
import html
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 검증된 8슬롯 범주형 팔레트. 9번째부터는 회색으로 두고 이름표로 구분한다(색 재사용 금지).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4", "#008300", "#eda100", "#e34948"]
EXTRA = "#8a8985"
INK, INK2, INK3, SURF, GRID = "#0b0b0b", "#52514e", "#8a8985", "#fcfcfb", "#e6e5e1"
CHROME = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome")


def color_of(i: int) -> str:
    return SLOTS[i] if i < len(SLOTS) else EXTRA


def mmss(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def _minmax(x: np.ndarray, sr: int, cols: int = 2400):
    cols = min(cols, max(len(x) // 8, 60))
    n = len(x) // cols * cols
    b = x[:n].reshape(cols, -1)
    return np.arange(cols) * (n / cols) / sr, b.min(1), b.max(1)


def _strip(ax):
    ax.set_facecolor(SURF)
    ax.grid(axis="x", color=GRID, lw=.7)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d2")
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)


def graph1(results: list[dict], out: Path) -> Path:
    n = len(results)
    h = 1.1 + 2.5 * n
    fig, axes = plt.subplots(n, 1, figsize=(13, h), facecolor=SURF, squeeze=False)
    axes = axes[:, 0]
    fig.subplots_adjust(hspace=.55, top=1 - .78 / h, bottom=.7 / h, left=.085, right=.98)
    fmt = FuncFormatter(lambda v, _: mmss(v))
    for ax, r, i in zip(axes, results, range(n)):
        c = color_of(i)
        tt, mn, mx = _minmax(r["wave"], r["sr"])
        tt = tt + r["t0"]
        _strip(ax)
        ax.fill_between(tt, mn, mx, color=c, lw=0, alpha=.9)
        lim = max(abs(mn).max(), mx.max(), 1e-6) * 1.35
        for t in r["abs_times"]:
            ax.plot(t, lim * .86, marker="v", ms=6, color=INK, mec="none", clip_on=False)
        ax.set_ylim(-lim, lim)
        ax.set_xlim(r["t0"], r["t1"])
        ax.xaxis.set_major_formatter(fmt)
        ax.set_title(f"{r['filename']}   |   analysed {mmss(r['t0'])}-{mmss(r['t1'])}"
                     f"   |   threshold {r['thr']:.0f}x noise floor   |   {len(r['times'])} spikes (v)",
                     loc="left", fontsize=10.5, color=INK, pad=7)
        ax.set_ylabel("amplitude", fontsize=8.5, color=INK2)
    axes[-1].set_xlabel("time in file (m:ss)", fontsize=9, color=INK2)
    fig.suptitle("Graph 1  -  Waveform per recording", x=.085, ha="left", fontsize=14, color=INK,
                 y=1 - .3 / h)
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)
    return out


def graph2(results: list[dict], out: Path) -> Path:
    n = len(results)
    h = 5.6 + .45 * n
    fig = plt.figure(figsize=(13, h), facecolor=SURF)
    lab = .055 + .006 * max(len(r["name"]) for r in results)   # 이름표가 길면 왼쪽을 더 준다
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, max(.8, .28 * n)], hspace=.16,
                          top=1 - 1.15 / h, bottom=.62 / h, left=min(lab, .16), right=.985)
    ax, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    span, ymax = 0.0, 10.0
    for i, r in enumerate(results):
        c = color_of(i)
        k = int(round(r["anchor"] / r["dt"]))
        e = r["env"][k:] / r["floor"]
        g = 10
        m = len(e) // g * g
        if m == 0:
            continue
        e = e[:m].reshape(-1, g).max(1)
        tr = np.arange(len(e)) * r["dt"] * g
        span, ymax = max(span, tr[-1]), max(ymax, e.max() * 1.25)
        ax.fill_between(tr, 1, e, color=c, lw=0, alpha=.22)
        kind = "first spike" if r["anchor_kind"] == "first_spike" else "max amplitude, no spike"
        ax.plot(tr, e, color=c, lw=1.0, alpha=.85,
                label=f"{r['name']}  ({kind} @ {mmss(r['abs_anchor'])})")
    for i, r in enumerate(results):
        c, y = color_of(i), n - 1 - i
        sp = [t - r["anchor"] for t in r["times"] if t >= r["anchor"]]
        axr.plot([0, r["t1"] - r["t0"] - r["anchor"]], [y, y], color=c, lw=.8, alpha=.28)
        axr.plot(sp, [y] * len(sp), "|", ms=13, mew=2, color=c)
        axr.text(-span * .006, y, r["name"], ha="right", va="center", fontsize=8.5, color=INK2)
    for a in (ax, axr):
        _strip(a)
        a.set_xlim(0, span or 1)
    ax.set_yscale("log")
    ax.set_ylim(1, ymax)
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(["1x", "10x", "100x"])
    ax.set_ylabel("peak envelope / own noise floor", fontsize=9, color=INK2)
    ax.tick_params(labelbottom=False)
    ax.legend(frameon=False, fontsize=8.8, labelcolor=INK2, ncols=min(4, max(1, n)),
              loc="lower left", bbox_to_anchor=(0, 1.005, 1, .08), mode="expand",
              borderaxespad=0, handlelength=1.6, columnspacing=1.2)
    axr.set_yticks([])
    axr.set_ylim(-.6, n - .4)
    axr.set_xlabel("time relative to each file's first spike (s)", fontsize=9, color=INK2)
    axr.set_title("spike times, same relative axis", loc="left", fontsize=9.5, color=INK2, pad=4)
    fig.suptitle("Graph 2  -  All recordings overlaid, aligned on their first spike (t = 0)",
                 x=min(lab, .16), ha="left", fontsize=14, color=INK, y=1 - .32 / h)
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)
    return out


def write_csv(results: list[dict], out: Path) -> Path:
    lines = ["file,index,spike_time_s,spike_time_mmss,peak_x_noise_floor,interval_from_prev_s"]
    for r in results:
        prev = None
        for i, (t, x) in enumerate(zip(r["abs_times"], r["ratios"]), 1):
            gap = "" if prev is None else f"{t - prev:.2f}"
            lines.append(f"{r['name']},{i},{t:.2f},{mmss(t)},{x:.1f},{gap}")
            prev = t
    out.write_text("\n".join(lines) + "\n")
    return out


def write_json(results: list[dict], params: dict, out: Path) -> Path:
    payload = dict(params=params, files=[
        {k: v for k, v in r.items() if k not in ("wave", "env")} for r in results])
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    return out


def observations(results: list[dict]) -> list[str]:
    """데이터에서 바로 계산되는 관찰만 만든다."""
    notes = []
    scored = [r for r in results if r["stats"] and r["stats"]["n"] >= 4]
    if scored:
        best = min(scored, key=lambda r: r["stats"]["cv"])
        s = best["stats"]
        notes.append(f"<b>{html.escape(best['name'])}가 가장 규칙적입니다.</b> 간격 {s['n']}개의 중앙값 "
                     f"{s['median']:.1f}초, 표준편차 {s['std']:.1f}초(변동계수 {s['cv']:.2f})로 "
                     f"이번 실행에서 가장 고른 간격을 보입니다.")
    weak = [r for r in results if r["times"] and r["max_ratio"] < 20]
    if weak:
        notes.append("<b>검출 임계에 겨우 걸친 파일이 있습니다.</b> " +
                     ", ".join(f"{html.escape(r['name'])}(최대 {r['max_ratio']:.0f}×)" for r in weak) +
                     " — 스파이크가 잡음 대비 크게 튀지 않아 개수·간격의 신뢰도가 낮습니다.")
    none = [r for r in results if not r["times"]]
    if none:
        notes.append("<b>스파이크가 하나도 잡히지 않은 파일:</b> " +
                     ", ".join(html.escape(r["name"]) for r in none) +
                     ". 그래프 2에서는 최대 진폭 지점을 기준점으로 썼습니다.")
    whole = [r for r in results if r["whole_file"]]
    if whole and len(whole) < len(results):
        notes.append("<b>구간이 다른 파일이 섞여 있습니다.</b> " +
                     ", ".join(html.escape(r["name"]) for r in whole) +
                     "은(는) 길이가 짧아 앞부분을 자르지 않고 전체를 썼습니다. 다른 파일에서 잘라낸 "
                     "구간이 여기에는 포함되므로 간격을 직접 비교할 때 주의해야 합니다.")
    if len(results) > len(SLOTS):
        notes.append(f"<b>색 슬롯을 넘는 파일이 있습니다.</b> 9번째부터는 그래프에서 회색으로 그려집니다"
                     f"(총 {len(results)}개). 이름표로 구분하세요.")
    return notes


def _card(r: dict, c: str) -> str:
    s = r["stats"]
    chips = "".join(f'<span class="iv">{v:.1f}</span>' for v in r["intervals"]) \
        or '<span class="none">간격 없음</span>'
    stat = (f'<span><b>{s["median"]:.1f}</b>초 중앙값</span><span><b>{s["mean"]:.1f}</b>초 평균</span>'
            f'<span>{s["min"]:.1f}–{s["max"]:.1f}초 범위</span><span>σ {s["std"]:.1f}초</span>'
            f'<span>변동계수 {s["cv"]:.2f}</span>') if s else '<span>간격 통계 없음</span>'
    times = " ".join(f'<span class="t">{mmss(t)}<i>{x:.0f}×</i></span>'
                     for t, x in zip(r["abs_times"], r["ratios"]))
    anchor = "첫 스파이크" if r["anchor_kind"] == "first_spike" else "최대 진폭(스파이크 없음)"
    return f'''<article class="card" style="--c:{c}">
  <header><h3>{html.escape(r["filename"])}</h3>
    <p class="meta">분석 구간 {mmss(r['t0'])}–{mmss(r['t1'])} · 임계 {r['thr']:.0f}× · 최대 {r['max_ratio']:.0f}×
      · 스파이크 {len(r['times'])}개 · 기준점 {mmss(r['abs_anchor'])} ({anchor})</p></header>
  <div class="stats">{stat}</div>
  <p class="lbl">스파이크 간격 (초, 앞 스파이크로부터)</p>
  <div class="chips">{chips}</div>
  <p class="lbl">스파이크 발생 시각 · 잡음바닥 대비 배율</p>
  <div class="times">{times or '<span class="none">검출 없음</span>'}</div>
</article>'''


def write_html(results: list[dict], params: dict, run_id: str, g1: Path, g2: Path, out: Path) -> Path:
    b64 = lambda p: base64.b64encode(p.read_bytes()).decode()
    rows = ""
    for i, r in enumerate(results):
        s = r["stats"]
        cells = (f"<td>{s['median']:.1f}</td><td>{s['mean']:.1f}</td>"
                 f"<td>{s['min']:.1f} / {s['max']:.1f}</td><td>{s['cv']:.2f}</td>") if s \
            else "<td>–</td><td>–</td><td>–</td><td>–</td>"
        rows += (f'<tr><td><span class="dot" style="background:{color_of(i)}"></span>'
                 f'{html.escape(r["name"])}</td><td>{len(r["times"])}</td><td>{r["thr"]:.0f}×</td>'
                 f'{cells}<td>{mmss(r["abs_anchor"])}</td></tr>')
    cards = "\n".join(_card(r, color_of(i)) for i, r in enumerate(results))
    notes = "".join(f"<li>{t}</li>" for t in observations(results)) or "<li>관찰할 항목이 없습니다.</li>"
    p = params
    win = (f"{p['skip_s']:.0f}초 이후 최대 {p['max_s'] / 60:.0f}분. 앞을 자르면 남는 길이가 "
           f"{p['min_keep_s']:.0f}초 미만인 파일은 전체 사용.")
    thr_rule = f"잡음 바닥 대비 max({p['k_abs']:.0f}×, {p['k_rel']:.2f} × p90) 초과, {p['refractory_s']:.0f}초 이내 병합."
    page = f'''<title>Spike Interval Report</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+KR:wght@400;500;600&display=swap">
<style>
:root{{--bg:#f2f4f6;--panel:#fff;--line:#dfe3e8;--ink:#12171c;--ink2:#48525e;--ink3:#77828f;
 --shadow:0 1px 2px rgba(18,23,28,.05),0 6px 22px -14px rgba(18,23,28,.28);--chip:#eaeef3;--figbg:#fcfcfb}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#12161a;--panel:#191e24;--line:#2a323b;
 --ink:#eef2f6;--ink2:#aab5c0;--ink3:#7d8894;--shadow:0 1px 2px rgba(0,0,0,.4);--chip:#232b33;--figbg:#f4f4f2}}}}
:root[data-theme="dark"]{{--bg:#12161a;--panel:#191e24;--line:#2a323b;--ink:#eef2f6;--ink2:#aab5c0;
 --ink3:#7d8894;--shadow:0 1px 2px rgba(0,0,0,.4);--chip:#232b33;--figbg:#f4f4f2}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans KR",system-ui,sans-serif;
 font-size:15.5px;line-height:1.65;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1080px;margin:0 auto;padding:56px 24px 96px;display:flex;flex-direction:column;gap:38px}}
header.top{{display:flex;flex-direction:column;gap:10px;border-bottom:1px solid var(--line);padding-bottom:24px}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3);margin:0}}
h1{{font-size:31px;font-weight:600;letter-spacing:-.015em;margin:0;text-wrap:balance}}
h2{{font-size:19px;font-weight:600;margin:0 0 2px}} h3{{font-size:16.5px;font-weight:600;margin:0;font-family:"IBM Plex Mono",monospace}}
p{{margin:0}} .sub{{color:var(--ink2);max-width:64ch}}
.rule{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}}
.rule div{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px 15px;box-shadow:var(--shadow)}}
.rule b{{display:block;font-family:"IBM Plex Mono",monospace;font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink3);font-weight:500;margin-bottom:4px}}
.rule span{{font-size:14px;color:var(--ink2)}}
figure{{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:var(--shadow)}}
figure .cap{{padding:16px 20px 4px;display:flex;flex-direction:column;gap:3px}}
figure .cap p{{color:var(--ink2);font-size:14px}}
figure .imgbox{{overflow-x:auto;padding:8px 12px 14px}}
figure img{{display:block;width:100%;min-width:760px;background:var(--figbg);border-radius:6px}}
.cards{{display:flex;flex-direction:column;gap:16px}}
.card{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--c);border-radius:10px;
 padding:18px 20px 20px;box-shadow:var(--shadow);display:flex;flex-direction:column;gap:12px}}
.card header{{display:flex;flex-direction:column;gap:3px}} .card h3{{color:var(--c)}}
.meta{{font-size:13.5px;color:var(--ink3)}}
.stats{{display:flex;flex-wrap:wrap;gap:8px 22px;font-size:14px;color:var(--ink2);font-variant-numeric:tabular-nums}}
.stats b{{color:var(--ink);font-family:"IBM Plex Mono",monospace;font-weight:500;font-size:16px}}
.lbl{{font-size:12px;letter-spacing:.06em;color:var(--ink3);text-transform:uppercase;font-family:"IBM Plex Mono",monospace}}
.chips{{display:flex;flex-wrap:wrap;gap:5px}}
.iv{{font-family:"IBM Plex Mono",monospace;font-size:12.5px;padding:3px 8px;border-radius:5px;background:var(--chip);color:var(--ink2);font-variant-numeric:tabular-nums}}
.none{{font-size:13px;color:var(--ink3)}}
.times{{display:flex;flex-wrap:wrap;gap:4px 10px;font-family:"IBM Plex Mono",monospace;font-size:12.5px;color:var(--ink2)}}
.times .t i{{font-style:normal;color:var(--ink3);margin-left:3px;font-size:11px}}
.tablewrap{{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}}
table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums;min-width:660px}}
th,td{{text-align:right;padding:11px 16px;border-bottom:1px solid var(--line);white-space:nowrap}}
th:first-child,td:first-child{{text-align:left;font-family:"IBM Plex Mono",monospace}}
th{{font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);font-weight:500}}
tbody tr:last-child td{{border-bottom:none}}
.dot{{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:8px;vertical-align:1px}}
.find{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px 22px;box-shadow:var(--shadow);display:flex;flex-direction:column;gap:9px}}
.find ul{{margin:0;padding-left:19px;display:flex;flex-direction:column;gap:7px;color:var(--ink2)}}
.find b{{color:var(--ink);font-weight:500}}
.note{{font-size:13.5px;color:var(--ink3);max-width:70ch}}
code{{font-family:"IBM Plex Mono",monospace;font-size:.92em;background:var(--chip);padding:1px 5px;border-radius:4px}}
@media print{{
 :root{{--bg:#fff;--panel:#fff;--line:#d7dbe0;--shadow:none;--figbg:#fff}}
 @page{{size:A4;margin:14mm 12mm}} body{{background:#fff;font-size:10.5px}}
 .wrap{{max-width:none;padding:0;gap:20px}} h1{{font-size:23px}} h2{{font-size:15px}} h3{{font-size:13px}}
 figure,.card,.find,.tablewrap,.rule div{{break-inside:avoid;box-shadow:none}}
 figure img{{min-width:0}} .imgbox,.tablewrap{{overflow:visible}}
 table{{min-width:0;font-size:9.5px}} th,td{{padding:7px 9px}}
 *{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
}}
</style>
<div class="wrap">
<header class="top">
  <p class="eyebrow">{len(results)} recordings · run {html.escape(run_id)}</p>
  <h1>파형 스파이크 간격 분석</h1>
  <p class="sub">각 녹음에서 파형이 튀는 순간을 검출해 그 간격을 정리했습니다. 그래프 1은 파일별 파형,
     그래프 2는 각 파일의 첫 스파이크를 t = 0에 맞춰 겹친 것입니다.</p>
</header>
<section class="rule">
  <div><b>분석 구간</b><span>{win}</span></div>
  <div><b>파형 측정</b><span>{p['hop_ms']} ms 프레임의 피크 엔벌로프({p['sr']} Hz 모노). 짧은 클릭이 평균에 묻히지 않습니다.</span></div>
  <div><b>스파이크 판정</b><span>{thr_rule}</span></div>
  <div><b>그래프 2 정렬</b><span>각 파일의 첫 스파이크를 t = 0으로 두고 상대 시간으로 겹쳐 그림.</span></div>
</section>
<figure>
  <div class="cap"><h2>그래프 1 — 파일별 파형</h2>
  <p>가로축은 파일 내 시간(m:ss), 세로축은 진폭. 검은 ▼ 가 검출된 스파이크입니다. 세로 눈금은 파일마다 다릅니다.</p></div>
  <div class="imgbox"><img src="data:image/png;base64,{b64(g1)}" alt="파일별 파형과 스파이크 표시"></div>
</figure>
<figure>
  <div class="cap"><h2>그래프 2 — 첫 스파이크 기준 겹쳐 보기</h2>
  <p>세로축은 각 파일의 잡음 바닥 대비 배율(로그). 녹음 레벨 차이가 커서 진폭 그대로 겹치면 작은 파일이 묻히므로
     자기 잡음 대비로 정규화했습니다. 아래 띠는 같은 상대 축 위의 스파이크 시각입니다.</p></div>
  <div class="imgbox"><img src="data:image/png;base64,{b64(g2)}" alt="첫 스파이크에 정렬해 겹친 엔벌로프"></div>
</figure>
<section class="find"><h2>자동 관찰</h2><ul>{notes}</ul></section>
<section class="tablewrap"><table>
  <thead><tr><th>파일</th><th>스파이크</th><th>임계</th><th>간격 중앙값(초)</th><th>평균(초)</th><th>최소 / 최대(초)</th><th>변동계수</th><th>첫 스파이크</th></tr></thead>
  <tbody>{rows}</tbody></table></section>
<section class="cards">{cards}</section>
<p class="note">같은 폴더에 <code>graph1_waveforms.png</code>, <code>graph2_overlay.png</code>,
 <code>spike_intervals.csv</code>, <code>spike_report.json</code>, <code>report.pdf</code> 가 함께 있습니다.</p>
</div>'''
    out.write_text(page)
    return out


def write_pdf(html_path: Path, out: Path) -> Path | None:
    exe = next((shutil.which(c) for c in CHROME if shutil.which(c)), None)
    if not exe:
        return None
    r = subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-sandbox",
                        "--no-pdf-header-footer", f"--print-to-pdf={out}",
                        "--virtual-time-budget=10000", html_path.resolve().as_uri()],
                       capture_output=True, text=True, timeout=180)
    return out if out.exists() else None
