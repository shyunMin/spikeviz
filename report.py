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
CSS = """:root{--bg:#f2f4f6;--panel:#fff;--line:#dfe3e8;--ink:#12171c;--ink2:#48525e;--ink3:#77828f;
 --shadow:0 1px 2px rgba(18,23,28,.05),0 6px 22px -14px rgba(18,23,28,.28);--chip:#eaeef3;--figbg:#fcfcfb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#12161a;--panel:#191e24;--line:#2a323b;
 --ink:#eef2f6;--ink2:#aab5c0;--ink3:#7d8894;--shadow:0 1px 2px rgba(0,0,0,.4);--chip:#232b33;--figbg:#f4f4f2}}
:root[data-theme="dark"]{--bg:#12161a;--panel:#191e24;--line:#2a323b;--ink:#eef2f6;--ink2:#aab5c0;
 --ink3:#7d8894;--shadow:0 1px 2px rgba(0,0,0,.4);--chip:#232b33;--figbg:#f4f4f2}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans KR",system-ui,sans-serif;
 font-size:15.5px;line-height:1.65;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:56px 24px 96px;display:flex;flex-direction:column;gap:38px}
header.top{display:flex;flex-direction:column;gap:10px;border-bottom:1px solid var(--line);padding-bottom:24px}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink3);margin:0}
h1{font-size:31px;font-weight:600;letter-spacing:-.015em;margin:0;text-wrap:balance}
h2{font-size:19px;font-weight:600;margin:0 0 2px} h3{font-size:16.5px;font-weight:600;margin:0;font-family:"IBM Plex Mono",monospace}
p{margin:0} .sub{color:var(--ink2);max-width:64ch}
.rule{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}
.rule div{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:13px 15px;box-shadow:var(--shadow)}
.rule b{display:block;font-family:"IBM Plex Mono",monospace;font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink3);font-weight:500;margin-bottom:4px}
.rule span{font-size:14px;color:var(--ink2)}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:var(--shadow)}
figure .cap{padding:16px 20px 4px;display:flex;flex-direction:column;gap:3px}
figure .cap p{color:var(--ink2);font-size:14px}
figure .imgbox{overflow-x:auto;padding:8px 12px 14px}
figure img{display:block;width:100%;min-width:760px;background:var(--figbg);border-radius:6px}
.cards{display:flex;flex-direction:column;gap:16px}
.card{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--c);border-radius:10px;
 padding:18px 20px 20px;box-shadow:var(--shadow);display:flex;flex-direction:column;gap:12px}
.card header{display:flex;flex-direction:column;gap:3px} .card h3{color:var(--c)}
.meta{font-size:13.5px;color:var(--ink3)}
.stats{display:flex;flex-wrap:wrap;gap:8px 22px;font-size:14px;color:var(--ink2);font-variant-numeric:tabular-nums}
.stats b{color:var(--ink);font-family:"IBM Plex Mono",monospace;font-weight:500;font-size:16px}
.verdict{display:flex;flex-wrap:wrap;align-items:center;gap:8px 18px;font-size:13.5px;
 color:var(--ink2);font-variant-numeric:tabular-nums;padding:9px 12px;border-radius:9px;
 background:var(--vbg);border:1px solid var(--vline)}
.verdict b{color:var(--ink);font-family:"IBM Plex Mono",monospace;font-weight:500;font-size:15px}
.verdict i{font-style:normal;color:var(--ink3);font-size:12.5px}
.badge{font-size:12.5px;font-weight:500;padding:2px 10px;border-radius:20px;
 background:var(--vink);color:var(--panel);white-space:nowrap}
.vok{--vbg:color-mix(in oklab,#0f8f5f 8%,var(--panel));--vline:color-mix(in oklab,#0f8f5f 28%,var(--panel));--vink:#0f8f5f}
.vwarn{--vbg:color-mix(in oklab,#b07d10 9%,var(--panel));--vline:color-mix(in oklab,#b07d10 28%,var(--panel));--vink:#b07d10}
.voff{--vbg:var(--chip);--vline:var(--line);--vink:var(--ink3)}
.bok{background:#0f8f5f;color:#fff} .bwarn{background:#b07d10;color:#fff}
.boff{background:var(--chip);color:var(--ink2)}
.bman{background:#3d5a80;color:#fff}
.manual{font-size:12.5px;color:#3d5a80;font-weight:500}
.guide{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px 22px;
 box-shadow:var(--shadow);display:flex;flex-direction:column;gap:12px}
.guide dl{margin:0;display:grid;grid-template-columns:minmax(130px,auto) 1fr;gap:10px 18px;font-size:14px}
.guide dt{font-weight:500;color:var(--ink)}
.guide dd{margin:0;color:var(--ink2)}
.stats .dom{background:var(--chip);border-radius:7px;padding:2px 10px;color:var(--ink)}
.stats .dom i{font-style:normal;color:var(--ink3);font-size:12.5px}
.stats .dom.none{color:var(--ink3)}
table i{font-style:normal;color:var(--ink3);font-size:12px}
.lbl{font-size:12px;letter-spacing:.06em;color:var(--ink3);text-transform:uppercase;font-family:"IBM Plex Mono",monospace}
.chips{display:flex;flex-wrap:wrap;gap:5px}
.iv{font-family:"IBM Plex Mono",monospace;font-size:12.5px;padding:3px 8px;border-radius:5px;background:var(--chip);color:var(--ink2);font-variant-numeric:tabular-nums}
.none{font-size:13px;color:var(--ink3)}
.times{display:flex;flex-wrap:wrap;gap:4px 10px;font-family:"IBM Plex Mono",monospace;font-size:12.5px;color:var(--ink2)}
.times .t i{font-style:normal;color:var(--ink3);margin-left:3px;font-size:11px}
.tablewrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums;min-width:660px}
th,td{text-align:right;padding:11px 16px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left;font-family:"IBM Plex Mono",monospace}
th{font-size:11.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);font-weight:500}
tbody tr:last-child td{border-bottom:none}
.dot{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:8px;vertical-align:1px}
.find{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:20px 22px;box-shadow:var(--shadow);display:flex;flex-direction:column;gap:9px}
.find ul{margin:0;padding-left:19px;display:flex;flex-direction:column;gap:7px;color:var(--ink2)}
.find b{color:var(--ink);font-weight:500}
.note{font-size:13.5px;color:var(--ink3);max-width:70ch}
code{font-family:"IBM Plex Mono",monospace;font-size:.92em;background:var(--chip);padding:1px 5px;border-radius:4px}
@media print{
 :root{--bg:#fff;--panel:#fff;--line:#d7dbe0;--shadow:none;--figbg:#fff}
 @page{size:A4;margin:14mm 12mm} body{background:#fff;font-size:10.5px}
 .wrap{max-width:none;padding:0;gap:20px} h1{font-size:23px} h2{font-size:15px} h3{font-size:13px}
 figure,.card,.find,.guide,.tablewrap,.rule div{break-inside:avoid;box-shadow:none}
 figure img{min-width:0;max-height:236mm;width:auto;max-width:100%;margin:0 auto} .imgbox,.tablewrap{overflow:visible}
 table{min-width:0;font-size:9.5px} th,td{padding:7px 9px}
 *{-webkit-print-color-adjust:exact;print-color-adjust:exact}
}"""

CHROME = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome")


BEFORE_C, AFTER_C = "#8a8985", "#e34948"   # 기준 시점 이전 / 이후


def pick_baseline(results: list[dict], baseline: str | None) -> str:
    """기준 파일 이름. 지정이 없으면 가장 오래된 '규칙적' 파일."""
    names = [r["name"] for r in results]
    if baseline in names:
        return baseline
    reg = [r["name"] for r in results
           if r["regularity"]["verdict"] in ("규칙적", "약한 규칙성")]
    return reg[0] if reg else names[0]


def colors_for(results: list[dict], baseline: str | None = None) -> list[str]:
    """기준 파일보다 앞선 녹음은 회색, 기준 파일부터는 빨강. 이름이 곧 시간순이다."""
    cut = pick_baseline(results, baseline)
    return [AFTER_C if r["name"] >= cut else BEFORE_C for r in results]


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


def graph1(results: list[dict], out: Path, baseline: str | None = None,
           per_page: int = 8) -> list[Path]:
    """파일별 파형. 한 장에 최대 per_page개씩 잘라 여러 장으로 만든다."""
    cols = colors_for(results, baseline)
    pages = [(results[i:i + per_page], cols[i:i + per_page], i)
             for i in range(0, len(results), per_page)]
    fmt = FuncFormatter(lambda v, _: mmss(v))
    out_paths = []
    for pno, (chunk, ccs, off) in enumerate(pages, 1):
        n = len(chunk)
        h = 1.1 + 2.5 * n
        fig, axes = plt.subplots(n, 1, figsize=(13, h), facecolor=SURF, squeeze=False)
        axes = axes[:, 0]
        fig.subplots_adjust(hspace=.55, top=1 - .78 / h, bottom=.7 / h, left=.085, right=.98)
        for ax, r, c in zip(axes, chunk, ccs):
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
            how = ("MANUAL count" if r.get("source") == "manual"
                   else f"threshold {r['thr']:.0f}x noise floor")
            ax.set_title(f"{r['filename']}   |   analysed {mmss(r['t0'])}-{mmss(r['t1'])}"
                         f"   |   {how}   |   {len(r['times'])} spikes (v)",
                         loc="left", fontsize=10.5, color=INK, pad=7)
            ax.set_ylabel("amplitude", fontsize=8.5, color=INK2)
        axes[-1].set_xlabel("time in file (m:ss)", fontsize=9, color=INK2)
        part = f"  ({pno}/{len(pages)}: files {off + 1}-{off + n} of {len(results)})" if len(pages) > 1 else ""
        fig.suptitle(f"Graph 1  -  Waveform per recording{part}",
                     x=.085, ha="left", fontsize=14, color=INK, y=1 - .3 / h)
        path = out if len(pages) == 1 else out.with_name(f"{out.stem}_{pno}{out.suffix}")
        fig.savefig(path, dpi=150, facecolor=SURF)
        plt.close(fig)
        out_paths.append(path)
    return out_paths


def _series(r: dict, normalise: bool, bin_n: int = 10):
    """기준점 이후의 엔벌로프를 표시용으로 100ms 단위 max-hold로 줄인다."""
    k = int(round(r["anchor"] / r["dt"]))
    e = r["env"][k:] / (r["floor"] if normalise else 1.0)
    m = len(e) // bin_n * bin_n
    if m == 0:
        return None, None
    e = e[:m].reshape(-1, bin_n).max(1)
    return np.arange(len(e)) * r["dt"] * bin_n, e


def _spike_marks(r: dict, normalise: bool, lift: float = 1.45):
    """검출된 스파이크의 (기준점 대비 시각, 표시 높이). 하나도 빠뜨리지 않는다.

    봉우리 끝에 정확히 겹치면 선에 묻히므로 조금 위에 찍는다.
    """
    scale = 1.0 if normalise else r["floor"]
    xs = [t - r["anchor"] for t in r["times"]]
    ys = [ratio * scale * lift for ratio in r["ratios"]]
    return xs, ys


def graph2(results: list[dict], out: Path, baseline: str | None = None) -> Path:
    n = len(results)
    cols = colors_for(results, baseline)
    names = [r["name"] for r in results]
    base_i = names.index(pick_baseline(results, baseline))
    # 위 패널: 정규화하지 않은 원래 크기 비교 — 기준 파일과 최신 1개만
    raw_idx = sorted({base_i, n - 1})

    h = 7.6 + .40 * n
    fig = plt.figure(figsize=(13, h), facecolor=SURF)
    lab = .055 + .006 * max(len(r["name"]) for r in results)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.0, 3.0, max(.8, .26 * n)], hspace=.34,
                          top=1 - 1.2 / h, bottom=.62 / h, left=min(lab, .16), right=.985)
    axr_, ax, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    span = 0.0

    # --- 패널 1: 원래 크기 (정규화 없음) ---
    ymin_raw, ymax_raw = 1.0, 0.0
    for i in raw_idx:
        r = results[i]
        c = cols[i]
        tr, e = _series(r, normalise=False)
        if tr is None:
            continue
        span = max(span, tr[-1])
        ymax_raw = max(ymax_raw, e.max() * 2.2)
        ymin_raw = min(ymin_raw, r["floor"] * .6)
        axr_.plot(tr, e, color=c, lw=1.0, alpha=.85,
                  label=f"{r['name']}" + ("  (baseline)" if i == base_i else "")
                        + ("  [manual]" if r.get("source") == "manual" else ""))
        xs, ys = _spike_marks(r, normalise=False)
        axr_.plot(xs, ys, "v", ms=5.5, color=c, mec=SURF, mew=.5, alpha=.95)
    axr_.set_yscale("log")
    axr_.set_ylim(max(ymin_raw, 1e-5), max(ymax_raw, 1e-4))
    axr_.tick_params(labelbottom=False)
    axr_.legend(frameon=False, fontsize=8.8, labelcolor=INK2, ncols=2, loc="lower left",
                bbox_to_anchor=(0, 1.005, 1, .08), mode="expand", borderaxespad=0, handlelength=1.6)
    axr_.set_title("", loc="left")

    # --- 패널 2: 자기 잡음 대비 정규화 (전체 파일) ---
    ymax = 10.0
    for i, r in enumerate(results):
        c = cols[i]
        tr, e = _series(r, normalise=True)
        if tr is None:
            continue
        span, ymax = max(span, tr[-1]), max(ymax, e.max() * 2.0)
        ax.fill_between(tr, 1, e, color=c, lw=0, alpha=.18)
        kind = "first spike" if r["anchor_kind"] == "first_spike" else "max amplitude, no spike"
        tag = " manual" if r.get("source") == "manual" else ""
        ax.plot(tr, e, color=c, lw=.9, alpha=.8,
                label=f"{r['name']}  ({kind} @ {mmss(r['abs_anchor'])}, {len(r['times'])}{tag} spikes)")
        xs, ys = _spike_marks(r, normalise=True)
        ax.plot(xs, ys, "v", ms=4.5, color=c, mec=SURF, mew=.4, alpha=.95)
    ax.set_yscale("log")
    ax.set_ylim(1, ymax)
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(["1x", "10x", "100x"])
    ax.set_ylabel("peak envelope / own noise floor", fontsize=9, color=INK2)
    ax.tick_params(labelbottom=False)
    ax.legend(frameon=False, fontsize=8.2, labelcolor=INK2, ncols=min(3, max(1, n)),
              loc="lower left", bbox_to_anchor=(0, 1.005, 1, .1), mode="expand",
              borderaxespad=0, handlelength=1.6, columnspacing=1.2)

    # --- 패널 3: 스파이크 시각 ---
    for i, r in enumerate(results):
        c, y = cols[i], n - 1 - i
        sp = [t - r["anchor"] for t in r["times"]]
        axr.plot([0, r["t1"] - r["t0"] - r["anchor"]], [y, y], color=c, lw=.8, alpha=.28)
        axr.plot(sp, [y] * len(sp), "|", ms=13, mew=2, color=c)
        axr.text(-span * .006, y, f"{r['name']} ({len(sp)})", ha="right", va="center",
                 fontsize=8.5, color=INK2)
    for a in (axr_, ax, axr):
        _strip(a)
        a.set_xlim(0, span or 1)
    axr.set_yticks([])
    axr.set_ylim(-.6, n - .4)
    axr.set_xlabel("time relative to each file's first spike (s)", fontsize=9, color=INK2)
    axr.set_title("spike times, same relative axis (all detected spikes)",
                  loc="left", fontsize=9.5, color=INK2, pad=4)
    fig.suptitle("Graph 2  -  aligned on each file's first spike (t = 0)   ·   "
                 "grey = before baseline, red = baseline onward",
                 x=min(lab, .16), ha="left", fontsize=13.5, color=INK, y=1 - .32 / h)
    axr_.set_ylabel("peak amplitude\n(raw, baseline vs newest)", fontsize=9, color=INK2)
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


def _period_groups(results: list[dict], tol: float = .15) -> list[list[dict]]:
    """검출된 주기가 서로 ±tol 안에 드는 파일끼리 묶는다(가까운 것부터 연결)."""
    have = sorted((r for r in results if r["regularity"]["period"]),
                  key=lambda r: r["regularity"]["period"])
    groups: list[list[dict]] = []
    for r in have:
        if groups and r["regularity"]["period"] <= groups[-1][-1]["regularity"]["period"] * (1 + tol):
            groups[-1].append(r)
        else:
            groups.append([r])
    return groups


def observations(results: list[dict], baseline: str | None = None) -> list[str]:
    """데이터에서 바로 계산되는 관찰만 만든다."""
    notes = []
    by = lambda v: [r for r in results if r["regularity"]["verdict"] == v]
    reg, weak, irr, unk = by("규칙적"), by("약한 규칙성"), by("불규칙"), by("판단 불가") + by("표본 부족")
    name = lambda rs: ", ".join(html.escape(r["name"]) for r in rs)

    if reg or weak:
        notes.append(f"<b>규칙적으로 판정된 파일 {len(reg) + len(weak)}개 / 전체 {len(results)}개.</b> " +
                     (f"규칙적(p≤0.01): {name(reg)}. " if reg else "") +
                     (f"약한 규칙성(p≤0.05): {name(weak)}. " if weak else ""))
    if irr:
        notes.append(f"<b>불규칙으로 판정된 파일:</b> {name(irr)} — 스파이크가 충분히 많은데도 "
                     "무작위로 뿌린 것과 구별되지 않습니다.")
    if unk:
        detail = ", ".join(f"{html.escape(r['name'])}(스파이크 {r['regularity']['n']}개, "
                           f"R {r['regularity']['R']:.2f} &lt; 필요값 {r['regularity']['r_needed']:.2f})"
                           if r["regularity"]["R"] else html.escape(r["name"]) for r in unk)
        notes.append(f"<b>판단할 수 없는 파일:</b> {detail} — 규칙적이 <i>아니라</i>는 뜻이 아니라, "
                     "스파이크가 적어 규칙적이어도 잡아낼 수 없다는 뜻입니다.")

    groups = _period_groups(results)
    big = max(groups, key=len) if groups else []
    if len(big) >= 2:
        lo = min(r["regularity"]["period"] for r in big)
        hi = max(r["regularity"]["period"] for r in big)
        notes.append(f"<b>{len(big)}개 파일의 주기가 같은 계열입니다.</b> " +
                     ", ".join(f"{html.escape(r['name'])} {r['regularity']['period']:.1f}초" for r in big) +
                     f" — {lo:.1f}–{hi:.1f}초 안에 모입니다.")

    skipped = [r for r in results if (r["regularity"]["skip_ratio"] or 0) >= 1.5]
    if skipped:
        notes.append("<b>주기를 건너뛰는 파일이 있습니다.</b> " +
                     ", ".join(f"{html.escape(r['name'])}(주기 {r['regularity']['period']:.1f}초, "
                               f"실제 간격은 그 {r['regularity']['skip_ratio']:.1f}배)" for r in skipped) +
                     " — 소리가 나는 자리는 일정한 격자 위에 있지만 상당수 주기는 비어 있습니다. "
                     "간격만 보면 격자 간격이 아니라 '보통 몇 초 만에 한 번'이 보입니다.")
    dense = [r for r in results if r["regularity"]["skip_ratio"] and r["regularity"]["skip_ratio"] <= .7]
    if dense:
        notes.append("<b>주기 사이에 부가 스파이크가 끼는 파일이 있습니다.</b> " +
                     ", ".join(f"{html.escape(r['name'])}(주기 {r['regularity']['period']:.1f}초, "
                               f"간격 중앙값 {r['stats']['median']:.1f}초)" for r in dense) +
                     " — 주기적인 스파이크 사이에 다른 소리가 섞여 간격 중앙값이 주기보다 짧게 나옵니다.")

    man = [r for r in results if r.get("source") == "manual"]
    if man:
        notes.append("<b>수동 측정을 쓴 파일:</b> " +
                     ", ".join(f"{html.escape(r['name'])}({len(r['times'])}개"
                               + (f"/기록 {r['manual_total']}개" if r.get("manual_total")
                                  and r["manual_total"] != len(r["times"]) else "") + ")"
                               for r in man) +
                     " — 자동 검출 대신 귀로 센 기록을 그대로 썼습니다. 나머지 파일은 자동 검출입니다.")
    weakdet = [r for r in results if r["times"] and r["max_ratio"] < 20 and r.get("source") != "manual"]
    if weakdet:
        notes.append("<b>검출 임계에 겨우 걸친 파일이 있습니다.</b> " +
                     ", ".join(f"{html.escape(r['name'])}(최대 {r['max_ratio']:.0f}×)" for r in weakdet) +
                     " — 스파이크가 잡음 대비 크게 튀지 않아 검출 자체의 신뢰도가 낮습니다.")
    none = [r for r in results if not r["times"]]
    if none:
        notes.append("<b>스파이크가 하나도 잡히지 않은 파일:</b> " + name(none) +
                     ". 그래프 2에서는 최대 진폭 지점을 기준점으로 썼습니다.")
    whole = [r for r in results if r["whole_file"]]
    if whole and len(whole) < len(results):
        notes.append("<b>구간이 다른 파일이 섞여 있습니다.</b> " + name(whole) +
                     "은(는) 길이가 짧아 앞부분을 자르지 않고 전체를 썼습니다. 다른 파일에서 잘라낸 "
                     "구간이 여기에는 포함되므로 직접 비교할 때 주의해야 합니다.")
    base = pick_baseline(results, baseline)
    before = [r for r in results if r["name"] < base]
    notes.append(f"<b>색은 기준 파일 {html.escape(base)}을 경계로 둘로 나눴습니다.</b> "
                 f"그 이전 {len(before)}개는 회색, 기준 파일부터 {len(results) - len(before)}개는 "
                 f"빨강입니다. 개별 파일 구분은 이름표로 하세요.")
    return notes


def _card(r: dict, c: str) -> str:
    s = r["stats"]
    chips = "".join(f'<span class="iv">{v:.1f}</span>' for v in r["intervals"]) \
        or '<span class="none">간격 없음</span>'
    g = r["regularity"]
    d = r["dominant"]
    dom = (f'<span class="dom">우세 간격 <b>{d["period"]:.1f}</b>초 '
           f'<i>{d["n"]}/{s["n"]}개</i></span>') if d and s else ''
    stat = (f'{dom}<span><b>{s["median"]:.1f}</b>초 중앙값</span><span><b>{s["mean"]:.1f}</b>초 평균</span>'
            f'<span>{s["min"]:.1f}–{s["max"]:.1f}초 범위</span><span>σ {s["std"]:.1f}초</span>'
            f'<span>변동계수 {s["cv"]:.2f}</span>') if s else '<span>간격 통계 없음</span>'
    times = " ".join(f'<span class="t">{mmss(t)}<i>{x:.0f}×</i></span>'
                     for t, x in zip(r["abs_times"], r["ratios"]))
    anchor = "첫 스파이크" if r["anchor_kind"] == "first_spike" else "최대 진폭(스파이크 없음)"
    vclass = {"규칙적": "ok", "약한 규칙성": "warn"}.get(g["verdict"], "off")
    is_manual = r.get("source") == "manual"
    how = "스파이크 수동 측정" if is_manual else f"임계 {r['thr']:.0f}×"
    extra = (f" (기록 {r['manual_total']}개 중 분석 구간 안 {len(r['times'])}개)"
             if r.get("manual_total") else "")
    manual_tag = (f'<p class="manual">스파이크 수동 측정 — 귀로 센 기록을 그대로 씁니다{extra}</p>'
                  if is_manual else "")

    if g["p"] is None:
        vmetrics = '<span>스파이크가 5개 미만이라 규칙성을 계산하지 않았습니다.</span>'
    else:
        per = (f'<span>주기 <b>{g["period"]:.1f}</b>초</span>') if g["period"] else ''
        vmetrics = (f'{per}<span>p <b>{g["p"]:.3f}</b></span>'
                    f'<span>CV <b>{g["cv"]:.2f}</b></span>'
                    f'<span>위상 집중도 R <b>{g["R"]:.2f}</b> <i>(이 표본에서 필요한 값 '
                    f'{g["r_needed"]:.2f})</i></span>')
    return f'''<article class="card" style="--c:{c}">
  <header><h3>{html.escape(r["filename"])}</h3>
    <p class="meta">분석 구간 {mmss(r['t0'])}–{mmss(r['t1'])} · {how} · 최대 {r['max_ratio']:.0f}×
      · 스파이크 {len(r['times'])}개 · 기준점 {mmss(r['abs_anchor'])} ({anchor})</p>
    {manual_tag}</header>
  <div class="verdict v{vclass}">
    <span class="badge">{g["verdict"]}</span>
    {vmetrics}
  </div>
  <div class="stats">{stat}</div>
  <p class="lbl">스파이크 간격 (초, 앞 스파이크로부터)</p>
  <div class="chips">{chips}</div>
  <p class="lbl">스파이크 발생 시각 · 잡음바닥 대비 배율</p>
  <div class="times">{times or '<span class="none">검출 없음</span>'}</div>
</article>'''


def write_html(results: list[dict], params: dict, run_id: str, g1, g2: Path, out: Path) -> Path:
    b64 = lambda p: base64.b64encode(p.read_bytes()).decode()
    g1_list = list(g1) if isinstance(g1, (list, tuple)) else [g1]
    baseline = params.get("baseline") or None
    cols = colors_for(results, baseline)
    rows = ""
    for i, r in enumerate(results):
        s = r["stats"]
        g = r["regularity"]
        vclass = {"규칙적": "ok", "약한 규칙성": "warn"}.get(g["verdict"], "off")
        num = lambda v, f="{:.2f}": f.format(v) if v is not None else "–"
        cells = (f'<td><span class="badge b{vclass}">{g["verdict"]}</span></td>'
                 f'<td>{num(g["p"], "{:.3f}")}</td><td>{num(g["cv"])}</td>'
                 f'<td>{num(g["R"])} <i>/ {num(g["r_needed"])}</i></td>'
                 f'<td>{num(g["period"], "{:.1f}")}</td>'
                 f'<td>{num(s["median"], "{:.1f}") if s else "–"}</td>')
        how = ('<span class="badge bman">수동</span>' if r.get("source") == "manual"
               else f'{r["thr"]:.0f}×')
        rows += (f'<tr><td><span class="dot" style="background:{cols[i]}"></span>'
                 f'{html.escape(r["name"])}</td><td>{len(r["times"])}</td><td>{how}</td>'
                 f'{cells}<td>{mmss(r["abs_anchor"])}</td></tr>')
    cards = "\n".join(_card(r, cols[i]) for i, r in enumerate(results))
    notes = "".join(f"<li>{t}</li>" for t in observations(results, baseline)) or "<li>관찰할 항목이 없습니다.</li>"
    g1_imgs = "\n  ".join(
        f'<div class="imgbox"><img src="data:image/png;base64,{b64(g)}" alt="파일별 파형과 스파이크 표시"></div>'
        for g in g1_list)
    p = params
    n_sur = next((r["regularity"]["n_sur"] for r in results if r["regularity"]["n_sur"]), 300)
    n_sur1, pmin = n_sur + 1, 1 / (n_sur + 1)
    n_man = sum(1 for r in results if r.get("source") == "manual")
    manual_note = (f" 단, 수동 측정 기록이 있는 {n_man}개 파일은 그 기록을 그대로 씁니다."
                   if n_man else "")
    win = (f"{p['skip_s']:.0f}초 이후 최대 {p['max_s'] / 60:.0f}분. 앞을 자르면 남는 길이가 "
           f"{p['min_keep_s']:.0f}초 미만인 파일은 전체 사용.")
    thr_rule = (f"잡음 바닥 대비 max({p['k_abs']:.0f}×, min({p.get('k_cap', 20):.0f}×, "
                f"{p['k_rel']:.2f} × p90)) 초과, {p['refractory_s']:.0f}초 이내 병합. "
                f"상한 {p.get('k_cap', 20):.0f}×는 큰 스파이크 몇 개 때문에 기준이 올라가 "
                f"나머지가 빠지는 것을 막습니다.")
    page = f'''<title>Spike Interval Report</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+KR:wght@400;500;600&display=swap">
<style>
{CSS}
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
  <div><b>스파이크 판정</b><span>{thr_rule}{manual_note}</span></div>
  <div><b>그래프 2 정렬</b><span>각 파일의 첫 스파이크를 t = 0으로 두고 상대 시간으로 겹쳐 그림.</span></div>
  <div><b>규칙성 판정</b><span>최적 주기에서의 위상 집중도를, 같은 개수를 같은 구간에 무작위로 뿌린
    {n_sur}번과 비교해 p값을 냅니다. p≤0.01 규칙적 · p≤0.05 약한 규칙성 · 스파이크 12개 미만이면 판단 불가.</span></div>
</section>
<figure>
  <div class="cap"><h2>그래프 1 — 파일별 파형</h2>
  <p>가로축은 파일 내 시간(m:ss), 세로축은 진폭. 검은 ▼ 가 검출된 스파이크입니다. 세로 눈금은 파일마다 다릅니다.
     한 장에 최대 8개씩 나눠 그렸습니다. <b>회색은 기준 파일 이전, 빨강은 기준 파일부터</b>입니다.</p></div>
  {g1_imgs}
</figure>
<figure>
  <div class="cap"><h2>그래프 2 — 첫 스파이크 기준 겹쳐 보기</h2>
  <p>세 칸 모두 각 파일의 첫 스파이크를 t = 0에 맞춘 상대 시간축입니다.
     <b>위</b>: 정규화하지 않은 원래 크기 — 기준 파일과 최신 1개만 겹쳤습니다.
     <b>가운데</b>: 자기 잡음 바닥 대비 배율(로그)로 정규화한 전 파일 비교. 녹음 레벨 차이가 커서
     진폭 그대로 겹치면 작은 파일이 묻히기 때문입니다.
     두 칸 모두 <b>▼ 가 검출된 스파이크 전부</b>입니다. <b>아래</b>: 같은 축 위의 스파이크 시각.
     색은 <b>회색 = 기준 파일 이전, 빨강 = 기준 파일부터</b>입니다.</p></div>
  <div class="imgbox"><img src="data:image/png;base64,{b64(g2)}" alt="첫 스파이크에 정렬해 겹친 엔벌로프"></div>
</figure>
<section class="guide"><h2>지표 읽는 법</h2>
  <p class="note">이 리포트가 답하려는 질문은 "주기가 몇 초인가"가 아니라
     <b>"이 녹음이 규칙적인가, 아니면 우연인가"</b>입니다. 아래 세 지표가 그 판단의 근거입니다.</p>
  <dl>
    <dt>CV (변동계수)</dt>
    <dd>간격의 표준편차 ÷ 평균. 시각이 완전히 무작위(포아송)면 1 부근, 규칙적일수록 0에 가깝습니다.
        간격만 보는 지표라 스파이크를 놓치거나 더 잡으면 흔들립니다.</dd>
    <dt>위상 집중도 R</dt>
    <dd>최적 주기를 찾아, 스파이크들이 그 주기의 <b>같은 위상</b>에 얼마나 모이는지를 0~1로 잰 값
        (모든 스파이크가 정확히 같은 위상이면 1). 스파이크를 하나 놓쳐 간격이 두 배가 되어도 위상은
        유지되므로, 검출 임계에 덜 흔들립니다.</dd>
    <dt>필요값 / p값</dt>
    <dd>같은 개수의 스파이크를 같은 구간에 <b>무작위로 뿌린 {n_sur}번</b>과 비교합니다.
        '필요값'은 그 무작위 시행의 상위 5% 지점 — 관측된 R이 이 값을 넘지 못하면 우연과 구별되지 않습니다.
        p값은 무작위 시행이 관측값 이상을 낸 비율이고, 최솟값은 1/{n_sur1}={pmin:.3f}입니다.</dd>
    <dt>판정</dt>
    <dd><span class="badge bok">규칙적</span> p ≤ 0.01 ·
        <span class="badge bwarn">약한 규칙성</span> p ≤ 0.05 ·
        <span class="badge boff">불규칙</span> 그 이상 ·
        <span class="badge boff">판단 불가</span> 스파이크가 12개 미만이라 규칙적이어도 검출할 힘이 없는 경우.
        '판단 불가'는 규칙적이 아니라는 뜻이 <b>아닙니다</b>.</dd>
    <dt>우세 간격</dt>
    <dd>간격 중 ±25% 안에 가장 많이 모이는 무리의 대표값. 참고용으로만 남겨둡니다 —
        무작위 스파이크열에서도 10~35% 확률로 값이 나오기 때문에 규칙성의 근거로는 쓸 수 없습니다.</dd>
  </dl>
</section>
<section class="find"><h2>자동 관찰</h2><ul>{notes}</ul></section>
<section class="tablewrap"><table>
  <thead><tr><th>파일</th><th>스파이크</th><th>임계</th><th>규칙성 판정</th><th>p값</th><th>CV</th><th>R / 필요값</th><th>주기(초)</th><th>간격 중앙값(초)</th><th>첫 스파이크</th></tr></thead>
  <tbody>{rows}</tbody></table></section>
<section class="cards">{cards}</section>
<p class="note">같은 폴더에 <code>graph1_waveforms*.png</code>, <code>graph2_overlay.png</code>,
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


# ---------------------------------------------------------------- 기능 2 (긴 파일)

def graph_long_timeline(r: dict, out: Path) -> Path:
    """전체 시간축: 엔벌로프, 국소 잡음 바닥, 살아남은 스파이크."""
    fig, ax = plt.subplots(figsize=(13, 4.4), facecolor=SURF)
    fig.subplots_adjust(top=.86, bottom=.13, left=.075, right=.985)
    h = lambda v: (r["t0"] + v) / 3600
    _strip(ax)
    ax.fill_between([h(v) for v in r["t_plot"]], 1e-6, r["env_plot"], color=SLOTS[0], lw=0, alpha=.45)
    ax.plot([h(v) for v in r["t_plot"]], r["floor_plot"], color=INK, lw=1.1, alpha=.75,
            label="local noise floor (10 s median)")
    top = max(r["env_plot"].max(), 1e-5)
    ax.plot([h(t) for t in r["times"]], [top * 1.35] * len(r["times"]), marker="v", ls="none",
            ms=4, color=SLOTS[1], mec="none", clip_on=False, label=f"kept spikes ({len(r['times'])})")
    ax.set_yscale("log")
    ax.set_ylim(max(r["floor_plot"].min() * .3, 1e-6), top * 1.8)
    ax.set_xlim(h(0), h(r["span"]))
    ax.set_ylabel("peak amplitude (log)", fontsize=9, color=INK2)
    ax.set_xlabel("time in file (hours)", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left",
              bbox_to_anchor=(0, 1.005, 1, .08), mode="expand", ncols=2, borderaxespad=0, handlelength=1.6)
    fig.suptitle(f"Long-form 1  -  {r['filename']}: envelope, local noise floor, kept spikes",
                 x=.075, ha="left", fontsize=13.5, color=INK, y=.965)
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)
    return out


def graph_long_trend(r: dict, out: Path) -> Path:
    """간격의 시간 변화와 구간별 빈도."""
    t = np.asarray(r["times"], dtype=float)
    iv = np.asarray(r["intervals"], dtype=float)
    tr, segs = r["trend"], r["segments"]
    fig = plt.figure(figsize=(13, 7.2), facecolor=SURF)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1], hspace=.28, top=.9, bottom=.08,
                          left=.075, right=.985)
    ax, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for a in (ax, axb):
        _strip(a)
        a.set_xlim(0, r["span"] / 3600)

    mid = (t[1:] + t[:-1]) / 2 / 3600
    ax.plot(mid, iv, "o", ms=3.2, color=SLOTS[0], alpha=.5, mec="none", label="interval")
    if len(iv) >= 9:                                   # 이동 중앙값
        w = max(5, len(iv) // 20 | 1)
        roll = np.array([np.median(iv[max(0, i - w // 2):i + w // 2 + 1]) for i in range(len(iv))])
        ax.plot(mid, roll, color=SLOTS[3], lw=1.8, label=f"rolling median ({w})")
    if tr["slope_per_hour"] is not None:
        x = np.array([0, r["span"] / 3600])
        y = np.median(iv) + tr["slope_per_hour"] * (x - np.median(mid))
        ax.plot(x, y, color=SLOTS[1], lw=2, ls=(0, (5, 3)),
                label=f"trend {tr['slope_per_hour']:+.2f} s/hour")
    ax.set_ylabel("interval between spikes (s)", fontsize=9, color=INK2)
    ax.set_ylim(0, np.percentile(iv, 99) * 1.25 if len(iv) else 1)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left",
              bbox_to_anchor=(0, 1.005, 1, .08), mode="expand", ncols=3, borderaxespad=0, handlelength=1.8)

    xs = [(s["t0"] + s["t1"]) / 2 / 3600 for s in segs]
    w = (segs[0]["t1"] - segs[0]["t0"]) / 3600 * .82
    axb.bar(xs, [s["rate_per_min"] for s in segs], width=w, color=SLOTS[2], alpha=.85)
    axb.set_ylabel("spikes per minute", fontsize=9, color=INK2)
    axb.set_xlabel("time in file (hours)", fontsize=9, color=INK2)
    axb.set_title("rate per segment", loc="left", fontsize=9.5, color=INK2, pad=4)
    fig.suptitle(f"Long-form 2  -  {r['filename']}: interval trend and rate over time",
                 x=.075, ha="left", fontsize=13.5, color=INK, y=.975)
    fig.savefig(out, dpi=150, facecolor=SURF)
    plt.close(fig)
    return out


def long_observations(r: dict) -> list[str]:
    notes = []
    tr, g = r["trend"], r["regularity"]
    if tr["verdict"] == "일정":
        notes.append(f"<b>빈도는 일정합니다.</b> 간격 추세의 기울기가 "
                     f"{tr['slope_per_hour']:+.2f}초/시간이고 p={tr['p']:.3f}로, 시간에 따른 변화를 "
                     f"우연과 구별할 수 없습니다.")
    elif tr["verdict"] != "표본 부족":
        notes.append(f"<b>빈도가 {tr['verdict']}.</b> 간격이 시간당 {tr['slope_per_hour']:+.2f}초씩 "
                     f"변합니다(p={tr['p']:.1e}). 처음 10% 구간 평균 {tr['first']:.1f}초 → "
                     f"마지막 10% {tr['last']:.1f}초, {tr['change_pct']:+.0f}%.")
    reg_segs = [s for s in r["segments"] if s["verdict"] in ("규칙적", "약한 규칙성")]
    if g["verdict"] not in ("규칙적", "약한 규칙성") and len(reg_segs) >= len(r["segments"]) / 2:
        notes.append(f"<b>전체로는 한 주기로 설명되지 않지만 구간별로는 규칙적입니다.</b> "
                     f"{len(r['segments'])}개 구간 중 {len(reg_segs)}개가 규칙적으로 판정됐습니다. "
                     f"주기가 시간에 따라 변하고 있다는 뜻입니다.")
    elif g["verdict"] in ("규칙적", "약한 규칙성"):
        notes.append(f"<b>전체 구간이 하나의 주기로 설명됩니다.</b> 주기 {g['period']:.1f}초, "
                     f"위상 집중도 {g['R']:.2f}(필요값 {g['r_needed']:.2f}), p={g['p']:.3f}.")
    rej = r["rejected"]
    tot = sum(rej.values()) + len(r["times"])
    if sum(rej.values()):
        notes.append(f"<b>잡음으로 제외한 것이 {sum(rej.values())}건입니다.</b> 후보 {tot}건 중 "
                     f"지속시간 초과 {rej['지속시간']}건(말소리·음악·문 닫는 소리처럼 길게 이어지는 소리), "
                     f"주변 소음 {rej['주변소음']}건(앞뒤가 조용하지 않아 판단할 수 없는 것). "
                     f"잡음 구간에 겹친 진짜 스파이크도 함께 빠지므로 그 구간의 빈도는 낮게 나올 수 있습니다.")
    if r["floor_max"] / max(r["floor_min"], 1e-9) > 5:
        notes.append(f"<b>녹음 환경이 시간에 따라 크게 변했습니다.</b> 국소 잡음 바닥이 "
                     f"{r['floor_min']:.5f}에서 {r['floor_max']:.5f}까지 "
                     f"{r['floor_max'] / r['floor_min']:.0f}배 움직입니다. 전역 단일 임계값을 썼다면 "
                     f"조용한 구간에서는 거짓 검출이, 시끄러운 구간에서는 누락이 났을 것입니다.")
    quiet = [s for s in r["segments"] if s["n"] == 0]
    if quiet:
        notes.append(f"<b>스파이크가 하나도 없는 구간이 {len(quiet)}개 있습니다.</b> " +
                     ", ".join(f"{s['t0'] / 3600:.1f}–{s['t1'] / 3600:.1f}h" for s in quiet[:6]) +
                     ("…" if len(quiet) > 6 else "") + ".")
    return notes


def write_long_csv(r: dict, out: Path) -> Path:
    lines = ["kind,index,time_s,time_hms,peak_x_local_floor,interval_from_prev_s"]
    hms = lambda t: f"{int(t) // 3600}:{int(t) % 3600 // 60:02d}:{int(t) % 60:02d}"
    prev = None
    for i, (t, x) in enumerate(zip(r["times"], r["ratios"]), 1):
        at = r["t0"] + t
        lines.append(f"spike,{i},{at:.2f},{hms(at)},{x:.1f},{'' if prev is None else f'{t - prev:.2f}'}")
        prev = t
    lines.append("")
    lines.append("kind,index,start_s,end_s,spikes,rate_per_min,median_interval_s,verdict")
    for i, s in enumerate(r["segments"], 1):
        med = f"{s['median_iv']:.2f}" if s["median_iv"] else ""
        lines.append(f"segment,{i},{r['t0'] + s['t0']:.0f},{r['t0'] + s['t1']:.0f},{s['n']},"
                     f"{s['rate_per_min']:.3f},{med},{s['verdict']}")
    out.write_text("\n".join(lines) + "\n")
    return out


def write_long_json(r: dict, params: dict, out: Path) -> Path:
    payload = dict(params=params, mode="long",
                   file={k: v for k, v in r.items() if k not in ("env_plot", "floor_plot", "t_plot")})
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    return out


def write_long_html(r: dict, params: dict, run_id: str, g1: Path, g2: Path, out: Path) -> Path:
    b64 = lambda p: base64.b64encode(p.read_bytes()).decode()
    tr, g, p = r["trend"], r["regularity"], params
    hm = lambda t: f"{int(t) // 3600}시간 {int(t) % 3600 // 60}분"
    tclass = {"빨라짐": "warn", "느려짐": "warn", "일정": "ok"}.get(tr["verdict"], "off")
    gclass = {"규칙적": "ok", "약한 규칙성": "warn"}.get(g["verdict"], "off")
    num = lambda v, f="{:.2f}": f.format(v) if v is not None else "–"

    seg_rows = ""
    for s in r["segments"]:
        cls = {"규칙적": "ok", "약한 규칙성": "warn"}.get(s["verdict"], "off")
        seg_rows += (f'<tr><td>{(r["t0"] + s["t0"]) / 3600:.2f}–{(r["t0"] + s["t1"]) / 3600:.2f}h</td>'
                     f'<td>{s["n"]}</td><td>{s["rate_per_min"]:.2f}</td>'
                     f'<td>{num(s["median_iv"], "{:.1f}")}</td><td>{num(s["cv"])}</td>'
                     f'<td><span class="badge b{cls}">{s["verdict"]}</span></td>'
                     f'<td>{num(s["p"], "{:.3f}")}</td></tr>')
    notes = "".join(f"<li>{t}</li>" for t in long_observations(r)) or "<li>관찰할 항목이 없습니다.</li>"
    rej = r["rejected"]
    page = f'''<title>Long Recording Trend</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+KR:wght@400;500;600&display=swap">
<style>
{CSS}
</style>
<div class="wrap">
<header class="top">
  <p class="eyebrow">long-form · {html.escape(run_id)}</p>
  <h1>{html.escape(r["filename"])} — 스파이크 빈도 변화</h1>
  <p class="sub">긴 녹음에서 스파이크가 아닌 소리를 걸러내고, 스파이크가 나는 빈도가 시간에 따라
     일정한지 빨라지는지 느려지는지 봤습니다. 녹음 길이 {hm(r["duration"])} 중
     {hm(r["span"])}을 분석했습니다.</p>
</header>

<section class="verdict v{tclass}" style="font-size:15px">
  <span class="badge">빈도 {tr["verdict"]}</span>
  <span>기울기 <b>{num(tr["slope_per_hour"], "{:+.2f}")}</b>초/시간</span>
  <span>p <b>{num(tr["p"], "{:.2g}")}</b></span>
  <span>처음 <b>{num(tr["first"], "{:.1f}")}</b>초 → 마지막 <b>{num(tr["last"], "{:.1f}")}</b>초
    <i>({num(tr["change_pct"], "{:+.0f}")}%)</i></span>
  <span class="badge b{gclass}">전체 {g["verdict"]}</span>
</section>

<section class="rule">
  <div><b>분석 구간</b><span>{p["skip_s"]:.0f}초 이후 전체. 스트리밍으로 읽어 길이에 관계없이 메모리를 적게 씁니다.</span></div>
  <div><b>국소 잡음 바닥</b><span>{p["floor_block_s"]:.0f}초 블록 중앙값을 이어 붙여 환경 변화를 따라갑니다.
    이번 파일에서는 {r["floor_min"]:.5f}~{r["floor_max"]:.5f}로 움직였습니다.</span></div>
  <div><b>잡음 제거</b><span>국소 바닥의 {p["k_local"]:.0f}배를 넘고, 지속폭이 {p["max_sustain_s"] * 1000:.0f}ms 이하이며,
    앞뒤 {p["guard_s"]:.0f}초가 조용한 것만 스파이크로 인정합니다.</span></div>
  <div><b>추세 판정</b><span>간격 대 시각의 Theil–Sen 기울기와 Mann–Kendall 검정(p≤0.05일 때만 변화로 판정).</span></div>
</section>

<figure>
  <div class="cap"><h2>전체 시간축</h2>
  <p>파란 영역이 소리의 세기(로그), 검은 선이 국소 잡음 바닥, 주황 ▼ 가 잡음 필터를 통과한 스파이크입니다.</p></div>
  <div class="imgbox"><img src="data:image/png;base64,{b64(g1)}" alt="전체 시간축 엔벌로프와 검출된 스파이크"></div>
</figure>

<figure>
  <div class="cap"><h2>간격 추세와 구간별 빈도</h2>
  <p>위: 스파이크 간격을 시각에 대해 찍고 이동 중앙값과 추세선을 겹쳤습니다. 아래: 구간별 분당 스파이크 수.</p></div>
  <div class="imgbox"><img src="data:image/png;base64,{b64(g2)}" alt="간격 추세와 구간별 빈도"></div>
</figure>

<section class="guide"><h2>지표 읽는 법</h2>
  <dl>
    <dt>국소 잡음 바닥</dt>
    <dd>몇 시간짜리 녹음은 시간대에 따라 배경 소음이 달라집니다. 전역 임계값 하나를 쓰면 조용한 구간에서는
        거짓 검출이, 시끄러운 구간에서는 누락이 납니다. 그래서 {p["floor_block_s"]:.0f}초 블록마다 중앙값을 재고
        그 선을 기준으로 몇 배 튀는지를 봅니다.</dd>
    <dt>지속폭 필터</dt>
    <dd>스파이크는 짧고 날카롭습니다. 피크 주변에서 바닥의 {p["sustain_k"]:.0f}배를 넘는 구간이
        {p["max_sustain_s"] * 1000:.0f}ms를 넘으면 말소리·음악·문 닫는 소리로 보고 버립니다.
        앞뒤 {p["guard_s"]:.0f}초가 조용하지 않아도 버립니다 — 잡음 속에 묻힌 것은 판단할 수 없기 때문입니다.</dd>
    <dt>Theil–Sen 기울기</dt>
    <dd>간격이 시간당 몇 초씩 변하는지. 점들의 모든 짝을 이어 만든 기울기의 중앙값이라, 이상치 몇 개에
        끌려가지 않습니다. 음수면 간격이 짧아지는 것 = 빨라지는 것입니다.</dd>
    <dt>Mann–Kendall p값</dt>
    <dd>그 추세가 우연인지 봅니다. 값의 크기가 아니라 순서만 보는 검정이라 분포를 가정하지 않습니다.
        p ≤ 0.05일 때만 빨라짐/느려짐으로 판정하고, 그렇지 않으면 '일정'입니다.</dd>
    <dt>구간별 규칙성</dt>
    <dd>구간마다 기능 1과 같은 방식(위상 집중도 + 무작위 대조)으로 규칙성을 판정합니다.
        주기가 서서히 변하는 녹음은 <b>전체로는 불규칙, 구간별로는 규칙적</b>으로 나옵니다 —
        이 조합이 곧 '주기가 변하고 있다'는 신호입니다.</dd>
  </dl>
</section>

<section class="find"><h2>자동 관찰</h2><ul>{notes}</ul></section>

<section class="tablewrap"><table>
  <thead><tr><th>구간</th><th>스파이크</th><th>분당 횟수</th><th>간격 중앙값(초)</th><th>CV</th>
    <th>구간 규칙성</th><th>p값</th></tr></thead>
  <tbody>{seg_rows}</tbody></table></section>

<p class="note">스파이크 {len(r["times"])}개 검출 · 잡음으로 제외 {sum(rej.values())}건
 (지속시간 {rej["지속시간"]} / 주변소음 {rej["주변소음"]}) · 엔벌로프 프레임 {r["n_frames"]:,}개.
 같은 폴더에 <code>timeline.png</code>, <code>trend.png</code>, <code>spikes_and_segments.csv</code>,
 <code>long_report.json</code>, <code>report.pdf</code> 가 있습니다.</p>
</div>'''
    out.write_text(page)
    return out
