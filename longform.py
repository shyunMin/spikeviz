"""기능 2: 긴 녹음(수 시간)의 잡음 제거와 스파이크 빈도 변화 분석.

기능 1과 다른 점 세 가지.
  1. 파일 전체를 메모리에 올리지 않고 스트리밍으로 엔벌로프만 뽑는다.
  2. 잡음 바닥을 전역 하나가 아니라 국소(블록 중앙값)로 잡아 환경 변화를 따라간다.
  3. 짧고 날카로운 충격음만 남기고 말소리·음악·문 닫는 소리 같은 것을 제외한다.
"""
from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from analysis import FFMPEG, probe_duration, regularity


@dataclass
class LongParams:
    skip_s: float = 60.0          # 앞부분 건너뛰기
    sr: int = 16000
    hop_ms: int = 10
    floor_block_s: float = 10.0   # 국소 잡음 바닥을 재는 블록
    k_local: float = 8.0          # 국소 바닥 대비 이만큼 넘어야 후보
    sustain_k: float = 3.0        # 지속폭을 재는 기준 배율
    max_sustain_s: float = 0.35   # 이보다 오래 이어지면 스파이크가 아니다 (실측: 실제 스파이크 0.12~0.34초)
    guard_s: float = 2.0          # 앞뒤 이만큼을 주변 소음으로 본다
    guard_max: float = 4.0        # 주변이 바닥의 이 배를 넘으면 제외
    refractory_s: float = 1.0
    seg_target: int = 24          # 구간 개수 목표
    seg_min_s: float = 300.0      # 구간 최소 길이

    def to_dict(self):
        return asdict(self)


def stream_envelope(path: str, p: LongParams) -> tuple[np.ndarray, float]:
    """디코딩을 파이프로 받아 프레임별 피크만 남긴다. 메모리는 길이에 비례하되 아주 작다."""
    hop = int(p.sr * p.hop_ms / 1000)
    cmd = [FFMPEG, "-v", "error", "-ss", str(p.skip_s), "-i", str(path),
           "-ac", "1", "-ar", str(p.sr), "-f", "s16le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 7)
    parts, rem = [], b""
    while True:
        buf = proc.stdout.read(hop * 2 * 6000)          # 60초씩
        if not buf:
            break
        buf = rem + buf
        nf = len(buf) // (hop * 2)
        use, rem = buf[:nf * hop * 2], buf[nf * hop * 2:]
        if nf:
            a = np.frombuffer(use, dtype=np.int16).astype(np.float32) / 32768.0
            parts.append(np.abs(a.reshape(-1, hop)).max(1))
    proc.stdout.close()
    proc.wait()
    if not parts:
        raise ValueError("오디오 스트림을 디코딩하지 못했습니다")
    return np.concatenate(parts), hop / p.sr


def local_floor(env: np.ndarray, dt: float, p: LongParams) -> np.ndarray:
    """블록 중앙값을 이어 붙인 국소 잡음 바닥. 몇 시간에 걸친 환경 변화를 따라간다."""
    blk = max(1, int(p.floor_block_s / dt))
    nb = max(1, len(env) // blk)
    med = np.median(env[:nb * blk].reshape(nb, blk), axis=1)
    if nb == 1:
        return np.full(len(env), max(float(med[0]), 1e-6))
    floor = np.interp(np.arange(len(env)), (np.arange(nb) + .5) * blk, med)
    return np.maximum(floor, 1e-6)


def detect_impulses(ratio: np.ndarray, dt: float, p: LongParams) -> tuple[list, list, dict]:
    """국소 바닥 대비 튀는 지점 중 '짧고 날카롭고 주변이 조용한' 것만 남긴다."""
    idx = np.where(ratio > p.k_local)[0]
    if not len(idx):
        return [], [], {"지속시간": 0, "주변소음": 0}
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    kept, rejected = [], {"지속시간": 0, "주변소음": 0}
    guard, pad = int(p.guard_s / dt), int(.3 / dt)
    for g in groups:
        k = int(g[int(np.argmax(ratio[g]))])
        lo = hi = k
        while lo > 0 and ratio[lo - 1] > p.sustain_k:
            lo -= 1
        while hi < len(ratio) - 1 and ratio[hi + 1] > p.sustain_k:
            hi += 1
        sustain = (hi - lo + 1) * dt
        if sustain > p.max_sustain_s:
            rejected["지속시간"] += 1
            continue
        ctx = np.concatenate([ratio[max(0, lo - guard):max(0, lo - pad)], ratio[hi + pad:hi + guard]])
        if len(ctx) and float(np.percentile(ctx, 90)) > p.guard_max:
            rejected["주변소음"] += 1
            continue
        kept.append((k * dt, float(ratio[k]), sustain))
    merged: list = []
    for t, r, sus in kept:
        if merged and t - merged[-1][0] <= p.refractory_s:
            if r > merged[-1][1]:
                merged[-1] = (t, r, sus)
        else:
            merged.append((t, r, sus))
    times = [m[0] for m in merged]
    ratios = [m[1] for m in merged]
    return times, ratios, rejected


def _theil_sen(x: np.ndarray, y: np.ndarray, max_pairs: int = 200_000) -> float:
    """짝의 기울기 중앙값. 점이 많으면 짝을 전부 만들지 않고 무작위로 뽑는다(메모리 상한)."""
    n = len(x)
    if n * (n - 1) // 2 <= max_pairs:
        i, j = np.triu_indices(n, 1)
    else:
        rng = np.random.default_rng(0)
        i = rng.integers(0, n, max_pairs)
        j = rng.integers(0, n, max_pairs)
        ok = i != j
        i, j = i[ok], j[ok]
    dx = x[j] - x[i]
    ok = dx != 0
    return float(np.median((y[j][ok] - y[i][ok]) / dx[ok]))


def _mann_kendall(y: np.ndarray) -> tuple[float, float]:
    """추세의 유의성. 값의 분포를 가정하지 않고 순서만 본다."""
    n = len(y)
    s = float(sum(np.sign(y[k + 1:] - y[k]).sum() for k in range(n - 1)))
    var = n * (n - 1) * (2 * n + 5) / 18
    if var <= 0:
        return 0.0, 1.0
    z = (s - np.sign(s)) / math.sqrt(var)
    return float(z), float(math.erfc(abs(z) / math.sqrt(2)))


def trend(times: list[float]) -> dict:
    """간격이 시간에 따라 늘어나는지 줄어드는지."""
    t = np.asarray(times, dtype=float)
    if len(t) < 8:
        return dict(verdict="표본 부족", slope_per_hour=None, z=None, p=None,
                    first=None, last=None, change_pct=None)
    iv = np.diff(t)
    mid = (t[1:] + t[:-1]) / 2
    slope = _theil_sen(mid, iv) * 3600
    z, p = _mann_kendall(iv)
    edge = max(3, len(iv) // 10)
    first, last = float(iv[:edge].mean()), float(iv[-edge:].mean())
    if p > .05:
        verdict = "일정"
    else:
        verdict = "느려짐" if slope > 0 else "빨라짐"
    return dict(verdict=verdict, slope_per_hour=slope, z=z, p=p, first=first, last=last,
                change_pct=(last - first) / first * 100 if first else None)


def segments(times: list[float], span: float, p: LongParams) -> list[dict]:
    """구간을 나눠 구간마다 빈도와 규칙성을 낸다."""
    seg_len = max(p.seg_min_s, span / p.seg_target)
    n_seg = max(1, int(math.ceil(span / seg_len)))
    t = np.asarray(times, dtype=float)
    out = []
    for k in range(n_seg):
        a, b = k * seg_len, min((k + 1) * seg_len, span)
        m = t[(t >= a) & (t < b)]
        iv = np.diff(m)
        reg = regularity(m - a, b - a, n_sur=200) if len(m) >= 5 else None
        out.append(dict(t0=a, t1=b, n=len(m), rate_per_min=len(m) / ((b - a) / 60) if b > a else 0,
                        median_iv=float(np.median(iv)) if len(iv) else None,
                        verdict=reg["verdict"] if reg else "표본 부족",
                        p=reg["p"] if reg else None, cv=reg["cv"] if reg else None))
    return out


def analyze_long(path: str, p: LongParams, plot_cols: int = 3000) -> dict:
    duration = probe_duration(path)
    env, dt = stream_envelope(path, p)
    span = len(env) * dt
    floor = local_floor(env, dt, p)
    ratio = env / floor
    times, ratios, rejected = detect_impulses(ratio, dt, p)

    cols = min(plot_cols, max(len(env) // 4, 60))
    cut = len(env) // cols * cols
    env_plot = env[:cut].reshape(cols, -1).max(1)
    floor_plot = floor[:cut].reshape(cols, -1).mean(1)
    t_plot = (np.arange(cols) + .5) * (cut / cols) * dt

    iv = np.diff(times)
    return dict(
        name=Path(path).stem, filename=Path(path).name, path=str(path),
        duration=duration, t0=p.skip_s, span=span, dt=dt,
        times=times, ratios=ratios, intervals=[float(v) for v in iv],
        rejected=rejected, n_frames=len(env),
        floor_median=float(np.median(floor)), floor_min=float(floor.min()), floor_max=float(floor.max()),
        trend=trend(times), segments=segments(times, span, p),
        regularity=regularity(times, span) if len(times) >= 5 else
                   dict(verdict="표본 부족", n=len(times), cv=None, cv2=None, period=None,
                        R=None, r_needed=None, p=None, n_sur=0, skip_ratio=None),
        env_plot=env_plot, floor_plot=floor_plot, t_plot=t_plot,
    )
