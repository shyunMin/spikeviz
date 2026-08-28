"""오디오 파형 스파이크 검출 - 디코딩, 피크 엔벌로프, 스파이크 판정."""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
AUDIO_EXT = {".m4a", ".mp3", ".wav", ".aac", ".flac", ".ogg", ".opus", ".wma",
             ".mp4", ".mov", ".3gp", ".amr", ".aif", ".aiff", ".caf", ".webm"}


@dataclass
class Params:
    """분석 기준값. 기본값은 최초 4개 파일 분석에 쓴 값과 같다."""
    skip_s: float = 60.0        # 앞부분 건너뛸 길이
    max_s: float = 600.0        # 읽을 최대 길이
    min_keep_s: float = 30.0    # 건너뛰고 남는 길이가 이보다 짧으면 파일 전체를 쓴다
    sr: int = 16000             # 분석용 리샘플 주파수
    hop_ms: int = 10            # 엔벌로프 프레임
    k_abs: float = 10.0         # 잡음 바닥 대비 최소 배율
    k_rel: float = 0.35         # 후보 피크 p90 대비 비율
    refractory_s: float = 1.0   # 이 안에 붙은 피크는 하나로 병합

    def to_dict(self):
        return asdict(self)


def probe_duration(path: str) -> float:
    out = subprocess.run([FFMPEG, "-hide_banner", "-i", str(path)],
                         capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", out)
    if not m:
        raise ValueError("오디오 길이를 읽지 못했습니다 (지원하지 않는 형식일 수 있습니다)")
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


def decode(path: str, sr: int) -> np.ndarray:
    """모노 float32 PCM. 12분짜리 파일이 1.4초쯤 걸려서 따로 캐시하지 않는다."""
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
        capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if x.size == 0:
        raise ValueError("오디오 스트림을 디코딩하지 못했습니다")
    return x


def window_of(duration: float, p: Params) -> tuple[float, float]:
    """분석 구간. 앞을 자르면 남는 게 min_keep_s보다 짧은 파일은 통째로 쓴다."""
    if duration - p.skip_s < p.min_keep_s:
        return 0.0, min(duration, p.max_s)
    return p.skip_s, min(duration, p.skip_s + p.max_s)


def _peaks(ratio: np.ndarray, dt: float, thr: float, refractory: float) -> list[int]:
    idx = np.where(ratio > thr)[0]
    if len(idx) == 0:
        return []
    out, grp = [], [idx[0]]
    for i in idx[1:]:
        if (i - grp[-1]) * dt <= refractory:
            grp.append(i)
        else:
            out.append(grp[int(np.argmax(ratio[grp]))])
            grp = [i]
    out.append(grp[int(np.argmax(ratio[grp]))])
    return out


def dominant_interval(intervals, tol: float = 0.25) -> dict | None:
    """간격 중 가장 큰 무리를 찾아 그 대표값을 돌려준다.

    짧은 부가 스파이크가 섞이면 전체 중앙값이 실제 주기보다 낮게 나온다.
    구간을 미리 정해두지 않고, 각 간격을 중심으로 ±tol 배 안에 들어오는
    간격이 가장 많은 무리를 고른다. 뚜렷한 무리가 없으면 None.
    """
    iv = np.array([v for v in intervals if v > 0], dtype=float)
    if len(iv) < 3:
        return None
    logs, width, best = np.log(iv), np.log(1 + tol), None
    for c in logs:
        grp = iv[np.abs(logs - c) <= width]
        score = (len(grp), -float(grp.std() / grp.mean()))
        if best is None or score > best[0]:
            best = (score, grp)
    grp = best[1]
    if len(grp) < 3 or len(grp) < 0.3 * len(iv):
        return None
    return dict(period=float(np.median(grp)), n=int(len(grp)), share=float(len(grp) / len(iv)),
                lo=float(grp.min()), hi=float(grp.max()), std=float(grp.std()))


def analyze(path: str, p: Params) -> dict:
    """한 파일의 분석 결과. 파형과 엔벌로프 배열까지 함께 돌려준다."""
    duration = probe_duration(path)
    t0, t1 = window_of(duration, p)
    x = np.asarray(decode(path, p.sr)[int(t0 * p.sr):int(t1 * p.sr)])

    hop = int(p.sr * p.hop_ms / 1000)
    m = len(x) // hop * hop
    env = np.abs(x[:m].reshape(-1, hop)).max(1)   # 피크 엔벌로프: 짧은 클릭이 평균에 묻히지 않는다
    dt = hop / p.sr

    floor = float(np.median(env))
    ratio = env / floor
    cand = _peaks(ratio, dt, 5.0, p.refractory_s)
    p90 = float(np.percentile(ratio[cand], 90)) if cand else 0.0
    thr = max(p.k_abs, p.k_rel * p90)

    ev = _peaks(ratio, dt, thr, p.refractory_s)
    times = [k * dt for k in ev]
    ratios = [float(ratio[k]) for k in ev]

    if times:
        anchor, anchor_kind = times[0], "first_spike"
    else:
        anchor, anchor_kind = float(np.argmax(env) * dt), "max_amplitude"

    iv = np.diff(times)
    long_iv = iv[iv >= 5.0]

    def stats(a):
        if len(a) == 0:
            return None
        return dict(n=int(len(a)), median=float(np.median(a)), mean=float(a.mean()),
                    min=float(a.min()), max=float(a.max()), std=float(a.std()),
                    cv=float(a.std() / a.mean()) if a.mean() else None)

    return dict(
        name=Path(path).stem, path=str(path), filename=Path(path).name,
        duration=duration, t0=t0, t1=t0 + len(x) / p.sr, sr=p.sr, dt=dt,
        floor=floor, thr=thr, max_ratio=max(ratios) if ratios else float(ratio.max()),
        whole_file=(t0 == 0.0),
        times=times, abs_times=[t0 + t for t in times], ratios=ratios,
        intervals=[float(v) for v in iv],
        stats=stats(iv), stats_long=stats(long_iv), dominant=dominant_interval(iv),
        anchor=anchor, abs_anchor=t0 + anchor, anchor_kind=anchor_kind,
        wave=x, env=env,
    )
