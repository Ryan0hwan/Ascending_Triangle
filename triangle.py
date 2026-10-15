"""
상승 삼각형 / 상승 수렴 스캐너 (추세선 방식)
- S&P 500 + 내 관심종목에서
- 고점들을 잇는 위쪽 추세선(수평 또는 완만한 상승)과, 저점들을 잇는 더 가파른 아래쪽 추세선이
  위로 수렴하는 패턴을 일봉·주봉에서 찾고
- 주가가 위쪽 선 근처(돌파 직전 또는 막 돌파)에 있는 종목을 디스코드로 보냅니다.
- 수동 실행 때 티커를 넣으면 그 종목이 왜 잡혔는지/안 잡혔는지 진단 결과를 보냅니다.
"""
import datetime as dt
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

# ───────────── 설정 (여기만 고치면 됩니다) ─────────────
GROUPS = os.environ.get("SCAN_GROUPS", "sp500,watchlist").split(",")

# 시간 단위별 설정: 패턴 길이 후보(봉 수), 고점·저점 판별 폭, 추세선 터치 허용 오차(%)
TIMEFRAMES = {
    "D": {"windows": [60, 90, 130, 180, 250], "pivot_k": 5, "tol": 2.0, "label": "일봉"},
    "W": {"windows": [52, 78, 104, 156],      "pivot_k": 3, "tol": 3.0, "label": "주봉"},
}
MIN_UPPER_TOUCH = int(os.environ.get("MIN_UPPER_TOUCH", "3"))   # 위쪽 선에 닿은 고점 최소 개수
MIN_LOWER_TOUCH = int(os.environ.get("MIN_LOWER_TOUCH", "3"))   # 아래쪽 선에 닿은 저점 최소 개수
MIN_TOTAL_TOUCH = int(os.environ.get("MIN_TOTAL_TOUCH", "6"))   # 위·아래 합계 최소 터치
MAX_UPPER_DROP = float(os.environ.get("MAX_UPPER_DROP", "2"))   # 위쪽 선이 패턴 기간 동안 이 % 넘게 내려가면 제외 (하락 삼각형 배제)
FLAT_PCT = float(os.environ.get("FLAT_PCT", "4"))               # 위쪽 선 상승폭이 이 % 이하면 FLAT(교과서형 상승 삼각형), 넘으면 RISE
SQUEEZE_RATIO = float(os.environ.get("SQUEEZE_RATIO", "0.5"))   # 지금 두 선 간격 ≤ 패턴 시작 때 간격 × 이 비율 (수렴)
NEAR_BELOW_PCT = float(os.environ.get("NEAR_BELOW_PCT", "3"))   # 위쪽 선 아래 이 % 이내 (돌파 직전)
NEAR_ABOVE_PCT = float(os.environ.get("NEAR_ABOVE_PCT", "3"))   # 위쪽 선 위 이 % 이내 (막 돌파·되돌림)
BREAKOUT_BARS = int(os.environ.get("BREAKOUT_BARS", "5"))       # 위쪽 선 위 마감은 최근 이 봉 수 안에서만 허용
MAX_BREAK_BELOW = float(os.environ.get("MAX_BREAK_BELOW", "0.05"))  # 아래쪽 선을 깨고 마감한 봉 비율 상한

REQUIRE_TREND = os.environ.get("REQUIRE_TREND", "1") == "1"     # 일봉 기준 주가 > 200일선, 50일선 > 200일선
MIN_PRICE = float(os.environ.get("MIN_PRICE", "5"))
MIN_DOLLAR_VOL = float(os.environ.get("MIN_DOLLAR_VOL", "5000000"))  # 50일 평균 거래대금($)

REALERT_DAYS = int(os.environ.get("REALERT_DAYS", "15"))        # 같은 종목·시간 단위·저항선은 이 기간 안에 다시 알리지 않음
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE = os.environ.get("FORCE", "0") == "1"                     # 주말에도 실행 (최근 거래일 기준)
DIAGNOSE = [t.strip().upper() for t in os.environ.get("DIAGNOSE", "").replace(" ", ",").split(",") if t.strip()]
# ──────────────────────────────────────────────────────

NY = ZoneInfo("America/New_York")
STATE = Path(__file__).parent / "state"
STATE.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (triangle-scanner)"}
GROUP_LABEL = {"sp500": "S&P 500", "watchlist": "Watchlist"}


# ───────────── 종목 목록 ─────────────
def _fetch_sp500() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers=UA, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0]
    return pd.DataFrame({"ticker": t["Symbol"], "name": t["Security"], "industry": t["GICS Sub-Industry"]})


def _load_watchlist() -> pd.DataFrame:
    p = Path(__file__).parent / "watchlist.txt"
    rows = []
    if p.exists():
        for ln in p.read_text(encoding="utf-8").splitlines():
            ln = ln.split("#")[0].strip()
            if not ln:
                continue
            tk, _, name = ln.partition(",")
            tk = re.sub(r"\.([A-Z])$", r"-\1", tk.strip().upper())  # MOG.A → MOG-A
            rows.append({"ticker": tk, "name": name.strip(), "industry": "Watchlist"})
    return pd.DataFrame(rows, columns=["ticker", "name", "industry"]).drop_duplicates("ticker")


def load_universe(group: str) -> pd.DataFrame:
    if group == "watchlist":
        return _load_watchlist()
    cache = STATE / f"universe_{group}.csv"
    fresh = cache.exists() and (time.time() - cache.stat().st_mtime) < 7 * 86400
    if not fresh:
        try:
            df = {"sp500": _fetch_sp500}[group]()
            if len(df) < 50:
                raise ValueError("종목 수가 비정상적으로 적음")
            df.to_csv(cache, index=False)
        except Exception as e:
            print(f"[경고] {group} 목록 갱신 실패: {e}")
    if not cache.exists():
        return pd.DataFrame(columns=["ticker", "name", "industry"])
    df = pd.read_csv(cache).fillna("")
    df["ticker"] = df["ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
    df = df[df["ticker"].str.fullmatch(r"[A-Z\-]{1,6}")]
    return df.drop_duplicates("ticker")


# ───────────── 시세 ─────────────
def download(tickers: list[str]):
    import yfinance as yf
    for i in range(0, len(tickers), 200):
        chunk = tickers[i:i + 200]
        try:
            data = yf.download(chunk, period="5y", interval="1d", group_by="ticker",
                               auto_adjust=True, threads=True, progress=False)
        except Exception as e:
            print(f"[경고] 시세 다운로드 실패({chunk[0]}…): {e}")
            continue
        if not isinstance(data.columns, pd.MultiIndex):
            data = pd.concat({chunk[0]: data}, axis=1)
        have = set(data.columns.get_level_values(0))
        for t in chunk:
            if t in have:
                yield t, data[t]
        time.sleep(1)


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    w = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    return w.dropna(subset=["Close"])


# ───────────── 추세선 찾기 ─────────────
def pivots(values: np.ndarray, k: int, kind: str) -> np.ndarray:
    """앞뒤 k봉보다 높은(kind='high') / 낮은(kind='low') 봉의 위치"""
    n = len(values)
    if n < 2 * k + 1:
        return np.array([], dtype=int)
    win = np.lib.stride_tricks.sliding_window_view(values, 2 * k + 1)
    centre = values[k:n - k]
    hit = centre == (win.max(axis=1) if kind == "high" else win.min(axis=1))
    idx = np.nonzero(hit)[0] + k
    # 같은 값이 연달아 나오면 하나만
    keep = [i for j, i in enumerate(idx) if j == 0 or i - idx[j - 1] > k]
    return np.array(keep, dtype=int)


def fit_line(xs: np.ndarray, ys: np.ndarray, kind: str, tol: float):
    """
    고점(kind='upper') 또는 저점(kind='lower') 두 개를 잇는 직선 중
    모든 고점이 선 아래(저점이면 위)에 있으면서 가장 많은 점이 선에 닿는 선을 고름.
    반환: (기울기, 절편, 터치 개수, 터치한 점 위치들) 또는 None
    """
    m = len(xs)
    if m < 2:
        return None
    a, b = np.triu_indices(m, 1)
    slope = (ys[b] - ys[a]) / (xs[b] - xs[a])
    inter = ys[a] - slope * xs[a]
    L = slope[:, None] * xs[None, :] + inter[:, None]           # 직선 후보 × 점
    if kind == "upper":
        valid = (ys[None, :] <= L * (1 + tol)).all(axis=1)
        touch = ys[None, :] >= L * (1 - tol)
    else:
        valid = (ys[None, :] >= L * (1 - tol)).all(axis=1)
        touch = ys[None, :] <= L * (1 + tol)
    cnt = np.where(valid, touch.sum(axis=1), -1)
    if cnt.max() < 2:
        return None
    span = xs[b] - xs[a]
    best = np.lexsort((span, cnt))[-1]                          # 터치 많은 순, 같으면 긴 선
    return float(slope[best]), float(inter[best]), int(cnt[best]), xs[touch[best]]


def find_pattern(df: pd.DataFrame, tf: str) -> tuple[dict | None, str]:
    """한 시간 단위에서 가장 좋은 패턴. (결과, 실패 이유)"""
    cfg = TIMEFRAMES[tf]
    tol = cfg["tol"] / 100
    h, l, c, v = df["High"].to_numpy(), df["Low"].to_numpy(), df["Close"].to_numpy(), df["Volume"].to_numpy()
    n = len(c)
    best, why = None, "시세 기간 부족"
    for W in cfg["windows"]:
        if n < W + 5:
            continue
        s = n - W
        hx = pivots(h[s:], cfg["pivot_k"], "high") + s
        lx = pivots(l[s:], cfg["pivot_k"], "low") + s
        up = fit_line(hx.astype(float), h[hx], "upper", tol)
        lo = fit_line(lx.astype(float), l[lx], "lower", tol)
        if up is None or lo is None:
            why = "고점 또는 저점이 2개 미만"
            continue
        su, iu, nu, tu = up
        sl, il, nl, tl = lo
        if nu < MIN_UPPER_TOUCH or nl < MIN_LOWER_TOUCH or nu + nl < MIN_TOTAL_TOUCH:
            why = f"추세선 터치 부족 (위 {nu}, 아래 {nl})"
            continue
        start = int(min(tu.min(), tl.min()))
        x = np.arange(n, dtype=float)
        U, Lw = su * x + iu, sl * x + il
        u_rise = (U[n - 1] / U[start] - 1) * 100
        if u_rise < -MAX_UPPER_DROP:
            why = f"위쪽 선이 하락 중 ({u_rise:+.1f}%)"
            continue
        if not sl > 0 or not sl > su:
            why = "아래쪽 선이 위쪽 선보다 가파르지 않음 (수렴 아님)"
            continue
        gap0 = (U[start] - Lw[start]) / U[start]
        gap1 = (U[n - 1] - Lw[n - 1]) / U[n - 1]
        if not (0 < gap1 <= gap0 * SQUEEZE_RATIO):
            why = f"수렴 부족 (간격 {gap0*100:.1f}% → {gap1*100:.1f}%)"
            continue
        seg = slice(start, n)
        above = c[seg] > U[seg] * (1 + tol)
        if above[:-BREAKOUT_BARS].any() if len(above) > BREAKOUT_BARS else False:
            why = "패턴 중간에 이미 위쪽 선을 크게 돌파한 적 있음"
            continue
        below_ratio = (c[seg] < Lw[seg] * (1 - tol)).mean()
        if below_ratio > MAX_BREAK_BELOW:
            why = f"아래쪽 선 이탈이 잦음 ({below_ratio*100:.0f}%)"
            continue
        pos = (c[-1] / U[-1] - 1) * 100
        if not (-NEAR_BELOW_PCT <= pos <= NEAR_ABOVE_PCT):
            why = f"주가가 위쪽 선에서 멂 ({pos:+.1f}%)"
            continue
        if c[-1] < Lw[-1] * (1 - tol):
            why = "주가가 아래쪽 선 아래"
            continue
        bars = n - start
        vbase = v[max(0, start - 50):start].mean() if start > 0 else np.nan
        cand = {
            "tf": tf, "type": "FLAT" if u_rise <= FLAT_PCT else "RISE",
            "bars": int(bars), "hi_t": nu, "lo_t": nl, "upper": float(U[-1]), "lower": float(Lw[-1]),
            "last": float(c[-1]), "pos": float(pos), "sqz": float(gap1 / gap0 * 100),
            "u_rise": float(u_rise), "vol_x": float(v[start:].mean() / vbase) if vbase else float("nan"),
        }
        score = (nu + nl, bars)
        if best is None or score > best[0]:
            best = (score, cand)
    return (best[1], "") if best else (None, why)


def check_ticker(df: pd.DataFrame, today: dt.date | None) -> tuple[list[dict], str, float]:
    """종목 하나: 기본 필터 → 일봉·주봉 패턴. (결과들, 실패 이유, RS 점수)"""
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if len(df) < 260:
        return [], "상장 기간 부족 (약 1년 미만)", np.nan
    c = df["Close"]
    score = rs_score(c)
    if today is not None and df.index[-1].date() != today:
        return [], "오늘 시세 없음", score
    last = float(c.iloc[-1])
    if last < MIN_PRICE or (c * df["Volume"]).tail(50).mean() < MIN_DOLLAR_VOL:
        return [], "주가 또는 거래대금 미달", score
    if REQUIRE_TREND:
        s50, s200 = c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
        if not (last > s200 and s50 > s200):
            return [], "상승 추세 아님 (주가 > 200일선, 50일선 > 200일선 불충족)", score
    out, reasons = [], []
    for tf in TIMEFRAMES:
        data = df if tf == "D" else to_weekly(df)
        r, why = find_pattern(data, tf)
        if r:
            out.append(r)
        else:
            reasons.append(f"{TIMEFRAMES[tf]['label']}: {why}")
    return out, " / ".join(reasons), score


def rs_score(c: pd.Series) -> float:
    num = den = 0.0
    for n, w in ((63, 2), (126, 1), (189, 1), (252, 1)):
        if len(c) > n:
            num += (c.iloc[-1] / c.iloc[-1 - n] - 1) * w
            den += w
    return num / den if den else np.nan


# ───────────── 디스코드 ─────────────
def company_name(ticker: str) -> str:
    cache_file = STATE / "names.json"
    try:
        cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    except Exception:
        cache = {}
    if ticker in cache:
        return cache[ticker]
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ""
    except Exception:
        name = ""
    cache[ticker] = name
    cache_file.write_text(json.dumps(cache))
    return name


def post(content: str):
    if not WEBHOOK:
        print(content)
        return
    for _ in range(5):
        r = requests.post(WEBHOOK, json={"content": content}, timeout=30)
        if r.status_code == 429:
            time.sleep(float(r.json().get("retry_after", 2)) + 0.5)
            continue
        r.raise_for_status()
        time.sleep(0.6)
        return


def send_table(title: str, rows: list[dict]):
    head = (f"{'TICKER':<7}{'NAME':<13}{'TYPE':<5}{'BARS':>5}{'HI':>3}{'LO':>3}"
            f"{'UPPER':>9}{'LAST':>9}{'POS%':>6}{'SQZ%':>5}{'RS':>4}{'VOLx':>5}")
    lines = [f"{r['ticker']:<7}{r['name'][:11]:<13}{r['type']:<5}{r['bars']:>5}{r['hi_t']:>3}{r['lo_t']:>3}"
             f"{r['upper']:>9.2f}{r['last']:>9.2f}{r['pos']:>+6.1f}{r['sqz']:>5.0f}{r['rs']:>4}{r['vol_x']:>5.2f}"
             for r in rows]
    first, part = True, 1
    while lines:
        block, size = [], 0
        while lines and size + len(lines[0]) + 1 < 1750:
            size += len(lines[0]) + 1
            block.append(lines.pop(0))
        header = f"**{title}**\n" if first else f"**{title} ({part})**\n"
        post(f"{header}```\n{head}\n" + "\n".join(block) + "\n```")
        first, part = False, part + 1


# ───────────── 실행 ─────────────
def main():
    now = dt.datetime.now(NY)
    if now.weekday() >= 5 and not FORCE and not DIAGNOSE:
        print("주말. 종료.")
        return
    today = None if (FORCE or DIAGNOSE) else now.date()

    sent_file = STATE / "sent.json"
    try:
        sent = json.loads(sent_file.read_text()) if sent_file.exists() else {}
    except Exception:
        sent = {}
    cutoff = (now.date() - dt.timedelta(days=REALERT_DAYS)).isoformat()
    sent = {k: v for k, v in sent.items() if v["date"] >= cutoff}

    universes = {g.strip(): load_universe(g.strip()) for g in GROUPS if g.strip()}
    meta, group_of = {}, {}
    for g, u in universes.items():
        for row in u.itertuples():
            if row.ticker not in meta:
                meta[row.ticker] = row
                group_of[row.ticker] = g
    tickers = DIAGNOSE or list(meta)
    print(f"스캔 대상 {len(tickers)}종목")
    if not tickers:
        post("⚠️ 삼각형 스캐너: 종목 목록을 불러오지 못했습니다. Actions 로그를 확인하세요.")
        return

    found, scores, diag, scanned = [], {}, {}, 0
    for ticker, df in download(tickers):
        scanned += 1
        res, why, score = check_ticker(df, today)
        scores[ticker] = score
        diag[ticker] = why
        for r in res:
            found.append({"ticker": ticker, **r})

    # RS 백분위 (진단 모드에서는 비교 대상이 적어 참고용)
    rs = (pd.Series(scores).rank(pct=True) * 98 + 1).round().fillna(0).astype(int)
    for f in found:
        f["rs"] = int(rs.get(f["ticker"], 0))
        m = meta.get(f["ticker"])
        f["name"] = (str(m.name) if m is not None else "") or company_name(f["ticker"])

    if DIAGNOSE:
        lines = []
        for t in DIAGNOSE:
            hits = [f for f in found if f["ticker"] == t]
            if t not in diag:
                lines.append(f"**{t}**: 시세를 받지 못함 (티커 확인 필요)")
                continue
            for f in hits:
                lines.append(f"**{t}** ✅ {TIMEFRAMES[f['tf']]['label']} {f['type']} · {f['bars']}봉 · 고점 {f['hi_t']}회 · 저점 {f['lo_t']}회 · "
                             f"위쪽 선 {f['upper']:.2f} · 종가 {f['last']:.2f} ({f['pos']:+.1f}%) · 수렴 {f['sqz']:.0f}%")
            if diag[t]:
                lines.append(f"**{t}** {'❌' if not hits else '참고'}: {diag[t]}")
        post("**삼각형 스캐너 진단**\n" + "\n".join(lines))
        return

    new = []
    for f in found:
        key = f"{f['ticker']}|{f['tf']}"
        prev = sent.get(key)
        if not FORCE and prev and abs(prev["upper"] / f["upper"] - 1) < 0.03:
            continue
        if not FORCE:
            sent[key] = {"upper": f["upper"], "date": now.date().isoformat()}
        new.append(f)

    post(f"**{now:%b} {now.day} | 상승 삼각형·상승 수렴**\n"
         f"일봉 {sum(f['tf'] == 'D' for f in new)}건 · 주봉 {sum(f['tf'] == 'W' for f in new)}건 · 확인 {scanned}종목")
    for tf, cfg in TIMEFRAMES.items():
        rows = sorted([f for f in new if f["tf"] == tf], key=lambda r: abs(r["pos"]))   # 위쪽 선에 가까운 순
        if rows:
            send_table(f"{cfg['label']} 패턴", rows)
    if scanned < len(tickers) * 0.5:
        post(f"⚠️ 삼각형 스캐너: 시세를 {scanned}/{len(tickers)}종목만 받았습니다. 일시적 오류일 수 있습니다.")
    if not FORCE:
        sent_file.write_text(json.dumps(sent))
    print(f"완료: 확인 {scanned}종목, 신규 {len(new)}건")


if __name__ == "__main__":
    sys.exit(main())
