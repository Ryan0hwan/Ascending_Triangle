"""
상승 삼각형(Ascending Triangle) 스캐너
- S&P 500 + 내 관심종목에서
- 수평 저항선 + 높아지는 저점으로 수렴 중이고, 아직 돌파 전인 종목을 찾아
- 디스코드로 표 형태로 보냅니다. (형성 중 알림)
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
GROUPS = os.environ.get("SCAN_GROUPS", "sp500,watchlist").split(",")  # 앞 그룹 우선 (겹치는 종목은 앞 그룹에만 표시)

# 패턴 정의
PIVOT_K = int(os.environ.get("PIVOT_K", "3"))                  # 고점·저점 판별: 앞뒤 K일보다 높은(낮은) 날
RES_TOL_PCT = float(os.environ.get("RES_TOL_PCT", "1.5"))       # ① 저항선: 고점들이 최고 고점 대비 이 % 안에 모여야 함
MIN_TOUCHES = int(os.environ.get("MIN_TOUCHES", "3"))           # ① 저항선 고점 최소 개수
MIN_HIGHER_LOWS = int(os.environ.get("MIN_HIGHER_LOWS", "2"))   # ② 높아지는 저점 최소 개수
LOW_STEP_PCT = float(os.environ.get("LOW_STEP_PCT", "1.0"))     # ② 저점이 직전 저점보다 최소 이 %만큼 높아야 함
MIN_DAYS = int(os.environ.get("MIN_DAYS", "15"))                # ③ 패턴 최소 기간(거래일)
MAX_DAYS = int(os.environ.get("MAX_DAYS", "60"))                # ③ 패턴 최대 기간(거래일)
SQUEEZE_RATIO = float(os.environ.get("SQUEEZE_RATIO", "0.5"))   # ④ 최근 10일 변동폭 ≤ 패턴 초반 깊이 × 이 비율
NEAR_RES_PCT = float(os.environ.get("NEAR_RES_PCT", "3.0"))     # ⑤ 현재가가 저항선 아래 이 % 이내 (돌파 전)

# 보강 조건
REQUIRE_TREND = os.environ.get("REQUIRE_TREND", "1") == "1"     # ⑥ 주가 > 200일선, 50일선 > 200일선
REQUIRE_VOL_DRY = os.environ.get("REQUIRE_VOL_DRY", "1") == "1" # ⑦ 패턴 기간 평균 거래량 < 패턴 직전 50일 평균
REQUIRE_RS = os.environ.get("REQUIRE_RS", "1") == "1"           # ⑧ 3개월·6개월 수익률이 모두 SPY보다 높음

# 기본 필터
MIN_PRICE = float(os.environ.get("MIN_PRICE", "5"))
MIN_DOLLAR_VOL = float(os.environ.get("MIN_DOLLAR_VOL", "5000000"))  # 50일 평균 거래대금($)

REALERT_DAYS = int(os.environ.get("REALERT_DAYS", "15"))        # 같은 종목·같은 저항선은 이 기간(달력일) 안에 다시 알리지 않음
WEBHOOK = os.environ.get("DISCORD_WEBHOOK_URL", "")
FORCE = os.environ.get("FORCE", "0") == "1"                     # 테스트용 수동 실행
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
            data = yf.download(chunk, period="2y", interval="1d", group_by="ticker",
                               auto_adjust=False, threads=True, progress=False)
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


# ───────────── 패턴 판별 ─────────────
def pivots(values: np.ndarray, k: int, kind: str) -> list[int]:
    """앞뒤 k일보다 높은(kind='high') / 낮은(kind='low') 날의 위치. 마지막 k일은 아직 확정 불가라 제외"""
    out = []
    for i in range(k, len(values) - k):
        win = values[i - k:i + k + 1]
        if (kind == "high" and values[i] == win.max()) or (kind == "low" and values[i] == win.min()):
            if not out or i - out[-1] > k:      # 같은 값이 연달아 나오면 하나만
                out.append(i)
    return out


def find_triangle(df: pd.DataFrame, today: dt.date | None, spy_ret: dict | None = None) -> dict | None:
    df = df.dropna(subset=["Close", "High", "Low", "Volume"])
    if len(df) < 260 or (today is not None and df.index[-1].date() != today):
        return None
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    last = float(c.iloc[-1])
    if last < MIN_PRICE or (c * v).tail(50).mean() < MIN_DOLLAR_VOL:
        return None

    # ⑥ 선행 추세
    s50, s200 = c.rolling(50).mean().iloc[-1], c.rolling(200).mean().iloc[-1]
    if REQUIRE_TREND and not (last > s200 and s50 > s200):
        return None

    # ⑧ 상대강도
    if REQUIRE_RS and spy_ret:
        r3, r6 = last / c.iloc[-64] - 1, last / c.iloc[-127] - 1
        if not (r3 > spy_ret["3m"] and r6 > spy_ret["6m"]):
            return None

    # 최근 MAX_DAYS + 여유 구간에서 고점·저점 찾기
    win = MAX_DAYS + PIVOT_K
    hh, ll = h.iloc[-win:].to_numpy(), l.iloc[-win:].to_numpy()
    n = len(hh)
    ph, pl = pivots(hh, PIVOT_K, "high"), pivots(ll, PIVOT_K, "low")
    if not ph:
        return None

    # ① 수평 저항선: 가장 높은 고점(R) 대비 RES_TOL_PCT 안에 드는 고점들
    res = max(hh[i] for i in ph)
    touches = [i for i in ph if hh[i] >= res * (1 - RES_TOL_PCT / 100)]
    if len(touches) < MIN_TOUCHES:
        return None
    start = touches[0]                      # 패턴 시작 = 첫 번째 저항선 고점
    days = n - 1 - start

    # ③ 기간
    if not (MIN_DAYS <= days <= MAX_DAYS):
        return None

    # 패턴 구간에서 저항선을 위로 뚫은 적이 없어야 함 (확정 전 마지막 며칠 포함)
    if hh[start:].max() > res * (1 + RES_TOL_PCT / 100) or c.iloc[-(n - start):].max() > res * 1.005:
        return None

    # ② 높아지는 저점: 패턴 시작 이후 저점들이 계속 높아져야 함
    lows = [i for i in pl if i > start]
    if len(lows) < MIN_HIGHER_LOWS:
        return None
    lv = [ll[i] for i in lows]
    if any(b < a * (1 + LOW_STEP_PCT / 100) for a, b in zip(lv, lv[1:])):
        return None
    # 마지막 저점 이후 그 아래로 깨진 적 없음
    if ll[lows[-1]:].min() < lv[-1] * 0.995:
        return None

    # ④ 수렴: 최근 10일 변동폭 ≤ 초반 깊이(저항선 - 첫 저점) × 비율
    first_depth = res - lv[0]
    recent_range = hh[-10:].max() - ll[-10:].min()
    if first_depth <= 0 or recent_range > first_depth * SQUEEZE_RATIO:
        return None

    # ⑤ 돌파 전, 저항선 근처
    gap = float((res - last) / res * 100)
    if not (0 <= gap <= NEAR_RES_PCT):
        return None

    # ⑦ 거래량 감소: 패턴 기간 평균 < 패턴 직전 50일 평균
    pv = v.iloc[-(n - start):].mean()
    before = v.iloc[-(n - start) - 50:-(n - start)].mean()
    if REQUIRE_VOL_DRY and not pv < before:
        return None

    return {"res": float(res), "last": last, "gap": gap, "touches": len(touches),
            "lows": len(lows), "days": int(days), "vol_ratio": float(pv / before) if before else float("nan")}


# ───────────── 기타 ─────────────
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
    head = (f"{'TICKER':<8}{'DESCRIPTION':<20}{'INDUSTRY':<18}"
            f"{'RESIST':>9}{'LAST':>9}{'GAP%':>6}{'HI':>4}{'LO':>4}{'DAYS':>5}{'VOL':>6}")
    lines = [
        f"{r['ticker']:<8}{r['name'][:18]:<20}{r['industry'][:16]:<18}"
        f"{r['res']:>9.2f}{r['last']:>9.2f}{r['gap']:>6.1f}{r['touches']:>4}{r['lows']:>4}{r['days']:>5}{r['vol_ratio']:>6.2f}"
        for r in rows
    ]
    first = True
    while lines:
        block, size = [], 0
        while lines and size + len(lines[0]) + 1 < 1700:
            size += len(lines[0]) + 1
            block.append(lines.pop(0))
        header = f"**{title}**\n" if first else ""
        post(f"{header}```\n{head}\n" + "\n".join(block) + "\n```")
        first = False


# ───────────── 실행 ─────────────
def main():
    now = dt.datetime.now(NY)
    if now.weekday() >= 5 and not FORCE:
        print("주말. 종료.")
        return
    today = now.date()

    # 최근 알림 기록 {티커: {"res": 저항선, "date": "YYYY-MM-DD"}}
    sent_file = STATE / "sent.json"
    try:
        sent = json.loads(sent_file.read_text()) if sent_file.exists() else {}
    except Exception:
        sent = {}
    cutoff = (today - dt.timedelta(days=REALERT_DAYS)).isoformat()
    sent = {k: v for k, v in sent.items() if v["date"] >= cutoff}

    universes = {g.strip(): load_universe(g.strip()) for g in GROUPS if g.strip()}
    meta, group_of = {}, {}
    for g, u in universes.items():
        for row in u.itertuples():
            if row.ticker not in meta:
                meta[row.ticker] = row
                group_of[row.ticker] = g
    print(f"스캔 대상 {len(meta)}종목 " + ", ".join(f"{g}={len(u)}" for g, u in universes.items()))
    if not meta:
        post("⚠️ 삼각형 스캐너: 종목 목록을 불러오지 못했습니다. Actions 로그를 확인하세요.")
        return

    spy_ret = None
    if REQUIRE_RS:
        for _, sdf in download(["SPY"]):
            sc = sdf["Close"].dropna()
            if len(sc) >= 127:
                spy_ret = {"3m": sc.iloc[-1] / sc.iloc[-64] - 1, "6m": sc.iloc[-1] / sc.iloc[-127] - 1}
        if spy_ret is None:
            print("[경고] SPY 시세를 받지 못해 상대강도 조건을 건너뜁니다.")

    found: dict[str, list] = {}
    scanned = 0
    for ticker, df in download(list(meta)):
        scanned += 1
        t = find_triangle(df, None if FORCE else today, spy_ret)
        if not t:
            continue
        prev = sent.get(ticker)
        if not FORCE and prev and abs(prev["res"] / t["res"] - 1) < RES_TOL_PCT / 100:
            continue                                    # 같은 저항선으로 최근에 이미 알림
        if not FORCE:
            sent[ticker] = {"res": t["res"], "date": today.isoformat()}
        m = meta[ticker]
        found.setdefault(group_of[ticker], []).append(
            {"ticker": ticker, "name": str(m.name) or company_name(ticker), "industry": str(m.industry), **t})

    total = sum(len(v) for v in found.values())
    print(f"시세 확인 {scanned}종목, 신규 후보 {total}건")
    for g in universes:
        rows = found.get(g)
        if rows:
            rows.sort(key=lambda r: r["gap"])           # 저항선에 가까운 순
            send_table(f"{now:%b} {now.day} | Ascending Triangle (형성 중) | {GROUP_LABEL.get(g, g)}", rows)

    if scanned < len(meta) * 0.5:
        post(f"⚠️ 삼각형 스캐너: 시세를 {scanned}/{len(meta)}종목만 받았습니다. 일시적 오류일 수 있습니다.")
    if FORCE:
        post(f"✅ 삼각형 스캐너 테스트 완료: {scanned}종목 확인, 후보 {total}건 (최근 거래일 기준)")
    else:
        sent_file.write_text(json.dumps(sent))


if __name__ == "__main__":
    sys.exit(main())
