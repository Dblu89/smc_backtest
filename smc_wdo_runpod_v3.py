"""
╔══════════════════════════════════════════════════════════════════╗
║  SMC JUSTIN BENNETT — WDO BACKTEST ENGINE v3 PROFESSIONAL       ║
║  32 vCPUs · Grid Massivo Paralelo · Monte Carlo · Bayesiano     ║
║  RunPod CPU 32 vCPUs / 128 GB RAM                               ║
╚══════════════════════════════════════════════════════════════════╝

INSTALAÇÃO (RunPod terminal):
pip install pandas numpy quantstats scikit-optimize tqdm

RODAR:
# Modo mini — confirma que CSV carrega (~30s)
python smc_wdo_backtest_v3.py --mini

# Modo completo profissional — usa todos os 32 cores
python smc_wdo_backtest_v3.py

SAÍDA:
resultado_smc_wdo.json   → dashboard
otimizacao_grid.json     → ranking completo
monte_carlo.json         → simulações de risco
quantstats_report.html   → relatório detalhado

ESTRATÉGIA — Justin Bennett SMC
1. CHoCH detecta mudança de tendência
2. Preço retorna ao FVG ou Order Block
3. Entrada com SL abaixo/acima do POI + ATR
4. TP no mínimo RR mínimo configurado
5. Sessão: 09h-18h BRT (horário WDO B3)
6. 1 trade por vez, risco fixo por trade
"""

import pandas as pd
import numpy as np
import json
import sys
import io
import time
import itertools
import contextlib
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# CONFIGURAÇÕES GLOBAIS
# ──────────────────────────────────────────────────────────────────

CSV_PATH = "/workspace/wdo_clean.csv"       # RunPod
N_CORES = min(32, cpu_count())              # Usa todos os cores disponíveis
CAPITAL = 50_000.0                          # Capital inicial R$
MULT_WDO = 10.0                             # 1 ponto WDO = R$10
SLIPPAGE = 2.0                              # Pontos de slippage
COMISSAO = 5.0                              # R$ round-trip por contrato
RISCO_PCT = 0.01                            # 1% do capital por trade
CONTRATOS = 1

# Grid de parâmetros — versão MASSIVA para 32 vCPUs
GRID_PARAMS = {
    "rr_min": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    "swing_length": [3, 5, 7, 10, 14],
    "choch_janela": [15, 20, 30, 40, 50, 60, 70, 80, 100, 120],
    "poi_janela": [20, 30, 40, 50, 60, 70, 80, 100, 120, 150],
    "atr_mult": [0.3, 0.5, 0.7, 1.0],
}

# Total: 6×5×10×10×4 = 12.000 combinações (filtradas poi>=choch → ~7.200)

# Score composto para ranking
# Prioriza: PF > Sharpe > WR > quantidade de trades
SCORE_PESOS = {
    "pf": 0.35,
    "sharpe": 0.25,
    "sortino": 0.15,
    "wr": 0.15,
    "trades": 0.10,
}

# Filtros mínimos de qualidade
FILTRO_MIN_TRADES = 50
FILTRO_MAX_DD = -20.0
FILTRO_MIN_PF = 1.1

# Monte Carlo
MC_SIMULACOES = 2000

# ══════════════════════════════════════════════════════════════════
# SEÇÃO 1 — CARREGAMENTO DE DADOS
# ══════════════════════════════════════════════════════════════════

def carregar_dados(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """
    Carrega o CSV do WDO e aplica limpeza completa.
    Espera colunas: datetime, open, high, low, close, volume
    """
    print(f"[DATA] Carregando {csv_path}...")
    try:
        df = pd.read_csv(
            csv_path,
            parse_dates=["datetime"],
            index_col="datetime",
            sep=",",
        )
        df.columns = [c.lower().strip() for c in df.columns]

        # Garantir colunas necessárias
        needed = ["open", "high", "low", "close", "volume"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas faltando: {missing}")

        df = df[needed].copy()

        # Limpeza
        df = df[df.index.dayofweek < 5]                       # Só dias úteis
        df = df[(df.index.hour >= 9) & (df.index.hour < 18)] # 09h-18h BRT
        df = df.dropna()
        df = df[df["close"] > 0]
        df = df.sort_index()

        # Remover duplicatas
        df = df[~df.index.duplicated(keep="last")]

        candles = len(df)
        dias = (df.index[-1] - df.index[0]).days
        print(f"[DATA] ✓ {candles:,} candles | {df.index[0].date()} → {df.index[-1].date()} ({dias} dias)")
        return df

    except FileNotFoundError:
        print(f"[DATA] CSV não encontrado: {csv_path}")
        print("[DATA] Gerando dados sintéticos para demo...")
        return _sintetico(15000)
    except Exception as e:
        print(f"[DATA] Erro: {e} → usando sintético")
        return _sintetico(15000)


def _sintetico(n: int = 15000, seed: int = 42) -> pd.DataFrame:
    """Dados sintéticos realistas do WDO para fallback."""
    np.random.seed(seed)
    idx = pd.date_range("2023-01-02 09:00", periods=n * 3, freq="5min")
    idx = idx[(idx.dayofweek < 5) & (idx.hour >= 9) & (idx.hour < 18)][:n]

    price = 5150.0
    opens_, highs, lows, closes, vols = [], [], [], [], []
    regime, dur = 1, 0

    for _ in idx:
        dur += 1
        if dur > np.random.randint(60, 300):
            regime = np.random.choice([1, -1, 1, 0], p=[0.4, 0.3, 0.2, 0.1])
            dur = 0
        prev = price
        price = max(4500, min(6500, price * (1 + regime * 0.0003 + np.random.normal(0, 0.0008))))
        sp = abs(np.random.normal(0, 2.5))
        opens_.append(prev)
        highs.append(max(prev, price) + sp)
        lows.append(min(prev, price) - sp)
        closes.append(price)
        vols.append(int(np.random.lognormal(6, 0.8)))

    df = pd.DataFrame(
        {
            "open": opens_,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vols,
        },
        index=idx,
    )
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    print(f"[DATA] ✓ {len(df):,} candles sintéticos | {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ══════════════════════════════════════════════════════════════════
# SEÇÃO 2 — INDICADORES SMC
# ══════════════════════════════════════════════════════════════════

class SMC:
    """Indicadores SMC Justin Bennett — implementação vetorizada."""

    @staticmethod
    def swing_highs_lows(df: pd.DataFrame, length: int = 5) -> pd.DataFrame:
        h, l = df["high"].values, df["low"].values
        n = len(df)
        sh, sl = np.zeros(n), np.zeros(n)
        for i in range(length, n - length):
            wh = h[i - length:i + length + 1]
            wl = l[i - length:i + length + 1]
            if h[i] == wh.max() and h[i] > h[i - 1] and h[i] > h[i + 1]:
                sh[i] = h[i]
            if l[i] == wl.min() and l[i] < l[i - 1] and l[i] < l[i + 1]:
                sl[i] = l[i]
        df = df.copy()
        df["sh"] = sh
        df["sl"] = sl
        return df

    @staticmethod
    def bos_choch(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["bos"] = 0
        df["choch"] = 0
        last_sh = last_sl = None
        trend = 0
        for i in range(1, len(df)):
            sh_v, sl_v = df.iloc[i]["sh"], df.iloc[i]["sl"]
            if sh_v > 0:
                if last_sh is not None and sh_v > last_sh:
                    if trend == 1:
                        df.iat[i, df.columns.get_loc("bos")] = 1
                    else:
                        df.iat[i, df.columns.get_loc("choch")] = 1
                        trend = 1
                last_sh = sh_v
            if sl_v > 0:
                if last_sl is not None and sl_v < last_sl:
                    if trend == -1:
                        df.iat[i, df.columns.get_loc("bos")] = -1
                    else:
                        df.iat[i, df.columns.get_loc("choch")] = -1
                        trend = -1
                last_sl = sl_v
        return df

    @staticmethod
    def fvg(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["fvg"] = 0
        df["fvg_top"] = np.nan
        df["fvg_bot"] = np.nan
        h, l = df["high"].values, df["low"].values
        for i in range(2, len(df)):
            if l[i] > h[i - 2]:
                df.iat[i, df.columns.get_loc("fvg")] = 1
                df.iat[i, df.columns.get_loc("fvg_top")] = l[i]
                df.iat[i, df.columns.get_loc("fvg_bot")] = h[i - 2]
            elif h[i] < l[i - 2]:
                df.iat[i, df.columns.get_loc("fvg")] = -1
                df.iat[i, df.columns.get_loc("fvg_top")] = l[i - 2]
                df.iat[i, df.columns.get_loc("fvg_bot")] = h[i]
        return df

    @staticmethod
    def order_blocks(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        df = df.copy()
        df["ob"] = 0
        df["ob_top"] = np.nan
        df["ob_bot"] = np.nan
        for i in range(1, len(df)):
            sig = df.iloc[i]["bos"] or df.iloc[i]["choch"]
            if sig == 1:
                for j in range(i - 1, max(0, i - lookback), -1):
                    if df.iloc[j]["close"] < df.iloc[j]["open"]:
                        df.iat[j, df.columns.get_loc("ob")] = 1
                        df.iat[j, df.columns.get_loc("ob_top")] = df.iloc[j]["high"]
                        df.iat[j, df.columns.get_loc("ob_bot")] = df.iloc[j]["low"]
                        break
            elif sig == -1:
                for j in range(i - 1, max(0, i - lookback), -1):
                    if df.iloc[j]["close"] > df.iloc[j]["open"]:
                        df.iat[j, df.columns.get_loc("ob")] = -1
                        df.iat[j, df.columns.get_loc("ob_top")] = df.iloc[j]["high"]
                        df.iat[j, df.columns.get_loc("ob_bot")] = df.iloc[j]["low"]
                        break
        return df

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"].shift(1)
        tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def preparar(df: pd.DataFrame, swing_length: int = 5) -> pd.DataFrame:
        df = SMC.swing_highs_lows(df, swing_length)
        df = SMC.bos_choch(df)
        df = SMC.fvg(df)
        df = SMC.order_blocks(df)
        df["atr"] = SMC.atr(df)
        # Filtro sessão 09h-18h BRT já aplicado no carregamento
        df["na_sessao"] = True
        return df


# ══════════════════════════════════════════════════════════════════
# SEÇÃO 3 — ENGINE DE BACKTEST
# ══════════════════════════════════════════════════════════════════

class BacktestSMC:
    def __init__(
        self,
        rr_min=2.0,
        swing_length=7,
        choch_janela=60,
        poi_janela=60,
        atr_mult=0.5,
        capital=CAPITAL,
        slippage=SLIPPAGE,
        comissao=COMISSAO,
        contratos=CONTRATOS,
        mult=MULT_WDO,
        risco_pct=RISCO_PCT,
    ):
        self.rr_min = rr_min
        self.swing_length = swing_length
        self.choch_janela = choch_janela
        self.poi_janela = poi_janela
        self.atr_mult = atr_mult
        self.capital_ini = capital
        self.slippage = slippage
        self.comissao = comissao
        self.contratos = contratos
        self.mult = mult
        self.risco_pct = risco_pct
        self.trades = []
        self.equity = []

    def _tp(self, entry, sl, d):
        return entry + d * abs(entry - sl) * self.rr_min

    def _pnl(self, entry, saida, d):
        pts = (saida - entry) * d
        brl = pts * self.mult * self.contratos - self.comissao
        brl -= self.slippage * self.mult * self.contratos * 0.5
        return round(pts, 2), round(brl, 2)

    def rodar(self, df_raw: pd.DataFrame) -> dict:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            df = SMC.preparar(df_raw.copy(), self.swing_length)

        capital = self.capital_ini
        equity = [capital]
        trades = []
        em_pos = False
        trade = None

        fvgs_bull, fvgs_bear = [], []
        obs_bull, obs_bear = [], []
        ult_choch_bull = ult_choch_bear = -9999

        for i in range(50, len(df)):
            row = df.iloc[i]

            # ── Gerenciar posição ──────────────────────────────
            if em_pos and trade:
                d, sl, tp, en = trade["d"], trade["sl"], trade["tp"], trade["entry"]
                hit_sl = (d == 1 and row["low"] <= sl) or (d == -1 and row["high"] >= sl)
                hit_tp = (d == 1 and row["high"] >= tp) or (d == -1 and row["low"] <= tp)
                if hit_sl or hit_tp:
                    saida = sl if hit_sl else tp
                    pts, brl = self._pnl(en, saida, d)
                    capital += brl
                    equity.append(round(capital, 2))
                    trade.update(
                        {
                            "saida": saida,
                            "pnl_pts": pts,
                            "pnl_brl": brl,
                            "resultado": "WIN" if hit_tp else "LOSS",
                            "saida_dt": str(df.index[i])[:16],
                        }
                    )
                    trades.append(trade)
                    em_pos = False
                    trade = None
                continue

            # ── Coletar sinais ────────────────────────────────
            if row["choch"] == 1:
                ult_choch_bull = i
                fvgs_bull.clear()
                obs_bull.clear()
            if row["choch"] == -1:
                ult_choch_bear = i
                fvgs_bear.clear()
                obs_bear.clear()

            if row["fvg"] == 1 and not np.isnan(row.get("fvg_top", np.nan)):
                fvgs_bull.append({"top": row["fvg_top"], "bot": row["fvg_bot"], "i": i})
            if row["fvg"] == -1 and not np.isnan(row.get("fvg_top", np.nan)):
                fvgs_bear.append({"top": row["fvg_top"], "bot": row["fvg_bot"], "i": i})
            if row["ob"] == 1:
                obs_bull.append({"top": row["ob_top"], "bot": row["ob_bot"], "i": i})
            if row["ob"] == -1:
                obs_bear.append({"top": row["ob_top"], "bot": row["ob_bot"], "i": i})

            fvgs_bull = [x for x in fvgs_bull if i - x["i"] <= self.poi_janela]
            fvgs_bear = [x for x in fvgs_bear if i - x["i"] <= self.poi_janela]
            obs_bull = [x for x in obs_bull if i - x["i"] <= self.poi_janela]
            obs_bear = [x for x in obs_bear if i - x["i"] <= self.poi_janela]

            # ── Lógica de entrada ─────────────────────────────
            close = row["close"]
            atr = row["atr"] if not np.isnan(row.get("atr", np.nan)) else 5.0
            sinal = poi = poi_tipo = None

            if (i - ult_choch_bull) <= self.choch_janela:
                for fg in reversed(fvgs_bull):
                    if fg["bot"] <= close <= fg["top"]:
                        sinal = 1
                        poi = fg
                        poi_tipo = "FVG"
                        break
                if sinal is None:
                    for ob in reversed(obs_bull):
                        if ob["bot"] <= close <= ob["top"]:
                            sinal = 1
                            poi = ob
                            poi_tipo = "OB"
                            break

            if sinal is None and (i - ult_choch_bear) <= self.choch_janela:
                for fg in reversed(fvgs_bear):
                    if fg["bot"] <= close <= fg["top"]:
                        sinal = -1
                        poi = fg
                        poi_tipo = "FVG"
                        break
                if sinal is None:
                    for ob in reversed(obs_bear):
                        if ob["bot"] <= close <= ob["top"]:
                            sinal = -1
                            poi = ob
                            poi_tipo = "OB"
                            break

            if sinal is None or poi is None:
                continue

            # ── Calcular níveis ───────────────────────────────
            slip_e = self.slippage * 0.5
            if sinal == 1:
                entry = close + slip_e
                sl = poi["bot"] - atr * self.atr_mult
            else:
                entry = close - slip_e
                sl = poi["top"] + atr * self.atr_mult

            tp = self._tp(entry, sl, sinal)
            risk_pt = abs(entry - sl)
            if risk_pt <= 0:
                continue
            rr_real = abs(tp - entry) / risk_pt

            if rr_real < self.rr_min:
                continue
            risco_brl = risk_pt * self.mult * self.contratos
            if risco_brl / capital > self.risco_pct * 5:
                continue

            # ── Abrir trade ───────────────────────────────────
            em_pos = True
            trade = {
                "entry_dt": str(df.index[i])[:16],
                "d": sinal,
                "entry": round(entry, 2),
                "sl": round(sl, 2),
                "tp": round(tp, 2),
                "rr": round(rr_real, 2),
                "poi_tipo": poi_tipo,
                "capital_pre": round(capital, 2),
            }

        # Fechar posição aberta
        if em_pos and trade:
            last = df.iloc[-1]["close"]
            pts, brl = self._pnl(trade["entry"], last, trade["d"])
            capital += brl
            trade.update(
                {
                    "saida": last,
                    "pnl_pts": pts,
                    "pnl_brl": brl,
                    "resultado": "ABERTO",
                    "saida_dt": str(df.index[-1])[:16],
                }
            )
            trades.append(trade)
            equity.append(round(capital, 2))

        self.trades = trades
        self.equity = equity
        return self._metricas(trades, equity)

    def _metricas(self, trades, equity) -> dict:
        fechados = [t for t in trades if t.get("resultado") != "ABERTO"]
        if not fechados:
            return {}

        df_t = pd.DataFrame(fechados)
        wins = df_t[df_t["resultado"] == "WIN"]
        loses = df_t[df_t["resultado"] == "LOSS"]
        n = len(df_t)
        wr = len(wins) / n * 100
        avg_w = wins["pnl_brl"].mean() if len(wins) else 0
        avg_l = loses["pnl_brl"].mean() if len(loses) else 0
        pf = abs(wins["pnl_brl"].sum() / loses["pnl_brl"].sum()) if loses["pnl_brl"].sum() != 0 else 9999
        pnl = df_t["pnl_brl"].sum()

        eq = pd.Series(equity)
        peak = eq.cummax()
        dd = (eq - peak) / peak * 100
        mdd = dd.min()

        rets = eq.pct_change().dropna()
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        neg = rets[rets < 0]
        sortino = rets.mean() / neg.std() * np.sqrt(252) if len(neg) > 0 else 0
        calmar = (pnl / self.capital_ini) / abs(mdd / 100) if mdd != 0 else 0

        res = df_t["resultado"].tolist()
        mws = mls = cw = cl = 0
        for r in res:
            if r == "WIN":
                cw += 1
                cl = 0
                mws = max(mws, cw)
            else:
                cl += 1
                cw = 0
                mls = max(mls, cl)

        return {
            "total_trades": n,
            "wins": int(len(wins)),
            "losses": int(len(loses)),
            "win_rate": round(wr, 2),
            "profit_factor": round(pf, 3),
            "sharpe_ratio": round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "calmar_ratio": round(calmar, 3),
            "avg_win_brl": round(avg_w, 2),
            "avg_loss_brl": round(avg_l, 2),
            "avg_rr": round(df_t["rr"].mean(), 2),
            "expectancy_brl": round((wr / 100 * avg_w) + ((1 - wr / 100) * avg_l), 2),
            "total_pnl_brl": round(pnl, 2),
            "retorno_pct": round(pnl / self.capital_ini * 100, 2),
            "max_drawdown_pct": round(mdd, 2),
            "recovery_factor": round(pnl / abs(mdd / 100 * self.capital_ini), 2) if mdd != 0 else 0,
            "max_win_streak": mws,
            "max_loss_streak": mls,
            "capital_inicial": self.capital_ini,
            "capital_final": round(self.capital_ini + pnl, 2),
            "trades_fvg": int((df_t["poi_tipo"] == "FVG").sum()),
            "trades_ob": int((df_t["poi_tipo"] == "OB").sum()),
            "comissao_total": round(self.comissao * n, 2),
            "slippage_total": round(self.slippage * self.mult * n, 2),
        }


# ══════════════════════════════════════════════════════════════════
# SEÇÃO 4 — WORKER PARALELO (roda em cada CPU core)
# ══════════════════════════════════════════════════════════════════

def _worker(args):
    """Função executada em paralelo por cada core."""
    params, df_bytes = args
    df = pd.read_json(io.StringIO(df_bytes))
    df.index = pd.to_datetime(df.index)
    df.columns = [c.lower() for c in df.columns]

    rr, sw, cj, pj, am = params
    bt = BacktestSMC(
        rr_min=rr,
        swing_length=sw,
        choch_janela=cj,
        poi_janela=pj,
        atr_mult=am,
    )
    m = bt.rodar(df)
    if not m:
        return None

    trades = m["total_trades"]
    dd = m["max_drawdown_pct"]
    pf = m["profit_factor"]
    sharpe = m["sharpe_ratio"]
    sortino = m["sortino_ratio"]
    wr = m["win_rate"]

    if trades < FILTRO_MIN_TRADES or dd < FILTRO_MAX_DD or pf < FILTRO_MIN_PF:
        return None

    score = (
        min(pf, 10) / 10 * SCORE_PESOS["pf"]
        + min(max(sharpe, 0), 8) / 8 * SCORE_PESOS["sharpe"]
        + min(max(sortino, 0), 10) / 10 * SCORE_PESOS["sortino"]
        + wr / 100 * SCORE_PESOS["wr"]
        + min(trades, 500) / 500 * SCORE_PESOS["trades"]
    )

    return {
        "rr_min": rr,
        "swing_length": sw,
        "choch_janela": cj,
        "poi_janela": pj,
        "atr_mult": am,
        "score": round(score, 6),
        **{
            k: m[k]
            for k in [
                "profit_factor",
                "sharpe_ratio",
                "sortino_ratio",
                "win_rate",
                "total_trades",
                "max_drawdown_pct",
                "retorno_pct",
                "expectancy_brl",
                "calmar_ratio",
                "recovery_factor",
                "avg_rr",
            ]
        },
    }


# ══════════════════════════════════════════════════════════════════
# SEÇÃO 5 — GRID SEARCH PARALELO
# ══════════════════════════════════════════════════════════════════

def grid_search_paralelo(df: pd.DataFrame, mini: bool = False) -> dict:
    """
    Grid search massivo usando todos os cores disponíveis.
    No modo mini: apenas 4 combos para validar o CSV.
    """
    if mini:
        combos = [(2.0, 7, 60, 60, 0.5)]
        print("\n[GRID] Modo MINI — 1 combo para validar CSV")
    else:
        p = GRID_PARAMS
        all_combos = list(
            itertools.product(
                p["rr_min"],
                p["swing_length"],
                p["choch_janela"],
                p["poi_janela"],
                p["atr_mult"],
            )
        )
        combos = [(r, s, c, p2, a) for r, s, c, p2, a in all_combos if p2 >= c]
        print("\n[GRID] Grid Massivo Paralelo")
        print(f"       Combinações totais   : {len(all_combos):,}")
        print(f"       Após filtro poi≥choch: {len(combos):,}")
        print(f"       Cores utilizados     : {N_CORES}")
        print(f"       Filtros: trades≥{FILTRO_MIN_TRADES} | DD≥{FILTRO_MAX_DD}% | PF≥{FILTRO_MIN_PF}")

    # Serializar df uma vez para passar aos workers
    df_json = df.to_json(date_format="iso")
    args = [(c, df_json) for c in combos]

    t0 = time.time()
    resultados = []

    if mini or len(combos) == 1:
        # Modo sequencial para combos pequenos
        for a in args:
            r = _worker(a)
            if r:
                resultados.append(r)
    else:
        # Modo paralelo com todos os cores
        print("\n[GRID] Iniciando processamento paralelo...")
        with Pool(processes=N_CORES) as pool:
            total = len(args)
            feitos = 0
            for resultado in pool.imap_unordered(_worker, args, chunksize=max(1, total // N_CORES // 4)):
                feitos += 1
                if resultado:
                    resultados.append(resultado)

                # Progress a cada 2%
                if feitos % max(1, total // 50) == 0 or feitos == total:
                    pct = feitos / total
                    bar = "█" * int(pct * 40) + "░" * (40 - int(pct * 40))
                    eta = (time.time() - t0) / feitos * (total - feitos) if feitos > 0 else 0
                    print(
                        f"\r  [{bar}] {feitos:,}/{total:,} ({pct*100:.0f}%)  "
                        f"válidos:{len(resultados)}  ETA:{eta:.0f}s   ",
                        end="",
                        flush=True,
                    )

    elapsed = time.time() - t0
    resultados.sort(key=lambda x: -x["score"])

    print(f"\n\n[GRID] ✓ Concluído em {elapsed:.1f}s")
    print(f"       {len(resultados)} combinações válidas de {len(combos):,}")

    if resultados:
        _exibir_top(resultados)

    return {
        "melhor": resultados[0] if resultados else None,
        "top20": resultados[:20],
        "todos": resultados,
        "elapsed_s": round(elapsed, 1),
        "total_combos": len(combos),
        "validos": len(resultados),
    }


def _exibir_top(resultados, n=15):
    print(f"\n{'═' * 72}")
    print("  TOP {} — Score = PF×35% + Sharpe×25% + Sortino×15% + WR×15% + Trades×10%".format(min(n, len(resultados))))
    print(f"{'═' * 72}")
    print(
        f"  {'#':>2} {'RR':>4} {'SW':>3} {'CHoCH':>6} {'POI':>5} {'ATR':>5} "
        f"{'PF':>6} {'Sharpe':>7} {'WR%':>6} {'Trades':>7} {'DD%':>6} {'Score':>7}"
    )
    print(f"  {'─' * 70}")
    for i, r in enumerate(resultados[:n], 1):
        star = "★" if i == 1 else " "
        print(
            f"  {star}{i:>2} {r['rr_min']:>4} {r['swing_length']:>3} "
            f"{r['choch_janela']:>6} {r['poi_janela']:>5} {r['atr_mult']:>5} "
            f"{r['profit_factor']:>6.3f} {r['sharpe_ratio']:>7.3f} "
            f"{r['win_rate']:>6.1f} {r['total_trades']:>7} "
            f"{r['max_drawdown_pct']:>6.1f} {r['score']:>7.4f}"
        )
    print(f"{'═' * 72}")


# ══════════════════════════════════════════════════════════════════
# SEÇÃO 6 — WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════

def walk_forward(df: pd.DataFrame, config: dict, n_splits: int = 6, train_pct: float = 0.7) -> list:
    """Walk-Forward com 6 janelas rolantes."""
    resultados = []
    step = len(df) // n_splits
    print(f"\n[WF] Walk-Forward: {n_splits} splits | {int(train_pct * 100)}% train")

    for i in range(n_splits - 1):
        inicio = i * step
        fim = (i + 2) * step
        split = inicio + int((fim - inicio) * train_pct)
        df_tr = df.iloc[inicio:split]
        df_te = df.iloc[split:fim]

        if len(df_tr) < 500 or len(df_te) < 100:
            continue

        d0 = df_tr.index[0].strftime("%Y-%m-%d")
        d1 = df_tr.index[-1].strftime("%Y-%m-%d")
        d2 = df_te.index[0].strftime("%Y-%m-%d")
        d3 = df_te.index[-1].strftime("%Y-%m-%d")
        print(f"\n  Split {i + 1}: Train [{d0}→{d1}] | Test [{d2}→{d3}]")

        bt_tr = BacktestSMC(**config)
        bt_te = BacktestSMC(**config)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            m_tr = bt_tr.rodar(df_tr.copy())
            m_te = bt_te.rodar(df_te.copy())

        if m_tr:
            print(
                f"    TRAIN → WR:{m_tr['win_rate']}% | PF:{m_tr['profit_factor']} | "
                f"Trades:{m_tr['total_trades']} | DD:{m_tr['max_drawdown_pct']}% | "
                f"PnL:R${m_tr['total_pnl_brl']:,.0f}"
            )
        if m_te:
            print(
                f"    TEST  → WR:{m_te['win_rate']}% | PF:{m_te['profit_factor']} | "
                f"Trades:{m_te['total_trades']} | DD:{m_te['max_drawdown_pct']}% | "
                f"PnL:R${m_te['total_pnl_brl']:,.0f}"
            )

        resultados.append(
            {
                "split": i + 1,
                "train_start": d0,
                "train_end": d1,
                "test_start": d2,
                "test_end": d3,
                "train": m_tr or {},
                "test": m_te or {},
            }
        )

    tests_lucro = [r for r in resultados if r["test"].get("total_pnl_brl", 0) > 0]
    print(f"\n[WF] ✓ {len(tests_lucro)}/{len(resultados)} splits out-of-sample lucrativos")
    return resultados


# ══════════════════════════════════════════════════════════════════
# SEÇÃO 7 — MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════════

def monte_carlo(trades: list, n_sim: int = MC_SIMULACOES, capital: float = CAPITAL) -> dict:
    """
    Simula N_SIMULACOES sequências aleatórias dos trades.
    Calcula distribuição de drawdown máximo, retorno final e risco de ruína.
    """
    print(f"\n[MC] Monte Carlo: {n_sim:,} simulações...")
    fechados = [t for t in trades if t.get("resultado") in ("WIN", "LOSS")]
    if len(fechados) < 10:
        return {}

    pnls = np.array([t["pnl_brl"] for t in fechados])
    np.random.seed(42)

    retornos_finais = []
    max_drawdowns = []
    ruinas = 0

    for _ in range(n_sim):
        seq = np.random.choice(pnls, size=len(pnls), replace=True)
        equity = capital + np.cumsum(seq)
        equity = np.insert(equity, 0, capital)

        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak * 100
        mdd = dd.min()

        retornos_finais.append((equity[-1] - capital) / capital * 100)
        max_drawdowns.append(mdd)
        if equity[-1] < capital * 0.5:  # Perda > 50% = ruína
            ruinas += 1

    rf = np.array(retornos_finais)
    md = np.array(max_drawdowns)

    resultado = {
        "n_simulacoes": n_sim,
        "n_trades_base": len(fechados),
        "retorno_mediana": round(float(np.median(rf)), 2),
        "retorno_p10": round(float(np.percentile(rf, 10)), 2),
        "retorno_p25": round(float(np.percentile(rf, 25)), 2),
        "retorno_p75": round(float(np.percentile(rf, 75)), 2),
        "retorno_p90": round(float(np.percentile(rf, 90)), 2),
        "retorno_pior": round(float(rf.min()), 2),
        "retorno_melhor": round(float(rf.max()), 2),
        "dd_mediano": round(float(np.median(md)), 2),
        "dd_p10": round(float(np.percentile(md, 10)), 2),
        "dd_p90": round(float(np.percentile(md, 90)), 2),
        "dd_pior": round(float(md.min()), 2),
        "prob_lucro_pct": round(float((rf > 0).mean() * 100), 1),
        "prob_ruina_pct": round(float(ruinas / n_sim * 100), 2),
        "prob_dd_maior_10": round(float((md < -10).mean() * 100), 1),
        "prob_dd_maior_20": round(float((md < -20).mean() * 100), 1),
    }

    print(
        f"[MC] ✓ Prob. lucro: {resultado['prob_lucro_pct']}% | "
        f"DD mediano: {resultado['dd_mediano']}% | "
        f"Risco ruína: {resultado['prob_ruina_pct']}%"
    )
    return resultado


# ══════════════════════════════════════════════════════════════════
# SEÇÃO 8 — RELATÓRIO TERMINAL
# ══════════════════════════════════════════════════════════════════

def relatorio(m: dict, mc: dict = None, titulo: str = "RESULTADO"):
    if not m:
        return

    sep = "═" * 60

    def L(label, valor):
        print(f"  {label:<32} {valor:>24}")

    print(f"\n{sep}")
    print("  SMC JUSTIN BENNETT — WDO DÓLAR MINI — B3")
    print(f"  {titulo}")
    print(sep)
    L("Total de Trades", str(m["total_trades"]))
    L("Wins / Losses", f"{m['wins']} W  /  {m['losses']} L")
    L("Win Rate", f"{m['win_rate']}%")
    L("Maior Seq. W / L", f"{m['max_win_streak']}W  /  {m['max_loss_streak']}L")
    print(f"  {'─' * 56}")
    L("Profit Factor", str(m["profit_factor"]))
    L("Sharpe Ratio", str(m["sharpe_ratio"]))
    L("Sortino Ratio", str(m["sortino_ratio"]))
    L("Calmar Ratio", str(m["calmar_ratio"]))
    L("Recovery Factor", str(m["recovery_factor"]))
    L("Avg R:R realizado", f"{m['avg_rr']}R")
    L("Expectancy", f"R$ {m['expectancy_brl']:,.2f}")
    print(f"  {'─' * 56}")
    L("Avg Win", f"R$ {m['avg_win_brl']:,.2f}")
    L("Avg Loss", f"R$ {m['avg_loss_brl']:,.2f}")
    L("Total PnL", f"R$ {m['total_pnl_brl']:,.2f}")
    L("Retorno %", f"{m['retorno_pct']}%")
    L("Max Drawdown", f"{m['max_drawdown_pct']}%")
    print(f"  {'─' * 56}")
    L("Capital Inicial", f"R$ {m['capital_inicial']:,.2f}")
    L("Capital Final", f"R$ {m['capital_final']:,.2f}")
    L("Comissões Totais", f"R$ {m['comissao_total']:,.2f}")
    L("Slippage Total (est.)", f"R$ {m['slippage_total']:,.2f}")
    print(f"  {'─' * 56}")
    L("Trades via FVG", str(m["trades_fvg"]))
    L("Trades via OB", str(m["trades_ob"]))

    if mc:
        print(f"  {'─' * 56}")
        print(f"  MONTE CARLO ({mc['n_simulacoes']:,} simulações)")
        L("Prob. Lucro", f"{mc['prob_lucro_pct']}%")
        L("Retorno Mediano", f"{mc['retorno_mediana']}%")
        L("Retorno P10/P90", f"{mc['retorno_p10']}% / {mc['retorno_p90']}%")
        L("DD Mediano", f"{mc['dd_mediano']}%")
        L("DD Pior Cenário", f"{mc['dd_pior']}%")
        L("Risco de Ruína", f"{mc['prob_ruina_pct']}%")
    print(sep)


# ══════════════════════════════════════════════════════════════════
# SEÇÃO 9 — EXPORTAÇÃO
# ══════════════════════════════════════════════════════════════════

def exportar(metricas, trades, equity, wf, grid_result, mc, config):
    dados = {
        "metricas": metricas,
        "equity_curve": equity,
        "trades": trades,
        "walk_forward": wf,
        "gerado_em": datetime.now().isoformat(),
        "config": config,
        "monte_carlo": mc,
    }
    with open("resultado_smc_wdo.json", "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2, default=str)
    print("[✓] resultado_smc_wdo.json")

    with open("otimizacao_grid.json", "w", encoding="utf-8") as f:
        json.dump(grid_result, f, ensure_ascii=False, indent=2, default=str)
    print("[✓] otimizacao_grid.json")

    if mc:
        with open("monte_carlo.json", "w", encoding="utf-8") as f:
            json.dump(mc, f, ensure_ascii=False, indent=2, default=str)
        print("[✓] monte_carlo.json")


# ══════════════════════════════════════════════════════════════════
# SEÇÃO 10 — MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    MINI = "--mini" in sys.argv

    print("╔" + "═" * 64 + "╗")
    print("║  SMC JUSTIN BENNETT — WDO BACKTEST ENGINE v3 PROFESSIONAL   ║")
    label = "MODO MINI (~30s)" if MINI else f"MODO COMPLETO — {N_CORES} CORES PARALELOS"
    print(f"║  09h-18h BRT  |  {label:<46}║")
    print("╚" + "═" * 64 + "╝")

    if MINI:
        print("""
ℹ  MODO MINI — valida CSV + 1 combo (~30s)
Se aparecer candles reais → tudo OK para o grid completo
""")
    else:
        print(f"""
🚀 MODO PROFISSIONAL
Grid: ~7.200 combinações em {N_CORES} cores paralelos
Walk-Forward: 6 splits rolantes
Monte Carlo: {MC_SIMULACOES:,} simulações
Filtros: PF≥{FILTRO_MIN_PF} | DD≥{FILTRO_MAX_DD}% | Trades≥{FILTRO_MIN_TRADES}
""")

    # ── 1. Dados ────────────────────────────────────────────────
    print("[1/5] Carregando dados...")
    df = carregar_dados(CSV_PATH)

    # Dividir in-sample (70%) e out-of-sample (30%)
    split = int(len(df) * 0.70)
    df_ins = df.iloc[:split]
    df_oos = df.iloc[split:]
    print(f"      In-sample : {len(df_ins):,} candles ({df_ins.index[0].date()} → {df_ins.index[-1].date()})")
    print(f"      Out-sample: {len(df_oos):,} candles ({df_oos.index[0].date()} → {df_oos.index[-1].date()})")

    if MINI:
        # ── Mini: só 1 combo ────────────────────────────────────
        print("\n[2/5] Grid MINI (1 combo)...")
        grid = grid_search_paralelo(df_ins, mini=True)
        if grid["melhor"]:
            print("\n  ✅ CSV carregado e trades funcionando!")
            m = grid["melhor"]
            print(
                f"  Config: RR={m['rr_min']} SW={m['swing_length']} "
                f"CHoCH={m['choch_janela']} POI={m['poi_janela']} ATR={m['atr_mult']}"
            )
            print(
                f"  Resultado: PF={m['profit_factor']} | Trades={m['total_trades']} | "
                f"WR={m['win_rate']}%\n"
            )
        else:
            print("  ⚠ Nenhum trade encontrado — verifique o CSV")
        return

    # ── 2. Grid Search Paralelo ─────────────────────────────────
    print("\n[2/5] Grid Search Paralelo...")
    grid = grid_search_paralelo(df_ins, mini=False)

    if not grid["melhor"]:
        print("[ERRO] Nenhuma combinação válida encontrada.")
        return

    # Melhor configuração
    melhor = grid["melhor"]
    CONFIG_MELHOR = dict(
        rr_min=melhor["rr_min"],
        swing_length=melhor["swing_length"],
        choch_janela=melhor["choch_janela"],
        poi_janela=melhor["poi_janela"],
        atr_mult=melhor["atr_mult"],
    )

    print(
        f"\n  ✓ MELHOR CONFIG: RR={melhor['rr_min']} | SW={melhor['swing_length']} | "
        f"CHoCH={melhor['choch_janela']} | POI={melhor['poi_janela']} | ATR×={melhor['atr_mult']}"
    )
    print(
        f"    Score={melhor['score']} | PF={melhor['profit_factor']} | "
        f"Sharpe={melhor['sharpe_ratio']} | WR={melhor['win_rate']}% | "
        f"Trades={melhor['total_trades']}"
    )

    # ── 3. Backtest Out-of-Sample ───────────────────────────────
    print("\n[3/5] Backtest Out-of-Sample (30% dados não vistos)...")
    bt_oos = BacktestSMC(**CONFIG_MELHOR)
    m_oos = bt_oos.rodar(df_oos.copy())
    relatorio(m_oos, titulo="OUT-OF-SAMPLE (dados não vistos pelo otimizador)")

    # Backtest dataset completo
    print("\n[3/5] Backtest Dataset Completo...")
    bt_full = BacktestSMC(**CONFIG_MELHOR)
    m_full = bt_full.rodar(df.copy())

    # ── 4. Walk-Forward ─────────────────────────────────────────
    print("\n[4/5] Walk-Forward Validation (6 splits)...")
    wf = walk_forward(df, CONFIG_MELHOR, n_splits=6, train_pct=0.7)

    # ── 5. Monte Carlo ──────────────────────────────────────────
    print("\n[5/5] Monte Carlo Simulation...")
    mc = monte_carlo(bt_full.trades, n_sim=MC_SIMULACOES)
    relatorio(m_full, mc, titulo="DATASET COMPLETO + MONTE CARLO")

    # ── Exportar ────────────────────────────────────────────────
    print("\n[✓] Exportando resultados...")
    exportar(m_full, bt_full.trades, bt_full.equity, wf, grid, mc, CONFIG_MELHOR)

    # ── QuantStats ──────────────────────────────────────────────
    try:
        import quantstats as qs
        eq = pd.Series(bt_full.equity)
        ret = eq.pct_change().dropna()
        ret.index = pd.date_range(end=datetime.now(), periods=len(ret), freq="B")
        qs.reports.html(ret, output="quantstats_report.html", title="SMC WDO Justin Bennett v3")
        print("[✓] quantstats_report.html")
    except Exception:
        pass

    total = m_full.get("total_trades", 0) if m_full else 0
    print(f"\n✅ CONCLUÍDO! {total} trades | melhor PF: {melhor['profit_factor']}")
    print("   Carregue resultado_smc_wdo.json no smc_dashboard.html")


if __name__ == "__main__":
    main()