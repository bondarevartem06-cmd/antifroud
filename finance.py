"""
Финансовые показатели прескоринга, формулировка фактов для языковой модели и автоматическая проверка чисел.
Все деньги считаются в Decimal и округляются до центов; допущения лежат в assumptions.py и сохраняются в отчёте.
"""
import hashlib
import json
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

import db

CENT = Decimal("0.01")


def _d(x: Any) -> Decimal:
    return Decimal(str(x))


def _q(x: Decimal) -> Decimal:
    return x.quantize(CENT, rounding=ROUND_HALF_UP)


def _pct(a: float, b: float) -> float:
    return round(100.0 * a / b, 2) if b else 0.0


# ----------------------------------------------------------------------------
# Экономика одного набора счётчиков
# ----------------------------------------------------------------------------
def economics(tp: int, fp: int, fn: int, tp_amt: Any, fp_amt: Any, fn_amt: Any, a: Dict[str, Any]) -> Dict[str, Decimal]:
    fee, review, fp_fixed, rate = (_d(a[k]) for k in ("chargeback_fee", "review_cost", "fp_fixed_cost", "fp_revenue_rate"))
    prevented = _q(_d(tp_amt) + fee * tp)
    incurred = _q(_d(fn_amt) + fee * fn)
    fp_cost = _q(fp_fixed * fp + rate * _d(fp_amt))
    review_cost = _q(review * (tp + fp))
    return {
        "prevented_loss": prevented,
        "incurred_loss": incurred,
        "fp_cost": fp_cost,
        "review_cost": review_cost,
        "baseline_loss": prevented + incurred,
        "loss_with_model": incurred + fp_cost + review_cost,
        "net_effect": prevented - fp_cost - review_cost,
    }


Q_AGG = """
SELECT count(*) AS n,
       count(*) FILTER (WHERE h.is_fraud) AS n_fraud,
       COALESCE(sum(h.amt), 0) AS total_amt,
       count(*) FILTER (WHERE r.is_fraud_pred AND h.is_fraud) AS tp,
       count(*) FILTER (WHERE r.is_fraud_pred AND NOT h.is_fraud) AS fp,
       count(*) FILTER (WHERE NOT r.is_fraud_pred AND h.is_fraud) AS fn,
       count(*) FILTER (WHERE NOT r.is_fraud_pred AND NOT h.is_fraud) AS tn,
       COALESCE(sum(h.amt) FILTER (WHERE r.is_fraud_pred AND h.is_fraud), 0) AS tp_amt,
       COALESCE(sum(h.amt) FILTER (WHERE r.is_fraud_pred AND NOT h.is_fraud), 0) AS fp_amt,
       COALESCE(sum(h.amt) FILTER (WHERE NOT r.is_fraud_pred AND h.is_fraud), 0) AS fn_amt,
       min(h.trans_time) AS t_min, max(h.trans_time) AS t_max,
       min(r.threshold) AS th_lo, max(r.threshold) AS th_hi
FROM prescoring_results r
JOIN holdout_transactions h ON h.trans_num = r.trans_num
WHERE r.model_version = :mv
"""

# Сценарий «что если порог другой»: счётчики по сетке порогов (по сырому скору)
Q_SCAN = """
WITH g AS (SELECT unnest(CAST(:grid AS double precision[])) AS th)
SELECT g.th,
       count(*) FILTER (WHERE r.fraud_score >= g.th AND h.is_fraud) AS tp,
       count(*) FILTER (WHERE r.fraud_score >= g.th AND NOT h.is_fraud) AS fp,
       count(*) FILTER (WHERE r.fraud_score < g.th AND h.is_fraud) AS fn,
       COALESCE(sum(h.amt) FILTER (WHERE r.fraud_score >= g.th AND h.is_fraud), 0) AS tp_amt,
       COALESCE(sum(h.amt) FILTER (WHERE r.fraud_score >= g.th AND NOT h.is_fraud), 0) AS fp_amt,
       COALESCE(sum(h.amt) FILTER (WHERE r.fraud_score < g.th AND h.is_fraud), 0) AS fn_amt
FROM g CROSS JOIN prescoring_results r
JOIN holdout_transactions h ON h.trans_num = r.trans_num
WHERE r.model_version = :mv
GROUP BY g.th ORDER BY g.th
"""

Q_RAW = """
SELECT h.amt, h.is_fraud, r.is_fraud_pred
FROM prescoring_results r JOIN holdout_transactions h ON h.trans_num = r.trans_num
WHERE r.model_version = :mv
"""


def compute_metrics(model_version: str, a: Dict[str, Any]) -> Dict[str, Any]:
    """Все показатели считаются SQL-запросами к prescoring_results и holdout_transactions."""
    with db.get_engine().connect() as c:
        g = c.execute(text(Q_AGG), {"mv": model_version}).mappings().one()
        if g["n"] == 0:
            raise RuntimeError(f"Нет результатов прескоринга для версии модели {model_version}")
        if g["th_lo"] != g["th_hi"]:
            raise RuntimeError("В результатах прескоринга встречаются разные пороги")
        threshold = float(g["th_hi"])
        grid = sorted({round(float(x), 3) for x in [0.005, 0.01, 0.02, 0.03]} |
                      {round(float(x), 2) for x in np.arange(0.05, 1.0, 0.05)} | {threshold})
        scan_rows = c.execute(text(Q_SCAN), {"mv": model_version, "grid": grid}).mappings().all()

    tp, fp, fn, tn = (int(g[k]) for k in ("tp", "fp", "fn", "tn"))
    eco = economics(tp, fp, fn, g["tp_amt"], g["fp_amt"], g["fn_amt"], a)

    scan, best = [], None
    for r in scan_rows:
        e = economics(int(r["tp"]), int(r["fp"]), int(r["fn"]), r["tp_amt"], r["fp_amt"], r["fn_amt"], a)
        item = {"threshold": round(float(r["th"]), 6), "tp": int(r["tp"]), "fp": int(r["fp"]), "fn": int(r["fn"]),
                "net_effect": float(e["net_effect"])}
        scan.append(item)
        if best is None or item["net_effect"] > best["net_effect"]:
            best = item

    f = {k: float(v) for k, v in eco.items()}
    return {
        "currency": a["currency"],
        "period_start": g["t_min"].date().isoformat(),
        "period_end": g["t_max"].date().isoformat(),
        "n_transactions": int(g["n"]),
        "n_fraud": int(g["n_fraud"]),
        "fraud_rate_pct": _pct(int(g["n_fraud"]), int(g["n"])),
        "total_amount": float(_q(_d(g["total_amt"]))),
        "threshold": threshold,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision_pct": _pct(tp, tp + fp),
        "recall_pct": _pct(tp, tp + fn),
        "baseline_loss": f["baseline_loss"],
        "prevented_loss": f["prevented_loss"],
        "incurred_loss": f["incurred_loss"],
        "fp_cost": f["fp_cost"],
        "review_cost": f["review_cost"],
        "loss_with_model": f["loss_with_model"],
        "net_effect": f["net_effect"],
        "loss_reduction_pct": _pct(f["prevented_loss"], f["baseline_loss"]),
        "best_threshold": best["threshold"],
        "net_effect_best": best["net_effect"],
        "threshold_scan": scan,
    }


def report_key(model_version: str, threshold: float, a: Dict[str, Any], n: int, n_fraud: int, llm_model: str) -> str:
    """Ключ идемпотентности: одни и те же входные данные, допущения и LLM дают один и тот же отчёт."""
    payload = json.dumps({"mv": model_version, "th": round(threshold, 6), "a": a, "n": n, "nf": n_fraud,
                          "llm": llm_model}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------
# Независимая перепроверка метрик (другой путь агрегации: pandas вместо SQL)
# ----------------------------------------------------------------------------
def recompute_independent(model_version: str, a: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.read_sql(text(Q_RAW), db.get_engine(), params={"mv": model_version})
    amt = df["amt"].astype("float64")
    fraud, pred = df["is_fraud"].astype(bool), df["is_fraud_pred"].astype(bool)
    m_tp, m_fp, m_fn = pred & fraud, pred & ~fraud, ~pred & fraud
    tp, fp, fn = int(m_tp.sum()), int(m_fp.sum()), int(m_fn.sum())
    eco = economics(tp, fp, fn, round(float(amt[m_tp].sum()), 2), round(float(amt[m_fp].sum()), 2),
                    round(float(amt[m_fn].sum()), 2), a)
    out = {k: float(v) for k, v in eco.items()}
    out.update({"tp": tp, "fp": fp, "fn": fn, "n_transactions": int(len(df))})
    return out


def check_identities(m: Dict[str, Any]) -> Dict[str, bool]:
    close = lambda x, y: abs(x - y) <= 0.011
    return {
        "baseline = prevented + incurred": close(m["baseline_loss"], m["prevented_loss"] + m["incurred_loss"]),
        "net = prevented - fp_cost - review": close(m["net_effect"], m["prevented_loss"] - m["fp_cost"] - m["review_cost"]),
        "loss_with_model = incurred + fp_cost + review":
            close(m["loss_with_model"], m["incurred_loss"] + m["fp_cost"] + m["review_cost"]),
        "net = baseline - loss_with_model": close(m["net_effect"], m["baseline_loss"] - m["loss_with_model"]),
        "tp + fn = n_fraud": m["tp"] + m["fn"] == m["n_fraud"],
        "tp + fp + fn + tn = n": m["tp"] + m["fp"] + m["fn"] + m["tn"] == m["n_transactions"],
    }


# ----------------------------------------------------------------------------
# Факты для языковой модели
# ----------------------------------------------------------------------------
def fmt_int(n: Any) -> str:
    return f"{int(n):,}".replace(",", " ")


def fmt_num(v: float, d: int = 2) -> str:
    return f"{v:,.{d}f}".replace(",", " ").replace(".", ",")


def fmt_money(v: float, cur: str = "USD") -> str:
    return f"{fmt_num(v)} {cur}"


def threshold_insight(m: Dict[str, Any]) -> str:
    """Вывод о пороге считается программно: языковая модель только пересказывает его."""
    cur = m["currency"]
    best = next((x for x in m["threshold_scan"] if abs(x["threshold"] - m["best_threshold"]) < 1e-9), None)
    if best is None or abs(m["best_threshold"] - round(m["threshold"], 6)) < 1e-9:
        return f"Текущий порог {fmt_num(m['threshold'], 3)} оптимален на этой выборке."
    gain = round(m["net_effect_best"] - m["net_effect"], 2)
    return (f"Текущий порог {fmt_num(m['threshold'], 3)} даёт чистый эффект {fmt_money(m['net_effect'], cur)}. "
            f"Порог {fmt_num(m['best_threshold'], 3)} дал бы чистый эффект {fmt_money(m['net_effect_best'], cur)}, "
            f"то есть на {fmt_money(gain, cur)} больше; при нём ложных блокировок было бы {fmt_int(best['fp'])} "
            f"вместо {fmt_int(m['fp'])}, пропусков {fmt_int(best['fn'])} вместо {fmt_int(m['fn'])}. "
            f"Порог подобран по этой же выборке, поэтому выигрыш оптимистичен.")


def build_facts(m: Dict[str, Any], a: Dict[str, Any]) -> str:
    cur = m["currency"]
    mon = lambda v: fmt_money(v, cur)
    lines = [
        f"Выборка: {fmt_int(m['n_transactions'])} операций, из них мошеннических {fmt_int(m['n_fraud'])} "
        f"({fmt_num(m['fraud_rate_pct'])}%). Период: {m['period_start']} – {m['period_end']}.",
        f"Порог блокировки модели: {fmt_num(m['threshold'], 3)}.",
        f"Мошенничество поймано в {fmt_int(m['tp'])} случаях, пропущено в {fmt_int(m['fn'])} случаях; "
        f"ложных блокировок {fmt_int(m['fp'])}. Точность блокировок {fmt_num(m['precision_pct'])}%, "
        f"полнота {fmt_num(m['recall_pct'])}%.",
        f"Потери без модели: {mon(m['baseline_loss'])}.",
        f"Предотвращённые потери: {mon(m['prevented_loss'])}.",
        f"Понесённые потери (пропущенное мошенничество): {mon(m['incurred_loss'])}.",
        f"Стоимость ложных срабатываний: {mon(m['fp_cost'])}.",
        f"Стоимость ручных проверок: {mon(m['review_cost'])}.",
        f"Чистый эффект модели: {mon(m['net_effect'])}.",
        f"Модель предотвращает {fmt_num(m['loss_reduction_pct'])}% потерь.",
        f"Вывод по порогу (рассчитан программно): {threshold_insight(m)}",
        f"Допущения: расход на пропущенный случай {mon(a['chargeback_fee'])}; ручная проверка одной операции "
        f"{mon(a['review_cost'])}; фиксированная стоимость ложной блокировки {mon(a['fp_fixed_cost'])}; "
        f"потерянная выручка при ложной блокировке {fmt_num(a['fp_revenue_rate'] * 100)}% суммы операции.",
    ]
    return "\n".join(lines)


def template_summary(m: Dict[str, Any]) -> str:
    """Резюме по шаблону: числа берутся из метрик, поэтому проверку проходит по построению."""
    cur = m["currency"]
    mon = lambda v: fmt_money(v, cur)
    tone = "положительный" if m["net_effect"] >= 0 else "отрицательный"
    return (
        f"По результатам прескоринга отложенной выборки ({fmt_int(m['n_transactions'])} операций, мошеннических "
        f"{fmt_int(m['n_fraud'])}) чистый эффект модели {tone}: {mon(m['net_effect'])}. "
        f"Модель предотвратила потери на {mon(m['prevented_loss'])}, это {fmt_num(m['loss_reduction_pct'])}% потерь "
        f"без модели ({mon(m['baseline_loss'])}). "
        f"Понесённые потери из-за пропущенного мошенничества составили {mon(m['incurred_loss'])}. "
        f"Ложные срабатывания обошлись в {mon(m['fp_cost'])}, ручные проверки в {mon(m['review_cost'])}. "
        f"При пороге {fmt_num(m['threshold'], 3)} поймано {fmt_int(m['tp'])} случаев мошенничества, "
        f"пропущено {fmt_int(m['fn'])}, ложных блокировок {fmt_int(m['fp'])}. "
        f"{threshold_insight(m)} Расчёт опирается на допущения, зафиксированные в отчёте."
    )


# ----------------------------------------------------------------------------
# Проверка чисел в тексте резюме
# ----------------------------------------------------------------------------
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}\.\d{1,2}\.\d{4}\b")
_LIST_MARK = re.compile(r"(?m)^\s*(?:\d+[.)]|[-•*])\s+")
_NUM = re.compile(r"(?<![\w.,])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?")
_UNIT = re.compile(r"\s{0,2}(%|тыс|млн|млрд)", re.IGNORECASE)
_SCALE = {"%": 1.0, "тыс": 1e3, "млн": 1e6, "млрд": 1e9}


@dataclass
class Num:
    raw: str
    value: float      # значение с учётом тыс./млн./млрд. (абсолютное)
    tol: float        # допуск округления: половина последнего показанного разряда
    percent: bool
    start: int = 0    # позиция в проверяемой строке
    end: int = 0


def parse_numbers(s: str) -> List[Num]:
    blank = lambda m_: " " * len(m_.group(0))
    s = _LIST_MARK.sub(blank, _DATE.sub(blank, s))
    out = []
    for m in _NUM.finditer(s):
        raw = m.group(0)
        digits = re.sub(r"[ \u00a0\u202f]", "", raw).replace(",", ".")
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        u = _UNIT.match(s, m.end())
        unit = u.group(1).lower() if u else ""
        scale = _SCALE.get(unit, 1.0)
        out.append(Num(raw=raw, value=float(digits) * scale, tol=0.5 * 10 ** (-decimals) * scale + 1e-9,
                       percent=(unit == "%"), start=m.start(), end=m.end()))
    return out


def _allowed_values(facts: str, m: Dict[str, Any]) -> List[float]:
    vals = [n.value for n in parse_numbers(facts)]
    vals += [v * 100 for v in vals if 0 < v < 1]                       # 0,02 -> «2%»
    for key in ("period_start", "period_end"):                          # компоненты дат (21 июня 2020)
        y, mo, d = m[key].split("-")
        vals += [float(y), float(mo), float(d)]
    return vals


def _matches(n: Num, target: float) -> bool:
    return abs(n.value - abs(target)) <= n.tol


_STEMS = {
    "net_effect": ("эффект",),
    "prevented_loss": ("предотвра",),
    "incurred_loss": ("понесен", "пропущен", "ущерб"),
    "fp_cost": ("ложн",),
    "review_cost": ("ручн",),
    "baseline_loss": ("без модел",),
}
LABEL_NAMES = {"net_effect": "чистый эффект", "prevented_loss": "предотвращённые потери",
               "incurred_loss": "понесённые потери", "fp_cost": "стоимость ложных срабатываний",
               "review_cost": "стоимость ручных проверок", "baseline_loss": "потери без модели"}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_SEP = re.compile(r"(?<!\d),|,(?!\d)|[;:()—–]")


def label_binding(summary: str, m: Dict[str, Any]):
    """Число, равное денежному показателю, должно стоять рядом с названием именно этого показателя
    (ближайшее название в той же части предложения). Ловит путаницу вроде «предотвратила потери в размере
    <чистый эффект>»."""
    mislabeled, bound = [], set()
    values = {k: m[k] for k in _STEMS}
    for sent in (x for x in _SENT_SPLIT.split(summary) if x.strip()):
        low = sent.lower().replace("ё", "е")
        occ = [(o.start(), o.end(), k) for k, stems in _STEMS.items() for st in stems
               for o in re.finditer(re.escape(st), low)]
        for n in parse_numbers(sent):
            hit = [k for k, v in values.items() if _matches(n, v)]
            if not hit:
                continue
            cs, ce = 0, len(sent)                      # границы части предложения, где стоит число
            for sep in _SEP.finditer(sent):
                if sep.end() <= n.start:
                    cs = sep.end()
                elif sep.start() >= n.end:
                    ce = sep.start()
                    break
            cands = [o for o in occ if o[0] >= cs and o[1] <= ce] or occ
            if not cands:
                continue
            dist = lambda o: n.start - o[1] if o[1] <= n.start else (o[0] - n.end if o[0] >= n.end else 0)
            label = min(cands, key=dist)[2]
            if label in hit:
                bound.add(label)
            else:
                mislabeled.append({"number": n.raw, "is": [LABEL_NAMES[k] for k in hit], "named": LABEL_NAMES[label]})
    unbound = [k for k in ("net_effect", "prevented_loss", "incurred_loss", "fp_cost") if k not in bound]
    return mislabeled, unbound


def verify_summary(summary: str, facts: str, m: Dict[str, Any]) -> Dict[str, Any]:
    """Каждое число в тексте должно совпадать (с точностью до показанного округления) с числом из фактов;
    ключевые показатели обязаны присутствовать; знак чистого эффекта не должен противоречить формулировкам."""
    nums = parse_numbers(summary)
    allowed = _allowed_values(facts, m)
    unknown = [n.raw for n in nums if not any(_matches(n, a) for a in allowed)]

    key = {"net_effect": m["net_effect"], "prevented_loss": m["prevented_loss"],
           "incurred_loss": m["incurred_loss"], "fp_cost": m["fp_cost"]}
    missing = [k for k, v in key.items() if not any(_matches(n, v) for n in nums)]

    low = summary.lower()
    neg_phrases = ("отрицательн", "в минус", "не окупа", "убыточ")
    if m["net_effect"] < 0:
        direction_ok = any(p in low for p in neg_phrases + ("убыт",))
    else:
        direction_ok = not any(p in low for p in neg_phrases)

    letters = [ch for ch in summary if ch.isalpha()]
    cyr = sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in letters)
    russian = bool(letters) and cyr / len(letters) >= 0.5
    length_ok = 150 <= len(summary) <= 3500

    mislabeled, unbound = label_binding(summary, m)

    ok = (not unknown and not missing and not mislabeled and not unbound and direction_ok and russian and length_ok)
    return {"ok": ok, "unknown_numbers": unknown, "missing_key_numbers": missing, "mislabeled_numbers": mislabeled,
            "unbound_key_numbers": unbound, "direction_ok": direction_ok, "russian": russian,
            "length_ok": length_ok, "numbers_found": len(nums)}


def validate_report(row: Dict[str, Any]) -> Dict[str, Any]:
    """Итоговая проверка отчёта из базы: тождества, независимый пересчёт, сверка порога, числа в тексте."""
    m, a = row["metrics"], row["assumptions"]["parameters"]
    checks: Dict[str, Any] = {}

    checks["identities"] = check_identities(m)

    ind = recompute_independent(row["model_version"], a)
    diffs = {k: abs(ind[k] - m[k]) for k in ("prevented_loss", "incurred_loss", "fp_cost", "review_cost", "net_effect")}
    checks["independent_recompute"] = {
        "max_abs_diff": round(max(diffs.values()), 4),
        "counts_equal": [ind["tp"], ind["fp"], ind["fn"], ind["n_transactions"]] ==
                        [m["tp"], m["fp"], m["fn"], m["n_transactions"]],
        "ok": max(diffs.values()) <= 0.011 and [ind["tp"], ind["fp"], ind["fn"]] == [m["tp"], m["fp"], m["fn"]],
    }

    at = [s for s in m["threshold_scan"] if abs(s["threshold"] - round(m["threshold"], 6)) < 1e-9]
    checks["scan_matches_headline"] = bool(at) and (at[0]["tp"], at[0]["fp"], at[0]["fn"]) == (m["tp"], m["fp"], m["fn"])

    if row.get("summary"):
        checks["summary_numbers"] = verify_summary(row["summary"], build_facts(m, a), m)
    else:
        checks["summary_numbers"] = {"ok": False, "error": "резюме отсутствует"}

    ok = (all(checks["identities"].values()) and checks["independent_recompute"]["ok"]
          and checks["scan_matches_headline"] and checks["summary_numbers"]["ok"])
    checks["ok"] = ok
    return checks
