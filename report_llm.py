"""
Резюме для руководства от компактной языковой модели (Ollama) с автоматической проверкой чисел.
Модель получает готовые факты и не считает ничего сама. Каждый вариант текста проверяется finance.verify_summary;
при ошибке модель получает список недопустимых чисел и пробует ещё раз. Если все попытки неудачны,
используется шаблонное резюме (в отчёте помечается summary_source = template_fallback).
"""
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import finance

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

SYSTEM_PROMPT = (
    "Ты финансовый аналитик банка. Пишешь краткое резюме для руководства на русском языке по готовым фактам. "
    "Строгие правила: 1) используй ТОЛЬКО числа из блока ФАКТЫ и копируй их без изменений, "
    "вместе с разделителями и валютой; 2) ничего не вычисляй, не округляй и не пересчитывай; "
    "3) не используй числа, которых нет в фактах, не нумеруй пункты и не пиши даты цифрами; "
    "4) не выдумывай причин и событий, которых нет в фактах; "
    "5) каждое число сопровождай названием показателя из фактов и не путай показатели: "
    "чистый эффект, предотвращённые потери, понесённые потери, стоимость ложных срабатываний и стоимость "
    "ручных проверок это РАЗНЫЕ числа; 6) вывод о пороге бери из строки «Вывод по порогу» без новых утверждений "
    "и не противоречь ей; "
    "7) формат: один связный абзац из 5-7 предложений: итог (чистый эффект), что дала модель "
    "(предотвращённые потери), цена ошибок (понесённые потери, ложные срабатывания, ручные проверки), "
    "вывод о пороге блокировки, оговорка о том, что расчёт опирается на допущения."
)


def _request(path: str, payload: Optional[dict] = None, timeout: int = 30) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"{OLLAMA_URL}{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_ready(model: str = OLLAMA_MODEL, timeout_s: int = 2700, poll_s: int = 10) -> None:
    """Ждёт, пока Ollama запустится и скачает модель (первый запуск: несколько минут)."""
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            names = [m.get("name", "") for m in _request("/api/tags").get("models", [])]
            if any(n == model or n.split(":")[0] == model for n in names):
                print(f"[llm] Ollama готов, модель {model} доступна")
                return
            last = f"модель {model} ещё не скачана (есть: {names})"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = f"Ollama недоступен: {exc}"
        print(f"[llm] ожидание: {last}", flush=True)
        time.sleep(poll_s)
    raise TimeoutError(f"Ollama не готов за {timeout_s} с: {last}")


def chat(messages: List[dict], model: str = OLLAMA_MODEL, seed: int = 7, timeout: int = 900) -> str:
    out = _request("/api/chat", {
        "model": model, "messages": messages, "stream": False,
        "options": {"temperature": 0.0, "seed": seed, "num_predict": 700, "num_ctx": 4096},
    }, timeout=timeout)
    return out["message"]["content"].strip()


def build_messages(facts: str, feedback: Optional[str] = None) -> List[dict]:
    user = f"ФАКТЫ:\n{facts}\n\nНапиши резюме для руководства."
    if feedback:
        user += f"\n\nИсправь ошибки предыдущего варианта: {feedback}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _feedback(check: Dict[str, Any]) -> str:
    parts = []
    if check.get("unknown_numbers"):
        parts.append("в тексте есть числа, которых нет в фактах: " + ", ".join(check["unknown_numbers"][:10]) +
                     ". Используй только числа из фактов, скопированные дословно")
    if check.get("mislabeled_numbers"):
        parts.append("числа названы не теми показателями: " + "; ".join(
            f"{x['number']} это {', '.join(x['is'])}, а в тексте названо «{x['named']}»" for x in check["mislabeled_numbers"][:4]))
    if check.get("unbound_key_numbers") and not check.get("missing_key_numbers"):
        names = {k: v for k, v in finance.LABEL_NAMES.items()}
        parts.append("не указано рядом с названием: " + ", ".join(names[k] for k in check["unbound_key_numbers"]))
    if check.get("missing_key_numbers"):
        names = {"net_effect": "чистый эффект", "prevented_loss": "предотвращённые потери",
                 "incurred_loss": "понесённые потери", "fp_cost": "стоимость ложных срабатываний"}
        parts.append("не указаны значения: " + ", ".join(names[k] for k in check["missing_key_numbers"]))
    if not check.get("direction_ok", True):
        parts.append("формулировка о знаке чистого эффекта противоречит фактам")
    if not check.get("russian", True):
        parts.append("пиши по-русски")
    if not check.get("length_ok", True):
        parts.append("объём должен быть 5-7 предложений")
    return "; ".join(parts) or "соблюдай правила"


def generate_verified_summary(metrics: Dict[str, Any], facts: str, model: str = OLLAMA_MODEL, max_attempts: int = 3,
                              allow_fallback: bool = True) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    feedback = None
    for i in range(1, max_attempts + 1):
        try:
            text = chat(build_messages(facts, feedback), model=model, seed=6 + i)
        except Exception as exc:  # сеть, таймаут, перегрузка
            attempts.append({"attempt": i, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[llm] попытка {i}: ошибка {exc}", flush=True)
            continue
        check = finance.verify_summary(text, facts, metrics)
        attempts.append({"attempt": i, "ok": check["ok"], "check": check, "text": text})
        print(f"[llm] попытка {i}: проверка {'пройдена' if check['ok'] else 'не пройдена'} "
              f"(лишние числа: {check['unknown_numbers']}, нет ключевых: {check['missing_key_numbers']}, "
              f"перепутаны: {[x['number'] for x in check['mislabeled_numbers']]})", flush=True)
        if check["ok"]:
            return {"summary": text, "source": "llm", "llm_model": model, "attempts": attempts}
        feedback = _feedback(check)

    if not allow_fallback:
        raise RuntimeError("Ни один вариант резюме от языковой модели не прошёл проверку чисел")
    print("[llm] все попытки неудачны: используется шаблонное резюме", flush=True)
    return {"summary": finance.template_summary(metrics), "source": "template_fallback", "llm_model": model,
            "attempts": attempts}
