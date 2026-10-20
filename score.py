"""
Запуск эксперимента одной кнопкой (Run в PyCharm): сначала compare_models.py, затем plot_results.py.
Меняйте только блок НАСТРОЙКИ. Файлы compare_models.py, fpgrowth_features.py, plot_results.py
должны лежать в одной папке со score.py. CSV берутся из DATA_DIR, результаты пишутся в RESULTS_DIR
(обе папки задаются переменными окружения; по умолчанию - текущая папка, как раньше).
"""
import os
import subprocess
import sys
import time

# ============================ НАСТРОЙКИ ============================
DATA_DIR = os.getenv("DATA_DIR", ".")          # в Docker: data
RESULTS_DIR = os.getenv("RESULTS_DIR", ".")    # в Docker: results
TRAIN = os.path.join(DATA_DIR, "fraudTrain.csv")
TEST = os.path.join(DATA_DIR, "fraudTest.csv")
PREFIX = "full"               # префикс результата: full_summary.csv, full_curve.csv; графики в папке out_full
MODELS = ["lr", "xgb", "lgbm", "rf"]  # доступно: lr, rf, xgb, lgbm
MAX_TRAIN_ROWS = 300_000      # 0 = весь train (долго)
RANK_BY = "cost"              # prob, money или cost
EXTRA_ARGS = []               # дополнительные параметры, например ["--cap", "0.005", "--seed", "1"]

BASE = "amt,log_amt,hour,dow,age,gender,dist_km,city_pop,category"   # card_all убран: раньше он дублировался во всех наборах
SETS = {                      # наборы признаков: имя -> список токенов
    "no_fp": BASE,
    "with_card": f"{BASE},card_all",
    "full": f"{BASE},card_all,fp_count,fp_max_lift",
}
RISK = "shopping_net=3,misc_net=3,grocery_pos=2"   # веса категорий покупок
BIG = "300-600=2,600-1000=3,>1000=4"                # веса размера платежа
SCENARIOS = {                 # сценарии весов (сценарий plain без весов добавляется сам)
    "cat_w": f"cat:{RISK}",
    "amt_w": f"amt:{BIG}",
    # "both": f"cat:{RISK}|amt:{BIG}",
}

RUN_PLOTS = True              # после обучения построить графики и таблицу
PLOT_SET = None               # набор для графиков; None = последний набор из SETS
# ===================================================================

HERE = os.path.dirname(os.path.abspath(__file__))  # пути считаются от папки со score.py


def show(cmd):
    """Команда в виде, пригодном для копирования в терминал (аргументы со спецсимволами в кавычках)."""
    return " ".join(f'"{c}"' if any(ch in c for ch in ' |><;&') else c for c in cmd)


def run(cmd, env):
    print("Запуск команды:")
    print(show(cmd))
    print("-" * 50, flush=True)
    code = subprocess.run(cmd, env=env, cwd=HERE).returncode  # вывод идёт в консоль в реальном времени
    if code != 0:
        print(f"\nКоманда завершилась с ошибкой (код {code}).")
        sys.exit(code)


def main():
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # чтобы русский текст не ломался в консоли Windows
    start = time.time()
    os.makedirs(os.path.join(HERE, RESULTS_DIR), exist_ok=True)
    out_prefix = os.path.join(RESULTS_DIR, PREFIX)

    cmd = [sys.executable, "compare_models.py", "--train", TRAIN, "--test", TEST]
    for name, tokens in SETS.items():
        cmd += ["--set", f"{name}={tokens}"]
    for name, body in SCENARIOS.items():
        cmd += ["--scenario", f"{name}|{body}"]
    cmd += ["--models", *MODELS, "--rank_by", RANK_BY, "--max_train_rows", str(MAX_TRAIN_ROWS),
            "--out_prefix", out_prefix, *EXTRA_ARGS]
    run(cmd, env)

    if RUN_PLOTS:
        plot = [sys.executable, "plot_results.py", "--file", f"{out_prefix}_summary.csv",
                "--out_dir", os.path.join(RESULTS_DIR, f"out_{PREFIX}")]
        if PLOT_SET:
            plot += ["--plot_set", PLOT_SET]
        print()
        run(plot, env)

    print(f"\nГотово за {(time.time() - start) / 60:.1f} мин. Результаты: {out_prefix}_summary.csv, "
          f"{out_prefix}_curve.csv, папка {os.path.join(RESULTS_DIR, 'out_' + PREFIX)}")


if __name__ == "__main__":
    main()
