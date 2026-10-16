import subprocess
import sys

def main():
    # Переменная с базовым набором признаков
    base = "amt,log_amt,hour,dow,age,gender,dist_km,city_pop,category"

    # Формируем список аргументов для вызова python-скрипта
    command = [
        sys.executable,  # Использует тот же интерпретатор Python, что и текущий скрипт
        "compare_models.py",
        "--train", "fraudTrain.csv",
        "--test", "fraudTest.csv",
        "--set", f"no_fp={base}",
        "--set", f"with_card={base},card_all",
        "--set", f"full={base},card_all,fp_count,fp_max_lift",
        "--scenario", "cat_w|cat:shopping_net=3,misc_net=3,grocery_pos=2",
        "--scenario", "amt_w|amt:300-600=2,600-1000=3,>1000=4",
        "--models", "lr", "xgb", "lgbm",
        "--rank_by", "cost",
        "--out_prefix", "full"
    ]

    print("Запуск команды:")
    print(" ".join(command))
    print("-" * 50)

    # Выполняем команду с выводом логов в реальном времени
    result = subprocess.run(command)

    # Возвращаем код завершения (0 — Успешно)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()