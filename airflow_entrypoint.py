"""
Точка входа контейнера airflow (запускается системным Python образа).
 - создаёт базу «airflow» в том же PostgreSQL (если её нет) и настраивает подключение, пароль экранируется;
 - применяет миграции метаданных Airflow, создаёт администратора с паролем из .env;
 - запускает scheduler и webserver; при остановке одного из процессов завершается контейнер.
Режим «cli» (python airflow_entrypoint.py cli <аргументы>) выполняет команду airflow с теми же настройками:
    docker compose exec airflow af dags trigger antifraud_prescoring_report
"""
import hashlib
import os
import signal
import subprocess
import sys
import time
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text

import db

AIRFLOW = "/opt/airflow-venv/bin/airflow"
AF_DB = "airflow"


def airflow_env() -> dict:
    u = db.database_url()
    conn = (f"postgresql+psycopg2://{quote_plus(u.username)}:{quote_plus(u.password)}"
            f"@{u.host}:{u.port or 5432}/{AF_DB}")
    secret = os.getenv("AIRFLOW__WEBSERVER__SECRET_KEY") or hashlib.sha256(
        (os.environ.get("AIRFLOW_ADMIN_PASSWORD", "") + "airflow-secret").encode()).hexdigest()
    return {**os.environ, "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN": conn, "AIRFLOW__WEBSERVER__SECRET_KEY": secret}


def ensure_database() -> None:
    eng = create_engine(db.database_url().set(database="postgres"), isolation_level="AUTOCOMMIT")
    with eng.connect() as c:
        if c.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": AF_DB}).first() is None:
            c.execute(text(f'CREATE DATABASE "{AF_DB}"'))
            print(f"[airflow] создана база {AF_DB}", flush=True)
    eng.dispose()


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        os.execve(AIRFLOW, [AIRFLOW] + sys.argv[2:], airflow_env())

    ensure_database()
    env = airflow_env()
    user, pwd = os.getenv("AIRFLOW_ADMIN_USER", "admin"), os.environ["AIRFLOW_ADMIN_PASSWORD"]

    subprocess.run([AIRFLOW, "db", "migrate"], check=True, env=env)
    # Пользователь НЕ удаляется и не пересоздаётся: иначе сессия в браузере указывает на несуществующий id,
    # и страницы Airflow отвечают Internal Server Error. Если пользователь есть, create завершится ошибкой (это нормально),
    # а reset-password всегда приводит пароль в соответствие с .env.
    subprocess.run([AIRFLOW, "users", "create", "--username", user, "--password", pwd, "--firstname", "Admin",
                    "--lastname", "User", "--role", "Admin", "--email", "admin@example.com"], env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # reset-password аннулирует открытые сессии, поэтому вызываем его только когда пароль в .env изменился
    # (отпечаток хранится в томе airflow_logs, который переживает перезапуски контейнера)
    marker = os.path.join(os.getenv("AIRFLOW_HOME", "/opt/airflow"), "logs", ".admin_password_sha256")
    digest = hashlib.sha256(f"{user}:{pwd}".encode()).hexdigest()
    applied = open(marker).read().strip() if os.path.exists(marker) else ""
    if applied != digest:
        subprocess.run([AIRFLOW, "users", "reset-password", "--username", user, "--password", pwd], check=True, env=env)
        with open(marker, "w") as f:
            f.write(digest)

    procs = [subprocess.Popen([AIRFLOW, "scheduler"], env=env),
             subprocess.Popen([AIRFLOW, "webserver", "--port", "8080"], env=env)]

    def stop(*_):
        for p in procs:
            p.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while all(p.poll() is None for p in procs):
        time.sleep(2)
    stop()
    for p in procs:
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            p.kill()
    return next((p.returncode for p in procs if p.returncode not in (0, None)), 0)


if __name__ == "__main__":
    sys.exit(main())
