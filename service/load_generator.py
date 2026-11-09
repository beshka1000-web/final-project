"""Генератор нагрузки для сервиса рекомендаций: вызовы идут заданное время (по умолчанию 30 минут).

Смесь запросов (чтобы на дашборде Grafana были все панели):
  55 %  GET  /recommend/{известный пользователь}?k=1..5   -> strategy=personalized
  15 %  POST /recommend (известный пользователь)          -> strategy=personalized
  20 %  GET  /recommend/{случайный новый id}              -> strategy=popular_fallback (холодный старт)
   6 %  некорректные запросы (abc, k=50, k=0, отрицательный id) -> 422
   4 %  GET  /health

Только стандартная библиотека Python 3. Список известных пользователей читается из known_users.txt
рядом со скриптом.

Запуск:  python3 load_generator.py --url http://localhost:8000 --minutes 30 --rps 2
"""
import argparse
import json
import math
import random
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://localhost:8000')
parser.add_argument('--minutes', type=float, default=30)
parser.add_argument('--rps', type=float, default=2.0, help='средняя частота запросов в секунду')
parser.add_argument('--workers', type=int, default=4)
args = parser.parse_args()

known = [int(x) for x in (Path(__file__).parent / 'known_users.txt').read_text().split()] or [8]
stats, lock = Counter(), threading.Lock()
t_start = time.time()
deadline = t_start + args.minutes * 60


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(args.url + path, data=data, method=method,
                                 headers={'Content-Type': 'application/json'} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
            return r.status
    except urllib.error.HTTPError as e:
        e.read()
        return e.code
    except Exception:
        return 0


def one_request():
    x = random.random()
    if x < 0.55:
        return 'personalized_get', call('GET', f'/recommend/{random.choice(known)}?k={random.randint(1, 5)}')
    if x < 0.70:
        return 'personalized_post', call('POST', '/recommend', {'user_id': random.choice(known), 'k': random.randint(1, 5)})
    if x < 0.90:
        return 'cold_start', call('GET', f'/recommend/{random.randint(10_000_000, 999_999_999)}')
    if x < 0.96:
        bad = random.choice(['/recommend/abc', '/recommend/8?k=50', '/recommend/8?k=0', '/recommend/-5'])
        return 'invalid', call('GET', bad)
    return 'health', call('GET', '/health')


def worker():
    while time.time() < deadline:
        # скорость плавно «дышит» (±60 % с периодом 10 минут), чтобы графики были живыми
        phase = (time.time() - t_start) / 600 * 2 * math.pi
        rate = max(args.rps * (1 + 0.6 * math.sin(phase)), 0.2) / args.workers
        time.sleep(random.expovariate(rate))
        kind, status = one_request()
        with lock:
            stats[(kind, status)] += 1


def report():
    while time.time() < deadline:
        time.sleep(60)
        with lock:
            total = sum(stats.values())
            by_status = Counter()
            for (_, s), n in stats.items():
                by_status[s] += n
        mins = (time.time() - t_start) / 60
        print(f'[{mins:5.1f} мин] всего {total} запросов, по статусам: {dict(sorted(by_status.items()))}', flush=True)


print(f'старт: {args.url}, {args.minutes} мин, ~{args.rps} req/s, известных пользователей: {len(known)}', flush=True)
threads = [threading.Thread(target=worker) for _ in range(args.workers)] + [threading.Thread(target=report, daemon=True)]
for t in threads:
    t.start()
for t in threads[:-1]:
    t.join()

print('итог:', flush=True)
for (kind, status), n in sorted(stats.items()):
    print(f'  {kind:<18} {status}: {n}', flush=True)
print(f'всего {sum(stats.values())} запросов за {(time.time() - t_start) / 60:.1f} мин', flush=True)
