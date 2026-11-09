## Докер-образ

Сервис упакован в Docker-образ `recsys:1.0` и передаётся файлом `recsys-1.0.tar` (186 МБ). Образ собран на базе `python:3.12-slim`. Модель XGBoost и таблицы признаков лежат внутри образа, поэтому для запуска нужен только Docker. Образ собирался и проверялся на внешнем сервере.

### 1. Передан docker image, экспортированный из докера (2 балла)

Образ экспортирован командой `docker save`:

```bash
docker save -o recsys-1.0.tar recsys:1.0
```

Файл `recsys-1.0.tar` скачан с сервера на локальную машину. Контрольная сумма SHA-256 на сервере и локально совпала, значит файл передан без повреждений.

### 2. Сервис запускается из командной строки, поднимается и принимает запросы (2 балла)

Запуск из переданного файла на любой машине с Docker:

```bash
docker load -i recsys-1.0.tar
docker run -d --name recsys -p 8000:8000 recsys:1.0
curl http://localhost:8000/health
```

Запуск проверен на сервере отдельным тестовым контейнером без подключённых volume. Сервис загрузил модель и ответил на `/health`:

```json
{"status":"ok","model":"xgboost classifier","precision_at_3_cv":0.01478}
```

В образе есть `HEALTHCHECK`: контейнер получает статус `healthy`, когда сервис готов. На это нужно от 10 до 60 секунд. Полный стек (сервис, Prometheus и Grafana) поднимается командой `docker compose up -d --build`.

### 3. Запросы к сервису по документации проходят и возвращают результат (3 балла)

Запросы проверены на запущенном контейнере.

| Запрос | Результат |
|---|---|
| `GET /recommend/8` | три товара, `strategy: personalized` |
| `GET /recommend/8?k=5` | пять товаров |
| `POST /recommend` с телом `{"user_id": 8, "k": 2}` | два товара, `strategy: personalized` |
| `GET /recommend/999999999` (новый пользователь) | три популярных товара, `strategy: popular_fallback` |
| `GET /health` | статус, модель и значение Precision@3 |

Пример ответа:

```json
{"user_id": 8, "strategy": "personalized",
 "recommendations": [{"rank": 1, "item_id": 461686, "score": 0.00039},
                     {"rank": 2, "item_id": 320130, "score": 0.00016},
                     {"rank": 3, "item_id": 384302, "score": 0.00016}]}
```

Описание всех эндпоинтов приведено в README (раздел 7). Интерактивная документация Swagger доступна по адресу `/docs`.

### 4. Сервис устойчив к некорректным запросам (1 балл)

Входные данные проверяются через pydantic: `user_id` должен быть целым числом не меньше 0, а `k` лежит в диапазоне от 1 до 10. Некорректные запросы получают ответ `422` с описанием ошибки, сервис продолжает работать.

| Запрос | Ответ |
|---|---|
| `GET /recommend/abc` | `422`, `Input should be a valid integer` |
| `GET /recommend/8?k=50` | `422`, `Input should be less than or equal to 10` |

После таких запросов сервис остаётся в статусе `healthy`. Ошибки учитываются в метрике `recsys_http_requests_total{status="422"}` и видны на дашборде Grafana.

### 5. У сервиса есть endpoint для снятия метрик (1 балл)

Эндпоинт `GET /metrics` отдаёт метрики в формате Prometheus:

```bash
curl http://localhost:8000/metrics
```

Метрики сервиса и модели:
- `recsys_requests_total{strategy}`: запросы по стратегиям;
- `recsys_request_seconds`: задержка расчёта;
- `recsys_top_score`: скор лучшего товара (индикатор дрейфа);
- `recsys_http_requests_total{path,status}`: HTTP-запросы по путям и статусам;
- `recsys_model_precision_at_3_cv`: качество модели;
- `recsys_model_info{model}`: информация о модели.

Prometheus опрашивает `/metrics` каждые 15 секунд, данные отображаются на дашборде Grafana «Recsys: метрики сервиса и модели».
