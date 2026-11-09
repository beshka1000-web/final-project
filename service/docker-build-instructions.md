# Сборка, запуск и экспорт Docker-образа сервиса рекомендаций

Состояние: сервис, модель и Dockerfile готовы, образ `recsys:1.0` собран и экспортирован в файл `recsys-1.0.tar`. Сборка, запуск и проверка выполняются на внешнем сервере, локальный Docker не нужен.

## 1. Сервер

| Параметр | Значение |
|---|---|
| Адрес | `155.138.220.26` |
| Пользователь | `root` |
| Доступ | SSH (по паролю или по ключу, пароль в документацию не включён) |
| Папка проекта на сервере | `/opt/service` |

Запущенные контейнеры (`docker compose`): `recsys` (:8000 и :80), `prometheus` (:9090), `grafana` (:3000).

```bash
ssh root@155.138.220.26
```

## 2. Загрузка проекта на сервер

Из папки `service/` на локальной машине (передаётся весь проект, кроме кэшей):

```bash
tar --exclude=.pytest_cache --exclude=__pycache__ -czf - . | ssh root@155.138.220.26 "mkdir -p /opt/service && tar -xzf - -C /opt/service"
```

## 3. Сборка и запуск на сервере

```bash
cd /opt/service
docker compose up -d --build
```

Поднимаются три контейнера: сервис, Prometheus и Grafana. Папка `artifacts/` смонтирована в контейнер сервиса, поэтому модель можно обновлять без пересборки образа (см. раздел 7).

Только образ сервиса, без мониторинга:

```bash
docker build -t recsys:1.0 .
docker run -d --name recsys -p 8000:8000 recsys:1.0
```

Если контейнер с таким именем уже есть: `docker rm -f recsys`.

## 4. Проверка работы

Сервис загружает модель около минуты, статус `healthy` появляется после этого (`docker ps`). Затем:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/recommend/8
curl "http://localhost:8000/recommend/8?k=5"
curl -i http://localhost:8000/recommend/abc
curl http://localhost:8000/metrics
```

Ожидаемо:
- `/health` возвращает статус, модель и значение Precision@3;
- `/recommend/8` возвращает три товара со `strategy: personalized`;
- неизвестный пользователь получает `strategy: popular_fallback`;
- `/recommend/abc` и `k` вне диапазона 1..10 возвращают `422`;
- `/metrics` отдаёт метрики `recsys_*`.

С локальной машины сервис доступен по адресу `http://155.138.220.26:8000` (документация Swagger: `/docs`), Grafana: `http://155.138.220.26:3000`, Prometheus: `http://155.138.220.26:9090`.

Логи при проблемах:

```bash
docker logs recsys
```

## 5. Экспорт образа в файл

На сервере:

```bash
docker save -o /root/recsys-1.0.tar recsys:1.0
sha256sum /root/recsys-1.0.tar
```

Скачивание на локальную машину (в папку проекта) и проверка контрольной суммы:

```bash
scp root@155.138.220.26:/root/recsys-1.0.tar .
sha256sum recsys-1.0.tar
```

Суммы на сервере и локально должны совпадать.

## 6. Запуск из переданного файла

На любой машине с Docker, исходники не нужны:

```bash
docker load -i recsys-1.0.tar
docker run -d --name recsys -p 8000:8000 recsys:1.0
curl http://localhost:8000/health
```

Модель и таблицы признаков лежат внутри образа. Остановка: `docker rm -f recsys`.

## 7. Обновление модели (переобучение)

1. Переобучить модель локально из корня проекта (нужны `events.csv` и `features/`): `python service/train.py`. Скрипт перезапишет `service/artifacts/` и `service/docs/experiments.json`.
2. Загрузить проект на сервер (раздел 2).
3. Перезапустить сервис, пересборка образа не нужна:

```bash
cd /opt/service
docker compose restart recsys
```

Новое значение Precision@3 автоматически появится на дашборде Grafana.

Чтобы получить новый экспортируемый образ, нужно выполнить `docker compose up -d --build` и повторить раздел 5.

## 8. Чек-лист этапа

- [x] сервис выдаёт рекомендации по id пользователя;
- [x] сервис обёрнут в Docker, образ собран, проверен и экспортирован в `recsys-1.0.tar`;
- [x] есть документация (`README.md`);
- [x] есть презентация (`recsys_presentation.pdf`);
- [x] мониторинг: Prometheus + Grafana + дашборд «Recsys: метрики сервиса и модели».

Следующий этап по плану (необязательно): коллаборативная фильтрация и факторизационные машины, они должны улучшить результат для новых и редких пользователей.
