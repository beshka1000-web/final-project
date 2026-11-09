import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query, Response
from fastapi import Path as PathParam
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from .recommender import Recommender

ART_DIR = Path(os.environ.get('ARTIFACTS_DIR', Path(__file__).resolve().parent.parent / 'artifacts'))

REQUESTS = Counter('recsys_requests_total', 'Запросы рекомендаций', ['strategy'])
LATENCY = Histogram('recsys_request_seconds', 'Время расчёта рекомендаций, с')

TOP_SCORE = Histogram('recsys_top_score', 'Скор лучшего товара в выдаче',
                      buckets=(1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.25, 0.5, 1.0))
HTTP = Counter('recsys_http_requests_total', 'HTTP-запросы', ['path', 'status'])
P_AT_3 = Gauge('recsys_model_precision_at_3_cv', 'Precision@3 модели по кросс-валидации')
MODEL_INFO = Gauge('recsys_model_info', 'Информация о модели', ['model'])

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state['rec'] = Recommender(ART_DIR)
    meta = state['rec'].meta
    P_AT_3.set(meta['precision_at_3_cv'])
    MODEL_INFO.labels(meta['model']).set(1)
    yield


app = FastAPI(title='Recsys service', version='1.0', lifespan=lifespan)


@app.middleware('http')
async def count_http(request, call_next):
    response = await call_next(request)
    route = request.scope.get('route')
    HTTP.labels(route.path if route else 'unmatched', str(response.status_code)).inc()
    return response


class RecommendRequest(BaseModel):
    user_id: int = Field(ge=0, description='visitorid')
    k: int = Field(3, ge=1, le=10, description='сколько товаров вернуть')


class Item(BaseModel):
    rank: int
    item_id: int
    score: float


class RecommendResponse(BaseModel):
    user_id: int
    strategy: str = Field(description='personalized — пользователь есть в истории, popular_fallback — холодный старт')
    recommendations: list[Item]


def _recommend(user_id: int, k: int) -> RecommendResponse:
    t = time.perf_counter()
    recs, strategy = state['rec'].recommend(user_id, k)
    LATENCY.observe(time.perf_counter() - t)
    REQUESTS.labels(strategy).inc()
    if recs:
        TOP_SCORE.observe(recs[0]['score'] if isinstance(recs[0], dict) else recs[0].score)
    return RecommendResponse(user_id=user_id, strategy=strategy, recommendations=recs)


@app.get('/health')
def health():
    meta = state['rec'].meta
    return {'status': 'ok', 'model': meta['model'], 'precision_at_3_cv': meta['precision_at_3_cv']}


@app.get('/recommend/{user_id}', response_model=RecommendResponse)
def recommend_get(user_id: int = PathParam(..., ge=0), k: int = Query(3, ge=1, le=10)):
    return _recommend(user_id, k)


@app.post('/recommend', response_model=RecommendResponse)
def recommend_post(req: RecommendRequest):
    return _recommend(req.user_id, req.k)


@app.get('/metrics')
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
