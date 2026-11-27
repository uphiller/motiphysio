# HTTP 추론 서버.
# GET  /health  모델이 GPU에 올라간 뒤 200, 그 전에는 503
# POST /infer   본문 = JPEG 바이트, 헤더 X-Request-Id
#
# 검출·관절 계산은 기준 파이프라인과 같다.
#   1) RTMDet-x(COCO)로 검출 → 사람(label 0) 중 점수 가장 높은 상자 1개 (점수 0.3 미만뿐이면 사진 전체를 상자로)
#   2) 그 상자로 RTMW3D-x top-down 추론 (flip test 없음, 공식 config 그대로)
# 정답지 설정은 fp32 / batch 1 / cv2 이다.
# 허용 오차를 측정한 변형이 fp16, batch 16 이므로 기본값은 그 범위에 둔다.
# JPEG 는 cv2.imdecode 로 읽는다 (cv2.imread 와 같은 디코더).

import asyncio
import logging
import os
import queue
import sys
import threading
import time
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

MMPOSE = Path("/opt/mmpose")
DET_CFG = "rtmdet_x_8xb32-300e_coco.py"
DET_CKPT = "/models/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth"
POSE_CFG = MMPOSE / "projects/rtmpose3d/configs/rtmw3d-x_8xb32_cocktail14-384x288.py"
POSE_CKPT = "/models/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth"
DET_SCORE_THR = 0.3

PRECISION = os.environ.get("PRECISION", "fp16")
MAX_BATCH = int(os.environ.get("MAX_BATCH", "16"))
MAX_WAIT = float(os.environ.get("MAX_WAIT_MS", "20")) / 1000.0
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", "2048"))
DEVICE = os.environ.get("DEVICE", "cuda:0")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("server")
models_ready = threading.Event()
batch_queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUE)
det_model = None
pose_model = None


def det_config_path() -> str:
    import mmdet
    p = Path(mmdet.__file__).parent / ".mim/configs/rtmdet" / DET_CFG
    if not p.exists():
        sys.exit(f"mmdet config 없음: {p}")
    return str(p)


def load_models(device: str):
    from mmdet.apis import init_detector
    from mmpose.apis import init_model
    from mmpose.utils import adapt_mmdet_pipeline
    import rtmpose3d  # noqa: F401  (PYTHONPATH=/opt/mmpose/projects/rtmpose3d) — 커스텀 모듈 등록

    det = init_detector(det_config_path(), DET_CKPT, device=device)
    det.cfg = adapt_mmdet_pipeline(det.cfg)
    pose = init_model(str(POSE_CFG), POSE_CKPT, device=device)
    return det, pose


def pick_boxes(det_results, shapes) -> list[np.ndarray]:
    out = []
    for res, (h, w) in zip(det_results, shapes):
        p = res.pred_instances.cpu().numpy()
        keep = (p.labels == 0) & (p.scores >= DET_SCORE_THR)
        if keep.any():
            i = int(np.argmax(np.where(keep, p.scores, -1)))
            out.append(p.bboxes[i].astype(np.float32))
        else:
            out.append(np.array([0, 0, w, h], dtype=np.float32))
    return out


def pose_batch(pose_model, imgs: list[np.ndarray], boxes: list[np.ndarray]):
    """mmpose.apis.inference_topdown 과 같은 처리를 여러 사진에 한 번에 (사진당 상자 1개)"""
    from mmengine.dataset import Compose, pseudo_collate
    pipeline = Compose(pose_model.cfg.test_dataloader.dataset.pipeline)
    data_list = []
    for img, box in zip(imgs, boxes):
        d = dict(img=img, bbox=box[None], bbox_score=np.ones(1, dtype=np.float32))
        d.update(pose_model.dataset_meta)
        data_list.append(pipeline(d))
    with torch.no_grad():
        return pose_model.test_step(pseudo_collate(data_list))


def half_features(det) -> None:
    """검출기는 특징 추출(backbone+neck)만 fp16, 상자 디코딩·NMS 는 fp32 (mmcv NMS 가 fp16 을 못 받음)"""
    orig = det.extract_feat

    def extract_feat(x):
        with torch.autocast("cuda", dtype=torch.float16):
            feats = orig(x)
        return tuple(f.float() for f in feats)
    det.extract_feat = extract_feat


def fail(status: int, detail: str) -> JSONResponse:
    return JSONResponse({"detail": detail}, status_code=status)


def decode_jpeg(raw: bytes):
    img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("사진을 읽지 못함")
    return img


def infer_images(imgs: list[np.ndarray]):
    """한 배치: 검출 → 상자 1개 → RTMW3D-x. 사진이 1장이면 리스트가 아닌 배열을 넘긴다."""
    amp = torch.autocast("cuda", dtype=torch.float16) if PRECISION == "fp16" else nullcontext()
    from mmdet.apis import inference_detector

    with torch.no_grad():
        dets = inference_detector(det_model, imgs if len(imgs) > 1 else imgs[0])
    dets = dets if isinstance(dets, list) else [dets]
    boxes = pick_boxes(dets, [im.shape[:2] for im in imgs])
    with amp, torch.no_grad():
        samples = pose_batch(pose_model, imgs, boxes)
    rows = []
    for sample in samples:
        pi = sample.pred_instances
        k3 = np.asarray(pi.keypoints, dtype=np.float64).reshape(-1, 133, 3)[0]
        k2 = np.asarray(pi.transformed_keypoints, dtype=np.float64).reshape(-1, 133, 2)[0]
        rows.append((np.round(k2, 2).tolist(), np.round(k3, 5).tolist()))
    return rows


def finish(loop, fut, payload, status=200):
    def _set():
        if not fut.done():
            fut.set_result(JSONResponse(payload, status_code=status))

    loop.call_soon_threadsafe(_set)


def handle_batch(loop, batch):
    good = []
    imgs = []
    for req_id, raw, fut in batch:
        try:
            imgs.append(decode_jpeg(raw))
        except ValueError:
            finish(loop, fut, {"detail": "사진을 읽지 못함"}, status=400)
            continue
        good.append((req_id, fut))
    if not imgs:
        return
    try:
        rows = infer_images(imgs)
    except Exception:
        log.exception("배치 추론 실패 (%d장)", len(imgs))
        for _, fut in good:
            finish(loop, fut, {"detail": "추론 실패"}, status=500)
        return
    for (req_id, fut), (k2, k3) in zip(good, rows):
        finish(loop, fut, {
            "request_id": req_id,
            "keypoints_2d": k2,
            "keypoints_3d": k3,
        })


def gpu_loop(loop):
    global det_model, pose_model
    if PRECISION not in ("fp32", "fp16"):
        log.error("PRECISION 은 fp32 또는 fp16 만 가능합니다: %s", PRECISION)
        os._exit(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    try:
        det_model, pose_model = load_models(DEVICE)
        if PRECISION == "fp16":
            half_features(det_model)
        infer_images([np.zeros((480, 640, 3), dtype=np.uint8)])
    except Exception:
        log.exception("모델 로딩 실패")
        os._exit(1)
    models_ready.set()
    log.info(
        "추론 준비 완료 device=%s precision=%s max_batch=%d max_wait_ms=%d",
        DEVICE, PRECISION, MAX_BATCH, int(MAX_WAIT * 1000),
    )
    while True:
        first = batch_queue.get()
        batch = [first]
        deadline = time.monotonic() + MAX_WAIT
        while len(batch) < MAX_BATCH:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(batch_queue.get(timeout=remaining))
            except queue.Empty:
                break
        handle_batch(loop, batch)


async def health(_: Request):
    if not models_ready.is_set():
        return fail(503, "모델을 올리고 있습니다")
    return JSONResponse({"status": "ok"})


async def infer(request: Request):
    if not models_ready.is_set():
        return fail(503, "모델을 올리고 있습니다")
    req_id = request.headers.get("x-request-id")
    if not req_id:
        return fail(400, "X-Request-Id 가 필요합니다")
    raw = await request.body()
    if not raw:
        return fail(400, "JPEG 본문이 비어 있습니다")
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    try:
        batch_queue.put_nowait((req_id, raw, fut))
    except queue.Full:
        return fail(503, "추론 대기열이 가득 찼습니다")
    return await fut


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=gpu_loop, args=(asyncio.get_running_loop(),), daemon=True).start()
    yield


app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/infer", infer, methods=["POST"]),
    ],
    lifespan=lifespan,
)
