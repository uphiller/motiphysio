# reference/Dockerfile — 기준 모델(RTMDet-x + RTMW3D-x, MMPose 공식 코드) 실행 환경
# 정답지·허용 오차를 만들 때 쓴다. 지원자도 같은 버전으로 맞추면 같은 결과가 나온다.
#   docker build -t dev_test_reference reference
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 wget \
    && rm -rf /var/lib/apt/lists/*

# OpenMMLab: mmcv 는 torch 2.1 + cu121 용 미리 빌드된 파일을 쓴다
RUN pip install "numpy<2" "opencv-python-headless<4.11" mmengine==0.10.7 \
    && pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html \
    && pip install mmdet==3.3.0 mmpose==1.3.2 pillow

# RTMPose3D 는 pip 패키지에 없고 MMPose 저장소 projects/ 에만 있다 → 커밋 고정
ARG MMPOSE_COMMIT=759b39c13fea6ba094afc1fa932f51dc1b11cbf9
RUN git clone https://github.com/open-mmlab/mmpose.git /opt/mmpose \
    && cd /opt/mmpose && git checkout ${MMPOSE_COMMIT}
ENV PYTHONPATH=/opt/mmpose/projects/rtmpose3d

# 가중치 (공개, 신청 불필요)
RUN mkdir -p /models && cd /models \
    && wget -q https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth \
    && wget -q https://download.openmmlab.com/mmpose/v1/wholebody_3d_keypoint/rtmw3d/rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.pth \
    && sha256sum *.pth > /models/SHA256SUMS

WORKDIR /work

# HTTP 추론 서버. 가중치 로딩이 끝나기 전에는 /health 가 503 이다.
RUN pip install "starlette==0.41.3" "uvicorn==0.32.1"
COPY server.py /work/
EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--timeout-keep-alive", "75", "--no-access-log"]

