# -*- coding: utf-8 -*-
"""
ReID Inference (single query) + Top-K rank strip + AP/CMC print

Usage:
python run_rank_strip.py --config_file path/to/config.yaml --query_path "D:\\ReID\\NightReID_LIME\\query\\0002R1C041.jpg" --topk 10 --size 128x256

Notes:
- cfg, make_dataloader, make_model, utils.metrics(euclidean_distance, re_ranking, eval_func) 필요
- cfg.TEST.FEAT_NORM (bool), cfg.TEST.RERANKING (bool, 선택) 사용 가능
- 저장 경로: cfg.OUTPUT_DIR/rank_strip_query_plus_topK.jpg
"""

import os
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

# 기존 프로젝트 유틸
from utils.metrics import euclidean_distance, re_ranking, eval_func
from utils.logger import setup_logger
from config import cfg
from datasets import make_dataloader
from model import make_model


# -----------------------------
# 0) 헬퍼들
# -----------------------------
def normpath_case(p: str) -> str:
    return os.path.normcase(os.path.normpath(p))

def parse_wh(s: str, default=(128, 256)):
    # "128x256" -> (128,256)
    try:
        w, h = s.lower().split("x")
        return (int(w), int(h))
    except Exception:
        return default


# -----------------------------
# 1) 이미지 유틸 (왜곡 방지)
# -----------------------------
def fit_with_padding(img: Image.Image, target_size=(128, 256), fill=(255, 255, 255)):
    """원본 비율 유지해서 축소/확대 후, 남는 영역은 패딩으로 채움."""
    tw, th = target_size  # (W, H)
    w, h = img.size
    scale = min(tw / w, th / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    img_resized = img.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (tw, th), fill)
    offset = ((tw - new_w) // 2, (th - new_h) // 2)
    canvas.paste(img_resized, offset)
    return canvas

def _draw_border(im, color=(0, 200, 0), thickness=6):
    im = im.copy()
    draw = ImageDraw.Draw(im)
    w, h = im.size
    for t in range(thickness):
        draw.rectangle([t, t, w - 1 - t, h - 1 - t], outline=color)
    return im


# -----------------------------
# 2) Rank Strip(이미지) 생성
# -----------------------------
def make_rank_strip(
    save_path: str,
    query_path: str,
    top_paths: list,
    q_pid: int,
    g_pids_top: list,
    q_camid: int = None,
    g_camids_top: list = None,
    size=(128, 256),
    gap=6,
    draw_rank_text=True,
):
    """
    쿼리 + Rank-K 가로 스트립 저장
    - 초록 테두리: 정답(g_pid == q_pid [필요시 동일카메라 제외])
    - 빨강 테두리: 오답
    """
    # 정답 여부
    correct_flags = []
    for i, gpid in enumerate(g_pids_top):
        ok = (gpid == q_pid)
        if (q_camid is not None) and (g_camids_top is not None):
            ok = ok and (g_camids_top[i] != q_camid)  # 동일 카메라 제외 규칙 필요시 유지
        correct_flags.append(ok)

    # 캔버스 준비 (QUERY 1 + TOP K)
    N = 1 + len(top_paths)
    w, h = size
    canvas_w = N * w + (N - 1) * gap
    canvas_h = h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    # 폰트 (환경에 따라 실패 가능)
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        pass

    # QUERY (파란 테두리)
    x = 0
    # q_img = fit_with_padding(Image.open(query_path).convert("RGB"), target_size=size)
    q_img = Image.open(query_path).convert("RGB").resize(size)
    q_img = _draw_border(q_img, color=(0, 128, 255), thickness=6)
    if draw_rank_text:
        d = ImageDraw.Draw(q_img)
        d.text((6, 6), "QUERY", fill=(255, 255, 255), font=font)
    canvas.paste(q_img, (x, 0))
    x += w + gap

    # RANK 1..K
    for r, (p, is_ok) in enumerate(zip(top_paths, correct_flags), start=1):
        # im = fit_with_padding(Image.open(p).convert("RGB"), target_size=size)
        im = Image.open(p).convert("RGB").resize(size)
        border_col = (0, 200, 0) if is_ok else (220, 0, 0)
        im = _draw_border(im, color=border_col, thickness=6)
        if draw_rank_text:
            d = ImageDraw.Draw(im)
            d.text((6, 6), f"R{r}", fill=(255, 255, 255), font=font)
        canvas.paste(im, (x, 0))
        x += w + gap

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.save(save_path)
    print(f"[saved] {save_path}")


# -----------------------------
# 3) Inference & Top-K 추출
# -----------------------------
@torch.no_grad()
def do_inference_single_query(cfg, model, val_loader, query_path, topk=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = logging.getLogger("transreid.test")
    logger.info('Enter inferencing')

    # 수집 버퍼: 쿼리 먼저, 갤러리 다음
    q_feats, q_pids, q_camids, q_paths = [], [], [], []
    g_feats, g_pids, g_camids, g_paths = [], [], [], []

    qpath_norm = normpath_case(query_path)

    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs for inference')
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    for _, (img, pid, camid, camids, target_view, imgpaths) in enumerate(val_loader):
        img = img.to(device, non_blocking=True)
        camids = camids.to(device, non_blocking=True)
        target_view = target_view.to(device, non_blocking=True)

        feats = model(img, cam_label=camids, view_label=target_view)  # [B, D]

        # 배치 내 개별 샘플 단위로 분기
        for j, p in enumerate(imgpaths):
            p_norm = normpath_case(p)
            f = feats[j].detach().cpu()
            pid_j = int(pid[j])
            camid_j = int(camid[j])
            if p_norm == qpath_norm:
                q_feats.append(f); q_pids.append(pid_j); q_camids.append(camid_j); q_paths.append(p)
            else:
                g_feats.append(f); g_pids.append(pid_j); g_camids.append(camid_j); g_paths.append(p)

    # 쿼리 확인
    if len(q_feats) != 1:
        raise RuntimeError(f"쿼리 경로를 정확히 1개 찾지 못했습니다. found={len(q_feats)} path={query_path}")

    # 특징 정규화
    if getattr(cfg.TEST, "FEAT_NORM", True):
        qf = torch.nn.functional.normalize(torch.stack(q_feats, dim=0), dim=1, p=2)  # [1, D]
        gf = torch.nn.functional.normalize(torch.stack(g_feats, dim=0), dim=1, p=2)  # [N, D]
    else:
        qf = torch.stack(q_feats, dim=0)
        gf = torch.stack(g_feats, dim=0)

    # 거리행렬 or 재랭킹
    use_rerank = bool(getattr(cfg.TEST, "RERANKING", False))
    if use_rerank:
        print("=> Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)  # [1, N] (ndarray)
        if isinstance(distmat, torch.Tensor):
            dist = distmat[0].cpu().numpy()
        else:
            dist = distmat[0]
    else:
        print("=> Computing DistMat with euclidean_distance")
        distmat = euclidean_distance(qf, gf)  # [1, N] (ndarray)
        dist = distmat[0]

    # AP & CMC 계산용 (원본) -------------------------
    q_pids_np   = np.asarray([q_pids[0]])
    q_camids_np = np.asarray([q_camids[0]])
    g_pids_np   = np.asarray(g_pids)
    g_camids_np = np.asarray(g_camids)

    cmc, mAP = eval_func(distmat, q_pids_np, g_pids_np, q_camids_np, g_camids_np)
    print("==== Evaluation Results ====")
    print(f"mAP: {mAP:.2%}")
    for r in [1, 5, 10]:
        if r <= len(cmc):
            print(f"Rank-{r:<3}: {cmc[r-1]:.2%}")

    # Top-K (시각화용) ------------------------------
    filter_same_cam = True   # ✅ 질문 요구: 동일 camid는 뒤로 밀기
    dist_vis = dist.copy()

    if filter_same_cam:
        same_cam_mask = (g_pids_np == q_pids_np[0]) & (g_camids_np == q_camids_np[0])
        dist_vis[same_cam_mask] = np.inf  # 동일 카메라는 전부 뒤로 밀기

    # (참고) Market1501 규칙만 적용하려면 아래를 사용하세요:
    # junk_mask = (g_pids_np == q_pids_np[0]) & (g_camids_np == q_camids_np[0])
    # dist_vis[junk_mask] = np.inf

    order_vis = np.argsort(dist_vis)  # same-cam은 ∞로 밀려 뒤로 감
    topk = min(topk, len(order_vis))
    top_idx = order_vis[:topk]

    # 만약 same-cam 제외로 topk가 너무 적다면, 남는 자리는 원 dist에서 채우기(선택)
    if np.isinf(dist_vis[order_vis[:topk]]).any():
        # 유효(비-∞)만 먼저 채움
        valid = order_vis[np.isfinite(dist_vis[order_vis])]
        need  = topk - len(valid[:topk])
        if need > 0:
            # 뒤에 same-cam에서도 추가로 채움
            rest = order_vis[np.isinf(dist_vis[order_vis])]
            top_idx = np.concatenate([valid[:topk], rest[:need]])[:topk]

    top_paths = [g_paths[i] for i in top_idx]

    print("\n[DEBUG] Top-{} list (pid/camid/path)".format(topk))
    hits = 0
    for rank, gi in enumerate(top_idx, 1):
        is_hit = (g_pids[gi] == q_pids[0]) and (g_camids[gi] != q_camids[0])
        hits += int(is_hit)
        print(f" R{rank:>2}: pid={g_pids[gi]} cam={g_camids[gi]} "
            f"hit={is_hit}  path={g_paths[gi]}")
    print(f"[DEBUG] hits in top-{topk} (CMC 기준): {hits}\n")


    return {
        "query_path": q_paths[0],
        "q_pid": q_pids[0],
        "q_camid": q_camids[0],
        "g_pids": g_pids,
        "g_camids": g_camids,
        "g_paths": g_paths,
        "top_idx": top_idx,
        "top_paths": top_paths,
        "distmat": distmat,  # (1, N)
    }


# -----------------------------
# 4) Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="ReID Inference + TopK Rank Strip + AP/CMC")
    parser.add_argument("--config_file", default="", type=str, help="path to config file")
    parser.add_argument("--query_path", default="", type=str, help="absolute path to the query image")
    parser.add_argument("--topk", default=10, type=int, help="K for rank visualization")
    parser.add_argument("--size", default="128x256", type=str, help="tile size WxH (e.g., 128x256)")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Modify config options from CLI")
    args = parser.parse_args()

    # cfg 로드
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    # 출력 디렉토리 및 로거
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)
    if args.config_file:
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r', encoding="utf-8") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # GPU 선택
    if hasattr(cfg.MODEL, "DEVICE_ID"):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.DEVICE_ID)

    # 데이터/모델
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)

    
    query_paths = [
        # Easy Queries
        "D:\ReID\NightReID\query\0010R2C161.jpg",
        "D:\ReID\NightReID\query\1024R3C038.jpg",
        "D:\ReID\NightReID\query\1010R1C041.jpg",
        "D:\ReID\NightReID\query\1115R3C062.jpg",
        "D:\ReID\NightReID\query\1256R3C072.jpg",
        "D:\ReID\NightReID\query\1097R3C056.jpg"

        # Hard Queries
        "D:\ReID\NightReID\query\0511R2C047.jpg",
        "D:\ReID\NightReID\query\0329R1C041.jpg",
        "D:\ReID\NightReID\query\0102R2C101.jpg",
        "D:\ReID\NightReID\query\0085R2C056.jpg",
        "D:\ReID\NightReID\query\0072R1C081.jpg",
        "D:\ReID\NightReID\query\0677R1C041.jpg"
    ]

    import os
    os.path.normpath()

    # query_path = args.query_path.strip() if args.query_path else "D:\\ReID\\NightReID_LIME\\query\\0137R3C008.jpg" # "D:\\ReID\\NightReID_LIME\\query\\0002R1C041.jpg"
    for i, query_path in enumerate(query_paths):

        # 실행
        size = parse_wh(args.size, default=(128, 256))
        result = do_inference_single_query(cfg, model, val_loader, query_path, topk=args.topk)

        # 스트립 저장
        top_paths = result["top_paths"]
        top_idx   = result["top_idx"]
        q_pid     = result["q_pid"]
        q_camid   = result["q_camid"]
        g_pids    = result["g_pids"]
        g_camids  = result["g_camids"]

        g_pids_top   = [g_pids[i] for i in top_idx]
        g_camids_top = [g_camids[i] for i in top_idx]

        save_path = os.path.join(cfg.OUTPUT_DIR, f"rank_strip_query{i:2d}_plus_top{len(top_paths)}.jpg")
        make_rank_strip(
            save_path=save_path,
            query_path=result["query_path"],
            top_paths=top_paths,
            q_pid=q_pid,
            g_pids_top=g_pids_top,
            # 동일 카메라 제외 규칙 쓰려면 아래 두 줄 활성화
            q_camid=q_camid,
            g_camids_top=g_camids_top,
            size=size,
            gap=6,
            draw_rank_text=True
        )
        print("[DONE] Saved:", save_path)


if __name__ == "__main__":
    main()
