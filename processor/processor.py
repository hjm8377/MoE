import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import torchvision.utils as vutils
import trackio

# ImageNet Mean/Std (ReID에서 표준으로 사용)
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # Clone to avoid modifying the original tensor
    t = tensor.clone().detach().cpu()
    
    # Reverse Normalization: x * std + mean
    for i in range(3):
        t[i] = t[i] * std[i] + mean[i]
    
    return t.clamp(0, 1) # 0~1 사이로 값 고정

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):

    accelerator = Accelerator(log_with="tensorboard", project_dir=cfg.OUTPUT_DIR)
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")

    output_dir = cfg.OUTPUT_DIR
    if accelerator.is_main_process:
        trackio.init(
            project="Night_ReID_MoE", 
            config=dict(cfg), 
        )
        
        log_file = os.path.join(cfg.OUTPUT_DIR, "train_log.txt")
        has_file_handler = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        
        if not has_file_handler:
            # 파일 핸들러 생성
            fh = logging.FileHandler(log_file, mode='a') # 'a'는 이어쓰기(append)
            fh.setLevel(logging.DEBUG)
            
            # 포맷 설정 (날짜, 시간, 레벨, 메시지)
            formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
            fh.setFormatter(formatter)
            
            # 로거에 핸들러 추가
            logger.addHandler(fh)
            logger.info(f"Log file is explicitly set to: {log_file}")



    model, optimizer, optimizer_center, train_loader, val_loader, scheduler, center_criterion = accelerator.prepare(
        model, optimizer, optimizer_center, train_loader, val_loader, scheduler, center_criterion
    )
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    
    if accelerator.is_main_process:
        logger.info('start training')
        # try:
        #     # 모델의 구조를 문자열로 변환
        #     # unwrap_model을 해야 DDP 껍데기 벗겨진 진짜 구조가 보임
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     model_str = str(unwrapped_model)
            
        #     # TensorBoard의 TEXT 탭에 저장
        #     writer.add_text('Model/Architecture', model_str.replace('\n', '  \n'), 0)
            
        #     logger.info("Model architecture logged as text.")
        # except Exception as e:
        #     logger.warning(f"Failed to log model architecture: {e}")

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()
        for n_iter, (img, vid, target_cam, target_view, domain) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            
            score_dict, feat, domain_logit = model(img, vid, cam_label=target_cam, view_label=target_view, domain_label=domain)
            
            loss_reid_shared = loss_fn(score_dict['shared'], feat, vid, target_cam)

            day_mask = (domain == 0)
            if day_mask.sum() > 0:
                loss_reid_day = loss_fn(score_dict['day'][day_mask], feat[day_mask], vid[day_mask], target_cam[day_mask])
            else:
                loss_reid_day = 0.0

            night_mask = (domain == 1)
            if night_mask.sum() > 0:
                loss_reid_night = loss_fn(score_dict['night'][night_mask], feat[night_mask], vid[night_mask], target_cam[night_mask])
            else:
                loss_reid_night = 0.0

            loss_reid = loss_reid_shared + loss_reid_day + loss_reid_night

            loss_domain = nn.CrossEntropyLoss()(domain_logit, domain)

            unwrapped_model = accelerator.unwrap_model(model)
            total_aux_loss = 0.
            total_ortho_loss = 0.
            for blk in model.base.blocks:
                if hasattr(blk.mlp, 'aux_loss'):
                    total_aux_loss += blk.mlp.aux_loss
                    total_ortho_loss += blk.mlp.ortho_loss

            loss = loss_reid + (0.1 * loss_domain) + (0.1 * total_aux_loss) + (0.1 * total_ortho_loss)

            accelerator.backward(loss)

            optimizer.step()

            if accelerator.is_main_process:
                # global_step 계산
                current_step = (epoch - 1) * len(train_loader) + n_iter

                trackio.log({
                    'train/loss_total': loss.item(),
                    'train/loss_reid': loss_reid.item(),
                    'train/loss_reid_shared': loss_reid_shared.item(),
                    'train/loss_reid_day': loss_reid_day.item(),
                    'train/loss_reid_night': loss_reid_night.item(),
                    'train/loss_domain': loss_domain.item(),
                    'train/loss_aux': total_aux_loss.item() if isinstance(total_aux_loss, torch.Tensor) else total_aux_loss,
                    'train/loss_ortho': total_ortho_loss.item() if isinstance(total_ortho_loss, torch.Tensor) else total_ortho_loss,
                })


            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    # param.grad는 accelerator.backward()에 의해 이미 채워져 있음
                    if param.grad is not None:
                        param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                optimizer_center.step()

            if isinstance(score_dict['shared'], list):
                acc = (score_dict['shared'][0].max(1)[1] == vid).float().mean()
            else:
                acc = (score_dict['shared'].max(1)[1] == vid).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            if accelerator.is_main_process:
                # writer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + n_iter)
                trackio.log({'train/acc': acc.item()})

            if (n_iter + 1) % log_period == 0:
                base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                
                if accelerator.is_main_process:
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr))
        scheduler.step(epoch)

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        
        if accelerator.is_main_process:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), cfg.SOLVER.IMS_PER_BATCH / time_per_batch))

        if epoch % checkpoint_period == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if accelerator.is_main_process:
                model.eval()
                # 평가 로더는 prepare하지 않았으므로 수동으로 device 이동 필요할 수 있음
                # (단, 위 prepare에서 제외했다면 device 이동 코드 유지)
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(accelerator.device)
                        camids = camids.to(accelerator.device)
                        target_view = target_view.to(accelerator.device)
                        
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                
                if accelerator.is_main_process:
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

    if accelerator.is_main_process:
        # writer.close()
        trackio.finish()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


