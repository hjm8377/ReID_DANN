import logging
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch import amp
import torch.distributed as dist
import numpy as np

def do_train(cfg,
             model,
             center_criterion,
             # train_loader,
             source_loader,
             target_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    def dann_lambda(step, total_steps, gamma=10.0):
        p = step/float(total_steps)
        return float(2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)
    
    total_steps = epochs * max(len(source_loader), len(target_loader))
    global_step = 0

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        model.train()

        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        iters_per_epoch = max(len(source_loader), len(target_loader))

        # for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
        for n_iter in range(iters_per_epoch):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            # img = img.to(device)
            # target = vid.to(device)
            # target_cam = target_cam.to(device)
            # target_view = target_view.to(device)

            try:
                s_img, s_vid, s_cam, s_view = next(source_iter)
            except:
                source_iter = iter(source_loader)
                s_img, s_vid, s_cam, s_view = next(source_iter)

            try:
                t_img, t_vid_dummy, t_cam, t_view = next(target_iter)
            except:
                target_iter = iter(target_loader)
                t_img, t_vid_dummy, t_cam, t_view = next(target_iter)

            s_img, s_vid, s_cam, s_view = s_img.to(device), s_vid.to(device), s_cam.to(device), s_view.to(device)
            t_img, t_cam, t_view = t_img.to(device), t_cam.to(device), t_view.to(device)
            
            lambda_d = dann_lambda(global_step, total_steps)
            if hasattr(model, "set_lambda_d"):
                model.set_lambda_d(lambda_d)
            elif hasattr(model, "module") and hasattr(model.module, "set_lambda_d"):
                model.module.set_lambda_d(lambda_d)

            with amp.autocast(enabled=True):
                out_s = model(s_img, s_vid, cam_label=s_cam, view_label=s_view, domain_only=False)
                score_s, feat_s, dom_s = out_s

                out_t = model(t_img, cam_label=t_cam, view_label=t_view, domain_only=True)
                _, feat_t, dom_t = out_t


                loss_id_tri = loss_fn(score_s,feat_s, s_vid, s_cam)

                # domain loss
                dom_label_s = torch.zeros(dom_s.shape[0]).long().to(device)
                dom_label_t = torch.ones(dom_t.shape[0]).long().to(device)
                
                # loss_dom_s = F.cross_entropy(dom_s, dom_label_s)
                # loss_dom_t = F.cross_entropy(dom_t, dom_label_t)

                # Ns, Nt = dom_s.size(0), dom_t.size(0)
                # loss_dom = (loss_dom_s * Ns + loss_dom_t * Nt) / (Ns + Nt)

                domain_logits = torch.cat([dom_s, dom_t], dim=0)
                domain_labels = torch.cat([
                    torch.zeros(dom_s.size(0)), 
                    torch.ones(dom_t.size(0))
                ], dim=0).long().to(device)

                loss_dom = F.cross_entropy(domain_logits, domain_labels)

                loss = loss_id_tri + 0.5 * loss_dom

                # score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
                # loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            # if isinstance(score, list):
            #     acc = (score[0].max(1)[1] == target).float().mean()
            # else:
            #     acc = (score.max(1)[1] == target).float().mean()

            # loss_meter.update(loss.item(), img.shape[0])
            # acc_meter.update(acc, 1)

            acc = (score_s.max(1)[1] == s_vid).float().mean()

            # 손실은 전체 배치 크기(소스+타겟) 기준으로 기록하는 것이 좋음
            total_batch_size = s_img.shape[0] + t_img.shape[0]
            loss_meter.update(loss.item(), total_batch_size)
            acc_meter.update(acc.item(), 1)

            global_step += 1

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    if (n_iter + 1) % log_period == 0:
                        base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                        logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), iters_per_epoch, loss_meter.avg, acc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), iters_per_epoch, loss_meter.avg, acc_meter.avg, base_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            total_bs = cfg.SOLVER.IMS_PER_BATCH
            speed = total_bs / time_per_batch
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), speed))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


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


