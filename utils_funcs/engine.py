import json
import math
import os
import shutil
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
from utils_funcs.coco_eval import CocoEvaluator
from utils_funcs.coco_utils import get_coco_api_from_dataset
from utils_funcs import utils, transforms, visualize
from utils_funcs.visualize import plot_log_per_epoch


def train_one_epoch(model, optimizer, data_loader, resolution, device, epoch, print_freq, log_dir, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    for images, targets in metric_logger.log_every(data_loader, print_freq, log_dir, header):
        # augment data
        images, targets = transforms.augment(images, targets)

        # preprocess image
        res_images, res_rois = transforms.preprocess(images, rois=[t["boxes"] for t in targets], device=device,
                                                     res=resolution)
        # update boxed according to the new resolution
        new_target = []
        for idx, target in enumerate(targets):
            if resolution is not None:
                target["boxes"] = torch.tensor(res_rois[idx])
            new_target.append({k: v.to(device) for k, v in target.items()})
        targets = new_target
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(res_images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # i = 0
        # for image, target in zip(res_images, targets):
        #     if i == 3:
        #         break
        #     i += 1
        #     visualize.plot_img_bbox(image, target)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, resolution, log_dir, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    for images, targets in metric_logger.log_every(data_loader, 10, log_dir, header):
        # preprocess image
        res_images, res_rois = transforms.preprocess(images, rois=[t["boxes"] for t in targets], device=device,
                                                     res=resolution)
        # update boxed according to the new resolution
        if resolution is not None:
            for idx, target in enumerate(targets):
                target["boxes"] = torch.tensor(res_rois[idx])

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(res_images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    avg_stats_str = f"Averaged stats: {metric_logger}"
    with open(f'{log_dir}/logs.txt', 'a', newline='\n', encoding='utf-8') as f:
        f.write(avg_stats_str + '\n')
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize(log_dir)
    # with open(f'{log_dir}/logs.txt', 'a', newline='\n', encoding='utf-8') as f:
    #     f.write(avg_stats_str + '\n')
    torch.set_num_threads(n_threads)
    return coco_evaluator


def train_model(model, train_ds, valid_ds, test_ds, model_dir, device, lr=1e-4, epochs=30, lr_decay=50, res=None,
                verbose=False):
    """
    Trains any model which takes (image, rois) and outputs class_logits.
    Expects dataset.pdosp.PDOSP datasets.
    Uses cross-entropy loss.
    """
    # transfer model to device
    model = model.to(device)
    model_dir = f'./{model_dir}'
    # construct an Adam optimizer
    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.AdamW(params, lr=lr)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay, gamma=0.1)

    # construct an SGD optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    # cosine lr shceduler
    # lr = 8e-5
    # plot losses
    # increase epochs

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,
    #                                                gamma=0.1)

    losses, mAP_results = [], []
    # train
    for epoch in range(1, epochs + 1):
        # train for one epoch

        # if this is the first epoch
        if epoch == 1:
            # ensure (an empty) model dir exists
            # shutil.rmtree(model_dir, ignore_errors=True)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            # os.makedirs(model_dir, exist_ok=False)

            create_logs_header(model_dir)

        print("*********** training step ***********")
        metric_logger = train_one_epoch(model, optimizer, train_ds, res, device, epoch, print_freq=10,
                                        log_dir=model_dir)
        lr_scheduler.step()
        epoch_losses = get_metric_epoch_losses(metric_logger)
        losses.append(epoch_losses)

        # save training losses
        save_metric_losses(f'{model_dir}/train_losses.csv', epoch_losses)

        # evaluate on the valid dataset
        with open(f'{model_dir}/logs.txt', 'a', newline='\n', encoding='utf-8') as f:
            f.write("*********** evaluation step ***********" + '\n')

        print("*********** evaluation step ***********")
        coco_eval = evaluate(model, valid_ds, res, model_dir, device)
        mAP_results.append(coco_eval.get_mAP_results()[0])

        # save mAP evaluation result
        save_evaluation_results(f'{model_dir}/evaluation_result.csv', mAP_results)

        # save weights
        torch.save(model.state_dict(), f'{model_dir}/weights_last_epoch.pt')

    # Plot training losses
    plot_log_per_epoch(range(1, epochs + 1), [round(loss[0], 3) for loss in losses], "Losses")

    # Plot evaluation AP results [IoU=0.50:0.95]
    plot_log_per_epoch(range(1, epochs + 1), [round(mAP[1], 3) for mAP in mAP_results], "mAPs")

    with open(f'{model_dir}/logs.txt', 'a', newline='\n', encoding='utf-8') as f:
        f.write("*********** testing step ***********" + '\n')

    # test model on test dataset
    print("*********** testing step ***********")
    evaluate(model, test_ds, res, model_dir, device)

    # delete model from memory
    del model


def create_logs_header(model_dir):
    with open(f'{model_dir}/logs.txt', 'w', newline='\n', encoding='utf-8') as f:
        f.write("*********** training step ***********" + '\n')

    # create loss log header
    with open(f'{model_dir}/train_losses.csv', 'w', newline='\n', encoding='utf-8') as f:
        f.write('loss,loss_classifier,loss_box_reg,loss_objectness,loss_rpn_box_reg\n')

    # create mAP log header
    with open(f'{model_dir}/evaluation_result.csv', 'w', newline='\n', encoding='utf-8') as f:
        f.write('AP [IoU=0.50:0.95], AP [IoU=0.50]\n')


def get_metric_epoch_losses(metric_logger):
    return [round(float(str(epoch_loss).split(" ")[0]), 3) for epoch_loss in metric_logger.meters.values()][1:6]


def save_metric_losses(log_file, metric_loss):
    with open(log_file, 'a', newline='\n', encoding='utf-8') as f:
        f.write(
            f'{metric_loss[0]},{metric_loss[1]},{metric_loss[2]},{metric_loss[3]},{metric_loss[4]}\n')


def save_evaluation_results(log_file, mAPs):
    print("maps : ",mAPs)
    with open(log_file, 'a', newline='\n', encoding='utf-8') as f:
        f.write(f'{mAPs[0]:.3f}, {mAPs[1]:.3f}\n')
