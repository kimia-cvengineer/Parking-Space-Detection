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
from utils_funcs import utils, transforms


def train_one_epoch(model, optimizer, data_loader, res, device, epoch, print_freq, scaler=None):
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
    i = 1
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        if i == 2:
            break
        # preprocess image
        res_images = transforms.preprocess(images, device=device, res=res)
        print("targets : ", targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(res_images, targets)
            losses = sum(loss for loss in loss_dict.values())

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
        i += 1

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
def evaluate(model, data_loader, res, device):
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
    i = 1
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        if i == 2:
            break
        # preprocess image
        res_images = transforms.preprocess(images, device=device, res=res)

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
        i+=1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


def train_model(model, train_ds, valid_ds, test_ds, model_dir, device, lr=1e-4, epochs=100, lr_decay=50, res=None,
                verbose=False):
    """
    Trains any model which takes (image, rois) and outputs class_logits.
    Expects dataset.pdosp.PDOSP datasets.
    Uses cross-entropy loss.
    """
    # transfer model to device
    model = model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay, gamma=0.1)

    # train
    for epoch in range(1, epochs + 1):
        # train for one epoch
        print("*********** training step ***********")
        train_one_epoch(model, optimizer, train_ds, res, device, epoch, print_freq=10)
        scheduler.step()

        # evaluate on the valid dataset
        print("*********** evaluation step ***********")
        evaluate(model, valid_ds, res, device)

        # save weights
        model_dir = f'./{model_dir}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), f'{model_dir}/weights_last_epoch.pt')

    # test model on test dataset
    print("*********** testing step ***********")
    evaluate(model, test_ds, res, device)
    # with open(f'{model_dir}/test_logs.json', 'w') as f:
    #     json.dump({'loss': test_loss, 'accuracy': test_accuracy}, f)

    # delete model from memory
    del model
