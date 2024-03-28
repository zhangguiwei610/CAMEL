import collections
import copy
import datetime
import glob
import logging
import os
import time
from os.path import join
from models.umt import UMT

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
# import wandb

from dataset import MetaLoader
from tasks.pretrain import setup_dataloaders
from tasks.retrieval_utils import evaluation_wrapper
from tasks.shared_utils import setup_model
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config import Config
from utils.config_utils import setup_main
from utils.distributed import get_rank, is_main_process
from utils.logger import log_dict_to_wandb, setup_wandb

logger = logging.getLogger(__name__)


def train(
    model,
    train_loaders,
    optimizer,
    tokenizer,
    epoch,
    global_step,
    device,
    scheduler,
    scaler,
    config,
):
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window=1, fmt="{value:.6f}"))
    metric_logger.add_meter("temperature", SmoothedValue(window=1, fmt="{value:.4f}"))
    loss_names = ["loss_" + k for k, v in config.criterion.loss_weight.items() if v != 0]

    media_types = [loader.dataset.media_type for loader in train_loaders]
    for name in loss_names:
        for m in media_types:
            metric_logger.add_meter(f"{m}-{name}", SmoothedValue(window=1, fmt="{value:.4f}"))

    header = f"Train Epoch: [{epoch}]"
    log_freq = config.log_freq

    if config.distributed:
        for d in train_loaders:
            d.sampler.set_epoch(epoch)
    train_loader = MetaLoader(name2loader=dict(list(zip(media_types, train_loaders))))

    model_without_ddp = model.module if config.distributed else model
    iterator = metric_logger.log_every(train_loader, log_freq, header)
    for i, (media_type, (image, text, idx)) in enumerate(iterator):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.max_txt_l,
            return_tensors="pt",
        ).to(device)

        with torch.cuda.amp.autocast(enabled=config.fp16):
            loss_dict = model(image, text_input, idx=idx)
            loss = sum(loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if config.optimizer.max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # logging
        for name in loss_names:
            value = loss_dict[name]
            value = value if isinstance(value, float) else value.item()
            metric_logger.update(**{f"{media_type}-{name}": value})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(temperature=model_without_ddp.temp.item())

        if is_main_process() and config.wandb.enable and global_step % log_freq == 0:
            logs = metric_logger.get_global_avg_dict()
            log_dict_to_wandb(logs, step=global_step, prefix="train/")

        global_step += 1

        if config.debug and (i + 1) % 5 == 0:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged train stats: {metric_logger.global_avg()}")
    return global_step


def main(config):
    if is_main_process() and config.wandb.enable:
        run = setup_wandb(config)

    logger.info(f"config: \n{config}")
    logger.info(f"train_file: {config.train_file}")

    setup_seed(config.seed + get_rank())
    device = torch.device(config.device)
    cudnn.benchmark = True

    train_loaders, test_name2loaders, train_media_types = setup_dataloaders(config, mode="ret")
    num_steps_per_epoch = sum(len(d) for d in train_loaders)
    config.scheduler.num_training_steps = num_steps_per_epoch * config.scheduler.epochs
    config.scheduler.num_warmup_steps = num_steps_per_epoch * config.scheduler.warmup_epochs

    model_cls = eval(config.model.get('model_cls', 'UMT'))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    ) = setup_model(
        config,
        model_cls=model_cls,
        has_decoder=False,
        pretrain=False,
        # find_unused_parameters=True,
        find_unused_parameters=False,
    )
    if is_main_process() and config.wandb.enable:
        wandb.watch(model)

    best = 0
    best_epoch = 0

    logger.info("Start " + "evaluation" if config.evaluate else "training")
    start_time = time.time()
    with torch.cuda.amp.autocast(enabled=config.fp16):
        eval_res = {}
        for test_name, test_loader in test_name2loaders.items():
            if test_name not in config.test_types:
                logger.info(
                    f"Skip eval {test_name} split. All test_types {config.test_types}"
                )
                continue
            test_loader.dataset.text = ['A person does a trick on a BMX bike at a snow covered park in Antarctica']
            test_loader.dataset.anno_list = [{'image': '/mnt/ve_perception/zhangguiwei/loveu-tgve-2023/outputs/loveu-tgve-2023/DAVIS_480p/bmx-rider/samples/sample-500/background/A person does a trick on a BMX bike at a snow covered park in Antarctica..gif'}]
            test_loader.dataset.image = [
                '/mnt/ve_perception/zhangguiwei/loveu-tgve-2023/outputs/loveu-tgve-2023/DAVIS_480p/bmx-rider/samples/sample-500/background/A person does a trick on a BMX bike at a snow covered park in Antarctica..gif']
            res = evaluation_wrapper(
                model_without_ddp, test_loader, tokenizer, device, config, prefix=test_name
            )
            eval_res.update(res)





        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
    logger.info(f"best epoch {best_epoch} [config.stop_key {config.stop_key}]")
    logger.info(f"Checkpoints and Logs saved at {config.output_dir}")

    if is_main_process() and config.wandb.enable:
        run.finish()

def calculate_umtscore(config,test_name2loaders,model_without_ddp,tokenizer,device,text,gif_path):
    with torch.cuda.amp.autocast(enabled=config.fp16):
        for test_name, test_loader in test_name2loaders.items():
            if test_name not in config.test_types:
                logger.info(
                    f"Skip eval {test_name} split. All test_types {config.test_types}"
                )
                continue
            test_loader.dataset.text = [text]
            test_loader.dataset.anno_list = [{'image': gif_path}]
            test_loader.dataset.image = [
                gif_path]
            res = evaluation_wrapper(
                model_without_ddp, test_loader, tokenizer, device, config, prefix=test_name
            )
    return res

def eval_after_training(train_config):
    # general config for all
    train_config.wandb.enable = False
    train_config.evaluate = True
    # train_config.pretrained_path = join(train_config.output_dir, "ckpt_best.pth")
    # train_config.pretrained_path = '/mnt/share_disk/zhangguiwei/l16_25m.pth'

    train_config.num_frames_test = train_config.num_frames
    train_config.inputs.video_input.num_frames_test = train_config.num_frames

    if train_config.get('num_frames_test_final', False):
        train_config.num_frames_test = train_config.num_frames_test_final
        train_config.batch_size = train_config.batch_size_final
        train_config.inputs.video_input.num_frames_test = train_config.num_frames_test_final
        train_config.model.vision_encoder.num_frames = train_config.num_frames_test_final

    eval_config = copy.deepcopy(train_config)
    eval_config.test_types = list(eval_config.test_file.keys())
    eval_config.output_dir = join(eval_config.output_dir, f"eval_after_training")
    eval_config.result_dir = eval_config.output_dir
    if is_main_process():
        os.makedirs(eval_config.output_dir, exist_ok=True)
        Config.dump(eval_config, os.path.join(eval_config.output_dir, "config.json"))

    if is_main_process() and eval_config.wandb.enable:
        run = setup_wandb(eval_config)


    setup_seed(eval_config.seed + get_rank())
    device = torch.device(eval_config.device)
    cudnn.benchmark = True

    train_loaders, test_name2loaders, train_media_types = setup_dataloaders(eval_config, mode="ret")
    num_steps_per_epoch = sum(len(d) for d in train_loaders)
    eval_config.scheduler.num_training_steps = num_steps_per_epoch * eval_config.scheduler.epochs
    eval_config.scheduler.num_warmup_steps = num_steps_per_epoch * eval_config.scheduler.warmup_epochs

    model_cls = eval(eval_config.model.get('model_cls', 'UMT'))
    (
        model,
        model_without_ddp,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        start_epoch,
        global_step,
    ) = setup_model(
        eval_config,
        model_cls=model_cls,
        has_decoder=False,
        pretrain=False,
        # find_unused_parameters=True,
        find_unused_parameters=False,
    )

    return eval_config,test_name2loaders,model_without_ddp,tokenizer,device

import os
if __name__ == "__main__":
    cfg = setup_main()
    # main(cfg)
    # if not cfg.evaluate:
    umt_score_dict=collections.defaultdict(list)
    eval_config,test_name2loaders,model_without_ddp,tokenizer,device=eval_after_training(cfg)
    # umt_score=calculate_umtscore(eval_config, test_name2loaders, model_without_ddp, tokenizer, device,
    #                              'A person does a trick on a BMX bike at a snow covered park in Antarctica',
    #                              '/mnt/ve_perception/zhangguiwei/loveu-tgve-2023/outputs/loveu-tgve-2023/DAVIS_480p/bmx-rider/samples/sample-500/background/A person does a trick on a BMX bike at a snow covered park in Antarctica..gif')

    for per_key in ['background','multiple','object','style']:
        dir_paths=glob.glob('/mnt/ve_perception/zhangguiwei/loveu-tgve-2023/outputs/loveu-tgve-2023/DAVIS_480p/*')
        for per_path in dir_paths:
            gif_path=glob.glob(per_path+'/samples/sample-500/'+per_key+'/*.gif')[0]
            text=os.path.basename(gif_path)[:-4]
            umt_score = calculate_umtscore(eval_config, test_name2loaders, model_without_ddp, tokenizer, device,
                                           text,
                                           gif_path)
            umt_score_dict[per_key].append(umt_score.item())
    c=1
