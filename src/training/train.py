import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F


from open_clip import ClipLoss
from .distributed import is_master
from .zero_shot import zero_shot_eval
from .precision import get_autocast
from sentence_transformers import SentenceTransformer, util
from itertools import *


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def data_wrapper(args, batch):
    if "CC3M" in args.train_data:
        if args.vl_pos or args.vl_negs:
            if args.mil_dense:
                images, texts, info_dict, mil_texts = batch
            else:
                images, texts, info_dict = batch
                mil_texts = None
        else:
            if args.mil_dense:
                images, texts, mil_texts = batch
            else:
                images, texts = batch
                mil_texts = None
            info_dict = {}
    elif "laion" in args.train_data and (args.vl_pos or args.vl_negs):
        images, texts, neg, pos = batch
        info_dict = {"negatives": neg, "positives": pos}
        mil_texts = None
    else:
        images, texts, neg, pos = batch
        info_dict = {}
        mil_texts = None

    negs = info_dict.get("negatives", None) if args.vl_negs else None
    poss = info_dict.get("positives", None) if (args.vl_pos or args.blip_cap) else None

    if args.calc_pos_sim:
        text_orig = info_dict.get("text", None)
        text_pos = info_dict.get("positives_text", None)
        return (
            images,
            texts,
            negs,
            poss,
            text_orig,
            text_pos,
            None,
            None,
            mil_texts,
            False,
        )

    return images, texts, negs, poss, None, None, None, None, mil_texts, False


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


def prepare_data_for_neg_loss(negs, args, device, texts):
    negs = negs.to(device=device, non_blocking=True)
    negs = negs.view(-1, negs.shape[-1])
    # clean non-negs that are zero. they r there because not every text has a negative
    pos_that_have_negs = [
        i for i, l in enumerate(list(negs[:: args.num_negs])) if l.nonzero().any()
    ]
    negs = [l for l in list(negs) if l.nonzero().any()]
    if len(negs) == 0:
        pos_that_have_negs = None
    else:
        texts = torch.cat((texts, torch.stack(negs)), dim=0)

    return texts, pos_that_have_negs


def prepare_data_for_pos_loss(poss, args, device, texts):
    if args.avg_pos_features:
        poss = torch.cat(poss)
    poss = poss.to(device=device, non_blocking=True)
    poss = poss.view(-1, poss.shape[-1])
    texts = torch.cat((poss, texts), dim=0)
    return texts


def loop_save_data(data, epoch, args):
    data["train"].set_epoch(
        epoch
    )  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["train"].dataloader
    for i, batch in enumerate(dataloader):
        continue


def train_one_epoch(
    model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None
):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.train()
    loss = ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        args=args,
    )
    if type(data["train"]) == list:
        num_batches_per_epoch = 0
        sample_digits = 0
        dataloader = []
        for d in data["train"]:
            d.set_epoch(
                epoch
            )  # set epoch in process safe manner via sampler or shared_epoch
            num_batches_per_epoch += d.dataloader.num_batches
            dataloader.append(d.dataloader)
            sample_digits += math.ceil(math.log(d.dataloader.num_samples + 1, 10))
    else:
        data["train"].set_epoch(
            epoch
        )  # set epoch in process safe manner via sampler or shared_epoch
        dataloader = data["train"].dataloader
        num_batches_per_epoch = dataloader.num_batches
        sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    loss_m = AverageMeter()
    loss_neg_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    total_time = 0

    if args.calc_pos_sim:
        simm_model = SentenceTransformer("all-MiniLM-L6-v2")
        cosine_score_all = 0

    for i, batch in enumerate(dataloader):
        # for i, batch in enumerate(roundrobin(*dataloader)):

        start_t = time.time()

        if epoch < args.warmup_ep_no_bn_update:
            optimizer.zero_grad()
            optimizer.step()
            continue

        step = num_batches_per_epoch * epoch + i
        scheduler(step)
        (
            images,
            texts,
            negs,
            poss,
            text_orig,
            text_pos,
            list_amount_of_pos,
            match_list,
            mil_texts,
            is_batch_mil_co_loader_v2,
        ) = data_wrapper(args, batch)

        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        if args.mil_dense:
            mil_texts = mil_texts.to(device=device, non_blocking=True)

        if args.calc_pos_sim:
            # Compute embedding for both lists
            embeddings1 = simm_model.encode(text_orig[0][0], convert_to_tensor=True)
            embeddings2 = simm_model.encode(text_pos[0][0], convert_to_tensor=True)
            # Compute cosine-similarities
            cosine_scores = util.cos_sim(embeddings1, embeddings2)
            cosine_score_all += cosine_scores
            if i == (dataloader.num_batches - 1) or i % 1000 == 0:
                print(f"{cosine_score_all/(i+1)}")
            continue
        pos_that_have_negs = None
        if poss is not None:
            texts = prepare_data_for_pos_loss(poss, args, device, texts)

        if negs is not None:
            texts, pos_that_have_negs = prepare_data_for_neg_loss(
                negs, args, device, texts
            )
        data_time_m.update(time.time() - end)
        # with autocast():
        #     if args.mil_co_loader:
        #         if texts[0].shape == 100:
        #             _, mil_texts_features, _ = model(images, texts)
        #             indices = expander_utils.choose_feat_mil(args, mil_texts_features)
        #             texts = [texts[ind] for ind in indices]

        optimizer.zero_grad()

        with autocast():
            if args.mil_dense:
                _, mil_texts_features, _ = model(images, mil_texts)
            else:
                mil_texts_features = None

            image_features, text_features, logit_scale = model(images, texts)
            # text_features = average_pos_feat(text_features,list_amount_of_pos,match_list) if args.avg_pos_features else text_features
            # text_features = choose_feat_mil(args, image_features, text_features,logit_scale,list_amount_of_pos) if (args.mil_co_loader and is_batch_mil_co_loader_v2) else text_features
            total_loss, loss_neg = loss(
                image_features,
                text_features,
                logit_scale,
                pos_that_have_negs,
                mil_texts_features,
                list_amount_of_pos,
                is_batch_mil_co_loader_v2,
            )
        if epoch < args.warmup_ep:
            total_loss = total_loss * 0.0

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if args.norm_gradient_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.norm_gradient_clip, norm_type=2.0
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if args.norm_gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.norm_gradient_clip, norm_type=2.0
                )
            optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        total_time = total_time + time.time() - start_t

        batch_time_m.update(time.time() - end)
        end = time.time()
        batch_count = i + 1

        if (
            is_master(args)
            and args.ZS_steps_eval
            and (epoch >= args.warmup_ep and epoch >= args.warmup_ep_no_bn_update)
            and i % args.ZS_freq == 0
        ):
            logging.info(f"eval batch {batch_count}")
            evaluate(model, data, batch_count, args, tb_writer)

        if is_master(args) and (
            (i % 100 == 0 and not type(data["train"]) == list)
            or (i % 100 == 1 and type(data["train"]) == list)
            or batch_count == num_batches_per_epoch
        ):
            # samples_per_epoch = 0
            # batch_size = 0
            # try:
            #     for d in dataloader:
            #         samples_per_epoch += d.num_samples
            #         batch_size += len(images)
            # except:
            samples_per_epoch = dataloader.num_samples
            batch_size = len(images)

            num_samples = batch_count * batch_size * args.world_size

            # samples_per_epoch = dataloader.num_samples
            percent_complete = 100.0 * batch_count / num_batches_per_epoch

            # NOTE loss is coarsely sampled, just master node and per log update
            loss_m.update(total_loss.item(), batch_size)

            loss_neg_m.update(loss_neg.item(), batch_size)
            logit_scale_scalar = logit_scale.item()
            logging.info(
                f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                f"Loss Neg: {loss_neg_m.val:#.5g} ({loss_neg_m.avg:#.4g}) "
                f"Data (t): {data_time_m.avg:.3f} "
                f"Batch (t): {batch_time_m.avg:.3f}, {args.batch_size*args.world_size / batch_time_m.val:#g}/s "
                f"LR: {optimizer.param_groups[0]['lr']:5f} "
                f"Logit Scale: {logit_scale_scalar:.3f}"
            )

            # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
            log_data = {
                "loss": loss_m.val,
                "data_time": data_time_m.val,
                "batch_time": batch_time_m.val,
                "samples_per_scond": args.batch_size
                * args.world_size
                / batch_time_m.val,
                "scale": logit_scale_scalar,
                "lr": optimizer.param_groups[0]["lr"],
                "loss_neg": loss_neg_m.val,
            }
            for name, val in log_data.items():
                name = "train/" + name
                if tb_writer is not None:
                    tb_writer.add_scalar(name, val, step)

            # resetting batch / data time meters per log window
            batch_time_m.reset()
            data_time_m.reset()
    # end for


def evaluate(model, data, epoch, args, tb_writer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()
    zero_shot_metrics = zero_shot_eval(model, data, epoch, args)

    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision)

    if "val" in data and (
        args.val_frequency
        and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)
    ):
        dataloader = data["val"].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if args.vl_pos or args.vl_negs:
                    images, texts, _ = batch
                else:
                    images, texts = batch

                images = images.to(device=device, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    image_features, text_features, logit_scale = model(images, texts)
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels)
                        + F.cross_entropy(logits_per_text, labels)
                    ) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Loss: {cumulative_loss / num_samples:.6f}\t"
                    )

            val_metrics = get_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {
                    **val_metrics,
                    "val_loss": loss.item(),
                    "epoch": epoch,
                    "num_samples": num_samples,
                }
            )

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    if args.save_logs:
        for name, val in metrics.items():
            if tb_writer is not None:
                tb_writer.add_scalar(f"val/{name}", val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    return metrics


def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics
