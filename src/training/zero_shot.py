import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from open_clip import tokenize
from .precision import get_autocast
from .imagenet_zeroshot_data import imagenet_classnames, openai_imagenet_template
import os
import json

def zero_shot_classifier(model, classnames, templates, args):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates]  # format with class
            texts = tokenize(texts).to(args.device)  # tokenize
            if args.distributed:
                class_embeddings = model.module.encode_text(texts)
            else:
                class_embeddings = model.encode_text(texts)
            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(args.device)
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    autocast = get_autocast(args.precision)
    with torch.no_grad():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(args.device)
            target = target.to(args.device)

            with autocast():
                # predict
                if args.distributed:
                    image_features = model.module.encode_image(images)
                else:
                    image_features = model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data, epoch, args):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and (args.ZS_steps_eval or epoch != args.epochs):# if ZS_steps_eval is true then epoch is a step
        return {}
    logging.info('Starting zero-shot imagenet.')

    logging.info('Building zero-shot classifier')
    classifier = zero_shot_classifier(model, imagenet_classnames, openai_imagenet_template, args)

    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5

    if args.eval_only:
        model_name = str.lower(model.model_name)

        try:
            path = os.path.join(args.logs.split('//')[0].split('src')[0], 'eval_jsons_correct',
                                model_name, args.resume.split('/')[-3])
        except:
            path = os.path.join(args.logs.split('//')[0].split('src')[0], 'eval_jsons_correct',
                                model_name, 'clip')
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'top1_zs_{epoch}.json'), 'w',
                  encoding='utf-8') as f:
            json.dump({
                "total_acc": round(top1, 4),}, f)
        with open(os.path.join(path, f'top5_zs_{epoch}.json'), 'w',
                  encoding='utf-8') as f:
            json.dump({
                "total_acc": round(top5, 4),}, f)
    logging.info('Finished zero-shot imagenet.')

    return results
