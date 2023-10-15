import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.split(os.getcwd())[0])
import logging
import random
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler
from torchvision.transforms import ToTensor

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

from open_clip import create_model_and_transforms, trace_model
from training.data import get_data
from training.distributed import is_master, init_distributed_device, world_info_from_env
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr
from training.train import train_one_epoch, evaluate, loop_save_data
from vl_checklist.evaluate_vl import EvaluateAllVL as EvaluateAllVL

os.environ["NCCL_DEBUG"] = "INFO"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def main():
    args = parse_args()

    if (
        args.blip_cap
        or args.noun_pos
        or args.use_extra_cc3m_expanders
        or args.use_extra_blip_cap_expanders
        or (
            args.use_v2_extra_blip_expanders
            and not args.use_expanders_as_additional_data
            and not args.mil_co_loader
            and not args.mil_dense
        )
        or args.calc_pos_sim
        or args.avg_pos_features
    ):  # evlk
        args.vl_pos = True
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
    args.model = args.model.replace("/", "-")

    # get the name of the experiments
    if args.name is None:
        args.name = "-".join(
            [
                datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
                f"model_{args.model}",
                f"lr_{args.lr}",
                f"b_{args.batch_size}",
                f"j_{args.workers}",
                f"p_{args.precision}",
            ]
        )

    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    args.log_path = None
    if is_master(args):
        if args.debug:
            # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            # os.environ['TORCH_USE_CUDA_DSA'] = '1'

            if args.debug_ip is None:
                import pydevd_pycharm

                pydevd_pycharm.settrace(
                    os.environ["SSH_CONNECTION"].split()[0],
                    port=args.debug_port,
                    stdoutToServer=True,
                    stderrToServer=True,
                    suspend=False,
                )


    if is_master(args, local=args.log_local):
        log_base_path = os.path.join(args.logs, args.name)
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f"out-{args.rank}" if args.log_local else "out.log"
        args.log_path = os.path.join(log_base_path, log_filename)

    # Set logger
    args.log_level = logging.INFO

    setup_logging(args.log_path, args.log_level)

    # fully initialize distributed device environment
    if args.save_data:
        data = get_data(args, (ToTensor(), ToTensor()), epoch=0)
        loop_save_data(data, 0, args)
        quit()

    device = init_distributed_device(args)

    args.tensorboard = "tensorboard" in args.report_to or "all" in args.report_to
    args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
    if is_master(args):
        args.tensorboard_path = (
            os.path.join(args.logs, args.name, "tensorboard")
            if args.tensorboard
            else ""
        )
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ""

    if args.copy_codebase:
        copy_codebase(args)

    assert args.precision in ["amp", "amp_bfloat16", "fp16", "fp32"]
    if args.precision == "fp16":
        logging.warning(
            "It is recommended to use AMP mixed-precision instead of FP16. "
            "FP16 support needs further verification and tuning, especially for train."
        )

    elif args.distributed:
        logging.info(
            f"Running in distributed mode with multiple processes. Device: {args.device}."
            f"Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}."
        )
    else:
        logging.info(f"Running with a single process. Device {args.device}.")

    random_seed(args.seed, 0)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        lora=args.lora,
        freeze_img=args.freeze_img,
        kqv_lora=args.kqv_lora,
    )
    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats,
        )

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args["static_graph"] = True
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], **ddp_args
        )

    # create optimizer and scaler
    optimizer = None
    scaler = None
    if args.train_data:
        assert not args.trace, "Cannot train with traced model"

        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        optimizer = optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {"params": rest_params, "weight_decay": args.wd},
            ],
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
        )

        scaler = GradScaler() if args.precision == "amp" else None

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        if args.resume == "auto":
            if os.path.exists(os.path.join(args.checkpoint_path, f"epoch_latest.pt")):
                args.resume = os.path.join(args.checkpoint_path, f"epoch_latest.pt")

        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location="cpu")
            if "epoch" in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                sd = checkpoint["state_dict"]
                if not args.distributed and next(iter(sd.items()))[0].startswith(
                    "module"
                ):
                    sd = {k[len("module.") :]: v for k, v in sd.items()}
                model.load_state_dict(sd)
                if optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler is not None and "scaler" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler"])
                logging.info(
                    f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})"
                )
            else:
                # loading a bare (model only) checkpoint for fine-tune or evaluation
                model.load_state_dict(checkpoint)
                logging.info(
                    f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})"
                )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    # initialize datasets
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch)
    assert len(data), "At least one train or eval dataset must be specified."

    # create scheduler if train
    scheduler = None
    if "train" in data and optimizer is not None:
        if type(data["train"]) == list:
            total_steps = 0
            for d in data["train"]:
                total_steps += d.dataloader.num_batches * args.epochs
        else:
            total_steps = data["train"].dataloader.num_batches * args.epochs

        scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != "none" and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.save_eval_model:
        model.eval()
        checkpoint_dict = {
            "epoch": start_epoch,
            "name": args.name,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(
            checkpoint_dict,
            os.path.join(args.checkpoint_path, f"eval_epoch_{start_epoch}.pt"),
        )
        print(
            f'saved in {os.path.join(args.checkpoint_path, f"eval_epoch_{start_epoch}.pt")}'
        )
        return
    if args.eval_only:
        if args.eval_vl_cklist:
            EvaluateAllVL(model, preprocess_val, start_epoch, args, writer)
        return

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f"Start epoch {epoch}")

        train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, writer)
        if args.calc_pos_sim:
            break

        completed_epoch = epoch + 1

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.save_most_recent:
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_latest.pt"),
                )


def copy_codebase(args):
    from shutil import copytree, ignore_patterns

    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns("log", "logs"))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main()
