import math
import sde
import ml_collections
import torch
from torch import multiprocessing as mp
from dataset.dataset import get_dataset
from torchvision.utils import make_grid, save_image
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
import tempfile
from absl import logging
import builtins
import os
import wandb
import libs.autoencoder



def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=32)

    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.init(dir=os.path.abspath(config.workdir), project=f'uvit_{config.dataset.name}', config=config.to_dict(),
                   name=config.hparams, job_type='train', mode='online')
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    dataset = get_dataset(**config.dataset)
    train_dataset = dataset.get_split(split='train', labeled=config.train.mode == 'cond')
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      num_workers=8, pin_memory=True, persistent_workers=True)

    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader)
    lr_scheduler = train_state.lr_scheduler
    # train_state.resume(config.ckpt_root)
    train_state.resume(
        config.ckpt_root,
        load_optimizer=False,
        load_lr_scheduler=True,
    )
    print("after resume step:", train_state.step)
    print("config lr:", config.optimizer.lr)
    print("train_state optimizer lr:", train_state.optimizer.param_groups[0]["lr"])
    print("prepared optimizer lr:", optimizer.param_groups[0]["lr"])

    if hasattr(train_state.lr_scheduler, "base_lrs"):
        print("scheduler base_lrs:", train_state.lr_scheduler.base_lrs)

    if hasattr(train_state.lr_scheduler, "last_epoch"):
        print("scheduler last_epoch:", train_state.lr_scheduler.last_epoch)

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder.to(device)
    autoencoder.eval()

    for p in autoencoder.parameters():
        p.requires_grad_(False)



    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    class DecoderFeatureHook:
        def __init__(self, autoencoder, layer_ids=('up_2',)):
            self.autoencoder = autoencoder
            self.layer_ids = layer_ids
            self.features = {}
            self.handles = []

            decoder = autoencoder.decoder

            for layer_id in layer_ids:
                name, layer = self._get_decoder_layer(decoder, layer_id)
                handle = layer.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

        def _get_decoder_layer(self, decoder, layer_id):
            # 允許舊寫法：layers=[2, 1]
            if isinstance(layer_id, int):
                name = f"up_{layer_id}"
                layer = decoder.up[layer_id].block[-1]
                return name, layer

            # 允許新寫法：layers=['up_2', 'up_1']
            if isinstance(layer_id, str) and layer_id.startswith("up_"):
                idx = int(layer_id.split("_")[-1])
                name = f"up_{idx}"
                layer = decoder.up[idx].block[-1]
                return name, layer

            # 新增：decoder mid block
            if layer_id == "mid":
                name = "mid"

                # Stable Diffusion / LDM 類型 decoder 通常有 mid.block_2
                if hasattr(decoder.mid, "block_2"):
                    layer = decoder.mid.block_2
                else:
                    # 如果你的 autoencoder 寫法不同，就退回 hook 整個 decoder.mid
                    layer = decoder.mid

                return name, layer

            raise ValueError(f"Unknown decoder layer_id: {layer_id}")

        def _make_hook(self, name):
            def hook(module, inputs, output):
                self.features[name] = output

            return hook

        def clear(self):
            self.features = {}

        def extract(self, z):
            self.clear()
            _ = decode(z)
            return {k: v for k, v in self.features.items()}

    # class DecoderFeatureHook:
    #     def __init__(self, autoencoder, layer_ids=(2,)):
    #         self.autoencoder = autoencoder
    #         self.layer_ids = layer_ids
    #         self.features = {}
    #         self.handles = []
    #
    #         decoder = autoencoder.decoder
    #
    #         for i in layer_ids:
    #             layer = decoder.up[i].block[-1]
    #             handle = layer.register_forward_hook(self._make_hook(f"up_{i}"))
    #             self.handles.append(handle)
    #
    #     def _make_hook(self, name):
    #         def hook(module, inputs, output):
    #             self.features[name] = output
    #
    #         return hook
    #
    #     def clear(self):
    #         self.features = {}
    #
    #     def extract(self, z):
    #         self.clear()
    #         _ = decode(z)
    #         return {k: v for k, v in self.features.items()}

    # def lpl_weight(step, total_steps, lambda_lpl=0.01, start=0.50, end=1.00):
    #     progress = float(step) / float(total_steps)
    #
    #     if progress < start or progress >= end:
    #         return 0.0
    #
    #     ratio = (progress - start) / (end - start)
    #     return lambda_lpl * 0.5 * (1.0 + math.cos(math.pi * ratio))
    def lpl_weight(step, total_steps, lambda_lpl=0.01, start=0.50, end=1.00, schedule='constant'):
        if schedule == 'constant':
            return lambda_lpl

        if schedule == 'cosine':
            progress = float(step) / float(total_steps)

            if progress < start or progress >= end:
                return 0.0

            ratio = (progress - start) / (end - start)
            return lambda_lpl * 0.5 * (1.0 + math.cos(math.pi * ratio))

        raise NotImplementedError(f"Unknown LPL weight schedule: {schedule}")

    def normalize_feature(pred_feat, target_feat, eps=1e-6):
        mean = pred_feat.mean(dim=[2, 3], keepdim=True)
        std = pred_feat.std(dim=[2, 3], keepdim=True).clamp_min(eps)

        pred_norm = (pred_feat - mean) / std
        target_norm = (target_feat - mean) / std

        return pred_norm, target_norm

    def masked_feature_mse(pred_n, target_n, outlier_threshold=4.0, min_valid_ratio=0.25):
        diff2 = (pred_n - target_n).pow(2)

        if outlier_threshold is None or outlier_threshold <= 0:
            valid_ratio = torch.ones((), device=pred_n.device)
            return diff2.mean(), valid_ratio

        with torch.no_grad():
            valid = (
                            pred_n.detach().abs() <= outlier_threshold
                    ) & (
                            target_n.detach().abs() <= outlier_threshold
                    )

            valid_ratio = valid.float().mean()

        if valid_ratio.item() < min_valid_ratio:
            return diff2.mean(), valid_ratio

        return diff2[valid].mean(), valid_ratio

    # def get_depth_weight(layer_name, ordered_keys, power=1.0, mid_weight=0.25):
    #     if layer_name == "mid":
    #         return mid_weight
    #     up_order = {
    #         "up_3": 0,
    #         "up_2": 1,
    #         "up_1": 2,
    #         "up_0": 3,
    #     }
    #
    #     if layer_name in up_order:
    #         depth_idx = up_order[layer_name]
    #         return 2.0 ** (-depth_idx * power)
    #
    #     depth_idx = ordered_keys.index(layer_name)
    #     return 2.0 ** (-depth_idx * power)
    def get_depth_weight(layer_name, ordered_keys, power=1.0):
        depth_idx = ordered_keys.index(layer_name)
        return 2.0 ** (-depth_idx * power)


    def compute_lpl_loss(pred_x0, target_x0, feature_hook):
        with torch.no_grad():
            target_feats = {
                k: v.detach()
                for k, v in feature_hook.extract(target_x0).items()
            }

        pred_feats = feature_hook.extract(pred_x0)

        keys = []

        for layer_id in feature_hook.layer_ids:
            if isinstance(layer_id, int):
                name = f"up_{layer_id}"
            else:
                name = layer_id

            if name in pred_feats and name in target_feats:
                keys.append(name)

        if len(keys) == 0:
            zero = torch.zeros((), device=pred_x0.device)
            return zero, {
                "lpl_valid_ratio": zero,
                "lpl_depth_weight_mean": zero,
            }

        base_hw = pred_feats[keys[0]].shape[-2:]

        total_loss = 0.0
        total_weight = 0.0
        valid_ratios = []
        depth_weights = []

        for k in keys:
            pred_f = pred_feats[k].float()
            target_f = target_feats[k].float()

            pred_n, target_n = normalize_feature(pred_f, target_f)

            if config.lpl.get('depth_weighting', True):
                depth_w = get_depth_weight(
                    k,
                    ordered_keys=keys,
                    power=config.lpl.get('depth_weight_power', 1.0),
                )
            else:
                depth_w = 1.0

            if config.lpl.get('outlier_mask', True):
                layer_loss, valid_ratio = masked_feature_mse(
                    pred_n,
                    target_n,
                    outlier_threshold=config.lpl.get('outlier_threshold', 4.0),
                    min_valid_ratio=config.lpl.get('min_valid_ratio', 0.25),
                )
            else:
                layer_loss = (pred_n - target_n).pow(2).mean()
                valid_ratio = torch.ones((), device=pred_x0.device)

            total_loss = total_loss + depth_w * layer_loss
            total_weight = total_weight + depth_w

            valid_ratios.append(valid_ratio.detach())
            depth_weights.append(torch.tensor(depth_w, device=pred_x0.device))

        loss = total_loss / max(total_weight, 1e-8)

        extra_metrics = {
            "lpl_valid_ratio": torch.stack(valid_ratios).mean(),
            "lpl_depth_weight_mean": torch.stack(depth_weights).mean(),
        }

        return loss, extra_metrics

    # def compute_lpl_loss(pred_x0, target_x0, feature_hook):
    #     # target_x0 不需要梯度
    #     with torch.no_grad():
    #         target_feats = {
    #             k: v.detach()
    #             for k, v in feature_hook.extract(target_x0).items()
    #         }
    #
    #     # pred_x0 要保留梯度，loss 才能回傳到 nnet
    #     pred_feats = feature_hook.extract(pred_x0)
    #
    #     loss = 0.0
    #     count = 0
    #
    #     for k in pred_feats.keys():
    #         pred_f = pred_feats[k].float()
    #         target_f = target_feats[k].float()
    #
    #         pred_n, target_n = normalize_feature(pred_f, target_f)
    #
    #         loss = loss + (pred_n - target_n).pow(2).mean()
    #         count += 1
    #
    #     return loss / max(count, 1)

    lpl_feature_hook = DecoderFeatureHook(
        autoencoder,
        layer_ids=tuple(config.lpl.layers),
    )

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()

    # set the score_model to train
    score_model = sde.ScoreModel(nnet, pred=config.pred, sde=sde.VPSDE())
    score_model_ema = sde.ScoreModel(nnet_ema, pred=config.pred, sde=sde.VPSDE())


    def train_step(prime_target, prime_anchor_view, prime_targe_pos, encode_anchor, encode_target):
        _metrics = dict()

        with accelerator.accumulate(nnet):
            # if config.train.mode == 'uncond':
            #     _z = autoencoder.sample(prime_target) if 'feature' in config.dataset.name else encode_target
            #     loss = sde.LSimple(score_model, _z, pred=config.pred)
            if config.train.mode == 'cond':
                _z = autoencoder.sample(prime_target) if 'feature' in config.dataset.name else encode_target

                if config.get('lpl', None) is not None and config.lpl.enable:
                    loss_main, pred_x0, diffusion_t = sde.LSimpleReturnX0(
                        score_model,
                        _z,
                        pred=config.pred,
                        conditions=[encode_anchor, prime_targe_pos],
                    )

                    w_lpl = lpl_weight(
                        step=train_state.step,
                        total_steps=config.train.n_steps,
                        lambda_lpl=config.lpl.lambda_lpl,
                        start=config.lpl.schedule_start,
                        end=config.lpl.schedule_end,
                        schedule=config.lpl.get('weight_schedule', 'constant'),
                    )

                    alpha = score_model.sde.cum_alpha(diffusion_t)
                    beta = score_model.sde.cum_beta(diffusion_t)
                    snr = alpha / beta.clamp_min(1e-8)

                    snr_mask = snr > config.lpl.snr_threshold

                    if w_lpl > 0.0 and snr_mask.any():
                        loss_lpl, lpl_extra = compute_lpl_loss(
                            pred_x0[snr_mask],
                            _z[snr_mask],
                            lpl_feature_hook,
                        )

                        loss = loss_main + w_lpl * loss_lpl

                        _metrics['loss_main'] = accelerator.gather(loss_main.detach()).mean()
                        _metrics['loss_lpl'] = accelerator.gather(loss_lpl.detach()).mean()
                        _metrics['lpl_weight'] = torch.tensor(w_lpl, device=device)
                        _metrics['lpl_effect'] = torch.tensor(w_lpl, device=device) * loss_lpl.detach()
                        _metrics['lpl_snr_ratio'] = accelerator.gather(snr_mask.float().detach()).mean()
                        _metrics['lpl_valid_ratio'] = accelerator.gather(lpl_extra['lpl_valid_ratio'].detach()).mean()
                        _metrics['lpl_depth_weight_mean'] = accelerator.gather(lpl_extra['lpl_depth_weight_mean'].detach()).mean()

                    else:
                        loss = loss_main

                        _metrics['loss_main'] = accelerator.gather(loss_main.detach()).mean()
                        _metrics['loss_lpl'] = torch.zeros((), device=device)
                        _metrics['lpl_weight'] = torch.tensor(w_lpl, device=device)
                        _metrics['lpl_effect'] = torch.zeros((), device=device)
                        _metrics['lpl_snr_ratio'] = accelerator.gather(snr_mask.float().detach()).mean()
                        _metrics['lpl_valid_ratio'] = torch.zeros((), device=device)
                        _metrics['lpl_depth_weight_mean'] = torch.zeros((), device=device)

                else:
                    loss = sde.LSimple(
                        score_model,
                        _z,
                        pred=config.pred,
                        conditions=[encode_anchor, prime_targe_pos],
                    )
            # elif config.train.mode == 'cond':
            #     _z = autoencoder.sample(prime_target) if 'feature' in config.dataset.name else encode_target
            #     loss = sde.LSimple(score_model, _z, pred=config.pred, conditions=[encode_anchor, prime_targe_pos])


            else:
                raise NotImplementedError(config.train.mode)

            _metrics['loss'] = accelerator.gather(loss.detach()).mean()
            accelerator.backward(loss.mean())

            optimizer.step()

            if accelerator.sync_gradients:
                lr_scheduler.step()
                train_state.ema_update(config.get('ema_rate', 0.9999))
                train_state.step += 1
                optimizer.zero_grad()

        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        batch = tree_map(lambda x: x.to(device), next(data_generator))
        batch = [batch[i].float() for i in range(len(batch))]
        prime_target, prime_anchor_view, prime_targe_pos = batch
        encode_anchor, encode_target = encode(prime_anchor_view), encode(prime_target)
        metrics = train_step(prime_target, prime_anchor_view, prime_targe_pos, encode_anchor, encode_target)

        nnet.eval()

        last_log_step = getattr(train_state, 'last_log_step', -1)
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0 and train_state.step != last_log_step:
            train_state.last_log_step = train_state.step
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        last_grid_step = getattr(train_state, 'last_grid_step', -1)
        if accelerator.is_main_process and train_state.step % config.train.eval_interval == 1 and train_state.step != last_grid_step:
            train_state.last_grid_step = train_state.step
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            z_init = torch.randn(encode_target.size(), device=device)
            if config.train.mode == 'uncond':
                z = sde.euler_maruyama(sde.ODE(score_model_ema), x_init=z_init, sample_steps=50)
            elif config.train.mode == 'cond':
                z = sde.euler_maruyama(sde.ODE(score_model_ema), x_init=z_init, sample_steps=50,
                                       conditions=[encode_anchor, prime_targe_pos])
            else:
                raise NotImplementedError

                # 【修正】：將所有 Tensor 轉回 fp32 (.float()) 並移至 CPU，確保 WandB 能夠正常渲染
            pred_target = decode(z).float().cpu()
            pred_target_grid = make_grid(dataset.unpreprocess(pred_target), 10)

            decode_target = decode(encode_target).float().cpu()
            decode_target_grid = make_grid(dataset.unpreprocess(decode_target), 10)

                # 將 DataLoader 來的資料取前 3 個通道，轉 float32 並放 CPU
            prime_target_rgb = prime_target[:, :3, :, :].float().cpu()
            prime_target_grid = make_grid(dataset.unpreprocess(prime_target_rgb), 10)

            decode_anchor = decode(encode_anchor).float().cpu()
            decode_anchor_grid = make_grid(dataset.unpreprocess(decode_anchor), 10)

            prime_anchor_view_rgb = prime_anchor_view[:, :3, :, :].float().cpu()
            prime_anchor_view_grid = make_grid(dataset.unpreprocess(prime_anchor_view_rgb), 10)

                # 統一使用後綴為 _grid 的變數進行儲存
            save_image(pred_target_grid, os.path.join(config.sample_dir, f'predict_target-{train_state.step}.png'))
            save_image(decode_target_grid, os.path.join(config.sample_dir, f'decode_target-{train_state.step}.png'))
            save_image(prime_target_grid, os.path.join(config.sample_dir, f'prime_target-{train_state.step}.png'))
            save_image(decode_anchor_grid, os.path.join(config.sample_dir, f'decode_anchor-{train_state.step}.png'))
            save_image(prime_anchor_view_grid,
                           os.path.join(config.sample_dir, f'prime_anchor-{train_state.step}.png'))

                # 安全地紀錄到 wandb
            wandb.log({'samples': wandb.Image(pred_target_grid)}, step=train_state.step)
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        last_eval = getattr(train_state, 'last_eval_step', -1)
        if ((
                    train_state.step % config.train.save_interval == 0 and train_state.step > 0) or train_state.step == config.train.n_steps) and train_state.step != last_eval:
            train_state.last_eval_step = train_state.step
            torch.cuda.empty_cache()
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
            accelerator.wait_for_everyone()
            torch.cuda.empty_cache()

            if accelerator.is_main_process:
                import subprocess
                import re
                logging.info(f"開始自動評估第 {train_state.step} 步的模型碼...")

                eval_dir = f"./eval_dir/scenery/1x_step{train_state.step}/"
                eval_cmd = (
                    f"torchrun --nproc_per_node=1 "
                    f"--master_addr=127.0.0.1 "
                    f"--master_port=36144 "
                    f"evaluate.py "
                    f"--target_expansion 0.25 0.25 0.25 0.25 "
                    f"--eval_dir {eval_dir} "
                    f"--size 128 "
                    f"--config flickr192_large "
                    f"--no-cfg "
                )

                # eval_cmd = f"torchrun --nproc_per_node=1 evaluate.py --target_expansion 0.25 0.25 0.25 0.25 --eval_dir {eval_dir} --size 128 --config flickr192_large --no-cfg"
                print(f"正在產圖: {eval_cmd}")
                subprocess.run(eval_cmd, shell=True)

                fid_cmd = f"python -m pytorch_fid {eval_dir}ori/ {eval_dir}gen/"
                print("正在計算 FID...")
                result = subprocess.run(fid_cmd, shell=True, capture_output=True, text=True)

                print(result.stdout)
                if result.stderr:
                    print(result.stderr)

                match = re.search(r"FID:\s+([0-9.]+)", result.stdout)
                if match:
                    fid_score = float(match.group(1))
                    print(f"成功擷取 FID 分數: {fid_score}，準備上傳 wandb!")
                    wandb.log({"eval/FID_1x": fid_score}, step=train_state.step)
                else:
                    print("警告: 無法從輸出中找到 FID 分數。")

    logging.info(f'Finish fitting, step={train_state.step}')
    accelerator.wait_for_everyone()


from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'x0pred'
    return hparams


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    config.workdir = FLAGS.workdir or os.path.join('workdir', config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)