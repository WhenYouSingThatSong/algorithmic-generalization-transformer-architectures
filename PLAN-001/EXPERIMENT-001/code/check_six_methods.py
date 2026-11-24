"""External, bounded-update checks of six existing model/loss branches.

This does NOT run the six original CLI programs, reproduce their evaluation
scripts, validate paper accuracy, or establish a self-correction mechanism.
All generated data, checkpoints, configurations, caches, and reports go under
the external --runs directory. Repository files are never intentionally written.

Prepare configurations with build_smoke_configs.py before running this script.
The module uses only the standard library until main() loads the ready runtime.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time
import traceback
from typing import Any

sys.dont_write_bytecode = True

METHODS = (
    "M1_feedforward", "M2_recurrent", "M3_cot", "M4_continuous",
    "M5_discrete", "M6_discrete_correction",
)
BOUNDARIES = [
    "Core-model/loss smoke check only: not successful execution of six original CLIs.",
    "Tiny diagnostic configurations and a few optimizer updates do not test paper accuracy.",
    "Repository data-generation functions are called directly; their CLI wrappers are bypassed.",
    "The CoT generator CLI prints five examples even for four samples; this runner bypasses that print loop.",
    "External Lightning subclasses call original compute_batch_loss methods with log=False.",
    "This bypasses the known fast-path logging-variable overwrite; original training logs are not validated.",
    "CoT uses the original CoTDataset with a bool-mask adapter; original buggy metrics are not accepted or reported.",
    "CoT padding/target-mask alignment and the underlying training loss are otherwise left unchanged.",
    "Correction probability 1 is a diagnostic override; tokenwise selection and original last-mask semantics are retained.",
    "Validation is disabled; bounded inference is autonomous but is not the official OOD evaluation.",
    "New checkpoints are reloaded with strict state_dict loading, not the original load_ckpt prefix-conversion helper.",
    "Only newly created trusted checkpoints are loaded with weights_only=False; no distributed demo checkpoint is loaded.",
]


def within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes(repo: Path) -> dict[str, str]:
    """Fingerprint source/config files, excluding Git metadata and runtime caches."""
    result = {}
    for directory, names, files in os.walk(repo):
        names[:] = sorted(n for n in names if n not in {".git", "__pycache__"})
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix.lower() in {".py", ".yaml", ".yml", ".md"}:
                result[path.relative_to(repo).as_posix()] = digest(path)
    return result


def git_info(repo: Path) -> dict[str, Any]:
    result = {}
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    for key, command in (
        ("commit", ["rev-parse", "HEAD"]),
        ("tracked_status", ["status", "--porcelain", "--untracked-files=no"]),
    ):
        try:
            process = subprocess.run(
                ["git", "-C", str(repo), *command], capture_output=True,
                text=True, timeout=30, env=env, check=False,
            )
            result[key] = process.stdout.strip() if process.returncode == 0 else None
            if process.returncode:
                result[key + "_error"] = process.stderr.strip()
        except Exception as error:
            result[key + "_error"] = str(error)
    return result


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_runtime(repo: Path, run_root: Path) -> dict[str, Any]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["MPLCONFIGDIR"] = str(run_root / "cache" / "matplotlib")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DIR"] = str(run_root / "logs" / "wandb")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for directory in (run_root / "cache" / "matplotlib", run_root / "logs" / "wandb"):
        directory.mkdir(parents=True, exist_ok=True)
    entries = [repo / "experiments", repo / "experiments" / "baselines",
               repo / "experiments" / "evaluation", repo]
    for entry in reversed(entries):
        sys.path.insert(0, str(entry))
    modules = {name: importlib.import_module(name) for name in ("torch", "yaml", "lightning.pytorch")}
    for name in ("tokenizers", "model", "train", "baseline_models", "baseline_cot_train",
                 "generate_data", "generate_cot_data", "Simtransformer.simtransformer.utils"):
        modules[name] = importlib.import_module(name)
    expected = repo / "experiments" / "tokenizers.py"
    actual = Path(modules["tokenizers"].__file__).resolve()
    if actual != expected.resolve():
        raise RuntimeError(f"Wrong tokenizers module: {actual}; expected {expected}")
    return modules


def make_classes(runtime: dict[str, Any]) -> dict[str, Any]:
    torch = runtime["torch"]
    base = runtime["baseline_models"]
    latent = runtime["model"]
    original_cot_dataset = runtime["baseline_cot_train"].CoTDataset

    class QuietSmokeMixin:
        # The CoT implementation calls self.log for some metrics even with log=False.
        # Suppress logging only; retain its original loss implementation.
        def log(self, *args, **kwargs):
            return None

        def remember_loss(self, loss):
            scalar = float(loss.detach().float().cpu())
            if not torch.isfinite(loss.detach()).all().item():
                raise RuntimeError(f"Non-finite training loss: {scalar}")
            self.smoke_losses.append(scalar)
            self.smoke_training_batches += 1

    class FactoredSmoke(QuietSmokeMixin, base.LitRecurrentTransformerModel):
        def training_step(self, batch, batch_idx):
            loss = self.compute_batch_loss(batch, log_prefix="train", log=False)
            self.remember_loss(loss)
            return loss

    class CoTSmoke(QuietSmokeMixin, base.LitCoTRecurrentTransformerModel):
        def training_step(self, batch, batch_idx):
            loss = self.compute_batch_loss(batch, log_prefix="train", log=False)
            self.remember_loss(loss)
            return loss

    class LatentSmoke(QuietSmokeMixin, latent.LitRecurrentTransformerModel):
        def training_step(self, batch, batch_idx):
            if self.nointerm:
                loss = self.compute_batch_loss_nointerm(batch, log_prefix="train", log=False)
            else:
                # Matches the generated diagnostic teacher=True, fast=True configuration.
                if not self.train_config.use_teacher_pred_for_next_loop or not self.train_config.use_fast_batch_forward:
                    raise RuntimeError("Discrete smoke requires teacher=True and fast=True")
                loss = self.compute_batch_loss_fast(batch, log_prefix="train", log=False)
            self.remember_loss(loss)
            # Original manual-optimization loss already performs backward/optimizer.step.
            return loss.detach()

        def randomly_corrupt_values(self, inputs, available):
            before = inputs.detach().clone()
            corrupted, mask = super().randomly_corrupt_values(inputs, available)
            value_index = self.model_config.factors.index("VALUE")
            old = before[..., value_index]
            new = corrupted[..., value_index]
            changed = old.ne(new)
            empty = self.teacher_model.value_tok2idx["EMPTY"]
            not_applicable = self.teacher_model.value_tok2idx["N/A"]
            entry = {
                "selected_positions": int(mask.sum().item()),
                "actual_changed_value_positions": int(changed.sum().item()),
                "selected_originally_empty": int((mask & old.eq(empty)).sum().item()),
                "changed_originally_empty": int((changed & old.eq(empty)).sum().item()),
                "selected_originally_numeric": int((mask & old.ne(empty) & old.ne(not_applicable)).sum().item()),
                "selected_unchanged": int((mask & ~changed).sum().item()),
                "changes_outside_selected_mask": int((changed & ~mask).sum().item()),
                "selected_per_depth": [int(x.sum().item()) for x in mask],
                "actual_changed_per_depth": [int(x.sum().item()) for x in changed],
            }
            other_factors = [i for i in range(before.shape[-1]) if i != value_index]
            entry["changed_non_value_components"] = int(
                before[..., other_factors].ne(corrupted[..., other_factors]).sum().item()
            )
            self.smoke_corruption_calls.append(entry)
            return corrupted, mask

    class BoolCoTDataset(original_cot_dataset):
        def __getitem__(self, index):
            xy, masks = super().__getitem__(index)
            return xy, tuple(mask.bool() for mask in masks)

    return {"factored": FactoredSmoke, "cot": CoTSmoke, "latent": LatentSmoke,
            "cot_dataset": BoolCoTDataset}


def read_configs(method: str, setup: Path, runtime: dict) -> dict:
    directory = setup / "experiments" / "smoke" / method
    configs = {}
    for name in ("data", "model", "train"):
        path = directory / f"{name}_config.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run build_smoke_configs.py first.")
        configs[name] = runtime["yaml"].safe_load(path.read_text(encoding="utf-8"))
    train = configs["train"]
    if train.get("compile") or train.get("precision", 32) not in (32, "32", "32-true"):
        raise ValueError("This runner requires compile=False and FP32 smoke configurations")
    if int(train.get("batch_size", 1)) != 1 or int(train.get("num_workers", 0)) != 0:
        raise ValueError("Smoke configuration must have batch_size=1 and num_workers=0")
    if train.get("lr_scheduler", "none") != "none":
        raise ValueError("Smoke uses the original optimizer with no scheduler")
    if method == "M1_feedforward" and configs["model"].get("n_iters") != 1:
        raise ValueError("Feedforward branch must configure n_iters=1")
    if method == "M2_recurrent" and int(configs["model"].get("n_iters", 1)) <= 1:
        raise ValueError("Fixed recurrent branch must configure n_iters>1")
    if method.startswith(("M4", "M5", "M6")):
        if bool(train.get("nointerm", False)) != (method == "M4_continuous"):
            raise ValueError("Method label disagrees with nointerm configuration")
        if bool(train.get("recorrection_args", {}).get("enable", False)) != (method == "M6_discrete_correction"):
            raise ValueError("Method label disagrees with correction configuration")
    if method == "M6_discrete_correction":
        correction = train.get("recorrection_args", {})
        if not correction.get("enable") or correction.get("corruption_prob") != 1.0:
            raise ValueError("Correction smoke must explicitly configure enable=True and corruption_prob=1")
        if correction.get("corruption_strategy") != "last":
            raise ValueError("Original last-mask semantics must remain unchanged in this diagnostic")
    return configs


def prepare_data(configs: dict, run_root: Path, cache: dict, runtime: dict, seed: int) -> dict:
    torch = runtime["torch"]
    pl = runtime["lightning.pytorch"]
    data = copy.deepcopy(configs["data"])
    is_cot = bool(data.get("cot", False))
    identity = json.dumps({"data": data, "seed": seed, "cot": is_cot}, sort_keys=True)
    key = hashlib.sha256(identity.encode()).hexdigest()[:12]
    if key in cache:
        return copy.deepcopy(cache[key])
    started = time.perf_counter()
    directory = run_root / "data" / (("cot-" if is_cot else "factored-") + key)
    directory.mkdir(parents=True, exist_ok=False)
    configured_data_dir = data["data_dir"]
    data["data_dir"] = str(directory)
    EasyDict = runtime["Simtransformer.simtransformer.utils"].EasyDict
    pl.seed_everything(seed, workers=True)
    tokenizer_args = dict(
        n_vars=max(data["dag_config"]["num_nodes"], data["val_dag_config"]["num_nodes"]),
        ops=data["dag_config"]["func_vocab"], mod_val=data["dag_config"]["mod_val"],
    )
    if is_cot:
        tokenizer = runtime["tokenizers"].CoTTokenizer(**tokenizer_args)
    else:
        tokenizer = runtime["tokenizers"].FactoredVocabTokenizer(
            **tokenizer_args, max_fan_in=data["dag_config"]["max_fan_in_deg"])
    tokenizer_path = directory / data["tokenizer_file_name"]
    with tokenizer_path.open("wb") as stream:
        pickle.dump(tokenizer, stream)
    summary = {}
    for split in ("dag_config", "val_dag_config"):
        cfg = EasyDict(data[split])
        if is_cot:
            values, info = runtime["generate_cot_data"].generate_cot_dataset(
                cfg, tokenizer, data.get("cot_type", "eq-val"))
        else:
            values, info = runtime["generate_data"].generate_dataset(cfg, tokenizer)
        path = directory / cfg.data_file_name
        if not within(path, directory):
            raise ValueError("Data filename escapes its external run directory")
        torch.save({"tokenized_dataset": values, "dataset_info": info}, path)
        summary[split] = {"path": str(path), "sha256": digest(path),
                          "shape": list(values.shape), "samples": len(values)}
        if not is_cot:
            summary[split]["maximum_zero_based_depth"] = int(info[..., 0].max().item())
    result = {"effective_data_config": data, "original_configured_data_dir": configured_data_dir,
              "note": "Per-run external data directory isolates generated diagnostic artifacts.",
              "tokenizer_path": str(tokenizer_path), "tokenizer_sha256": digest(tokenizer_path),
              "splits": summary, "generation_seconds": time.perf_counter() - started}
    cache[key] = copy.deepcopy(result)
    return result


def tensor_tree_to(value, device):
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, (tuple, list)):
        return type(value)(tensor_tree_to(item, device) for item in value)
    if isinstance(value, dict):
        return {key: tensor_tree_to(item, device) for key, item in value.items()}
    return value


def gradients(model, torch) -> dict:
    items = [parameter.grad.detach() for parameter in model.parameters() if parameter.grad is not None]
    return {"tensors": len(items),
            "all_finite": bool(items) and all(torch.isfinite(item).all().item() for item in items),
            "nonzero_tensors": sum(bool(item.ne(0).any().item()) for item in items),
            "max_abs": max((float(item.abs().max().cpu()) for item in items), default=0.0)}


def cuda_sync(torch, device) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def cuda_memory(torch, device) -> dict:
    if device != "cuda":
        return {"device": "cpu", "peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {"device": "cuda", "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def bounded_forward(model, probe, method, tokenizer, torch, device, steps, cot_tokens):
    """Use actual model forwards, with no teacher states during continuation."""
    model.to(device)
    model.eval()
    probe = tensor_tree_to(probe, device)
    signatures = []
    summary = {"paper_accuracy_measured": False}

    def remember(logits):
        items = logits if isinstance(logits, (tuple, list)) else [logits]
        for item in items:
            if not torch.isfinite(item).all().item():
                raise RuntimeError("Non-finite bounded-forward logits")
            signatures.append(item.detach().cpu().clone())

    with torch.no_grad():
        if method == "M3_cot":
            x = probe[0][0]
            cot_id = tokenizer.token2idx[tokenizer.COT]
            positions = x[0].eq(cot_id).nonzero(as_tuple=False)
            if positions.numel() == 0:
                raise RuntimeError("CoT example contains no COT delimiter")
            prefix_length = int(positions[0, 0]) + 1
            tokens = x[:, :prefix_length].clone()
            count = min(cot_tokens, int(model.model_config.max_seq_len) - prefix_length)
            if count <= 0:
                raise RuntimeError("No context budget for bounded CoT generation")
            generated = []
            for _ in range(count):
                logits = model(tokens)[:, -1, :]
                remember(logits)
                token = logits.argmax(dim=-1, keepdim=True)
                generated.append(int(token[0, 0]))
                tokens = torch.cat([tokens, token], dim=1)
            summary.update(kind="bounded_autoregressive_generation", prefix_tokens=prefix_length,
                           generated_tokens=count, generated_ids=generated,
                           generated_text=tokenizer.decode_tokens(generated, return_string=True))
        else:
            # Keep repository trimming and token conventions; there is no new evaluator here.
            batch = model.reduce_batch(probe)
            inputs = batch[..., :model.model_config.data_dim]
            if method in ("M1_feedforward", "M2_recurrent"):
                logits = model(inputs)
                remember(logits)
                summary.update(kind="fixed_budget_final_forward", configured_n_iters=model.model_config.n_iters,
                               input_shape=list(inputs.shape))
            elif method == "M4_continuous":
                hidden = model.model.token_embedder(inputs)
                for _ in range(steps):
                    hidden, logits = model.model.TF_forward(hidden)
                    remember(logits)
                summary.update(kind="continuous_hidden_recurrence", loops=steps,
                               input_shape=list(inputs.shape), teacher_feedback=False)
            else:
                state = inputs.clone()
                for _ in range(steps):
                    logits = model(state)
                    remember(logits)
                    state = torch.stack([item.argmax(dim=-1) for item in logits], dim=-1)
                    # This matches the repository's loop_forward auxiliary-factor reset.
                    for factor, mapping in (("REGRET", tokenizer.regret_tok2idx),
                                            ("MODIFICATION", tokenizer.modification_tok2idx)):
                        state[..., model.model_config.factors.index(factor)] = mapping["FALSE"]
                summary.update(kind="discrete_argmax_recurrence", loops=steps,
                               input_shape=list(inputs.shape), teacher_feedback=False,
                               auxiliary_factors_reset_to_false=True)
    return signatures, summary


def compare_outputs(reference, actual, torch) -> dict:
    if len(reference) != len(actual):
        raise AssertionError("Different number of inference outputs after reload")
    maximum = 0.0
    for before, after in zip(reference, actual):
        torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
        maximum = max(maximum, float((before - after).abs().max()))
    return {"passed": True, "compared_tensors": len(reference),
            "max_absolute_difference": maximum, "rtol": 1e-5, "atol": 1e-6}


def run_method(method, args, run_root, runtime, classes, data_cache, result) -> dict:
    torch, pl = runtime["torch"], runtime["lightning.pytorch"]
    EasyDict = runtime["Simtransformer.simtransformer.utils"].EasyDict
    started = time.perf_counter()
    configs = read_configs(method, args.setup, runtime)
    if (method == "M3_cot") != bool(configs["data"].get("cot", False)):
        raise ValueError("Method and data tokenization configuration disagree")
    seed = args.seed if args.seed is not None else int(configs["train"].get("seed", 42))
    configs["train"]["seed"] = seed
    prepared = prepare_data(configs, run_root, data_cache, runtime, seed)
    configs["data"] = prepared["effective_data_config"]
    directory = run_root / "methods" / method
    directory.mkdir(parents=True, exist_ok=False)
    for name, config in configs.items():
        (directory / f"effective_{name}_config.yaml").write_text(
            runtime["yaml"].safe_dump(config, sort_keys=False), encoding="utf-8")
    result.update(seed=seed, data=prepared, effective_configurations=configs,
                  original_configuration_directory=str(args.setup / "experiments" / "smoke" / method))
    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("highest")
    data_cfg, model_cfg, train_cfg = (EasyDict(copy.deepcopy(configs[name])) for name in ("data", "model", "train"))
    train_cfg.seed = seed
    tokenizer_kwargs = dict(n_vars=max(data_cfg.dag_config.num_nodes, data_cfg.val_dag_config.num_nodes),
                            ops=data_cfg.dag_config.func_vocab, mod_val=data_cfg.dag_config.mod_val)
    if method == "M3_cot":
        tokenizer = runtime["tokenizers"].CoTTokenizer(**tokenizer_kwargs)
        dataset = classes["cot_dataset"](Path(data_cfg.data_dir) / data_cfg.dag_config.data_file_name)
        model_cfg.vocab_size = len(tokenizer.vocab)
        model_cfg.max_seq_len = max(model_cfg.max_seq_len, data_cfg.dag_config.max_length,
                                   data_cfg.val_dag_config.max_length)
        result["cot_mask_adapter"] = {
            "original_mask_dtypes": {name: str(dataset.ds_info[name].dtype)
                                     for name in ("cot_mask", "value_mask", "var_mask", "operation_mask")},
            "returned_masks": "torch.bool", "mask_positions_changed": False,
            "original_accuracy_metrics_accepted": False,
        }
    else:
        tokenizer = runtime["tokenizers"].FactoredVocabTokenizer(
            **tokenizer_kwargs, max_fan_in=data_cfg.dag_config.max_fan_in_deg)
        data_module = runtime["train"].DAGDataModule(
            data_path=[str(Path(data_cfg.data_dir) / data_cfg.dag_config.data_file_name),
                       str(Path(data_cfg.data_dir) / data_cfg.val_dag_config.data_file_name)],
            batch_size=1, num_workers=0, factored_tokenizer=tokenizer)
        data_module.setup()
        model_cfg.vocab_sizes, model_cfg.factors = tokenizer.vocab_sizes, tokenizer.factors
        model_cfg.data_dim, model_cfg.label_dim, model_cfg.info_dim = (
            data_module.data_dim, data_module.label_dim, data_module.info_dim)
        dataset = data_module.train_dataloader().dataset
    result["resolved_model_configuration"] = dict(model_cfg)
    # Keep one explicit sample for round-trip comparisons; training shuffles four examples.
    probe_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    probe = next(iter(probe_loader))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    if method == "M3_cot" and not probe[1][0].any().item():
        raise RuntimeError("Empty CoT training mask")

    def construct():
        mc, dc, tc = (copy.deepcopy(item) for item in (model_cfg, data_cfg, train_cfg))
        if method in ("M1_feedforward", "M2_recurrent"):
            module = classes["factored"](mc, dc, tc)
            module.factored_tokenizer = tokenizer
        elif method == "M3_cot":
            module = classes["cot"](mc, dc, tc)
            module.factored_tokenizer = tokenizer
        else:
            teacher = runtime["model"].DAGTeacherModel(mc, tokenizer)
            module = classes["latent"](mc, dc, tc, teacher)
        module.smoke_losses = []
        module.smoke_training_batches = 0
        module.smoke_corruption_calls = []
        return module

    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = construct()
    initial = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    result["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    result["parameter_dtypes"] = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("Expected FP32 model parameters")
    result["training_branch"] = (
        "CoT next-token loss" if method == "M3_cot" else
        "final factored-label loss" if method in ("M1_feedforward", "M2_recurrent") else
        "continuous per-loop supervised loss" if method == "M4_continuous" else
        "discrete teacher-forced fast loss"
    )
    trainer = pl.Trainer(
        accelerator="gpu" if args.device == "cuda" else "cpu", devices=1,
        precision="32-true", logger=False, enable_checkpointing=False,
        enable_model_summary=False, enable_progress_bar=False,
        max_epochs=1, limit_train_batches=args.train_batches, limit_val_batches=0,
        num_sanity_val_steps=0, accumulate_grad_batches=1,
        default_root_dir=str(directory), log_every_n_steps=1,
    )
    cuda_sync(torch, args.device)
    training_started = time.perf_counter()
    trainer.fit(model, train_dataloaders=loader)
    cuda_sync(torch, args.device)
    result["training_seconds"] = time.perf_counter() - training_started
    result["training_cuda_memory"] = cuda_memory(torch, args.device)
    result["losses"] = list(model.smoke_losses)
    result["training_batches"] = model.smoke_training_batches
    result["trainer_global_step"] = int(trainer.global_step)
    result["gradients"] = gradients(model, torch)
    if model.smoke_training_batches != args.train_batches or trainer.global_step != args.train_batches:
        raise RuntimeError(f"Expected exactly {args.train_batches} batches and optimizer updates")
    if not result["gradients"]["all_finite"] or not result["gradients"]["nonzero_tensors"]:
        raise RuntimeError("Expected finite, nonzero gradients after the one training update")
    changed = 0
    maximum_change = 0.0
    for name, parameter in model.named_parameters():
        now = parameter.detach().cpu()
        if not torch.isfinite(now).all().item():
            raise RuntimeError(f"Non-finite trained parameter: {name}")
        if not torch.equal(initial[name], now):
            changed += 1
            maximum_change = max(maximum_change, float((now - initial[name]).abs().max()))
    result["parameter_update"] = {"changed_tensors": changed, "max_absolute_change": maximum_change}
    if not changed:
        raise RuntimeError("No parameters changed")
    result["corruption"] = {
        "probability_override": 1.0 if method == "M6_discrete_correction" else None,
        "original_last_mask_retained": method == "M6_discrete_correction",
        "function_calls": len(model.smoke_corruption_calls),
        "events": copy.deepcopy(model.smoke_corruption_calls),
        "actual_changed_value_positions": sum(event["actual_changed_value_positions"]
                                               for event in model.smoke_corruption_calls),
    }
    if method == "M6_discrete_correction":
        if result["corruption"]["function_calls"] != args.train_batches:
            raise RuntimeError("Expected one original corruption call per training batch")
        if not result["corruption"]["actual_changed_value_positions"]:
            raise RuntimeError("Corruption was called but changed no values; retry with another --seed")
    elif result["corruption"]["function_calls"]:
        raise RuntimeError("Unexpected corruption in a non-correction branch")

    inference_started = time.perf_counter()
    reference, inference = bounded_forward(model, probe, method, tokenizer, torch, args.device,
                                           args.bounded_steps, args.cot_new_tokens)
    cuda_sync(torch, args.device)
    result["bounded_inference"] = inference
    result["inference_seconds"] = time.perf_counter() - inference_started
    checkpoint = directory / f"{args.train_batches}_updates.ckpt"
    state_path = directory / "model_state_dict.pt"
    save_started = time.perf_counter()
    trainer.save_checkpoint(str(checkpoint))
    cpu_state = {key: value.detach().cpu().clone() for key, value in model.model.state_dict().items()}
    torch.save(cpu_state, state_path)
    result["checkpoint_save_seconds"] = time.perf_counter() - save_started
    del initial

    reload_started = time.perf_counter()
    fresh = construct()
    # We created this checkpoint in this process; it is not an untrusted input.
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fresh.load_state_dict(checkpoint_data["state_dict"], strict=True)
    actual, _ = bounded_forward(fresh, probe, method, tokenizer, torch, args.device,
                                args.bounded_steps, args.cot_new_tokens)
    full_comparison = compare_outputs(reference, actual, torch)
    fresh_state = fresh.model.state_dict()
    if set(cpu_state) != set(fresh_state) or not all(
            torch.equal(cpu_state[key], fresh_state[key].detach().cpu()) for key in cpu_state):
        raise AssertionError("Full checkpoint did not preserve exact model state")
    del fresh, actual, checkpoint_data, fresh_state
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    fresh = construct()
    saved_state = torch.load(state_path, map_location="cpu", weights_only=True)
    fresh.model.load_state_dict(saved_state, strict=True)
    actual, _ = bounded_forward(fresh, probe, method, tokenizer, torch, args.device,
                                args.bounded_steps, args.cot_new_tokens)
    state_comparison = compare_outputs(reference, actual, torch)
    cuda_sync(torch, args.device)
    result["reload_seconds"] = time.perf_counter() - reload_started
    result["strict_reload"] = {"lightning_checkpoint": full_comparison,
                               "pure_model_state_dict": state_comparison,
                               "original_load_ckpt_helper_used": False}
    result["artifacts"] = {"checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
                           "state_dict": str(state_path), "state_dict_sha256": digest(state_path)}
    result["whole_method_cuda_memory"] = cuda_memory(torch, args.device)
    result["total_seconds"] = time.perf_counter() - started
    result["passed"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--setup", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--train-batches", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--bounded-steps", type=int, default=3)
    parser.add_argument("--cot-new-tokens", type=int, default=4)
    args = parser.parse_args()
    args.repo, args.setup = args.repo.resolve(), args.setup.resolve()
    args.runs = (args.runs or args.setup.parent / "repro_runs").resolve()
    if not 1 <= args.bounded_steps <= 16 or not 1 <= args.cot_new_tokens <= 16:
        parser.error("Bounded inference lengths must be in 1..16")
    if len(set(args.methods)) != len(args.methods):
        parser.error("Do not repeat a method")
    if within(args.runs, args.repo) or within(args.setup, args.repo):
        parser.error("--setup and --runs must be outside the code repository")
    if not (args.repo / "experiments" / "model.py").is_file():
        parser.error("--repo does not contain experiments/model.py")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_root = args.runs / "core_smoke" / run_id
    args.output = (args.output or run_root / "six_methods_report.json").resolve()
    if within(args.output, args.repo):
        parser.error("--output must be outside the code repository")
    run_root.mkdir(parents=True, exist_ok=False)
    before = source_hashes(args.repo)
    report = {
        "schema_version": 2, "scope": "six existing core model/loss branches, bounded updates",
        "expected_optimizer_steps_per_method": args.train_batches,
        "diagnostic_script_sha256": digest(Path(__file__)),
        "boundaries": BOUNDARIES, "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "repo": str(args.repo), "setup": str(args.setup),
        "run_root": str(run_root), "output": str(args.output), "device": args.device,
        "git_before": git_info(args.repo), "selected_methods": args.methods,
        "trainer_policy": {"logger": False, "limit_train_batches": args.train_batches, "max_epochs": 1,
                           "limit_val_batches": 0, "num_workers": 0, "precision": "32-true",
                           "compile": False, "automatic_checkpointing": False},
        "methods": [], "passed": False,
    }
    write_report(args.output, report)
    runtime = None
    try:
        runtime = load_runtime(args.repo, run_root)
        torch = runtime["torch"]
        torch.set_num_threads(4)
        report["packages"] = {}
        for package in ("torch", "lightning", "pytorch-lightning", "torchmetrics", "numpy", "PyYAML"):
            try:
                report["packages"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                report["packages"][package] = None
        report["repository_imports"] = {
            name: str(Path(module.__file__).resolve())
            for name, module in runtime.items() if name not in ("torch", "yaml", "lightning.pytorch")
        }
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; no silent CPU fallback")
            report["gpu"] = {"name": torch.cuda.get_device_name(0),
                             "capability": list(torch.cuda.get_device_capability(0)),
                             "torch_cuda": torch.version.cuda,
                             "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}
        classes = make_classes(runtime)
        data_cache = {}
        for method in args.methods:
            print(f"[core smoke] {method}", flush=True)
            method_started = time.perf_counter()
            result = {"method": method, "passed": False}
            try:
                result = run_method(method, args, run_root, runtime, classes, data_cache, result)
            except Exception:
                result.update(passed=False, error=traceback.format_exc(),
                              elapsed_seconds_before_failure=time.perf_counter() - method_started)
            report["methods"].append(result)
            write_report(args.output, report)
            print(f"[core smoke] {method}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
            if not result["passed"]:
                print(result["error"], flush=True)
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        after = source_hashes(args.repo)
        changed = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
        report["repository_source_check"] = {
            "unchanged": not changed, "changed_source_or_config_files": changed,
            "fingerprinted_files": len(before), "bytecode_writes_disabled": True,
            "note": "Checks .py/.yaml/.yml/.md content; Git metadata may be unavailable.",
        }
        report["git_after"] = git_info(args.repo)
        report["passed"] = (not report.get("error") and not changed
                            and len(report["methods"]) == len(args.methods)
                            and all(item["passed"] for item in report["methods"]))
        write_report(args.output, report)
    print(json.dumps({"passed": report["passed"], "report": str(args.output),
                      "methods": [{"method": item["method"], "passed": item["passed"]}
                                  for item in report["methods"]]}, ensure_ascii=True, indent=2))
    if report.get("error"):
        print(report["error"], file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
