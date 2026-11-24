import argparse
import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '..')))

import torch
import numpy as np
import pandas as pd
import plotly.express as px

import yaml
import pickle
import wandb
import wandb.plot

from ansi2html import Ansi2HTMLConverter

from model import LitRecurrentTransformerModel, DAGTeacherModel
from Simtransformer.simtransformer.utils import EasyDict

from metric_utils import calc_metrics_across_batches, print_model_steps, calc_batch_metrics_fast, aggregate_formal_metrics, aggregate_batch_metrics
import metric_utils

from train import DAGDataModule
from tokenizers import build_factored_tokenizer
from datetime import datetime


parser = argparse.ArgumentParser(description="Training parameters")
parser.add_argument('--wandb_project', type=str, default="RecTransformer-Nointerm-Eval-v1")
parser.add_argument('--wandb_entity', type=str, default="transformer-computation-graph")
parser.add_argument('--group_name', type=str, required=True)
parser.add_argument('--run_name', type=str, required=True)
parser.add_argument('--n_additional_steps', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_batches', type=int, default=None)
parser.add_argument('--n_samples', type=int, default=5)
parser.add_argument('--out_path', type=str, default='results/nointerm')
parser.add_argument('--overwrite_ok', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--log_wandb', action='store_true')
parser.add_argument('--nointerm', action='store_true')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--ckpt_base_dir', type=str, default=None,
                    help='Base dir of checkpoints; default: experiments/checkpoints')
parser.add_argument('--ckpt_name', type=str, default='last.ckpt',
                    help='Checkpoint filename inside <base>/<group>-<run>/')
parser.add_argument('--config_dir_name', type=str, default=None,
                    help='Explicit config dir under experiments/configs; default: derive from group_name')
parser.add_argument('--val_ds_path', type=str, default=None,
                    help='Val datasets dir; default: <this script dir>/val_datasets')
parser.add_argument('--compile', action='store_true',
                    help='torch.compile the recurrent model (eval speed only; semantics-neutral)')
parser.add_argument('--no_attn_metrics', action='store_true',
                    help='skip attention entropy/max-score metrics (main accuracy metrics unchanged)')

args = parser.parse_args()
if args.no_attn_metrics:
    metric_utils.SKIP_ATTN = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = os.path.dirname(current_dir)
configs_dir = os.path.join(base_dir, 'configs')

val_ds_path = args.val_ds_path or os.path.join(current_dir, 'val_datasets')

group_name = args.group_name
run_name = args.run_name

# load model, train, and data config
config_dir = args.config_dir_name or group_name.replace(' ', '') # calc config dir from wandb group name
# if config_dir has '_nointerm', remove it
config_dir = config_dir.replace('_nointerm', '')
# if config_dir has '_interm', remove it

with open(os.path.join(configs_dir, config_dir, 'model_config.yaml')) as f:
    model_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

with open(os.path.join(configs_dir, config_dir, 'train_config.yaml')) as f:
    train_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

with open(os.path.join(configs_dir, config_dir, 'data_config.yaml')) as f:
    data_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

if hasattr(args, 'nointerm') and args.nointerm:
    train_config.nointerm = True
# load factored_tokenizer
# with open(os.path.join(base_dir, data_config.data_dir, data_config.tokenizer_file_name), 'rb') as f:
#     factored_tokenizer = pickle.load(f)
factored_tokenizer = build_factored_tokenizer(data_config)

# set vocab_size and factors in model_config
model_config.vocab_sizes = factored_tokenizer.vocab_sizes
model_config.factors = factored_tokenizer.factors

# load model
ckpt_base_dir = args.ckpt_base_dir or os.path.join(base_dir, 'checkpoints')
ckpt_fname = args.ckpt_name
model_ckpt_path = os.path.join(ckpt_base_dir, f"{group_name}-{run_name}", ckpt_fname) # model checkpoint path from group name
# model_ckpt = torch.load(model_ckpt_path, weights_only=False)

teacher_model = DAGTeacherModel(model_config, factored_tokenizer)
litmodel = LitRecurrentTransformerModel(model_config, data_config, train_config, teacher_model)

litmodel.load_ckpt(model_ckpt_path)

# move model to device
litmodel.to(device)
if args.compile:
    print('compiling recurrent model for eval ...')
    litmodel.model = torch.compile(litmodel.model)

# create the data module
func_vocab_val_dir = '_'.join(data_config.dag_config.func_vocab)
# get list of .pt files in val_ds_path/func_vocab_val_dir
val_ds_dir = os.path.join(val_ds_path, func_vocab_val_dir)
val_ds_files = [os.path.join(val_ds_dir, f) for f in os.listdir(val_ds_dir) if f.endswith('.pt')]
# match pattern from test_data_N{a}D{b}.pt and get N and D values
val_sizes = sorted([int(fname.split('_')[-1].split('.')[0].split('N')[1]) for fname in val_ds_files])


# set up wandb logging
if args.log_wandb:
    experiment_config = dict(
        model_config=model_config,
        train_config=train_config,
        data_config=data_config,
        val_sizes=val_sizes,
        val_ds_dir=val_ds_dir,
        val_ds_files=val_ds_files,
        val_ds_path=val_ds_path,
        group_name=group_name,
        run_name=run_name,
    )
    run = wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        group=group_name, name=run_name,
        config=experiment_config)



val_metrics_by_size = dict()
val_metrics_by_size_agg = dict()
val_formal_by_size = dict()

sample_preds = dict({n: [] for n in val_sizes})

specs = dict()

start_time = datetime.now()
# evaluate model on different datasets
if args.debug:
    val_sizes = val_sizes[:4] # only evaluate on first few dataset for debugging

for n_nodes in val_sizes:

    torch.cuda.empty_cache()
    print()
    print('-'*50)
    print(f"# Nodes: {n_nodes}")
    print(f"Current time: {datetime.now()}")
    print(f'Elapsed time: {datetime.now() - start_time}')

    ds_file = os.path.join(val_ds_dir, f"testdata_N{n_nodes}.pt")
    print(f"Evaluating model on {ds_file}")

    data_module = DAGDataModule(
        data_path=[ds_file, ds_file],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        factored_tokenizer=factored_tokenizer)


    # add the dimension of data, label, and info to model config
    data_module.setup()
    model_config.data_dim = data_module.data_dim
    model_config.label_dim = data_module.label_dim
    model_config.info_dim = data_module.info_dim

    specs[n_nodes] = data_module.val_spec

    # evaluate model
    # n_steps = None infers the number of steps from the data
    max_depth = data_module.val_spec['max_depth']
    n_steps = max_depth + args.n_additional_steps
    val_metrics = calc_metrics_across_batches(
        litmodel, data_module.val_dataloader(), factored_tokenizer=factored_tokenizer,
        n_steps=n_steps, n_batches=args.n_batches, factors_slice=slice(0, 4), verbose=False, func_to_call=calc_batch_metrics_fast)

    val_metrics_by_size[n_nodes] = val_metrics
    val_formal_by_size[n_nodes] = aggregate_formal_metrics(val_metrics)

    val_metrics_by_size_agg[n_nodes] = aggregate_batch_metrics(val_metrics)

    for k, v in val_metrics_by_size_agg[n_nodes].items():
        print(f"{k}: {v}")
    print(f"formal_metrics N={n_nodes}: {val_formal_by_size[n_nodes]}")

    # get sample predictions
    with torch.no_grad():
        val_iterator = iter(data_module.val_dataloader())
        for i in range(args.n_samples):
            batch = next(val_iterator).to(device)
            sample_step_pred_str = print_model_steps(litmodel, batch, return_str=True, sample=0, n_steps=None)
            sample_preds[n_nodes].append(sample_step_pred_str)

print('DONE')
print(f'Elapsed time: {datetime.now() - start_time}')

# recursively cast Tensor to np.ndarray
def to_numpy(d):
    if isinstance(d, dict):
        return {k: to_numpy(v) for k, v in d.items()}
    elif isinstance(d, torch.Tensor):
        return d.cpu().numpy()
    else:
        return d

val_metrics_by_size_np = to_numpy(val_metrics_by_size)
val_metrics_by_size_agg_np = to_numpy(val_metrics_by_size_agg)
formal_metrics_by_size_np = to_numpy(val_formal_by_size)

# save parsed results

run_name_ = run_name
if not args.overwrite_ok and os.path.exists(os.path.join(args.out_path, group_name, run_name_)):
    suffix = 1
    while os.path.exists(os.path.join(args.out_path, group_name, f"{run_name}_v{suffix}")):
        suffix += 1
    run_name_ = f"{run_name}_v{suffix}"

out_path = os.path.join(args.out_path, group_name, run_name_)
os.makedirs(out_path, exist_ok=True)

np.save(os.path.join(out_path, 'val_metrics_by_size.npy'), val_metrics_by_size_np)
np.save(os.path.join(out_path, 'val_metrics_by_size_agg.npy'), val_metrics_by_size_agg_np)
np.save(os.path.join(out_path, 'formal_metrics_by_size.npy'), formal_metrics_by_size_np)
np.save(os.path.join(out_path, 'sample_preds.npy'), sample_preds)
np.save(os.path.join(out_path, 'specs.npy'), specs)
print(f'Saved to {out_path}')

# get size of depth_val_metrics_np in megabytes

def calc_size(d):
    if isinstance(d, dict):
        return sum([calc_size(v) for v in d.values()])
    elif isinstance(d, np.ndarray):
        return d.nbytes
    else:
        return 0

val_metrics_by_size_mem = calc_size(val_metrics_by_size_np) / 1e6
val_metrics_by_size_agg_mem = calc_size(val_metrics_by_size_agg_np) / 1e6

# [fix-plot] 多 batch 时 raw 指标是 3D（n_batches, n_steps, ...），曲线统一用 batch 均值；
# raw 已单独保存到 val_metrics_by_size.npy，不受影响。
val_metrics_by_size_np = val_metrics_by_size_agg_np

# [no-attn] 跳过注意力指标时，给画图用的副本补零占位（已保存的 .npy 不受影响）
if args.no_attn_metrics:
    for _n in val_sizes:
        _ref = val_metrics_by_size_np[_n]['fully_correct_acc']
        for _k in ['max_attn_score', 'attn_score_entropy', 'attn_score_normalized_entropy']:
            val_metrics_by_size_np[_n][_k] = np.zeros_like(_ref)
sample_preds_mem = sum([sys.getsizeof(s) for s in sample_preds.values()]) / 1e6
specs_mem = sys.getsizeof(specs) / 1e6

print(f"depth_val_metrics_np size: {val_metrics_by_size_mem:.2f} MB")
print(f"depth_val_metrics_agg_np size: {val_metrics_by_size_agg_mem:.2f} MB")
print(f"sample_preds size: {sample_preds_mem:.2f} MB")
print(f"specs size: {specs_mem:.2f} MB")

def add_vline(fig, x=data_config.dag_config.num_nodes, annotation_text='Max Train # Nodes', line_color='black'):
    fig.add_vline(x=x, line_dash='dash', annotation_text=annotation_text, line_color=line_color, annotation_position='bottom left')

print('Generating plots for results...')

# per graph-size figures
print('Per-Size Metrics')
for n_nodes in val_sizes:
    # factor accuracy
    df = pd.DataFrame(val_metrics_by_size_np[n_nodes]['factor_acc'], columns=factored_tokenizer.factors[:4])
    df.insert(0, 'Iteration', range(1, len(val_metrics_by_size_np[n_nodes]['factor_acc']) + 1))
    fig = px.line(df, x='Iteration', y=factored_tokenizer.factors[:4], title=f'Factor Accuracy (# Nodes {n_nodes})')
    fig.update_layout(yaxis_tickformat = '.1%')
    if args.log_wandb:
        wandb.log({f'N = {n_nodes}/Factor Accuracy Table': wandb.Table(dataframe=df)})
        wandb.log({f'N = {n_nodes}/Factor Accuracy': fig})
    # fig.show()

    # fully correct accuracy
    df = pd.DataFrame({'Iteration': range(1, len(val_metrics_by_size_np[n_nodes]['fully_correct_acc']) + 1), 'Fully Correct Accuracy': val_metrics_by_size_np[n_nodes]['fully_correct_acc']})
    fig = px.line(df, x='Iteration', y='Fully Correct Accuracy', title=f'Fully Correct Accuracy (# Nodes {n_nodes})')
    fig.update_layout(yaxis_tickformat = '.1%')
    if args.log_wandb:
        wandb.log({f'N = {n_nodes}/Fully Correct Accuracy Table': wandb.Table(dataframe=df)})
        wandb.log({f'N = {n_nodes}/Fully Correct Accuracy': fig})
    # fig.show()

    # recorrection
    df = pd.DataFrame({'Iteration': range(1, len(val_metrics_by_size_np[n_nodes]['recorrection']) + 1), 'Recorrection %': val_metrics_by_size_np[n_nodes]['recorrection']})
    fig = px.line(df, x='Iteration', y='Recorrection %', title=f'Recorrection % (# Nodes {n_nodes})')
    fig.update_layout(yaxis_tickformat = '.1%')
    if args.log_wandb:
        wandb.log({f'N = {n_nodes}/Recorrection % Table': wandb.Table(dataframe=df)})
        wandb.log({f'N = {n_nodes}/Recorrection %': fig})
    # fig.show()

    # max_attn_score（--no_attn_metrics 时跳过）
    if 'max_attn_score' in val_metrics_by_size_np[n_nodes]:
        df = pd.DataFrame({'Iteration': range(1, len(val_metrics_by_size_np[n_nodes]['max_attn_score']) + 1), 'Max Attention Score': val_metrics_by_size_np[n_nodes]['max_attn_score']})
        fig = px.line(df, x='Iteration', y='Max Attention Score', title=f'Max Attention Score (# Nodes {n_nodes})')
        fig.update_layout(yaxis_tickformat = '.1%')
        if args.log_wandb:
            wandb.log({f'N = {n_nodes}/Max Attention Score Table': wandb.Table(dataframe=df)})
            wandb.log({f'N = {n_nodes}/Max Attention Score': fig})

    # attn_score_entropy（--no_attn_metrics 时跳过）
    if 'attn_score_entropy' in val_metrics_by_size_np[n_nodes]:
        df = pd.DataFrame({'Iteration': range(1, len(val_metrics_by_size_np[n_nodes]['attn_score_entropy']) + 1), 'Attention Score Entropy': val_metrics_by_size_np[n_nodes]['attn_score_entropy']})
        fig = px.line(df, x='Iteration', y='Attention Score Entropy', title=f'Attention Score Entropy (# Nodes {n_nodes})')
        fig.update_layout(yaxis_tickformat = '.1%')
        if args.log_wandb:
            wandb.log({f'N = {n_nodes}/Attention Score Entropy Table': wandb.Table(dataframe=df)})
            wandb.log({f'N = {n_nodes}/Attention Score Entropy': fig})

    # position accuracy
    df = pd.DataFrame({'Position': range(1, val_metrics_by_size_np[n_nodes]['position_acc'].shape[1] + 1), 'Position-wise Accuracy': val_metrics_by_size_np[n_nodes]['position_acc'].mean(axis=0)})
    fig = px.line(df, x='Position', y='Position-wise Accuracy', title=f'Positionwise Average Accuracy (# Nodes {n_nodes})')
    fig.update_layout(yaxis_tickformat = '.1%')
    if args.log_wandb:
        wandb.log({f'N = {n_nodes}/Position-wise Accuracy Table': wandb.Table(dataframe=df)})
        wandb.log({f'N = {n_nodes}/Position-wise Accuracy': fig})
    # fig.show()

# Final Prediction
print('Final Prediction Metrics')
df = pd.DataFrame({'# Nodes': val_sizes, '% Completely Solved': [formal_metrics_by_size_np[d]['final_strict_graph_acc'] for d in val_sizes]})
for spec_key in ['min_depth', 'mean_depth', 'max_depth', 'min_non_pad_len', 'mean_non_pad_len', 'max_non_pad_len']:
    df.insert(1, spec_key, [specs[d][spec_key] for d in val_sizes])

# Formal table: exactly the three acceptance numbers.
formal_df = pd.DataFrame({
    '# Nodes': val_sizes,
    'Value Node Acc': [formal_metrics_by_size_np[d]['final_value_node_acc'] for d in val_sizes],
    'Graph Value All': [formal_metrics_by_size_np[d]['final_value_graph_all_acc'] for d in val_sizes],
    'Strict First-4': [formal_metrics_by_size_np[d]['final_strict_graph_acc'] for d in val_sizes],
})
print('Formal metrics:')
print(formal_df.to_string(index=False))
if args.log_wandb:
    wandb.log({'Formal Metrics/Formal Table': wandb.Table(dataframe=formal_df)})

fig = px.line(df, x='# Nodes', y='% Completely Solved', title='% Completely Solved by Graph Size')
add_vline(fig, x=data_config.dag_config.num_nodes, annotation_text='Max Train # Nodes')
fig.update_layout(yaxis_tickformat = '.1%')
# fig.show()
if args.log_wandb:
    table = wandb.Table(dataframe=df)
    wandb.log({'Summary Metrics/Final Prediction Average Accuracy': np.mean([formal_metrics_by_size_np[d]['final_strict_graph_acc'] for d in val_sizes])})
    wandb.log({'Final Prediction/Final Full Accuracy Table': table})
    wandb.log({'Final Prediction/Final Full Accuracy': fig})
    wandb.log({'Final Prediction/Final Full Accuracy (W&B)': wandb.plot.line(table, x='# Nodes', y='% Completely Solved', title='% Completely Solved by # Nodes')})

# Final Prediction: Factor Accuracy
df = pd.DataFrame(
    {factor: [val_metrics_by_size_np[d]['final_label_factor_acc'][factor_idx] for d in val_sizes] for factor_idx, factor in enumerate(factored_tokenizer.factors[:4])})
df['Graph Size (# Nodes)'] = val_sizes

for spec_key in ['min_depth', 'mean_depth', 'max_depth', 'min_non_pad_len', 'mean_non_pad_len', 'max_non_pad_len']:
    df.insert(1, spec_key, [specs[d][spec_key] for d in val_sizes])

table = wandb.Table(dataframe=df)

fig = px.line(df, x='Graph Size (# Nodes)', y=factored_tokenizer.factors[:4], title='Final Factor Accuracy by Graph Size')
add_vline(fig, x=data_config.dag_config.num_nodes, annotation_text='Train # Nodes')
fig.update_layout(yaxis_tickformat = '.1%')
# fig.show()
if args.log_wandb:
    wandb.log({'Final Prediction/Final Factor Accuracy Table': table})
    wandb.log({'Final Prediction/Final Factor Accuracy': fig})
    wandb.log({'Final Prediction/Final Factor Accuracy (W&B)': wandb.plot.line(table, x='# Nodes', y='% Completely Solved', title='Final Factor Accuracy by # Nodes')})

# attn_score_entropy by graph size
df = pd.DataFrame({'# Nodes': val_sizes, 'Attention Score Entropy': [np.mean(val_metrics_by_size_np[d]['attn_score_entropy']) for d in val_sizes]})
for spec_key in ['min_depth', 'mean_depth', 'max_depth', 'min_non_pad_len', 'mean_non_pad_len', 'max_non_pad_len']:
    df.insert(1, spec_key, [specs[d][spec_key] for d in val_sizes])

fig = px.line(df, x='# Nodes', y='Attention Score Entropy', title='Attention Score Entropy by Graph Size')
add_vline(fig, x=data_config.dag_config.num_nodes, annotation_text='Max Train # Nodes')
# fig.show()
if args.log_wandb:
    table = wandb.Table(dataframe=df)
    wandb.log({'Attention Scores/Entropy Table': table})
    wandb.log({'Attention Scores/Entropy': fig})
    wandb.log({'Attention Scores/Entropy (W&B)': wandb.plot.line(table, x='# Nodes', y='Attention Score Entropy', title='Attention Score Entropy by # Nodes')})

# attn_score_normalized_entropy by graph size
df = pd.DataFrame({'# Nodes': val_sizes, 'Normalized Attention Score Entropy': [np.mean(val_metrics_by_size_np[d]['attn_score_normalized_entropy']) for d in val_sizes]})
for spec_key in ['min_depth', 'mean_depth', 'max_depth', 'min_non_pad_len', 'mean_non_pad_len', 'max_non_pad_len']:
    df.insert(1, spec_key, [specs[d][spec_key] for d in val_sizes])

fig = px.line(df, x='# Nodes', y='Normalized Attention Score Entropy', title='Normalized Attention Score Entropy by Graph Size')
add_vline(fig, x=data_config.dag_config.num_nodes, annotation_text='Max Train # Nodes')
# fig.show()
if args.log_wandb:
    table = wandb.Table(dataframe=df)
    wandb.log({'Attention Scores/Normalized Entropy Table': table})
    wandb.log({'Attention Scores/Normalized Entropy': fig})
    wandb.log({'Attention Scores/Normalized Entropy (W&B)': wandb.plot.line(table, x='# Nodes', y='Normalized Attention Score Entropy', title='Normalized Attention Score Entropy by # Nodes')})

# max_attn_score by graph size
df = pd.DataFrame({'# Nodes': val_sizes, 'Max Attention Score': [np.mean(val_metrics_by_size_np[d]['max_attn_score']) for d in val_sizes]})
for spec_key in ['min_depth', 'mean_depth', 'max_depth', 'min_non_pad_len', 'mean_non_pad_len', 'max_non_pad_len']:
    df.insert(1, spec_key, [specs[d][spec_key] for d in val_sizes])

fig = px.line(df, x='# Nodes', y='Max Attention Score', title='Max Attention Score by Graph Size')
add_vline(fig, x=data_config.dag_config.num_nodes, annotation_text='Max Train # Nodes')
# fig.show()
if args.log_wandb:
    table = wandb.Table(dataframe=df)
    wandb.log({'Attention Scores/Max Score Table': table})
    wandb.log({'Attention Scores/Max Score': fig})
    wandb.log({'Attention Scores/Max Score (W&B)': wandb.plot.line(table, x='# Nodes', y='Max Attention Score', title='Max Attention Score by # Nodes')})

# log sample predictions
print('Logging sample predictions...')
print(f'Current Time: {datetime.now()}')
conv = Ansi2HTMLConverter()

if args.log_wandb:
    for n_nodes in val_sizes:
        for i, sample_pred in enumerate(sample_preds[n_nodes]):
            wandb.log({f'Sample Predictions/N = {n_nodes}/Sample {i}': wandb.Html(conv.convert(sample_pred))})

if args.log_wandb:
    wandb.finish()