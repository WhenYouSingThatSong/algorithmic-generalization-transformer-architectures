import torch
import copy
from tqdm import tqdm
import numpy as np
from Simtransformer.simtransformer.utils import attach_hooks, EasyDict

# [eval-speed] 注意力熵/最大注意力指标在每个 batch 的每个 step 都要对 [B,H,T,T]
# 重算一遍，是 C0 伴随矩阵评测的主要开销（且与主指标无关）。评测脚本可置 True 跳过。
SKIP_ATTN = False


def attach_hooks_to_model(litmodel):
    HookDict = EasyDict()

    for i in range(litmodel.model_config.num_layers):
        HookDict[f'L{i}_R0'] = f'blocks.layer_{i}.input'
        HookDict[f'L{i}_R1'] = f'blocks.layer_{i}.attn_res_output'
        HookDict[f'L{i}_R2'] = f'blocks.layer_{i}.output'
        HookDict[f'L{i}_Ain'] = f'blocks.layer_{i}.attn.input'
        HookDict[f'L{i}_Aout'] = f'blocks.layer_{i}.attn.output'
        HookDict[f'L{i}_Min'] = f'blocks.layer_{i}.mlp.input'
        HookDict[f'L{i}_Mout'] = f'blocks.layer_{i}.mlp.output'
    for i in range(litmodel.model_config.num_layers):
        HookDict[f'L{i}_attn_prob'] = f'blocks.layer_{i}.attn.attn_prob'

    # attach hooks to the TF model
    buffer, hook_handles = attach_hooks(litmodel.model.transformer, HookDict)
    return buffer, hook_handles

def compute_attention_metrics(buffer):
    # attn_score keys
    attn_score_keys = [k for k in buffer.keys() if 'attn_prob' in k]

    layerwise_max_attn_score = {k: buffer[k].max(dim=-1).values.mean().item() for k in attn_score_keys}
    avg_max_attn_score = torch.tensor(list(layerwise_max_attn_score.values())).mean()

    layerwise_attn_score_entropy = {k: -(buffer[k] * torch.log(buffer[k] + 1e-6)).sum(dim=-1).mean().item() for k in attn_score_keys}
    avg_attn_score_entropy = torch.tensor(list(layerwise_attn_score_entropy.values())).mean()

    # Entropy / log(n), where n = number of tokens
    layerwise_attn_score_normalized_netropy = {k: -(buffer[k] * torch.log(buffer[k] + 1e-6)).sum(dim=-1).mean().item() / np.log(buffer[k].shape[-1]) for k in attn_score_keys}
    avg_attn_score_normalized_entropy = torch.tensor(list(layerwise_attn_score_normalized_netropy.values())).mean()

    attention_metrics = dict(
        attn_score_entropy=avg_attn_score_entropy,
        attn_score_normalized_entropy=avg_attn_score_normalized_entropy,
        max_attn_score=avg_max_attn_score)
    return attention_metrics

def one_step_prediction(litmodel, batch, step, outputs=None, step_stack=None):

    batch = batch.to(litmodel.device)
    # print('inputs', inputs.shape)
    if hasattr(litmodel, 'nointerm') and litmodel.nointerm:
        if step == 0:
            with torch.no_grad():
                input_emb = litmodel.model.token_embedder(batch)
        else:
            input_emb = outputs
        # if input only has two dimensions, add a dimension
        if input_emb.dim() == 2:
            input_emb = input_emb.unsqueeze(0)
        outputs, logitss = litmodel.predict_step(input_emb)
        return outputs, logitss
    else:
        inputs = step_stack[-1]
        inputs.to(litmodel.device)
        # if input only has two dimensions, add a dimension
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        logitss = litmodel.predict_step(inputs)
        return None, logitss



def calc_batch_metrics(litmodel, batch, factored_tokenizer, n_steps=None, factors_slice=None, verbose=False):
    """Compatibility wrapper around the corrected fast implementation.

    The old implementation had a separate batch-size-1 branch and counted PAD
    as correct in factor/position statistics.  Keep the public name, but use one
    implementation so slow/fast eval cannot diverge.
    """
    return calc_batch_metrics_fast(
        litmodel, batch, factored_tokenizer,
        n_steps=n_steps, factors_slice=factors_slice, verbose=verbose)


def calc_metrics_across_batches(litmodel, dataloader, factored_tokenizer, n_batches=None, n_steps=None, factors_slice=None, verbose=False, func_to_call=calc_batch_metrics):

    if factors_slice is None:
        factors_slice = slice(0, 6) # consider all factors

    metrics = dict(factor_acc=[], position_acc=[], fully_correct_acc=[], recorrection=[])

    for i, batch in tqdm(enumerate(dataloader)):
        if n_batches is not None and i >= n_batches:
            break

        # move batch to device
        if batch.device != litmodel.device:
            batch = batch.to(litmodel.device)

        batch_metrics = func_to_call(litmodel, batch, factored_tokenizer, n_steps=n_steps, factors_slice=factors_slice, verbose=verbose)

        for k, v in batch_metrics.items():
            if k not in metrics:
                metrics[k] = []
            metrics[k].append(v)

    metrics['final_label_full_acc'] = torch.tensor(metrics['final_label_full_acc'])

    for k, v in metrics.items():
        if isinstance(v, list):
            if isinstance(v[0], torch.Tensor):
                metrics[k] = torch.stack(v, dim=0)

    return metrics


def aggregate_formal_metrics(metrics):
    """Exact count-weighted formal metrics from per-batch metric tensors.

    ``calc_batch_metrics_fast`` returns raw counts alongside rates.  Averaging
    per-batch rates is not exact when the last batch is smaller, so the formal
    table uses total_correct / total_count here.

    Returns:
      final_value_node_acc: numeric-node VALUE token accuracy (non-PAD only)
      final_value_graph_all_acc: per-graph all numeric VALUEs correct
      final_strict_graph_acc: per-graph all non-PAD first-four factors correct
      final_label_full_acc: alias of final_strict_graph_acc for backwards compat
      plus the raw counts used to compute them.
    """
    def total(key):
        value = metrics.get(key)
        if value is None:
            return 0.0
        return float(torch.as_tensor(value).sum().item())

    value_node_num = total('final_value_node_num')
    value_node_correct = total('final_value_node_correct')
    value_graph_num = total('final_value_graph_num')
    value_graph_correct = total('final_value_graph_correct')
    strict_graph_num = total('final_strict_graph_num')
    strict_graph_correct = total('final_strict_graph_correct')

    def ratio(num, den):
        return (num / den) if den > 0 else float('nan')

    return {
        'final_value_node_acc': ratio(value_node_correct, value_node_num),
        'final_value_graph_all_acc': ratio(value_graph_correct, value_graph_num),
        'final_strict_graph_acc': ratio(strict_graph_correct, strict_graph_num),
        'final_label_full_acc': ratio(strict_graph_correct, strict_graph_num),
        'final_value_node_correct': value_node_correct,
        'final_value_node_num': value_node_num,
        'final_value_graph_correct': value_graph_correct,
        'final_value_graph_num': value_graph_num,
        'final_strict_graph_correct': strict_graph_correct,
        'final_strict_graph_num': strict_graph_num,
    }


FORMAL_COUNT_KEYS = frozenset({
    'final_value_node_correct', 'final_value_node_num',
    'final_value_graph_correct', 'final_value_graph_num',
    'final_strict_graph_correct', 'final_strict_graph_num',
})


def aggregate_batch_metrics(metrics):
    """Aggregate across batches for plotting/printing.

    Count keys are summed (their per-batch values are integer counts); all
    other keys are averaged with nanmean.  Formal rates should still be
    computed by ``aggregate_formal_metrics`` (total_correct / total_count), not
    from the sum of per-batch rates.
    """
    aggregated = {}
    for key, value in metrics.items():
        if key in FORMAL_COUNT_KEYS:
            aggregated[key] = value.sum() if torch.is_tensor(value) else value
        else:
            aggregated[key] = torch.nanmean(value, dim=0)
    return aggregated

# region prediction on sample

def print_step(inputs, preds, factored_tokenizer, label=None, stop_at_pad=True, col_width=12, factors_slice=None, return_str=False):

    if return_str:
        str_out = ''
        def print(*args, **kwargs):
            nonlocal str_out
            end = kwargs.get('end', '\n')
            str_out += ' '.join(map(str, args)) + end

    factored_input = factored_tokenizer.decode_tokens(inputs.tolist())
    factored_preds = factored_tokenizer.decode_tokens(preds.tolist())
    if label is not None:
        factored_label = factored_tokenizer.decode_tokens(label.tolist())

    if factors_slice is None:
        factors_slice = slice(0, 6) # don't ignore any factors
    factors = factored_tokenizer.factors[factors_slice]
    # factors_slice = slice(0, 4) # ignore REGRET and MODIFICATION for now

    if label is not None:
        print('INPUT -> PREDICTION | LABEL')
    else:
        print('INPUT -> PREDICTION')
    num_sections = 3 if label is not None else 2
    print('-'*(col_width+3)*len(factors)*num_sections)

    factored_input.insert(0, factors)
    factored_preds.insert(0, factors)
    if label is not None:
        factored_label.insert(0, factors)

    factored_input.insert(1, ['-'*col_width]*len(factors))
    factored_preds.insert(1, ['-'*col_width]*len(factors))
    if label is not None:
        factored_label.insert(1, ['-'*col_width]*len(factors))

    if label is None:
        for x, yhat in zip(factored_input, factored_preds):
            if stop_at_pad and x[0] == factored_tokenizer.PAD:
                break
            x, yhat = list(x)[factors_slice], list(yhat)[factors_slice]
            for idx, (xfactor, yhatfactor) in enumerate(zip(x, yhat)):
                x[idx] = x[idx].center(col_width)
                yhat[idx] = yhat[idx].center(col_width)
                if xfactor != yhatfactor:
                    x[idx] = red(x[idx])
                    yhat[idx] = green(yhat[idx])

            for xfactor in x:
                print(xfactor, end=' | ')
            print('\b| -> |', end='| ')
            for yhatfactor in yhat:
                print(yhatfactor, end=' | ')
            print()
    else:
        for x, yhat, y in zip(factored_input, factored_preds, factored_label):
            if stop_at_pad and x[0] == factored_tokenizer.PAD:
                break
            x, yhat, y = list(x)[factors_slice], list(yhat)[factors_slice], list(y)[factors_slice]

            for idx, (xfactor, yhatfactor, yfactor) in enumerate(zip(x, yhat, y)):
                # center the text within column
                x[idx] = x[idx].center(col_width)
                yhat[idx] = yhat[idx].center(col_width)
                y[idx] = y[idx].center(col_width)

                # color-code labels based in change and correctness
                if xfactor != yhatfactor:
                    x[idx] = violet(x[idx])
                    yhat[idx] = violet(yhat[idx])
                if xfactor != yhatfactor and yhatfactor != yfactor:
                    y[idx] = red(y[idx])
                elif xfactor != yhatfactor and yhatfactor == yfactor:
                    y[idx] = green(y[idx])

            for i, xfactor in enumerate(x):
                end = ' | ' if i < len(x) - 1 else ' |'
                print(xfactor, end=end)
            print('| -> |', end='| ')
            for i, yhatfactor in enumerate(yhat):
                end = ' | ' if i < len(yhat) - 1 else ' |'
                print(yhatfactor, end=end)
            print('| || |', end='| ')
            for yfactor in y:
                print(yfactor, end=' | ')
            print()

    if return_str:
        return str_out


def print_model_steps(litmodel, batch, return_str=False, sample=0, n_steps=None):

    if return_str:
        str_out = ''
        def print(*args, **kwargs):
            nonlocal str_out
            end = kwargs.get('end', '\n')
            str_out += ' '.join(map(str, args)) + end

    # process batch
    batch = batch.to(litmodel.device)
    batch = litmodel.reduce_batch(copy.deepcopy(batch))
    model_config = litmodel.model_config
    factored_tokenizer = litmodel.teacher_model.factored_tokenizer

    batch, batch_label, batch_info = batch[..., :model_config.data_dim], batch[..., model_config.data_dim:model_config.data_dim + model_config.label_dim], batch[..., model_config.data_dim + model_config.label_dim:]
    # note: batch_label is final label

    # get teacher batch
    depths = batch_info[..., 0]
    if n_steps is None:
        n_steps = depths.max().item() + 1

    batch_ext, value_avail_pos, value_to_compute_pos = litmodel.get_teacher_batch_with_depth(batch, batch_label, depths)

    # inputs = batch[0]
    inputs_stack = [batch[sample]]
    batch_ext = batch_ext[:, sample]
    # labels = batch_ext[0, 1:]

    factor_slice = slice(0, 4) # ignore REGRET and MODIFICATION for now
    print_width = (12 + 3) * 4 * 3 # (col_width+3)*len(factors)*num_sections

    fully_corrects = []
    outputs = None
    for step in range(0, n_steps):
        inputs = inputs_stack[-1]

        # logitss = litmodel.predict_step(inputs.unsqueeze(0))
        # one_step_prediction
        outputs, logitss = one_step_prediction(litmodel, batch, step, outputs=outputs, step_stack=inputs_stack)

        preds = torch.stack([logits.argmax(-1) for logits in logitss], dim=-1)[0] # argmax prediction for each factoro
        # preds = litmodel.SAE_forward(inputs.unsqueeze(0))

        if step + 1 <= len(batch_ext) - 1:
            step_label = batch_ext[step + 1]
        else:
            step_label = batch_ext[-1]

        fully_correct = (preds == step_label)[:, factor_slice].all().detach().cpu().numpy()
        fully_corrects.append(fully_correct)

        inputs_stack.append(preds)

        print('='*print_width)
        print(f'STEP: {step + 1} (Fully Correct: {fully_correct} )')
        if return_str:
            out_str = print_step(inputs=inputs, preds=preds, factored_tokenizer=factored_tokenizer, label=step_label, factors_slice=factor_slice, return_str=True)
            print(out_str)
        else:
            print_step(inputs=inputs, preds=preds, factored_tokenizer=factored_tokenizer, label=step_label, factors_slice=factor_slice, return_str=False)
        print('='*print_width)
        print()

    # print the final label
    print('='*print_width)
    print('IS FULLY CORRECT AT ALL STEPS:', all(fully_corrects))

    matches_final_label = (inputs_stack[-1] == batch_label[sample])[:, factor_slice].all().detach().cpu().numpy()
    print('MATCHES FINAL LABEL:', matches_final_label)

    if return_str:
        return str_out

# endregion

def print_sample(sample, factored_tokenizer, print_=True):

    # decode factored tokens in sample
    factored_tokens = factored_tokenizer.decode_tokens(sample.tolist())

    # translate to readable string
    sample_string = ' '.join(factored_tokenizer.de_factor_tokens(factored_tokens))
    sample_string = sample_string.replace(f' {factored_tokenizer.EQ_SEP} ', '\n') # separate equations
    sample_string = sample_string.replace(factored_tokenizer.PAD, '.') # replace padding with a dot

    # replace op names with more readable symbols
    op_dict = {'ADD': '+', 'SUB': '-', 'MUL': '*', 'DIV': '/'}
    for op_old, op_new in op_dict.items():
        sample_string = sample_string.replace(op_old, op_new)

    if print_:
        print(sample_string)
    else:
        return sample_string

# helper functions to color text (used to highlight changes in predictions)
def blue(text):
    return f'\033[34m{text}\033[0m'

def green(text):
    return f'\033[32m{text}\033[0m'

def red(text):
    return f'\033[31m{text}\033[0m'

def yellow(text):
    return f'\033[33m{text}\033[0m'

def cyan(text):
    return f'\033[36m{text}\033[0m'

def magenta(text):
    return f'\033[35m{text}\033[0m'

def purple(text):
    return f'\033[95m{text}\033[0m'

def violet(text):
    return f'\033[35m{text}\033[0m'

def orange(text):
    return f'\033[33m{text}\033[0m'


# endregion


def calc_batch_metrics_fast(litmodel, batch, factored_tokenizer, n_steps=None, factors_slice=None, verbose=False):

    litmodel.eval()
    if SKIP_ATTN:
        buffer, hook_handles = {}, []
    else:
        buffer, hook_handles = attach_hooks_to_model(litmodel)

    model_config = litmodel.model_config

    if factors_slice is None:
        factors_slice = slice(0, 6) # consider all factors

    with torch.no_grad():
        # process batch
        batch = litmodel.reduce_batch(copy.deepcopy(batch))
        batch, batch_label, batch_info = batch[..., :model_config.data_dim], batch[..., model_config.data_dim:model_config.data_dim + model_config.label_dim], batch[..., model_config.data_dim + model_config.label_dim:]
        # note: batch_label is final label

        # get teacher batch
        depths = batch_info[..., 0]
        if n_steps is None:
            n_steps = depths.max().item() + 1
            print('inferred max steps:', n_steps)

        batch_ext, value_avail_pos, value_to_compute_pos = litmodel.get_teacher_batch_with_depth(batch, batch_label, depths)
        # batch_ext.shape: [n_steps, batch_size, seq_len, n_factors]

        # infer which tokens are padding, and ignore them in computing accuracy
        pad_encoding = torch.tensor(factored_tokenizer.encode_string(factored_tokenizer.PAD)[0], device=batch.device)
        is_pad = torch.eq(batch, pad_encoding).all(dim=-1)  # [B, T]

        # get first pad for each sample
        orig_seq_len = batch.size(1)
        seq_lens = (~is_pad).sum(dim=1)
        max_seq_len = seq_lens.max()
        if verbose:
            print('max non-pad length:', max_seq_len)
            print('orig seq len:', orig_seq_len)

        # Truncate to the longest non-pad sequence in this batch.
        batch = batch[:, :max_seq_len]
        batch_ext = batch_ext[:, :, :max_seq_len]
        batch_label = batch_label[:, :max_seq_len]
        is_pad = is_pad[:, :max_seq_len]

        valid = ~is_pad                               # [B, T]
        valid_f = valid.unsqueeze(-1)                 # [B, T, 1]
        value_idx = factored_tokenizer.factors.index('VALUE')
        # Numeric node: non-PAD position whose FINAL label is a value token
        # (0..mod_val-1).  Excludes N/A, EMPTY, PAD and structural tokens.
        numeric = valid & (batch_label[..., value_idx] < factored_tokenizer.mod_val)

        step_stack = [batch]
        metrics = dict(
            factor_acc=[], position_acc=[], fully_correct_acc=[], recorrection=[],
            step_final_label_full_acc=[], step_final_label_factor_acc=[],
            step_final_label_persymbol_avg_acc=[], step_last_var_factor_acc=[],
            step_last_var_fully_correct=[])

        last_mistakes = None
        outputs = None
        last_non_pad_by_sample = valid.sum(dim=1).clamp_min(1) - 1
        batch_indices = torch.arange(batch.size(0), device=batch.device)

        for step in range(0, n_steps):
            if verbose:
                print('step', step)

            # one_step_prediction
            outputs, logitss = one_step_prediction(litmodel, batch, step, outputs=outputs, step_stack=step_stack)

            preds = torch.stack([logits.argmax(-1) for logits in logitss], dim=-1)  # [B, T, F]
            step_stack.append(preds)

            if step + 1 <= len(batch_ext) - 1:
                step_label = batch_ext[step + 1]
            else:
                step_label = batch_ext[-1]

            # Graph-level strict: PAD is ignored (OR pad).  Keep this for
            # strict whole-graph metrics only; do NOT use it for factor stats.
            raw_correct = (preds == step_label)[..., factors_slice]
            strict_correct = torch.logical_or(raw_correct, is_pad.unsqueeze(-1))

            # percent recorrection
            if last_mistakes is None or last_mistakes.sum() == 0:
                recorrection = torch.nan
            else:
                recorrection = torch.masked_select(strict_correct, last_mistakes).float().mean().item()
            metrics['recorrection'].append(recorrection)

            # Per-factor accuracy over non-PAD tokens only.
            factor_acc = (raw_correct & valid_f).sum(dim=(0, 1)).float() / valid.sum().clamp_min(1)
            metrics['factor_acc'].append(factor_acc)

            # Position-wise accuracy over non-PAD samples only.  Positions with
            # no valid sample in the batch stay NaN and are ignored by nanmean.
            pos_correct = (raw_correct.all(dim=-1) & valid).sum(dim=0).float()
            pos_total = valid.sum(dim=0).float()
            position_acc = torch.where(
                pos_total > 0,
                pos_correct / pos_total.clamp_min(1),
                torch.full_like(pos_correct, float('nan')))
            position_acc = torch.cat([
                position_acc,
                torch.full((orig_seq_len - position_acc.size(0),), fill_value=torch.nan,
                           device=batch.device)])
            metrics['position_acc'].append(position_acc)

            # Full-sequence correctness (strict graph-level, PAD ignored).
            fully_correct = strict_correct.all(axis=(1, 2))
            metrics['fully_correct_acc'].append(fully_correct.float().mean())

            # Per-symbol accuracy over non-PAD tokens.
            metrics['persymbol_avg_acc'] = (
                (raw_correct.all(dim=-1) & valid).sum().float()
                / valid.sum().clamp_min(1))

            # Check whether final label is correct (strict graph-level).
            final_raw = (preds == batch_label)[..., factors_slice]
            final_strict = torch.logical_or(final_raw, is_pad.unsqueeze(-1))
            metrics['step_final_label_full_acc'].append(
                final_strict.all(axis=(1, 2)).float().mean())
            metrics['step_final_label_factor_acc'].append(
                (final_raw & valid_f).sum(dim=(0, 1)).float()
                / valid.sum().clamp_min(1))
            metrics['step_final_label_persymbol_avg_acc'].append(
                (final_raw.all(dim=-1) & valid).sum().float()
                / valid.sum().clamp_min(1))

            # Last non-PAD token per sample (not the physical last column).
            last_var_is_correct = (
                preds[batch_indices, last_non_pad_by_sample, :]
                == batch_label[batch_indices, last_non_pad_by_sample, :]
            )[..., factors_slice]
            metrics['step_last_var_factor_acc'].append(
                last_var_is_correct.float().mean(dim=0))
            metrics['step_last_var_fully_correct'].append(
                last_var_is_correct.all(dim=-1).float().mean())

            # calculate attention metrics
            if not SKIP_ATTN:
                attn_metrics = compute_attention_metrics(buffer)
                for k, v in attn_metrics.items():
                    if k not in metrics:
                        metrics[k] = []
                    metrics[k].append(v)

            # set boolean mask of mistakes to check re-correction at next iteration
            last_mistakes = torch.logical_not(strict_correct)

            if verbose:
                print('factor_acc', factor_acc)
                print('fully_correct_acc', fully_correct.float().mean())

    metrics['recorrection'] = torch.tensor(metrics['recorrection'])

    for k, v in metrics.items():
        if isinstance(v, list):
            metrics[k] = torch.stack(v)

    final_preds = step_stack[-1]

    # Strict whole-graph metric: all non-PAD positions, first factors_slice.
    final_raw = (final_preds == batch_label)[..., factors_slice]
    final_strict = torch.logical_or(final_raw, is_pad.unsqueeze(-1))
    final_strict_graph = final_strict.all(axis=(1, 2))
    metrics['final_label_full_acc'] = final_strict_graph.float().mean().item()
    metrics['final_label_factor_acc'] = (
        (final_raw & valid_f).sum(dim=(0, 1)).float()
        / valid.sum().clamp_min(1))

    # Formal node VALUE accuracy: numeric nodes only, non-PAD only.
    value_correct = (final_preds[..., value_idx] == batch_label[..., value_idx])
    value_correct_numeric = value_correct & numeric
    metrics['final_value_node_acc'] = (
        value_correct_numeric.sum().float() / numeric.sum().clamp_min(1))
    # Formal whole-graph VALUE all-correct: every numeric node correct;
    # PAD/structural positions are ignored.
    value_graph_ok = (value_correct | ~numeric).all(dim=1)
    metrics['final_value_graph_all_acc'] = value_graph_ok.float().mean()

    # Raw counts for exact aggregation across batches (not batch-mean of means).
    metrics['final_value_node_correct'] = value_correct_numeric.sum().detach()
    metrics['final_value_node_num'] = numeric.sum().detach()
    metrics['final_value_graph_correct'] = value_graph_ok.sum().detach()
    metrics['final_value_graph_num'] = torch.tensor(batch.size(0), device=batch.device)
    metrics['final_strict_graph_correct'] = final_strict_graph.sum().detach()
    metrics['final_strict_graph_num'] = torch.tensor(batch.size(0), device=batch.device)

    # Last non-PAD token, not the physical last column (which can be PAD).
    last_var_is_correct = (
        final_preds[batch_indices, last_non_pad_by_sample, :]
        == batch_label[batch_indices, last_non_pad_by_sample, :]
    )[..., factors_slice]
    metrics['last_var_factor_acc'] = last_var_is_correct.float().mean(dim=0)
    metrics['last_var_fully_correct'] = last_var_is_correct.all(dim=-1).float().mean()

    if verbose:
        print('final_label_full_acc', metrics['final_label_full_acc'])
        print('final_label_factor_acc', metrics['final_label_factor_acc'])
        print('final_value_node_acc', metrics['final_value_node_acc'])

    return metrics
