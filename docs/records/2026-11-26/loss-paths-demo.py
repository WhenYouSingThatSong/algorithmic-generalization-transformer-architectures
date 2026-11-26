"""对照 A/B 与 C 两条损失路径真正喂给模型的东西。

用法（仓库根目录，需要 tmp/runs 下的本地权重）：
    python docs/records/2026-11-26/loss-paths-demo.py

三节：
  1. 从 Tr32Test128-ADD 取一张真图，打印题面与 32 个结果位置；
  2. 复现 model.py:get_teacher_batch_with_depth，展示 A/B 每一轮输入里的 VALUE 是真值（教师强制）；
  3. 载入原始 C 的 e143 权重，真实跑 compute_batch_loss_nointerm 的循环，
     展示 depth_batch 被自己的隐藏输出覆盖，并逐轮读出 VALUE。

只读；不写任何文件。
"""
import sys
import os
import yaml
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, 'experiments'))
from tokenizers import FactoredVocabTokenizer          # noqa: E402
from model import RecurrentTransformer                 # noqa: E402

DATA = os.path.join(REPO, 'experiments/data/Tr32Test128-ADD/train_data.pt')
CFG_C = os.path.join(REPO, 'experiments/configs/C0-M4-Nodes32-ADD-L4H16D384-DeBERTa')
CKPT_C = os.path.join(
    REPO, 'tmp/runs/oldC0-resume-150ep/checkpoints',
    'Nodes32-ADD - L4H16D384_DeBERTa-seed-42 - 2026-11-14-02:35:10/epoch=143-val_loss=0.0000.ckpt')
SAMPLE = 0


class EasyDict(dict):
    """Minimal stand-in for the config wrapper train.py uses."""
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)

    def __setattr__(self, k, v):
        self[k] = v


def _ed(o):
    if isinstance(o, dict):
        return EasyDict({k: _ed(v) for k, v in o.items()})
    if isinstance(o, list):
        return [_ed(v) for v in o]
    return o


def main():
    tok = FactoredVocabTokenizer(n_vars=128, ops=['ADD'], mod_val=23, max_fan_in=3)
    F = {n: i for i, n in enumerate(tok.factors)}
    EMPTY = tok.value_tok2idx['EMPTY']
    say = lambda L, i: L[int(i)] if 0 <= int(i) < len(L) else f'<{int(i)}>'

    d = torch.load(DATA, map_location='cpu', weights_only=True)
    label, info = d['tokenized_dataset'][SAMPLE], d['dataset_info'][SAMPLE]
    batch = label.clone()
    batch[batch[:, F['SYNTAX']] == tok.syntax_tok2idx['VARIABLE'], F['VALUE']] = EMPTY
    depths = info[:, 0]
    nonpad = label[:, F['SYNTAX']] != tok.syntax_tok2idx['<PAD>']
    nodes = sorted([p for p in range(len(depths)) if nonpad[p] and int(depths[p]) >= 0],
                   key=lambda p: (int(depths[p]), p))
    K = int(depths.max()) + 1

    print(f"真实图：样本 #{SAMPLE}，{len(nodes)} 个结果位置，depth 0..{K-1}")
    print(f"  depth=-1 是题面里的字面量；depth>=0 是结果位置（输入为空、标签是真值）")
    print()

    # ---- 1. A/B：get_teacher_batch_with_depth（照抄 model.py:468-480）----
    b, l = batch.unsqueeze(0), label.unsqueeze(0)
    de = depths.unsqueeze(0).unsqueeze(0).repeat(K, 1, 1)
    dt = torch.arange(K).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len(depths))
    ext = b.unsqueeze(0).repeat(K, 1, 1, 1)
    ext[..., F['VALUE']][torch.logical_and(de.le(dt), de.ne(-1))] = \
        l.unsqueeze(0).repeat(K, 1, 1, 1)[..., F['VALUE']][torch.logical_and(de.le(dt), de.ne(-1))]
    bx = torch.cat([b.unsqueeze(0), ext], dim=0)

    print("=== A/B（compute_batch_loss_fast）第 k 轮输入里的 VALUE ===")
    print("行=结果位置，列=第 k 轮输入 batch_ext[k+1]；'空'=EMPTY")
    print(f"{'node':>7} {'dep':>4} {'真值':>4} |" + "".join(f"{'k'+str(k):>5}" for k in range(K)))
    for p in nodes:
        cells = "".join("   空" if int(bx[k + 1, 0, p, F['VALUE']]) == EMPTY
                        else f"{int(bx[k + 1, 0, p, F['VALUE']]):>5}" for k in range(K))
        print(f"{say(tok.variable_toks, label[p, F['VARIABLE']]):>7} {int(depths[p]):>4} "
              f"{say(tok.value_toks, label[p, F['VALUE']]):>4} |{cells}")
    print("⇒ 第 k 轮输入中 depth<=k 的位置已经是真值：教师强制，且所有 depth 拼成一个 batch。")
    print()

    # ---- 2. C：compute_batch_loss_nointerm 的循环 ----
    mc = _ed(yaml.load(open(os.path.join(CFG_C, 'model_config.yaml')), Loader=yaml.FullLoader))
    mc.vocab_sizes, mc.factors = tok.vocab_sizes, tok.factors
    mc.data_dim = mc.label_dim = mc.info_dim = 6
    net = RecurrentTransformer(mc).eval()
    if not os.path.exists(CKPT_C):
        print(f"找不到 C 权重，跳过第 2 节：{CKPT_C}")
        return
    sd = torch.load(CKPT_C, map_location='cpu', weights_only=False)['state_dict']
    sd = {(k[len('model.'):] if k.startswith('model.') else k).replace('_orig_mod.', ''): v
          for k, v in sd.items()}
    miss, unexp = net.load_state_dict(sd, strict=False)
    print(f"载入原始 C e143：缺 {len(miss)} 键，多 {len(unexp)} 键")

    with torch.no_grad():
        db = net.token_embedder(b)                      # 唯一一次使用离散输入
        print(f"  循环前 depth_batch={tuple(db.shape)}  ← token_embedder(batch)")
        rows = []
        for loop in range(K):
            db, preds = net(db, True)                   # 喂自己的隐藏态，不重新嵌入
            rows.append(preds[F['VALUE']].argmax(-1)[0])
            ok = sum(int(rows[-1][p]) == int(label[p, F['VALUE']]) for p in nodes)
            print(f"  loop {loop}: 输出 hidden {tuple(db.shape)}，{len(nodes)} 个位置对了 {ok} 个")

    print()
    print("=== C（compute_batch_loss_nointerm）每轮自己读出的 VALUE ===")
    print(f"{'node':>7} {'dep':>4} {'真值':>4} |" + "".join(f"{'L'+str(k):>5}" for k in range(K)))
    for p in nodes:
        cells = "".join("   空" if int(rows[k][p]) == EMPTY else f"{int(rows[k][p]):>5}"
                        for k in range(K))
        print(f"{say(tok.variable_toks, label[p, F['VARIABLE']]):>7} {int(depths[p]):>4} "
              f"{say(tok.value_toks, label[p, F['VALUE']]):>4} |{cells}")
    print("⇒ 输入是模型自己的上一轮状态；未算到的位置它自己读出 EMPTY。没有任何真值被喂回。")


if __name__ == '__main__':
    main()
