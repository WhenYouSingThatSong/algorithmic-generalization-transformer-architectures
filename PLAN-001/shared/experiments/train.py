import os
# [switch] torch 2.6 的 weights_only 默认会阻断 Lightning resume 的 optimizer/hparam 反序列化
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import argparse
import yaml
import pickle
from datetime import datetime

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch import Trainer

from model import LitRecurrentTransformerModel, get_experiment_name, DAGTeacherModel
import os
from importlib.util import spec_from_file_location, module_from_spec

current_dir = os.path.dirname(os.path.realpath(__file__))
# # Load the local factored_tokenizers.py from the current directory to avoid conflict with the standard package.
# spec = spec_from_file_location("local_tokenizers", os.path.join(current_dir, "factored_tokenizers.py"))
# local_tokenizers = module_from_spec(spec)
# spec.loader.exec_module(local_tokenizers)
from tokenizers import FactoredVocabTokenizer
from Simtransformer.simtransformer.utils import EasyDict

import copy
current_dir = os.path.dirname(os.path.realpath(__file__))

class CustomDataset(Dataset):
    def __init__(self, *args):
        self.data = args

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return tuple(data[idx] for data in self.data)

# Load the data module
class DAGDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_path,
                 batch_size,
                 factored_tokenizer,
                 num_workers=4,
                 tr_val_test_split=(0.9, 0.05, 0.05),
                 val2_path=None,
                 val2_batch_size=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tr_val_test_split = tr_val_test_split
        self.factored_tokenizer = factored_tokenizer
        self.val2_path = val2_path
        self.val2_batch_size = val2_batch_size or batch_size

    def setup(self, stage=None):
        # Load your dataset here
        if isinstance(self.data_path, str):
            dataset_dict = torch.load(self.data_path, weights_only=True)
            dataset = dataset_dict['tokenized_dataset']
            dataset_info = dataset_dict['dataset_info']

            tr_split, val_split, test_split = self.tr_val_test_split

            num_data_train = int(len(dataset) * tr_split)
            num_data_val = int(len(dataset) * val_split)
            num_data_test = len(dataset) - num_data_train - num_data_val

            self.data_train_label = dataset[:num_data_train]
            self.data_val_label = dataset[num_data_train:num_data_train+num_data_val]
            self.data_test_label = dataset[-num_data_test:]

            self.data_train_info = dataset_info[:num_data_train]
            self.data_val_info = dataset_info[num_data_train:num_data_train+num_data_val]
            self.data_test_info = dataset_info[-num_data_test:]

        elif isinstance(self.data_path, list) or isinstance(self.data_path, tuple):
            if len(self.data_path) == 3:
                data_train_path, data_val_path, data_test_path = self.data_path
            elif len(self.data_path) == 2:
                data_train_path, data_val_path = self.data_path
                data_test_path = None

            # load the data from the paths
            self.data_train = torch.load(data_train_path, weights_only=True)
            self.data_val = torch.load(data_val_path, weights_only=True)
            self.data_test = self.data_val if data_test_path is None else torch.load(data_test_path, weights_only=True)

            self.data_train_label, self.data_train_info = self.data_train['tokenized_dataset'], self.data_train['dataset_info']
            self.data_val_label, self.data_val_info = self.data_val['tokenized_dataset'], self.data_val['dataset_info']
            self.data_test_label, self.data_test_info = self.data_test['tokenized_dataset'], self.data_test['dataset_info']

            if 'spec' in self.data_train:
                self.train_spec = self.data_train['spec']
            if 'spec' in self.data_val:
                self.val_spec = self.data_val['spec']
            if 'spec' in self.data_test:
                self.test_spec = self.data_test['spec']

        # set the variable tokens to empty
        self.data_train = copy.deepcopy(self.data_train_label)
        value_idx = self.data_train[:, :, self.factored_tokenizer.factors.index('SYNTAX')].eq(self.factored_tokenizer.syntax_tok2idx['VARIABLE'])
        data_value = self.data_train[:, :, self.factored_tokenizer.factors.index('VALUE')]
        data_value[value_idx] = self.factored_tokenizer.value_tok2idx['EMPTY']

        self.data_val = copy.deepcopy(self.data_val_label)
        value_idx_val = self.data_val[:, :, self.factored_tokenizer.factors.index('SYNTAX')].eq(self.factored_tokenizer.syntax_tok2idx['VARIABLE'])
        data_value_val = self.data_val[:, :, self.factored_tokenizer.factors.index('VALUE')]
        data_value_val[value_idx_val] = self.factored_tokenizer.value_tok2idx['EMPTY']

        self.data_test = copy.deepcopy(self.data_test_label)
        value_idx_test = self.data_test[:, :, self.factored_tokenizer.factors.index('SYNTAX')].eq(self.factored_tokenizer.syntax_tok2idx['VARIABLE'])
        data_value_test = self.data_test[:, :, self.factored_tokenizer.factors.index('VALUE')]
        data_value_test[value_idx_test] = self.factored_tokenizer.value_tok2idx['EMPTY']

        # optional second validation stream (e.g. N128 generalization)
        self.data_val2 = None
        if self.val2_path is not None:
            data_val2 = torch.load(self.val2_path, weights_only=True)
            self.data_val2_label = data_val2['tokenized_dataset']
            self.data_val2_info = data_val2['dataset_info']
            self.data_val2_spec = data_val2.get('spec')
            if self.data_val2_spec is None:
                self.data_val2_spec = self._make_spec(self.data_val2_info, self.data_val2_label)
            self.data_val2 = copy.deepcopy(self.data_val2_label)
            value_idx_val2 = self.data_val2[:, :, self.factored_tokenizer.factors.index('SYNTAX')].eq(self.factored_tokenizer.syntax_tok2idx['VARIABLE'])
            data_value_val2 = self.data_val2[:, :, self.factored_tokenizer.factors.index('VALUE')]
            data_value_val2[value_idx_val2] = self.factored_tokenizer.value_tok2idx['EMPTY']

        # Old evaluation files (e.g. Tr32Test128-ADD/val_data.pt) have no
        # `spec` key. Compute the same spec from dataset_info so eval scripts
        # can always use val_spec['max_depth'].
        if not hasattr(self, 'train_spec'):
            self.train_spec = self._make_spec(self.data_train_info, self.data_train_label)
        if not hasattr(self, 'val_spec'):
            self.val_spec = self._make_spec(self.data_val_info, self.data_val_label)
        if not hasattr(self, 'test_spec'):
            self.test_spec = self._make_spec(self.data_test_info, self.data_test_label)

        self.data_dim = self.data_train.shape[-1]
        self.label_dim = self.data_train_label.shape[-1]
        self.info_dim = self.data_train_info.shape[-1]

    def _make_spec(self, info, labels):
        depths = info[..., 0]
        per = depths.max(dim=1).values
        pad = self.factored_tokenizer.syntax_tok2idx[self.factored_tokenizer.PAD]
        nonpad = (labels[..., 0] != pad).sum(dim=1)
        return dict(
            min_depth=int(per.min()), mean_depth=float(per.float().mean()),
            max_depth=int(per.max()), min_non_pad_len=int(nonpad.min()),
            mean_non_pad_len=float(nonpad.float().mean()),
            max_non_pad_len=int(nonpad.max()), orig_seq_len=int(labels.shape[1]))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.concat([self.data_train, self.data_train_label, self.data_train_info], dim=-1),
            batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
            pin_memory=True)

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            torch.concat([self.data_val, self.data_val_label, self.data_val_info], dim=-1),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        if self.data_val2 is None:
            return val_loader
        val2_loader = torch.utils.data.DataLoader(
            torch.concat([self.data_val2, self.data_val2_label, self.data_val2_info], dim=-1),
            batch_size=self.val2_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return [val_loader, val2_loader]

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            torch.concat([self.data_test, self.data_test_label, self.data_test_info], dim=-1),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )



def load_litmodel(args, seed=0):
    # load model, train, and data config
    with open(os.path.join(args.config_dir, 'model_config.yaml')) as f:
        model_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    with open(os.path.join(args.config_dir, 'train_config.yaml')) as f:
        train_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    with open(os.path.join(args.config_dir, 'data_config.yaml')) as f:
        data_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    train_config.seed = seed
    if hasattr(args, 'nointerm') and args.nointerm:
        train_config.nointerm = True

    # load tokenizer
    # with open(os.path.join(current_dir, data_config.data_dir, data_config.tokenizer_file_name), 'rb') as f:
        # factored_tokenizer = pickle.load(f)
    # tokenizer_n_vars is an explicit override; otherwise keep the original
    # max(train.N, val.N) rule. The override is needed when the validation
    # stream is N128 but the in-range val file is N32, and when data was
    # generated with the full 128-name pool.
    n_vars = getattr(data_config, 'tokenizer_n_vars', None)
    if n_vars is None:
        n_vars = max(data_config.dag_config.num_nodes, data_config.val_dag_config.num_nodes)
    factored_tokenizer = FactoredVocabTokenizer(n_vars=n_vars, ops=data_config.dag_config.func_vocab, mod_val=data_config.dag_config.mod_val, max_fan_in=data_config.dag_config.max_fan_in_deg)

    model_config.vocab_sizes = factored_tokenizer.vocab_sizes
    model_config.factors = factored_tokenizer.factors

    val2_path = None
    val2_file_name = getattr(data_config, 'val2_data_file_name', None)
    if val2_file_name:
        val2_dir = getattr(data_config, 'val2_data_dir', data_config.data_dir)
        val2_path = os.path.join(current_dir, val2_dir, val2_file_name)

    # create the data module
    data_module = DAGDataModule(
        data_path=[os.path.join(current_dir, data_config.data_dir, data_config.dag_config.data_file_name), os.path.join(current_dir, data_config.data_dir, data_config.val_dag_config.data_file_name)],
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers if not args.debug else 0,
        factored_tokenizer=factored_tokenizer,
        val2_path=val2_path,
        val2_batch_size=getattr(train_config, 'val2_batch_size', None))

    # add the dimension of data, label, and info to model config
    data_module.setup()
    model_config.data_dim = data_module.data_dim
    model_config.label_dim = data_module.label_dim
    model_config.info_dim = data_module.info_dim


    # set matmul precision
    if getattr(train_config, 'matmul_precision', None) is not None:
        torch.set_float32_matmul_precision(train_config.matmul_precision)
        print(f'Matmul precision set to {train_config.matmul_precision}.')

    # create the model and Lightning module
    litmodel = LitRecurrentTransformerModel(model_config, data_config, train_config, DAGTeacherModel(model_config, factored_tokenizer))

    experiment_config = EasyDict(dict(train_config=train_config, model_config=model_config))

    return litmodel, data_module, factored_tokenizer, model_config, train_config, data_config

def str_or_none(value):
    if value == 'None':
        return None
    else:
        return value

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--nointerm', action='store_true', help='Run without intermediate DecEnc.')
    parser.add_argument('--config_dir', type=str, default=os.path.join(current_dir, 'configs/Nodes32-ADD-L2H16D256_DeBERTa'))
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (no logging, no checkpoints).')
    parser.add_argument('--use_teacher_pred_for_next_loop', action='store_true', help='Use teacher prediction for next loop.')
    parser.add_argument('--use_fast_batch_forward', action='store_true', help='Use fast batch for forward.')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--checkpoint_dir', type=str_or_none, help='Dir to the checkpoint file.', default=None)
    parser.add_argument('--seed', default=None)
    parser.add_argument('--max_epochs', type=int, default=None,
                        help='override train_config.max_epochs (for smoke tests)')
    parser.add_argument('--limit_train_batches', type=int, default=None,
                        help='for smoke tests; passed to Trainer')
    parser.add_argument('--limit_val_batches', type=int, default=None,
                        help='for smoke tests; passed to Trainer')
    parser.add_argument('--save_top_k', type=int, default=None,
                        help='override ModelCheckpoint save_top_k (default 3)')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to a Lightning checkpoint to resume training from.')

    args = parser.parse_args()

    # set the seed for reproducibility
    if args.seed is not None:
        seed = int(args.seed)
    else:
        seed = np.random.randint(0, 2**32) # randomly sample a seed
    pl.seed_everything(seed) # sets the seed for all random number generators

    # load model, train, and data config
    with open(os.path.join(args.config_dir, 'model_config.yaml')) as f:
        model_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    with open(os.path.join(args.config_dir, 'train_config.yaml')) as f:
        train_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    with open(os.path.join(args.config_dir, 'data_config.yaml')) as f:
        data_config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    train_config.seed = seed

    # convience functions to set/overwrite the config
    if hasattr(args, 'nointerm') and args.nointerm:
        train_config.nointerm = True

    if hasattr(args, 'use_teacher_pred_for_next_loop') and args.use_teacher_pred_for_next_loop:
        train_config.use_teacher_pred_for_next_loop = True

    if hasattr(args, 'use_fast_batch_forward') and args.use_fast_batch_forward:
        train_config.use_fast_batch_forward = True

    if args.wandb_project is not None:
        train_config.wandb_config.wandb_project = args.wandb_project

    # load tokenizer
    # with open(os.path.join(current_dir, data_config.data_dir, data_config.tokenizer_file_name), 'rb') as f:
        # factored_tokenizer = pickle.load(f)
    # tokenizer_n_vars is an explicit override; otherwise keep the original
    # max(train.N, val.N) rule. The override is needed when the validation
    # stream is N128 but the in-range val file is N32, and when data was
    # generated with the full 128-name pool.
    n_vars = getattr(data_config, 'tokenizer_n_vars', None)
    if n_vars is None:
        n_vars = max(data_config.dag_config.num_nodes, data_config.val_dag_config.num_nodes)
    factored_tokenizer = FactoredVocabTokenizer(n_vars=n_vars, ops=data_config.dag_config.func_vocab, mod_val=data_config.dag_config.mod_val, max_fan_in=data_config.dag_config.max_fan_in_deg)

    model_config.vocab_sizes = factored_tokenizer.vocab_sizes
    model_config.factors = factored_tokenizer.factors

    val2_path = None
    val2_file_name = getattr(data_config, 'val2_data_file_name', None)
    if val2_file_name:
        val2_dir = getattr(data_config, 'val2_data_dir', data_config.data_dir)
        val2_path = os.path.join(current_dir, val2_dir, val2_file_name)

    # create the data module
    data_module = DAGDataModule(
        data_path=[os.path.join(current_dir, data_config.data_dir, data_config.dag_config.data_file_name), os.path.join(current_dir, data_config.data_dir, data_config.val_dag_config.data_file_name)],
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers if not args.debug else 0,
        factored_tokenizer=factored_tokenizer,
        val2_path=val2_path,
        val2_batch_size=getattr(train_config, 'val2_batch_size', None))

    # add the dimension of data, label, and info to model config
    data_module.setup()
    model_config.data_dim = data_module.data_dim
    model_config.label_dim = data_module.label_dim
    model_config.info_dim = data_module.info_dim


    # set matmul precision
    if getattr(train_config, 'matmul_precision', None) is not None:
        torch.set_float32_matmul_precision(train_config.matmul_precision)
        print(f'Matmul precision set to {train_config.matmul_precision}.')

    # create the model and Lightning module
    litmodel = LitRecurrentTransformerModel(model_config, data_config, train_config, DAGTeacherModel(model_config, factored_tokenizer))

# if __name__ == '__main__':
    # Group: Data Config - Model Config ... Run name: Seed + Date-Time
    group_name, run_name = get_experiment_name(model_config, data_config, train_config)

    train_config.experiment_run_name = run_name
    train_config.experiment_group = group_name

    experiment_config = EasyDict(dict(train_config=train_config, model_config=model_config))
    print(experiment_config)

    wandb_project = train_config.wandb_config.wandb_project
    if hasattr(args, 'nointerm') and args.nointerm:
        logger = pl.loggers.WandbLogger(
        entity=train_config.wandb_config.wandb_entity, project=wandb_project,
        name=train_config.experiment_run_name, group=train_config.experiment_group,
        config=experiment_config) if not args.debug else None
    else:
        logger = pl.loggers.WandbLogger(
        entity=train_config.wandb_config.wandb_entity, project=wandb_project,
        name=train_config.experiment_run_name, group=train_config.experiment_group,
        config=experiment_config) if not args.debug else None

    # watch gradients and model params
    if logger is not None and getattr(train_config, 'wandb_watch', True):
        logger.watch(litmodel.model, log='all', log_graph=True)

    # callbacks: checkpoint and lr monitor
    checkpoint_dir = os.path.join(current_dir, f'checkpoints', f'{train_config.experiment_group}-{train_config.experiment_run_name}') if args.checkpoint_dir is None else os.path.join(args.checkpoint_dir, f'{train_config.experiment_group}-{train_config.experiment_run_name}')

    # Keep more than the historical top-3 when explicitly requested, so a
    # continuation run can preserve intermediate candidates for later formal
    # metric-based checkpoint selection.
    save_top_k = args.save_top_k if args.save_top_k is not None else 3

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch}-{val_loss:.4f}',
        monitor='total_loss/val', # this depends on logging in the LightningModule
        mode='min',
        save_top_k=save_top_k,
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor, checkpoint_callback]

    # optional second best-checkpoint stream for the N128 val set.
    # Main checkpoint remains total_loss/val (in-range); this one only saves
    # extra weights for the supplementary N128-selection analysis.
    checkpoint_callback_n128 = None
    if val2_path is not None and getattr(train_config, 'save_n128_best', True):
        checkpoint_callback_n128 = ModelCheckpoint(
            dirpath=checkpoint_dir + '-n128',
            filename='{epoch}-{val_loss:.4f}',
            monitor='total_loss/val_n128',
            mode='min',
            save_top_k=save_top_k,
            save_last=True
        )
        callbacks.append(checkpoint_callback_n128)

    if args.debug:
        callbacks = []
        logger = None

    # [fix-env] torch>=2.6 的 compiled backward 默认启用 donated buffers，与本模型
    # 逐因子 manual_backward(retain_graph=True) 不兼容（A线seed42首训39s崩溃于此）。
    # 禁用 donated_buffer：仅放弃显存复用优化，不改变训练语义。
    torch._functorch.config.donated_buffer = False

    # compile the model
    if getattr(train_config, 'compile', False):
        # litmodel = torch.compile(litmodel)
        litmodel.model = torch.compile(litmodel.model)
        print('Model compiled.')

    trainer_kwargs = dict(
        max_epochs=args.max_epochs if args.max_epochs is not None else train_config.max_epochs,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=getattr(train_config, 'val_check_interval', 1.0),
        precision=getattr(train_config, 'precision', '32'),
        accelerator='gpu',
        # devices=1,
    )
    if args.limit_train_batches is not None:
        trainer_kwargs['limit_train_batches'] = args.limit_train_batches
    if args.limit_val_batches is not None:
        trainer_kwargs['limit_val_batches'] = args.limit_val_batches

    if getattr(trainer_kwargs, 'precision', None) is not None:
        print(f'Precision set to {trainer_kwargs.precision}.')

    trainer = Trainer(**trainer_kwargs)

    # train the model
    trainer.fit(litmodel, datamodule=data_module, ckpt_path=args.resume_from_checkpoint)

    # Force-save the true final weights. `save_last` in this Lightning version
    # only updates when a top-k checkpoint is written, so the last epoch can be
    # missing (observed in old C0 runs). `final.ckpt` always exists after a
    # normal completion.
    if not args.debug and trainer.is_global_zero:
        os.makedirs(checkpoint_dir, exist_ok=True)
        final_path = os.path.join(checkpoint_dir, 'final.ckpt')
        trainer.save_checkpoint(final_path)
        print(f'Saved final checkpoint: {final_path}')
        if checkpoint_callback_n128 is not None:
            n128_dir = checkpoint_dir + '-n128'
            os.makedirs(n128_dir, exist_ok=True)
            n128_final = os.path.join(n128_dir, 'final.ckpt')
            trainer.save_checkpoint(n128_final)
            print(f'Saved final N128 checkpoint: {n128_final}')