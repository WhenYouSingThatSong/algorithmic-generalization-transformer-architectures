"""Original four-layer forward; observation-only layer-2 hook, no gating."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import protocol as P
_READY=False

def setup():
    global _READY
    if _READY:return
    sys.path.insert(0,str(HERE.parent/'b_mechanism'))
    from trajectory_io import configure_runtime
    configure_runtime()
    _READY=True

class Runtime:
    def __init__(self, spec, device='cuda', initialize_update=False):
        setup()
        import torch
        import b_mechanism_common as B
        self.torch=torch; self.spec=spec
        mc,tc,dc=B.load_configs(spec['config'])
        tok=B.build_factored_tokenizer(dc)
        if list(tok.vocab_sizes)!=[7,129,2,25,2,2]: raise ValueError('wrong vocab')
        mc.vocab_sizes,mc.factors=tok.vocab_sizes,tok.factors
        self.lit=B.LitRecurrentTransformerModel(mc,dc,tc,B.DAGTeacherModel(mc,tok))
        if P.sha(spec['path'])!=spec['sha256']: raise ValueError('checkpoint hash')
        self.lit.load_ckpt(spec['path'],map_location='cpu')
        self.model=self.lit.model; self.tok=tok; self.device=torch.device(device)
        self.model.to(self.device).eval()
        self.continuous=bool(self.lit.nointerm); self.update=spec.get('arm')=='update'
        if self.continuous!=(spec['family']=='C'):raise ValueError('family / feedback mismatch')
        self.calls=0; self.capture=False; self.middle=None
        if self.update:
            blocks=self.model.transformer.blocks
            if len(blocks)!=4 or 'layer_1' not in blocks: raise ValueError('requires four blocks')
            if initialize_update: self.model.output_layers[4].reset_parameters()
            self.hook=blocks['layer_1'].register_forward_hook(self._capture)
        torch.set_float32_matmul_precision('medium')

    def _capture(self, module, args, output):
        if self.capture: self.middle=output[0]

    def init(self,x): return self.model.token_embedder(x) if self.continuous else x.clone()
    def decode(self,h):
        return self.torch.stack([head(h).argmax(-1) for head in self.model.output_layers],-1) if self.continuous else h

    def forward(self,h, middle=False):
        self.capture=middle and self.update; self.middle=None
        try:
            if self.continuous: out,logits=self.model(h,True)
            else:
                logits=self.model(h);out=self.torch.stack([x.argmax(-1) for x in logits],-1);out[...,4:]=0
            if self.capture and self.middle is None: raise RuntimeError('layer-2 capture absent')
            mid=self.model.output_layers[4](self.middle) if self.capture else None
            self.calls+=len(h)
            return out,logits,mid
        finally: self.capture=False;self.middle=None

    def donor(self,x,g,pos,edits,rounds):
        """Fixed absolute source round; never wait until source becomes correct."""
        if edits is None:return None,None
        x=x.clone()
        for p,delta in edits.items():x[0,p,3]=(x[0,p,3]+delta)%23
        training=self.model.training
        try:
            self.model.eval()
            with self.torch.no_grad():
                h=self.init(x)
                for _ in range(rounds):h,_,_=self.forward(h)
                return h[0,pos].detach().clone(),self.decode(h)[0,pos,:4].tolist()
        finally:self.model.train(training)

    def donors(self,x,events,chunk=8):
        """Batch independent source forwards; fixed rounds, no success search."""
        results={};training=self.model.training
        try:
            self.model.eval()
            with self.torch.no_grad():
                for rounds in sorted({e['donor_round'] for e in events}):
                    ee=[e for e in events if e['donor_round']==rounds]
                    for off in range(0,len(ee),chunk):
                        group=ee[off:off+chunk];good=[];inputs=[]
                        for e in group:
                            key=e['bi'],e['pos']
                            if e['edits'] is None:results[key]=(None,None);continue
                            xx=x[e['bi']].clone()
                            for p,d in e['edits'].items():xx[p,3]=(xx[p,3]+d)%23
                            inputs.append(xx);good.append(e)
                        if not good:continue
                        h=self.init(self.torch.stack(inputs))
                        for _ in range(rounds):h,_,_=self.forward(h)
                        decoded=self.decode(h)
                        for i,e in enumerate(good):
                            p=e['pos'];results[e['bi'],p]=(h[i,p].detach().clone(),decoded[i,p,:4].tolist())
            return results
        finally:self.model.train(training)

def load_data(path,tok):
    import torch
    blob=torch.load(path,map_location='cpu',weights_only=False)
    y=blob['tokenized_dataset'].long();info=blob['dataset_info'].long();x=y.clone()
    x[...,3][x[...,0]==tok.syntax_tok2idx['VARIABLE']]=tok.value_tok2idx['EMPTY']
    x[...,4:]=0
    return x,y,info

def dataset_graphs(y,info):return [P.graph(a.tolist(),b.tolist()) for a,b in zip(y,info)]

def node_masks(gs,length,device):
    import torch
    mask=torch.zeros(len(gs),length,dtype=torch.bool,device=device)
    for i,g in enumerate(gs):mask[i,g['nodes']]=True
    return mask
