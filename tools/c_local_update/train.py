"""Three independent original-C starts. Identical data, corruption draws and D+1 budgets."""
import argparse
import json
import random
from pathlib import Path

import local_protocol as S
P = S.P


def dev_check(rt, plan, frozen, out, epoch):
    from assess import focused_split
    from evaluate import small_check
    import torch
    # Diagnostics may construct a frozen receiver, consuming initialisation RNG.
    devices = [rt.device.index if rt.device.index is not None else torch.cuda.current_device()] if rt.device.type=="cuda" else []
    with torch.random.fork_rng(devices=devices):
        rt.model.eval()
        try:
            small_check(rt,plan,epoch,out.with_suffix(".json"))
            focused_split(rt,plan,frozen,"dev",out.with_name(out.name+"-classes.json"))
        finally:
            rt.model.train()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--model",choices=S.MODELS,required=True)
    parser.add_argument("--out-root",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--micro-batch",type=int,choices=(8,16,32),default=32)
    parser.add_argument("--donor-batch",type=int,choices=(8,16,32),default=32)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--smoke",action="store_true",help="three forced modes on a maximum-depth batch in a separate directory")
    args=parser.parse_args()
    plan=S.validate(args.plan,args.manifest)
    frozen=P.read(args.manifest)
    from runtime import Runtime,load_data,dataset_graphs,setup
    setup()
    import numpy as np
    import torch
    from local_loss import recurrent_loss
    from tensor_targets import Topology
    from process_loss import materialize
    torch.set_num_threads(4)
    seed=P.seed(S.VERSION,args.model,"training")%(2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    source=next(m for m in plan["models"] if m["id"]==args.model)
    if source["sha256"]!=frozen["sources"][args.model]:
        raise ValueError("original checkpoint identity mismatch")
    identity=dict(version=S.VERSION,model=args.model,source=source["sha256"],manifest=P.sha(args.manifest),
                  seed=seed,micro_batch=args.micro_batch,donor_batch=args.donor_batch)
    dest=args.out_root/(args.model+"-smoke" if args.smoke else args.model)
    if args.smoke and args.resume:
        parser.error("smoke is a separate, fresh run")
    if args.resume:
        if P.read(dest/"meta.json")["identity"]!=identity:
            raise ValueError("resume identity mismatch")
        if (dest/"final.ckpt").exists() and P.read(dest/"meta.json")["status"]=="complete":
            return
    else:
        dest.mkdir(parents=True,exist_ok=False)
        P.write(dest/"meta.json",dict(status="running",identity=identity))
    rt=Runtime(dict(source,arm="update"),args.device,initialize_update=True)
    x,y,info=load_data(plan["train"]["path"],rt.tok)
    graphs=dataset_graphs(y,info)
    del info
    depths=[g["D"] for g in graphs]
    if len(graphs)!=50000 or max(depths)>7 or any(len(g["nodes"])!=32 for g in graphs):
        raise ValueError("expected original N32 training data")
    topology=Topology(graphs)
    middle=list(rt.model.output_layers[4].parameters())
    ids_mid={id(p) for p in middle}
    optimizer=torch.optim.AdamW([
        {"params":[p for p in rt.model.parameters() if id(p) not in ids_mid],"base_lr":1e-4},
        {"params":middle,"base_lr":5e-4}],betas=(.9,.999),eps=1e-8,weight_decay=.01)
    step=completed=0
    total_steps=len(P.batches(depths,1))*50
    if args.resume and (dest/"last.ckpt").exists():
        ck=torch.load(dest/"last.ckpt",map_location="cpu",weights_only=False)
        if ck["identity"]!=identity:
            raise ValueError("wrong last checkpoint")
        rt.model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        completed,step=ck["epoch"],ck["step"]
        torch.set_rng_state(ck["rng"])
        if rt.device.type=="cuda":
            torch.cuda.set_rng_state(ck["gpu_rng"],device=rt.device)
        del ck

    def check(epoch):
        if epoch in P.CHECKS and not (dest/f"dev-{epoch:02d}-classes.json").exists():
            dev_check(rt,plan,frozen,dest/f"dev-{epoch:02d}",epoch)

    def save(name, epoch):
        temp=dest/(name+".tmp")
        torch.save(dict(identity=identity,epoch=epoch,step=step,state_dict=rt.model.state_dict(),
                        optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),
                        gpu_rng=torch.cuda.get_rng_state(rt.device) if rt.device.type=="cuda" else None),temp)
        temp.replace(dest/name)

    rt.model.train()
    if args.smoke:
        maximum = max(depths)
        ids = [i for i,d in enumerate(depths) if d==maximum][:32]
        evidence = []
        for j,mode in enumerate(S.PROTOCOL["modes"]):
            d = depths[ids[0]]
            settings = dict(mode=mode,enabled=True,rounds=d+1,
                            start=None if mode=="frontier" else d if mode=="completed" else max(1,d//2))
            gs = [graphs[i] for i in ids]
            _,ev = S.events(gs,ids,1,j,settings)
            audit = []
            optimizer.zero_grad(set_to_none=True)
            for off in range(0,len(ids),args.micro_batch):
                ii = ids[off:off+args.micro_batch]
                xx = x[ii].to(rt.device)
                loss,rr = recurrent_loss(rt,xx,y[ii].to(rt.device),gs[off:off+len(ii)],
                                         topology.batch(ii,xx.shape[1],rt.device),ev,settings,off,args.donor_batch)
                (loss*len(ii)/len(ids)).backward()
                audit.extend(materialize(rr))
            norm = torch.nn.utils.clip_grad_norm_(middle,float("inf"),error_if_nonfinite=True)
            if not float(norm)>0:
                raise ValueError("no layer-2 UPDATE gradient")
            evidence.append(dict(mode=mode,events=len(ev),accepted=sum(bool(r["accepted"]) for r in audit),
                                 changed=sum(bool(r.get("actual_value_changed")) for r in audit),
                                 head_grad_norm=float(norm),rounds=settings["rounds"]))
        P.write(dest/"meta.json",dict(status="smoke_complete",identity=identity,evidence=evidence))
        return
    check(completed)
    for epoch in range(completed+1,51):
        # An interrupted epoch is replayed from its last complete boundary.
        with (dest/f"epoch-{epoch:02d}.jsonl").open("w",encoding="utf-8") as log:
            for batch, ids in enumerate(P.batches(depths,epoch)):
                settings=S.schedule(depths[ids[0]],epoch,batch)
                gs=[graphs[i] for i in ids]
                candidates,events=S.events(gs,ids,epoch,batch,settings)
                for group in optimizer.param_groups:
                    group["lr"]=group["base_lr"]*P.lr_factor(step,total_steps)
                optimizer.zero_grad(set_to_none=True)
                loss_value=0
                audit=[]
                for off in range(0,len(ids),args.micro_batch):
                    ii=ids[off:off+args.micro_batch]
                    xx=x[ii].to(rt.device)
                    block=topology.batch(ii,xx.shape[1],rt.device)
                    loss,rows=recurrent_loss(rt,xx,y[ii].to(rt.device),gs[off:off+len(ii)],block,
                                                 events,settings,off,args.donor_batch)
                    weight=len(ii)/len(ids)
                    (loss*weight).backward()
                    loss_value=loss_value+loss.detach()*weight
                    audit.extend(dict(z,bi=z["bi"]+off) for z in rows)
                torch.nn.utils.clip_grad_norm_(rt.model.parameters(),float("inf"),error_if_nonfinite=True)
                optimizer.step()
                step+=1
                record=materialize(dict(epoch=epoch,step=step,batch=batch,graphs=len(ids),D=depths[ids[0]],
                                        loss=loss_value,candidates=candidates,**settings,injections=audit))
                log.write(json.dumps(record,allow_nan=False)+"\n")
                if step%100==0:
                    log.flush()
                    print(f"{args.model} epoch {epoch} step {step}/{total_steps} loss {record['loss']:.4f}",flush=True)
        if epoch in P.CHECKS:
            save(f"epoch-{epoch:02d}.ckpt",epoch)
        save("last.ckpt",epoch)
        check(epoch)
    save("final.ckpt",50)
    P.write(dest/"meta.json",dict(status="complete",identity=identity,steps=step,epochs=50,
                                  final_sha256=P.sha(dest/"final.ckpt")))


if __name__=="__main__":
    main()
