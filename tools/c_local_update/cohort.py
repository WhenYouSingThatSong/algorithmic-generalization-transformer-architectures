"""Combine three registered final evaluations, retaining each seed and its denominator."""
import argparse
from pathlib import Path
from collections import defaultdict
import local_protocol as S


def combine(paths):
    rows = defaultdict(dict)
    sources = {}
    manifest = None
    for path in paths:
        meta = S.P.read(path/"meta.json")
        ident = meta["source"]
        model = ident["model"]
        if (meta["status"]!="complete" or ident["version"]!=S.VERSION or model in sources
                or (manifest is not None and manifest!=ident["manifest"])):
            raise ValueError("incomplete, duplicate or mismatched cohort")
        manifest = ident["manifest"]
        sources[model] = meta["checkpoint"]
        for split in S.P.STRATA:
            result = S.P.read(path/(split+"-classes.json"))["summary"]
            for r in result["graph_equal"]:
                rows[tuple(r["key"])][model] = [r["mean"],r["graphs"]]
            for r in S.P.read(path/(split+"-standard-summary.json"))["clean"]:
                for metric in ("mean_value","mean_first4","graph_value_all"):
                    rows[split,"clean",r["horizon"],metric][model] = [r[metric],r["graphs"]]
    if set(sources)!=set(S.MODELS):
        raise ValueError("three distinct registered seeds required")
    result = []
    for key,v in sorted(rows.items()):
        values = [z[0] for z in v.values()]
        result.append(dict(key=key,by_seed=v,mean_three=sum(values)/3 if len(v)==3 else None))
    return dict(manifest=manifest,sources=sources,metrics=result)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True,nargs=3)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    S.P.write(args.output,combine(args.results))
