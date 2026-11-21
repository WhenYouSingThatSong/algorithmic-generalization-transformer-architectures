"""Prespecified contrasts computed from per-graph cells (blank for zero denominator)."""
from collections import Counter, defaultdict


def rate(n, d):
    return n/d if d else None


def matched_fanout(rows):
    """Aggregate over the same graph IDs accepted at all three fan-out levels."""
    by_graph = defaultdict(lambda: defaultdict(Counter))
    for r in rows:
        key = (r["model"], r["width"], r["depth"], r["timing"], r["gid"])
        c = by_graph[key][int(r["profile"])]
        if r["role"] == "attempt":
            c["attempted"] += r["attempts"]
            c["accepted"] += r["accepted"]
        elif r["role"] != "source":
            c["downstream_node_rounds"] += r["observed"]
            c["downstream_clean_correct"] += r["clean_correct"]
            c["downstream_harmed"] += r["harmed"]
            c["specific"] += r["specific"]
            if r["role"] == "child" and r["coefficient"] == 0:
                c["unexposed_clean_correct"] += r["clean_correct"]
                c["unexposed_harmed"] += r["harmed"]
            if r["role"] == "sink":
                c["sink_specific_rounds"] += r["specific"]
    totals = defaultdict(Counter)
    fields = ("downstream_node_rounds","downstream_clean_correct","downstream_harmed","specific",
              "unexposed_clean_correct","unexposed_harmed")
    for key, settings in by_graph.items():
        if set(settings)!={1,2,4} or any(c["attempted"]!=1 for c in settings.values()):
            raise ValueError("incomplete matched fan-out design")
        accepted = all(c["accepted"]==1 for c in settings.values())
        for f,c in settings.items():
            dest = totals[(*key[:-1],f)]
            dest["candidate_graphs"] += 1
            dest["common_graphs"] += accepted
            for field in fields:
                dest[field] += c[field] if accepted else 0
            dest["graphs_with_specific_sink_damage"] += accepted and c["sink_specific_rounds"]>0
    cols=("model","width","depth","timing","fanout")
    return [dict(**dict(zip(cols,key)),**c) for key,c in sorted(totals.items())]


def contrasts(stage, rows):
    bins = defaultdict(Counter)
    for r in rows:
        common = (r["model"], r["width"], r["depth"], r["profile"])
        if stage == "transitions":
            key = common+(r["arm"], r["window"])
            c = bins[key]
            old, new, n = r["previous"], r["following"], r["observations"]
            c.update(observations=n, numeric_error_inputs=n*(old=="wrong_number"),
                     numeric_errors_to_correct=n*(old=="wrong_number" and new=="correct"),
                     non_numeric_to_correct=n*(old in ("empty","non_numeric") and new=="correct"),
                     correct_inputs=n*(old=="correct"), correct_to_error=n*(old=="correct" and new!="correct"),
                     first_correct=r["first_correct"],returned=r["returned_to_correct"],
                     stable_numeric_correct=r["numeric_correct_at_both"],
                     lost_identity_while_correct=r["identity_lost_while_value_correct"])
        elif stage == "signals":
            key = common+(r["arm"],r["window"],r["previous"],r["margin_bin"],r["repeat"])
            c=bins[key]
            for update in (0,1):
                n=r["observations"] if r["update"]==update else 0
                c[f"update_{update}_observed"]+=n
                c[f"update_{update}_next_correct"]+=n*r["next_correct"]
        elif stage == "identity":
            key = common+(r["extra"],r["step"],r["condition"])
            c=bins[key]
            keep=r["common_eligible"]
            c.update(attempts=r["attempts"],single_eligible=r["eligible"],common_eligible=keep*r["attempts"],
                     paired_clean_correct=keep*r["clean_pair_correct"],paired_both_readers_correct=keep*r["actual_pair_correct"],
                     paired_spoiled=keep*r["clean_pair_spoiled"],u_reads_other=keep*r["u_reader_specific_other"],
                     v_reads_other=keep*r["v_reader_specific_other"],u_reads_plus_one=keep*r["u_reader_value_plus_one"])
        else:
            key = common+(r["timing"],r["role"],r["coefficient"],r["step"])
            bins[key].update({k:r[k] for k in ("attempts","accepted","observed","clean_correct","harmed","specific")})
    suffixes={"transitions":("arm","window"),"signals":("arm","window","previous","margin_bin","repeat"),
              "identity":("extra","step","condition"),"fanout":("timing","role","coefficient","step")}
    table=[]
    for key,c in sorted(bins.items()):
        row=dict(zip(("model","width","depth","profile",*suffixes[stage]),key))
        row.update(c)
        if stage=="transitions":
            row["numeric_error_recovery_rate"]=rate(c["numeric_errors_to_correct"],c["numeric_error_inputs"])
            row["correct_to_error_rate"]=rate(c["correct_to_error"],c["correct_inputs"])
        elif stage=="signals":
            a=rate(c["update_0_next_correct"],c["update_0_observed"])
            b=rate(c["update_1_next_correct"],c["update_1_observed"])
            row.update(next_correct_no_update=a,next_correct_update=b,
                       update_minus_no_update=b-a if a is not None and b is not None else None)
        elif stage=="identity":
            row["spoiled_given_both_clean_correct"]=rate(c["paired_spoiled"],c["paired_clean_correct"])
        else:
            row["harmed_given_clean_correct"]=rate(c["harmed"],c["clean_correct"])
        table.append(row)
    return table
