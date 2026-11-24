"""Fixed sustained-parent and noise-dose protocols; torch imported only on use."""
import suite_core as U
C = U.C
HORIZON = 16
DURATIONS = (2, 4)
SCALES = (0.5, 1., 2., 3., 4., 6.)
REPLICATES = 2


def branches(mode):
    if mode == 'sustained':
        return [dict(name=f'{u}_{q}', condition=f'{u}_{q}', u_mode=u,
                     q_wrong=q == 'q_wrong', replicate=0)
                for u in ('free', 'original', 'same', 'wrong')
                for q in ('q_clean', 'q_wrong')]
    if mode == 'noise':
        controls = [dict(name=n, condition=n, u_mode=n, q_wrong=False, replicate=0)
                    for n in ('free', 'same', 'wrong')]
        return controls + [dict(name=f'noise_s{s:g}_r{r}', condition=f'noise_s{s:g}',
                                u_mode='noise', q_wrong=False, scale=s, replicate=r)
                           for s in SCALES for r in range(REPLICATES)]
    raise ValueError(mode)


def phase(mode, step, duration):
    if step < 1:
        raise ValueError('outputs start at 1')
    if mode == 'sustained':
        return ('forced', step) if step <= duration else ('released', step-duration)
    return 'post', step


def should_replace(mode, step, duration):
    return step <= duration if mode == 'sustained' else step == 1


def apply_inputs(states, sources, specs, u, q, replace_u, inject_q, seed):
    """sources order: original/u_same/u_wrong/q_same/q_wrong, current absolute time.

    original is a clean replay row, NOT a sham after histories have diverged.
    q is replaced once at the first input, never clamped thereafter.
    """
    out = states.clone()
    dose = []
    for i, b in enumerate(specs):
        old_u, old_q = states[i, u], states[i, q]
        radius = float((sources[2, u]-old_u).norm()) if replace_u else None
        if replace_u:
            um = b['u_mode']
            if um in ('original', 'same', 'wrong'):
                out[i, u] = sources[{'original': 0, 'same': 1, 'wrong': 2}[um], u]
            elif um == 'noise':
                changed, _, _ = C.intervene(states, i, u, sources[2, u], 'noise',
                    C.stable_seed(seed, b['replicate']), b['scale'])
                out[i, u] = changed[i, u]
        if inject_q and b['q_wrong']:
            out[i, q] = sources[4, q]
        dose.append(dict(reference_norm=radius,
                         u_norm=float((out[i, u]-old_u).norm()),
                         q_norm=float((out[i, q]-old_q).norm()),
                         u_applied=replace_u and b['u_mode'] != 'free',
                         q_applied=inject_q and b['q_wrong']))
    return out, dose


def cohort_keys(records, arms, durations):
    """Common qualifications across every model AND duration, not noise outcomes."""
    cases = {}
    for r in records:
        key = (r['graph_id'], r['timing'])
        slot = (r['arm'], r['duration'])
        entry = cases.setdefault(key, {})
        if slot in entry:
            raise ValueError('duplicate graph/timing/arm/duration')
        entry[slot] = r
    expected = {(a, d) for a in arms for d in durations}
    if any(set(rs) != expected for rs in cases.values()):
        raise ValueError('unpaired case')
    return {k: (all(r['value_qualified'] for r in rs.values()),
                all(r['strict_qualified'] for r in rs.values())) for k, rs in cases.items()}


def event_windows(r, branch, reference, correct):
    """Repeated writes are doses, not outputs; release starts after last forced output."""
    b, ref = r['branches'][branch], r['branches'][reference]
    xs = [correct(b['inputs'][0])] + [correct(x) for x in b['outputs']]
    bs = [correct(ref['inputs'][0])] + [correct(x) for x in ref['outputs']]
    windows = {'full': U.S.trajectory_event(xs, bs)}
    if r['mode'] == 'sustained':
        # xs[L] = last forced forward's output, xs[L+1] = first unforced output.
        l = r['duration']
        windows['release'] = U.S.trajectory_event(xs[l:], bs[l:])
    return windows
