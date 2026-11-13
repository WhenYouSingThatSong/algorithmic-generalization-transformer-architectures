# -*- coding: utf-8 -*-
"""Summarize fixed reruns and compare with the pre-fix JSONs.

Usage (from repo root):
    python tmp/scripts/summarize_fixed_20261113.py

Reads:
    tmp/analysis/parent-2x2-fixed/**/*.json   (new)
    tmp/analysis/inject-v2-fixed/*.json       (new)
Old files are looked up by basename under:
    tmp/analysis/parent-2x2/ (including P1/)
    tmp/analysis/inject-v2/
"""
import json, os, glob, collections

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OLD_DIRS = [
    os.path.join(ROOT, 'tmp/analysis/parent-2x2'),
    os.path.join(ROOT, 'tmp/analysis/parent-2x2/P1'),
    os.path.join(ROOT, 'tmp/analysis/inject-v2'),
]

def find_old(name):
    for d in OLD_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None

def load(p):
    with open(p) as f:
        return json.load(f)

def cnt(d):
    return sum(d.values())

def fmt_u(x):
    return ' '.join(f'{k}={v}' for k, v in x.items() if v)

print('=' * 72)
print('PARENT 2x2 fixed reruns')
print('=' * 72)
parent_files = sorted(glob.glob(os.path.join(ROOT, 'tmp/analysis/parent-2x2-fixed/**/*.json'), recursive=True))
parent_files = [p for p in parent_files if '/smoke/' not in p]
for p in parent_files:
    d = load(p); s = d.get('summary', {})
    name = os.path.basename(p)
    print(f'\n{name}')
    print(f"  n_graphs={s.get('n_graphs')} n_records={s.get('n_records')} "
          f"start={s.get('start')} timing={s.get('timing')} mode={s.get('value_mode')}")
    if 'post1' not in s:
        continue
    old = find_old(name)
    old_s = load(old).get('summary', {}) if old else {}
    for cond in ['clean', 'u', 'r', 'ur']:
        u = s['post1']['by_cond'][cond]['u']
        line = f"  post1 {cond:5s} u: " + fmt_u(u)
        if old_s:
            ou = old_s.get('post1', {}).get('by_cond', {}).get(cond, {}).get('u', {})
            line += '   | old: ' + (fmt_u(ou) if ou else '-')
        print(line)
    for post in ['post2', 'post4']:
        if post in s:
            u = s[post]['by_cond']['r']['u'] if 'r' in s[post].get('by_cond', {}) else {}
            print(f"  {post} R-only u: " + fmt_u(u))

print('\n' + '=' * 72)
print('V2 teacher-state fixed reruns')
print('=' * 72)
v2_files = sorted(glob.glob(os.path.join(ROOT, 'tmp/analysis/inject-v2-fixed/*.json')))
v2_files = [p for p in v2_files if '/smoke/' not in p]
for p in v2_files:
    d = load(p); s = d.get('summary', {})
    name = os.path.basename(p)
    print(f'\n{name}')
    print(f"  n_graphs={s.get('n_graphs')} n_records={s.get('n_records')} k={s.get('k')}")
    for key in ['u_r1', 'u_r2', 'u_r4', 'q_r1', 'q_r2', 'q_r4']:
        print(f"  {key}: {s.get(key)}")
    old = find_old(name)
    if old:
        o = load(old).get('summary', {})
        print('  OLD u_r1:', o.get('u_r1'), 'O q_r1:', o.get('q_r1'))
        print('  OLD u_r2:', o.get('u_r2'), 'O q_r2:', o.get('q_r2'))