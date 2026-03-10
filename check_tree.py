import sys
sys.path.insert(0, '.')

with open('src/web/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

marker = 'RESEARCH_TREE: List[Dict] = '
start = content.index(marker) + len(marker)
depth = 0
in_list = False
end = start
for i, c in enumerate(content[start:], start):
    if c == '[':
        depth += 1
        in_list = True
    elif c == ']':
        depth -= 1
        if in_list and depth == 0:
            end = i + 1
            break

tree_str = content[start:end]
RESEARCH_TREE = eval(tree_str)

branches = {}
for n in RESEARCH_TREE:
    b = n['branch']
    if b not in branches:
        branches[b] = []
    branches[b].append(n)

print('Branch counts:')
for b, nodes in branches.items():
    print(f'  {b}: {len(nodes)}')

all_ids = {n['id'] for n in RESEARCH_TREE}
errors = []
for n in RESEARCH_TREE:
    if n['requires'] and n['requires'] not in all_ids:
        errors.append(f'BROKEN requires: {n["id"]} -> {n["requires"]}')

dups = {}
for n in RESEARCH_TREE:
    dups[n['id']] = dups.get(n['id'], 0) + 1
for nid, cnt in dups.items():
    if cnt > 1:
        errors.append(f'DUPLICATE id: {nid}')

# Cross-branch check
for n in RESEARCH_TREE:
    if n['requires']:
        parent = next((x for x in RESEARCH_TREE if x['id'] == n['requires']), None)
        if parent and parent['branch'] != n['branch']:
            errors.append(f'CROSS-BRANCH: {n["id"]} ({n["branch"]}) requires {n["requires"]} ({parent["branch"]})')

if errors:
    print('ERRORS:')
    for e in errors:
        print(' ', e)
else:
    print('Structure: OK')

# Tier distribution
print('\nTier distribution:')
for b, nodes in branches.items():
    tiers = {}
    for n in nodes:
        tiers[n['tier']] = tiers.get(n['tier'], 0) + 1
    print(f'  {b}: {sorted(tiers.items())}')

# Chain structure per branch
print('\nChains per branch:')
for b, nodes in branches.items():
    roots = [n for n in nodes if n['requires'] is None]
    print(f'  {b}: {len(roots)} root(s) -> ', end='')
    for root in roots:
        chain = [root['id']]
        cur = root
        while True:
            nxt = next((n for n in nodes if n['requires'] == cur['id']), None)
            if not nxt:
                break
            chain.append(nxt['id'])
            cur = nxt
        print(f'{len(chain)} long', end=' ')
    print()

# Check bonus types
print('\nBonus types used:')
bonus_types = {}
for n in RESEARCH_TREE:
    for k in n.get('bonus', {}):
        bonus_types[k] = bonus_types.get(k, 0) + 1
for k, cnt in sorted(bonus_types.items()):
    print(f'  {k}: {cnt} nodes')
