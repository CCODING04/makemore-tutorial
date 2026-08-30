import re
def shingles(text, k=3):
    words = re.findall(r"[a-z]+", text.lower())
    return {" ".join(words[i:i+k]) for i in range(len(words)-k+1)}
def jaccard(s1, s2):
    if not s1 and not s2: return 1.0
    return len(s1 & s2)/len(s1 | s2)
def signature_agreement(s1, s2): return sum(1 for a,b in zip(s1,s2) if a==b)/len(s1)
def lsh_hit_probability(j, bands, rows): return 1 - (1 - j**rows) ** bands
def choose_bands_for_recall(j, rows, target=0.99):
    if j <= 0: return 1
    if j >= 1: return 1
    for b in range(1, 10001):
        if lsh_hit_probability(j, b, rows) >= target: return b
    return 10000
def keep_first_per_cluster(names, pairs):
    parent = {n: n for n in names}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
    first = {}
    for n in names:
        r = find(n)
        if r not in first: first[r] = n
    kept = [n for n in names if first[find(n)] == n]
    dropped = [n for n in names if first[find(n)] != n]
    return kept, dropped
