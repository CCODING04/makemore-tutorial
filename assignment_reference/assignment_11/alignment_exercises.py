import math, re
def math_reward(response, ground_truth):
    for pat in (r"\\boxed\{(-?[\d,\.]+)\}", r"####\s*(-?[\d,\.]+)", r"(-?\d[\d,]*(?:\.\d+)?)"):
        m = re.findall(pat, response, flags=re.I)
        if m:
            try:
                pred = float(m[-1].replace(",", "").rstrip("."))
                return 1.0 if abs(pred - float(ground_truth)) < 1e-4 else 0.0
            except ValueError:
                continue
    return 0.0
def group_advantages(rewards, eps=1e-6):
    mean = sum(rewards)/len(rewards)
    std = max(sum((x-mean)**2 for x in rewards)/len(rewards), eps) ** 0.5
    return [(x-mean)/std for x in rewards]
def zero_gradient_groups(rm): return [i for i,g in enumerate(rm) if max(g)==min(g)]
def k3_kl(lr, ln):
    return sum(math.exp(a-b)-a+b-1 for a,b in zip(lr,ln))/len(lr)
def kl_budget_ok(lr, ln, budget=0.05): return k3_kl(lr,ln) <= budget
