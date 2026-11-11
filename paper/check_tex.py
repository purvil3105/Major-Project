"""Static checks for main.tex when no LaTeX compiler is available: python check_tex.py"""
import os
import re

t = open("main.tex", encoding="utf-8").read()
b = open("refs.bib", encoding="utf-8").read()
problems = []

if any(ord(c) < 32 and c not in "\n\r" for c in t):
    problems.append("control characters present")
depth = 0
for c in re.sub(r"\\[{}%$&#_]", "", t):
    depth += (c == "{") - (c == "}")
    if depth < 0:
        problems.append("unbalanced } ")
        break
if depth:
    problems.append(f"brace depth {depth}")
stack = []
for kind, env in re.findall(r"\\(begin|end)\{([^}]+)\}", t):
    if kind == "begin":
        stack.append(env)
    elif not stack or stack.pop() != env:
        problems.append(f"environment mismatch at {env}")
if stack:
    problems.append(f"unclosed {stack}")
body = re.sub(r"\\\$", "", t)
if len(re.findall(r"\$", body)) % 2:
    problems.append("odd number of $")
keys = set(re.findall(r"@\w+\{([^,]+),", b))
cites = {k.strip() for g in re.findall(r"\\cite\{([^}]+)\}", t) for k in g.split(",")}
if cites - keys:
    problems.append(f"missing bib keys {cites - keys}")
figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t)
missing = [f for f in figs if not os.path.exists(os.path.join("figs", f))]
if missing:
    problems.append(f"missing figures {missing}")
labels = re.findall(r"\\label\{([^}]+)\}", t)
refs = set(re.findall(r"\\ref\{([^}]+)\}", t))
if refs - set(labels):
    problems.append(f"undefined refs {refs - set(labels)}")
if len(labels) != len(set(labels)):
    problems.append("duplicate labels")
words = len(re.sub(r"\\[a-zA-Z]+|[{}$\\]", " ", t.split("\\begin{document}")[1]).split())
print("unused bib entries:", sorted(keys - cites))
print(f"approx. words: {words}; figures: {len(figs)}; citations: {len(cites)}")
print("OK" if not problems else "PROBLEMS: " + "; ".join(problems))
