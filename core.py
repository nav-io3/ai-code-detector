# Auto-extracted from ai_code_detector.ipynb so the app uses exactly the same features.
import ast, io, re, hashlib, keyword, tokenize
import numpy as np
import pandas as pd

from collections import Counter
NEST = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.FunctionDef, ast.ClassDef)
SNAKE = re.compile(r"^[a-z_][a-z0-9_]*$")

def depth(node, d=0):
    m = d
    for c in ast.iter_child_nodes(node):
        m = max(m, depth(c, d + 1 if isinstance(c, NEST) else d))
    return m

def features(src):
    lines = src.splitlines() or [""]
    nb = [l for l in lines if l.strip()]
    lens = [len(l) for l in nb] or [0]
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except Exception:
        toks = []
    comments = [t for t in toks if t.type == tokenize.COMMENT]
    try:
        tree = ast.parse(src)
    except SyntaxError:
        tree = ast.parse("pass")
    names = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name): names.append(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)): names.append(n.name)
        elif isinstance(n, ast.arg): names.append(n.arg)
    nodes = [type(n).__name__ for n in ast.walk(tree)]
    total = max(len(nodes), 1)
    f = {
        "n_lines": len(nb),
        "avg_line_len": float(np.mean(lens)), "max_line_len": max(lens), "std_line_len": float(np.std(lens)),
        "blank_ratio": 1 - len(nb) / max(len(lines), 1),
        "comment_density": len(comments) / max(len(nb), 1),
        "inline_comment_ratio": sum(1 for t in comments if t.line.strip() and not t.line.strip().startswith("#")) / max(len(comments), 1),
        "avg_ident_len": float(np.mean([len(x) for x in names])) if names else 0,
        "unique_ident_ratio": len(set(names)) / max(len(names), 1),
        "snake_ratio": float(np.mean([bool(SNAKE.match(x)) for x in names])) if names else 0,
        "single_char_ratio": float(np.mean([len(x) == 1 for x in names])) if names else 0,
        "n_functions": nodes.count("FunctionDef"),
        "max_nesting": depth(tree),
        "trailing_ws": sum(1 for l in lines if l != l.rstrip()),
        "tab_lines": sum(1 for l in lines if l.startswith("\t")),
        "has_type_hints": int(any(isinstance(n, ast.arg) and n.annotation for n in ast.walk(tree)) or
                              any(isinstance(n, ast.FunctionDef) and n.returns for n in ast.walk(tree))),
        "n_docstrings": sum(1 for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(n)),
        "uses_main_guard": int("__name__" in names),
        "n_fstrings": nodes.count("JoinedStr"),
        "n_prints": sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "print"),
    }
    for k, v in Counter(nodes).items():
        f["ast_" + k] = v / total
    return f



def featurize(codes, cols=None):
    X = pd.DataFrame([features(s) for s in codes]).fillna(0)
    if cols is not None:
        X = X.reindex(columns=cols, fill_value=0)
    return X


def code_tokens(src):
    out = []
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.NAME:
            import keyword
            out.append(t.string if keyword.iskeyword(t.string) else "ID")
        elif t.type == tokenize.NUMBER: out.append("NUM")
        elif t.type == tokenize.STRING: out.append("STR")
        elif t.type == tokenize.OP: out.append(t.string)
    return out

def fingerprints(src, k=5, w=4):
    toks = code_tokens(src)
    if len(toks) < k: return set()
    hashes = [int(hashlib.md5(" ".join(toks[i:i + k]).encode()).hexdigest()[:8], 16) for i in range(len(toks) - k + 1)]
    fp = set()
    for i in range(max(len(hashes) - w + 1, 1)):
        win = hashes[i:i + w]
        fp.add(min(win))                       # winnowing: keep the minimum hash of each window
    return fp

def _unused(a, b):
    fa, fb = fingerprints(a), fingerprints(b)
    return len(fa & fb) / max(min(len(fa), len(fb)), 1)    # containment of the smaller one



def containment(fa, fb):
    return len(fa & fb) / max(min(len(fa), len(fb)), 1)
