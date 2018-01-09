import itertools
import functools
import operator
from collections import namedtuple

def all_combinations(params):
    items = sorted(list(params.items()))
    keys = [k for k,v in items]
    vals = [v for k,v in items]
    ranges = [range(len(v)) for k,v in items]
    Combo = namedtuple('Combo', keys)
    def _build_tuple(choice):
        t = [vals[i][j] for i,j in enumerate(choice)]
        return Combo(*t)
    res = [_build_tuple(choice) for choice in itertools.product(*ranges)]
    assert len(res) == functools.reduce(operator.mul, map(len, vals), 1)
    return res

