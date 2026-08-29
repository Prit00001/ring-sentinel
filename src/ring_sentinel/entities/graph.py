"""Time-sliced ring graph.

The graph is rebuilt once per day `d` using only transactions in
[d - LOOKBACK_DAYS, d). Connected components over surviving edges are candidate
rings. NO EDGE MAY EXIST BETWEEN A TRANSACTION AND ANY TRANSACTION THAT
HAPPENED AFTER IT — the window is half-open on the right for exactly this
reason, and tests/test_leakage.py checks it.

networkx is deliberately not used here. At 590k rows a path-compressed
union-find in thirty lines is faster and has no hidden ordering behaviour.
networkx is used only for the small per-case subgraph rendered in the console.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .resolve import ENTITY_COLUMNS


class UnionFind:
    """Path-compressed union-find with union by size."""

    __slots__ = ("parent", "size")

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = self.parent
        root = x
        while p[root] != root:
            root = p[root]
        # path compression
        while p[x] != root:
            p[x], x = root, p[x]
        return int(root)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def roots(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))], dtype=np.int64)


@dataclass
class WindowGraph:
    """Components of one daily window, plus the entity->component lookup.

    `entity_to_comp` is what lets a transaction on day d be attached to a
    component built purely from days before d, without that transaction
    influencing the component structure. That asymmetry is the whole causality
    argument for the ring features.
    """

    day: int
    window_start_day: int
    row_positions: np.ndarray            # positions into the full sorted frame
    comp_of_row: np.ndarray              # component id per window row
    entity_to_comp: dict[tuple[str, str], int] = field(default_factory=dict)
    entity_degree: dict[tuple[str, str], int] = field(default_factory=dict)
    n_components: int = 0
    pruned_counts: dict[str, int] = field(default_factory=dict)
    kept_counts: dict[str, int] = field(default_factory=dict)


def build_window_graph(
    day: int,
    window_start_day: int,
    row_positions: np.ndarray,
    entity_values: dict[str, np.ndarray],
    max_degree: int,
    min_degree: int,
    max_component_size: int | None = None,
) -> WindowGraph:
    """Union-find over rows that share a surviving entity value.

    entity_values maps entity type -> array of values aligned to row_positions,
    with None/NaN for absent.

    Hub pruning: an entity value whose degree exceeds max_degree is dropped
    entirely (gmail.com would otherwise merge the whole graph into one
    component and every ring feature would become a constant). An entity value
    with degree below min_degree creates no edge and is dropped too.

    Per-entity degree capping alone bounds no single entity value's blast
    radius, but it does not bound the *component* that emerges from chaining
    many entities together: a few hundred moderate-degree device or address
    values, each individually well under max_degree, can still transitively
    merge most of a window into one component (the classic random-graph
    giant-component effect). `max_component_size`, when given, is a second,
    coarser safety net: entity groups are unioned in ascending-degree order
    (the most specific, most ring-like signals first) and a union that would
    push the resulting component above the cap is skipped rather than applied
    — the same rows simply don't get that entity's edge, exactly as if that
    one entity value had been pruned for degree reasons.
    """
    n = len(row_positions)
    uf = UnionFind(n)
    entity_to_comp: dict[tuple[str, str], int] = {}
    entity_degree: dict[tuple[str, str], int] = {}
    pruned: dict[str, int] = {}
    kept: dict[str, int] = {}

    surviving: dict[str, dict[str, list[int]]] = {}
    # (etype, value, members) for every group that passed the degree filter,
    # collected across ALL entity types so the component-size cap below can
    # be enforced globally rather than per type.
    degree_ok: list[tuple[str, str, list[int]]] = []

    for etype, values in entity_values.items():
        groups: dict[str, list[int]] = {}
        for i, v in enumerate(values):
            if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
                continue
            groups.setdefault(v, []).append(i)

        n_pruned = 0
        for v, members in groups.items():
            deg = len(members)
            entity_degree[(etype, v)] = deg
            if deg > max_degree or deg < min_degree:
                n_pruned += 1
                continue
            degree_ok.append((etype, v, members))
        pruned[etype] = n_pruned
        kept[etype] = 0
        surviving[etype] = {}

    for etype, v, members in sorted(degree_ok, key=lambda g: len(g[2])):
        if max_component_size is not None:
            roots_here = {uf.find(m) for m in members}
            merged_size = sum(int(uf.size[r]) for r in roots_here)
            if merged_size > max_component_size:
                pruned[etype] += 1
                continue
        keep = surviving[etype]
        keep[v] = members
        kept[etype] += 1
        # Union the members into one component: chain them, which is O(k).
        first = members[0]
        for j in members[1:]:
            uf.union(first, j)

    roots = uf.roots()
    # Relabel roots to a dense 0..k-1 component id.
    uniq, comp_of_row = np.unique(roots, return_inverse=True)

    for etype, keep in surviving.items():
        for v, members in keep.items():
            entity_to_comp[(etype, v)] = int(comp_of_row[members[0]])

    return WindowGraph(
        day=day,
        window_start_day=window_start_day,
        row_positions=row_positions,
        comp_of_row=comp_of_row.astype(np.int64),
        entity_to_comp=entity_to_comp,
        entity_degree=entity_degree,
        n_components=int(len(uniq)),
        pruned_counts=pruned,
        kept_counts=kept,
    )


def extract_entity_values(df: pd.DataFrame, positions: np.ndarray) -> dict[str, np.ndarray]:
    """Pull the entity columns for a set of row positions, as object arrays."""
    out = {}
    for etype, col in ENTITY_COLUMNS.items():
        vals = df[col].to_numpy(dtype=object)[positions]
        out[etype] = vals
    return out


def pruning_report(
    df: pd.DataFrame, max_degree: int, min_degree: int
) -> pd.DataFrame:
    """Global (not windowed) view of how aggressive the degree cap is.

    Reported so a reviewer can see the pruning decision rather than infer it.
    """
    rows = []
    n = len(df)
    for etype, col in ENTITY_COLUMNS.items():
        vc = df[col].value_counts(dropna=True)
        if vc.empty:
            rows.append({
                "entity": etype, "n_distinct": 0, "median_degree": 0.0,
                "p99_degree": 0.0, "max_degree": 0,
                "n_values_over_cap": 0, "rows_losing_this_entity": 0,
                "share_rows_dropped_pct": 0.0,
            })
            continue
        over = vc[vc > max_degree]
        under = vc[vc < min_degree]
        rows_lost = int(over.sum() + under.sum())
        rows.append({
            "entity": etype,
            "n_distinct": int(vc.size),
            "median_degree": float(vc.median()),
            "p99_degree": float(vc.quantile(0.99)),
            "max_degree": int(vc.max()),
            "n_values_over_cap": int(over.size),
            "rows_losing_this_entity": rows_lost,
            "share_rows_dropped_pct": round(100.0 * rows_lost / n, 2) if n else 0.0,
        })
    return pd.DataFrame(rows)


def case_subgraph(df: pd.DataFrame, member_positions: np.ndarray, max_nodes: int = 60):
    """Small networkx graph for one case, for SVG rendering in the console.

    This is the only place networkx is used. Nodes are transactions and
    entities; edges connect a transaction to each entity it carries.
    """
    import networkx as nx

    g = nx.Graph()
    positions = member_positions[:max_nodes]
    for pos in positions:
        row = df.iloc[int(pos)]
        tx_node = f"tx:{int(row['TransactionID'])}"
        g.add_node(tx_node, kind="tx", amount=float(row["TransactionAmt"]),
                   dt=int(row["TransactionDT"]))
        for etype, col in ENTITY_COLUMNS.items():
            v = row[col]
            if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
                continue
            ent_node = f"{etype}:{v}"
            g.add_node(ent_node, kind=etype)
            g.add_edge(tx_node, ent_node)
    return g
