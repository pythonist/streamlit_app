import numpy as np
import pandas as pd
import networkx as nx

from config import CONFIG
from utils import print_step


class GraphAnalytics:
    SAMPLE_CAPS = {
        "graph_sample_size": 900,
        "ring_sample_size": 450,
    }
    PAGERANK_NODE_LIMIT = 850
    CLUSTERING_NODE_LIMIT = 950
    CORE_NUMBER_NODE_LIMIT = 1200
    CYCLE_BASIS_NODE_LIMIT = 40

    def _node_type(self, node_id):
        text = str(node_id)
        prefix, _, _ = text.partition("::")
        return {
            "cust": "customer",
            "acct": "account",
            "dev": "device",
            "ip": "ip",
            "cp": "counterparty",
            "mch": "merchant",
        }.get(prefix, prefix or "node")

    def _pick_sample(self, df, sample_key):
        configured_size = int(CONFIG.get(sample_key, len(df)))
        hard_cap = int(self.SAMPLE_CAPS.get(sample_key, configured_size))
        sample_size = min(configured_size, hard_cap, len(df))
        if sample_size <= 0:
            return df.head(0).copy()
        if sample_size == len(df):
            return df.copy()
        return df.sample(sample_size, random_state=CONFIG["random_state"]).copy()

    def _ip_column(self, df):
        return "device_ip_address" if "device_ip_address" in df.columns else "ip_address"

    def _build_edge_frame(self, left, right, weights):
        edge_df = pd.DataFrame({"u": left, "v": right, "weight": weights})
        edge_df = edge_df.dropna(subset=["u", "v"])
        if edge_df.empty:
            return edge_df

        edge_df["u"] = edge_df["u"].astype(str)
        edge_df["v"] = edge_df["v"].astype(str)
        edge_df = edge_df[(edge_df["u"] != "nan") & (edge_df["v"] != "nan")]
        if edge_df.empty:
            return edge_df

        return edge_df.groupby(["u", "v"], as_index=False, sort=False)["weight"].sum()

    def _build_graph(self, df):
        graph = nx.Graph()
        ip_col = self._ip_column(df)

        def node_key(prefix, series):
            return prefix + "::" + series.astype(str)

        amount = df.get("amount", pd.Series(0, index=df.index)).fillna(0).astype(float)
        edge_frames = [
            self._build_edge_frame(node_key("cust", df["customer_id"]), node_key("acct", df["account_id"]), amount),
        ]

        if "device_id" in df.columns:
            edge_frames.append(self._build_edge_frame(node_key("cust", df["customer_id"]), node_key("dev", df["device_id"]), amount))
        if ip_col in df.columns:
            edge_frames.append(self._build_edge_frame(node_key("cust", df["customer_id"]), node_key("ip", df[ip_col]), amount))
        if "counterparty_id" in df.columns:
            edge_frames.append(self._build_edge_frame(node_key("acct", df["account_id"]), node_key("cp", df["counterparty_id"]), amount))
        if "merchant_id" in df.columns:
            edge_frames.append(self._build_edge_frame(node_key("cust", df["customer_id"]), node_key("mch", df["merchant_id"]), amount))

        edge_frames = [frame for frame in edge_frames if frame is not None and not frame.empty]
        if not edge_frames:
            return graph, ip_col

        edge_df = pd.concat(edge_frames, ignore_index=True)
        if len(edge_frames) > 1:
            edge_df = edge_df.groupby(["u", "v"], as_index=False, sort=False)["weight"].sum()

        graph.add_weighted_edges_from(edge_df.itertuples(index=False, name=None))

        return graph, ip_col

    def model1_graph_analytics(self, df):
        print_step("STEP 8: MODEL1 GRAPH ANALYTICS")

        if df.empty:
            return df.copy(), pd.DataFrame()

        sample = self._pick_sample(df, "graph_sample_size")
        graph, ip_col = self._build_graph(sample)
        if graph.number_of_nodes() == 0:
            return df.copy(), pd.DataFrame()

        degree_map = dict(graph.degree())
        denom = max(graph.number_of_nodes() - 1, 1)
        degree_cent = {node: degree / denom for node, degree in degree_map.items()}
        weighted_degree = dict(graph.degree(weight="weight"))
        clustering = nx.clustering(graph) if graph.number_of_nodes() <= self.CLUSTERING_NODE_LIMIT else {}
        if graph.number_of_nodes() <= self.PAGERANK_NODE_LIMIT:
            try:
                pagerank = nx.pagerank(graph, alpha=0.85, max_iter=25, tol=1.0e-04)
            except Exception:
                pagerank = {}
        else:
            pagerank = {}
        if not pagerank:
            total_weight = sum(weighted_degree.values()) or 1.0
            pagerank = {node: float(weighted_degree.get(node, 0.0) / total_weight) for node in graph.nodes()}
        core_number = nx.core_number(graph) if graph.number_of_nodes() <= self.CORE_NUMBER_NODE_LIMIT and graph.number_of_edges() > 0 else {}

        community_map = {}
        component_size_map = {}
        component_edge_map = {}
        component_density_map = {}
        cycle_nodes = set()
        for component_id, component in enumerate(nx.connected_components(graph)):
            subgraph = graph.subgraph(component)
            component_size = len(component)
            component_edges = subgraph.number_of_edges()
            component_density = float(nx.density(subgraph)) if component_size > 1 else 0.0
            for node in component:
                community_map[node] = component_id
                component_size_map[node] = component_size
                component_edge_map[node] = component_edges
                component_density_map[node] = component_density
            if component_edges >= component_size:
                cycle_nodes.update(component)

        nodes = list(graph.nodes())
        node_types = [self._node_type(node) for node in nodes]
        degree_values = [graph.degree(node) for node in nodes]
        max_degree = max(degree_values) if degree_values else 1
        graph_features = pd.DataFrame(
            {
                "node_id": nodes,
                "graph_node_type": node_types,
                "graph_degree_centrality": [degree_cent.get(node, 0.0) for node in nodes],
                "graph_clustering": [clustering.get(node, component_density_map.get(node, 0.0)) for node in nodes],
                "graph_pagerank": [pagerank.get(node, 0.0) for node in nodes],
                "graph_community_id": [community_map.get(node, -1) for node in nodes],
                "graph_cycle_flag": [int(node in cycle_nodes) for node in nodes],
                "graph_component_size": [component_size_map.get(node, 1) for node in nodes],
                "graph_component_edges": [component_edge_map.get(node, 0) for node in nodes],
                "graph_component_density": [component_density_map.get(node, 0.0) for node in nodes],
                "graph_core_number": [core_number.get(node, 0) for node in nodes],
                "graph_degree_rank": [round((degree_map.get(node, 0) / max_degree), 4) if max_degree else 0.0 for node in nodes],
            }
        )

        out = df.copy().reset_index(drop=True)
        graph_index = graph_features.set_index("node_id")
        merged_frames = []

        for prefix, raw_col in [
            ("cust", "customer_id"),
            ("acct", "account_id"),
            ("dev", "device_id"),
            ("ip", ip_col),
            ("cp", "counterparty_id"),
            ("mch", "merchant_id"),
        ]:
            if raw_col not in out.columns:
                continue
            node_col = f"{prefix}_node"
            out[node_col] = prefix + "::" + out[raw_col].astype(str)
            merged = graph_index.reindex(out[node_col]).reset_index(drop=True).add_prefix(f"{prefix}_")
            merged_frames.append(merged)

        if merged_frames:
            out = pd.concat([out] + merged_frames, axis=1)

        return out, graph_features

    def model2_ring_detection(self, df):
        print_step("STEP 9: MODEL2 RING DETECTION")

        if df.empty:
            out = df.copy()
            out["ring_count"] = 0
            out["ring_max_risk_score"] = 0.0
            out["ring_max_member_count"] = 0
            return out, pd.DataFrame()

        sample = self._pick_sample(df, "ring_sample_size")
        graph = nx.Graph()
        ring_sample = sample[["account_id", "counterparty_id", "amount"]].dropna(subset=["account_id", "counterparty_id"]).copy()
        if ring_sample.empty:
            out = df.copy()
            out["ring_count"] = 0
            out["ring_max_risk_score"] = 0.0
            out["ring_max_member_count"] = 0
            return out, pd.DataFrame()

        ring_sample["account_id"] = ring_sample["account_id"].astype(str)
        ring_sample["counterparty_id"] = ring_sample["counterparty_id"].astype(str)
        ring_sample["amount"] = pd.to_numeric(ring_sample["amount"], errors="coerce").fillna(0.0)
        ring_edges = ring_sample.groupby(["account_id", "counterparty_id"], as_index=False, sort=False)["amount"].sum()
        graph.add_weighted_edges_from(ring_edges.itertuples(index=False, name=None))

        candidates = []
        max_rings = int(CONFIG.get("max_rings", 25))
        for component in nx.connected_components(graph):
            if len(component) < 3:
                continue
            subgraph = graph.subgraph(component)
            if subgraph.number_of_edges() < len(component):
                continue

            cycles = nx.cycle_basis(subgraph) if subgraph.number_of_nodes() <= self.CYCLE_BASIS_NODE_LIMIT else []
            if not cycles and subgraph.number_of_edges() < len(component) + 1:
                continue

            member_list = sorted(component)
            cycle_path = cycles[0] if cycles else member_list[: min(6, len(member_list))]
            total_amount = sum(data.get("weight", 0.0) for _, _, data in subgraph.edges(data=True))
            density = nx.density(subgraph)
            risk_score = float(
                min(
                    0.99,
                    0.25
                    + min(0.25, len(member_list) / 40.0)
                    + min(0.20, density)
                    + min(0.29, np.log1p(total_amount) / 25.0),
                )
            )
            candidates.append(
                {
                    "ring_id": f"RING_{len(candidates) + 1:03d}",
                    "ring_member_count": len(member_list),
                    "ring_edge_count": subgraph.number_of_edges(),
                    "ring_density": round(float(density), 4),
                    "ring_total_amount": round(float(total_amount), 2),
                    "ring_avg_edge_weight": round(float(total_amount / max(subgraph.number_of_edges(), 1)), 2),
                    "ring_risk_score": round(risk_score, 4),
                    "ring_path_signature": "->".join(cycle_path),
                    "ring_members": member_list,
                    "ring_type": "cycle",
                }
            )

        if not candidates:
            fallback_groups = (
                sample.groupby("counterparty_id")
                .agg(
                    ring_member_count=("account_id", "nunique"),
                    ring_total_amount=("amount", "sum"),
                    ring_edge_count=("account_id", "count"),
                )
                .reset_index()
            )
            fallback_groups = fallback_groups[fallback_groups["ring_member_count"] >= 3].sort_values(
                ["ring_member_count", "ring_total_amount"], ascending=[False, False]
            ).head(max_rings)

            for _, row in fallback_groups.iterrows():
                cp_id = row["counterparty_id"]
                members = sorted(set(sample.loc[sample["counterparty_id"] == cp_id, "account_id"].astype(str).head(10))) + [str(cp_id)]
                density = min(1.0, row["ring_edge_count"] / max(row["ring_member_count"], 1))
                risk_score = float(min(0.92, 0.30 + min(0.32, row["ring_member_count"] / 18.0) + min(0.25, np.log1p(row["ring_total_amount"]) / 22.0)))
                candidates.append(
                    {
                        "ring_id": f"HUB_{len(candidates) + 1:03d}",
                        "ring_member_count": int(row["ring_member_count"] + 1),
                        "ring_edge_count": int(row["ring_edge_count"]),
                        "ring_density": round(float(density), 4),
                        "ring_total_amount": round(float(row["ring_total_amount"]), 2),
                        "ring_avg_edge_weight": round(float(row["ring_total_amount"] / max(row["ring_edge_count"], 1)), 2),
                        "ring_risk_score": round(risk_score, 4),
                        "ring_path_signature": "->".join(members[:6]),
                        "ring_members": members,
                        "ring_type": "hub",
                    }
                )

        if candidates:
            ring_df = pd.DataFrame(candidates).sort_values(
                ["ring_risk_score", "ring_member_count"], ascending=[False, False]
            ).head(max_rings)
        else:
            ring_df = pd.DataFrame(
                columns=[
                    "ring_id",
                    "ring_member_count",
                    "ring_edge_count",
                    "ring_density",
                    "ring_total_amount",
                    "ring_avg_edge_weight",
                    "ring_risk_score",
                    "ring_path_signature",
                    "ring_members",
                    "ring_type",
                ]
            )

        out = df.copy()
        out["ring_count"] = 0
        out["ring_max_risk_score"] = 0.0
        out["ring_max_member_count"] = 0

        if ring_df.empty:
            return out, ring_df

        member_to_ring_ids = {}
        ring_stats = {}
        for ring_idx, ring in enumerate(ring_df.itertuples(index=False)):
            members = set(ring.ring_members)
            ring_stats[ring_idx] = {
                "risk_score": float(ring.ring_risk_score),
                "member_count": int(ring.ring_member_count),
            }
            for member in members:
                member_to_ring_ids.setdefault(str(member), set()).add(ring_idx)

        account_sets = out["account_id"].astype(str).map(lambda value: member_to_ring_ids.get(value, set()))
        counterparty_sets = out["counterparty_id"].astype(str).map(lambda value: member_to_ring_ids.get(value, set()))
        matched_ring_sets = [acct_set | cp_set for acct_set, cp_set in zip(account_sets, counterparty_sets)]

        out["ring_count"] = [len(ring_ids) for ring_ids in matched_ring_sets]
        out["ring_max_risk_score"] = [
            max((ring_stats[ring_id]["risk_score"] for ring_id in ring_ids), default=0.0)
            for ring_ids in matched_ring_sets
        ]
        out["ring_max_member_count"] = [
            max((ring_stats[ring_id]["member_count"] for ring_id in ring_ids), default=0)
            for ring_ids in matched_ring_sets
        ]

        if "ring_members" in ring_df.columns:
            ring_df = ring_df.copy()
            ring_df["ring_members"] = ring_df["ring_members"].apply(lambda members: ", ".join(members[:12]))

        return out, ring_df.reset_index(drop=True)
