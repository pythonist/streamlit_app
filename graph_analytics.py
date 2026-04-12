import numpy as np
import pandas as pd
import networkx as nx

from config import CONFIG
from utils import print_step


class GraphAnalytics:
    def _pick_sample(self, df, sample_key):
        sample_size = min(int(CONFIG.get(sample_key, len(df))), len(df))
        if sample_size <= 0:
            return df.head(0).copy()
        if sample_size == len(df):
            return df.copy()
        return df.sample(sample_size, random_state=CONFIG["random_state"]).copy()

    def _ip_column(self, df):
        return "device_ip_address" if "device_ip_address" in df.columns else "ip_address"

    def _build_graph(self, df):
        graph = nx.Graph()
        ip_col = self._ip_column(df)

        def node_key(prefix, series):
            return prefix + "::" + series.astype(str)

        def add_edges(left, right, weights):
            for u, v, weight in zip(left, right, weights):
                if pd.isna(u) or pd.isna(v):
                    continue
                if graph.has_edge(u, v):
                    graph[u][v]["weight"] += float(weight)
                else:
                    graph.add_edge(u, v, weight=float(weight))

        amount = df.get("amount", pd.Series(0, index=df.index)).fillna(0).astype(float)
        add_edges(node_key("cust", df["customer_id"]), node_key("acct", df["account_id"]), amount)

        if "device_id" in df.columns:
            add_edges(node_key("cust", df["customer_id"]), node_key("dev", df["device_id"]), amount)
        if ip_col in df.columns:
            add_edges(node_key("cust", df["customer_id"]), node_key("ip", df[ip_col]), amount)
        if "counterparty_id" in df.columns:
            add_edges(node_key("acct", df["account_id"]), node_key("cp", df["counterparty_id"]), amount)
        if "merchant_id" in df.columns:
            add_edges(node_key("cust", df["customer_id"]), node_key("mch", df["merchant_id"]), amount)

        return graph, ip_col

    def model1_graph_analytics(self, df):
        print_step("STEP 8: MODEL1 GRAPH ANALYTICS")

        if df.empty:
            return df.copy(), pd.DataFrame()

        sample = self._pick_sample(df, "graph_sample_size")
        graph, ip_col = self._build_graph(sample)
        if graph.number_of_nodes() == 0:
            return df.copy(), pd.DataFrame()

        degree_cent = nx.degree_centrality(graph)
        clustering = nx.clustering(graph) if graph.number_of_nodes() <= 5000 else {}
        pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100) if graph.number_of_nodes() <= 6000 else {}

        community_map = {}
        cycle_nodes = set()
        for component_id, component in enumerate(nx.connected_components(graph)):
            subgraph = graph.subgraph(component)
            for node in component:
                community_map[node] = component_id
            if subgraph.number_of_edges() >= subgraph.number_of_nodes():
                cycle_nodes.update(component)

        nodes = list(graph.nodes())
        graph_features = pd.DataFrame(
            {
                "node_id": nodes,
                "graph_degree_centrality": [degree_cent.get(node, 0.0) for node in nodes],
                "graph_clustering": [clustering.get(node, 0.0) for node in nodes],
                "graph_pagerank": [pagerank.get(node, 0.0) for node in nodes],
                "graph_community_id": [community_map.get(node, -1) for node in nodes],
                "graph_cycle_flag": [int(node in cycle_nodes) for node in nodes],
            }
        )

        out = df.copy()
        graph_index = graph_features.set_index("node_id")

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
            out = pd.concat([out.reset_index(drop=True), merged.reset_index(drop=True)], axis=1)

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
        edge_amount = {}

        for _, row in sample.iterrows():
            account_id = row.get("account_id")
            counterparty_id = row.get("counterparty_id")
            if pd.isna(account_id) or pd.isna(counterparty_id):
                continue
            u = str(account_id)
            v = str(counterparty_id)
            amount = float(row.get("amount", 0) or 0)
            graph.add_edge(u, v, weight=graph.get_edge_data(u, v, default={}).get("weight", 0.0) + amount)
            edge_amount[(u, v)] = edge_amount.get((u, v), 0.0) + amount

        candidates = []
        max_rings = int(CONFIG.get("max_rings", 25))
        for component in nx.connected_components(graph):
            if len(component) < 3:
                continue
            subgraph = graph.subgraph(component)
            if subgraph.number_of_edges() < len(component):
                continue

            cycles = nx.cycle_basis(subgraph) if subgraph.number_of_nodes() <= 150 else []
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
                    "ring_risk_score": round(risk_score, 4),
                    "ring_path_signature": "->".join(cycle_path),
                    "ring_members": member_list,
                }
            )

        ring_df = pd.DataFrame(candidates).sort_values(
            ["ring_risk_score", "ring_member_count"], ascending=[False, False]
        ).head(max_rings)

        out = df.copy()
        out["ring_count"] = 0
        out["ring_max_risk_score"] = 0.0
        out["ring_max_member_count"] = 0

        if ring_df.empty:
            return out, ring_df

        ring_memberships = []
        for _, ring in ring_df.iterrows():
            members = set(ring["ring_members"])
            ring_memberships.append(
                {
                    "members": members,
                    "risk_score": ring["ring_risk_score"],
                    "member_count": ring["ring_member_count"],
                }
            )

        counts = []
        max_scores = []
        max_sizes = []
        for _, row in out.iterrows():
            row_members = {str(row.get("account_id", "")), str(row.get("counterparty_id", ""))}
            matched = [ring for ring in ring_memberships if row_members & ring["members"]]
            counts.append(len(matched))
            max_scores.append(max([ring["risk_score"] for ring in matched], default=0.0))
            max_sizes.append(max([ring["member_count"] for ring in matched], default=0))

        out["ring_count"] = counts
        out["ring_max_risk_score"] = max_scores
        out["ring_max_member_count"] = max_sizes

        if "ring_members" in ring_df.columns:
            ring_df = ring_df.copy()
            ring_df["ring_members"] = ring_df["ring_members"].apply(lambda members: ", ".join(members[:12]))

        return out, ring_df.reset_index(drop=True)
