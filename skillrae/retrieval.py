import json
import collections
import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Configuration
REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT
MODEL_NAME = os.environ.get("SKILLSBENCH_EMBED_MODEL", "BAAI/bge-small-en")
HUB_THRESHOLD = 15
TOP_N_SUBUNITS = 30
TOP_N_L2_CLUSTERS = 2
TOP_SELECTED_SUBUNITS_PER_SKILL = 3
RESCUE_CANDIDATE_POOL_MULTIPLIER = 4
MIN_RESCUE_CANDIDATE_POOL = 10
REQUIRED_RETRIEVAL_ARTIFACTS = (
    "skill_nodes.json",
    "subunit_nodes.json",
    "edges.json",
    "canonical_skill_representations.json",
    "skill_l2_mapping.json",
    "subunit_ids.json",
    "subunit_embeddings.npy",
)


def _local_snapshot_candidates(model_name: str):
    """Return likely local HF snapshot directories for a model."""
    candidates = []
    if "/" not in model_name:
        return candidates

    org, model = model_name.split("/", 1)
    repo_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org}--{model}"
    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.exists():
        # Prefer latest snapshot first.
        snapshot_dirs = sorted(
            [p for p in snapshots_dir.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(snapshot_dirs)
    return candidates


def resolve_data_dir(data_dir=None) -> Path:
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()

    configured_data_dir = os.environ.get("SKILLSBENCH_RETRIEVAL_DATA_DIR")
    if configured_data_dir:
        return Path(configured_data_dir).expanduser().resolve()

    configured_repo_root = os.environ.get("SKILLSBENCH_REPO_ROOT")
    if configured_repo_root:
        return Path(configured_repo_root).expanduser().resolve()

    return DATA_DIR


def ensure_required_retrieval_artifacts(data_dir: Path) -> None:
    missing = [name for name in REQUIRED_RETRIEVAL_ARTIFACTS if not (data_dir / name).exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Retrieval artifacts missing under {data_dir}: {missing_list}"
        )

class SkillRetriever:
    def __init__(self, data_dir=None):
        self.data_dir = resolve_data_dir(data_dir)
        ensure_required_retrieval_artifacts(self.data_dir)
        self.skill_nodes = {}
        self.skill_name_to_id = {}
        self.subunit_nodes = {}
        self.skill_to_subunits = collections.defaultdict(list)
        self.subunit_to_skills = collections.defaultdict(list)

        # L2 (Capability) Data
        self.l2_clusters = {}
        self.skill_to_l2 = {}
        self.l2_ids_in_order = []
        self.l2_texts = []
        self.l2_embeddings = None

        # L1 (Skill) Data
        self.skill_texts = []
        self.skill_name_texts = [] # For Semantic Prior
        self.skill_ids_in_order = []
        self.skill_representations = {}
        self.skill_embeddings_l1 = None
        self.skill_name_embeddings = None # For Semantic Prior

        # L0 (Subunit) Data
        self.subunit_ids = []
        self.subunit_embeddings = None

        self.model = None

        # Pipeline
        self.load_data()
        self.load_skill_representations()
        self.load_l2_mapping() # Load L2
        self.load_embeddings() # Load L0
        self.load_model()      # Load Model & Build L1/L2/Prior
        self.validate_loaded_state()

    def validate_loaded_state(self):
        if self.model is None:
            raise RuntimeError(f"Embedding model failed to load for retrieval data dir {self.data_dir}.")
        if not self.skill_nodes:
            raise RuntimeError(f"No skills were loaded from {self.data_dir}.")
        if not self.skill_representations:
            raise RuntimeError(f"No skill representations were loaded from {self.data_dir}.")
        if not self.skill_ids_in_order:
            raise RuntimeError(f"No ranked skill ids were prepared from {self.data_dir}.")
        if self.skill_embeddings_l1 is None or self.skill_name_embeddings is None:
            raise RuntimeError(f"Skill embeddings were not built for retrieval data dir {self.data_dir}.")
        if self.subunit_embeddings is None or not self.subunit_ids:
            raise RuntimeError(f"Subunit embeddings were not loaded from {self.data_dir}.")

    def load_data(self):
        """
        Load graph nodes and edges.
        """
        try:
            # 1. Load Nodes & Edges
            with open(self.data_dir / "skill_nodes.json", "r") as f:
                skills = json.load(f)
                for s in skills:
                    self.skill_nodes[s["id"]] = s["name"]
                    self.skill_name_to_id[s["name"]] = s["id"]

            with open(self.data_dir / "subunit_nodes.json", "r") as f:
                subunits = json.load(f)
                for s in subunits:
                    self.subunit_nodes[s["id"]] = s["text"]

            with open(self.data_dir / "edges.json", "r") as f:
                edges = json.load(f)
                for e in edges:
                    sid = e["skill_id"]
                    uid = e["subunit_id"]
                    self.skill_to_subunits[sid].append(uid)
                    self.subunit_to_skills[uid].append(sid)

            print(f"Loaded {len(self.skill_nodes)} skills, {len(self.subunit_nodes)} subunits.")

        except FileNotFoundError as e:
            print(f"Error loading graph data: {e}")

    def load_skill_representations(self):
        """
        Load deterministic skill representations.
        """
        try:
            with open(self.data_dir / "canonical_skill_representations.json", "r") as f:
                self.skill_representations = json.load(f)
            print(f"Loaded {len(self.skill_representations)} skill representations.")
        except FileNotFoundError:
            print("Warning: canonical_skill_representations.json not found.")

    def load_l2_mapping(self):
        """
        Load L2 capability clusters.
        """
        try:
            with open(self.data_dir / "skill_l2_mapping.json", "r") as f:
                self.l2_clusters = json.load(f)

            # Build skill -> l2 mapping
            for l2_id, data in self.l2_clusters.items():
                for sid in data["skills"]:
                    self.skill_to_l2[sid] = l2_id

            print(f"Loaded {len(self.l2_clusters)} L2 clusters.")

        except FileNotFoundError:
            print("Warning: skill_l2_mapping.json not found.")

    def load_embeddings(self):
        """
        Load precomputed L0 subunit embeddings.
        """
        try:
            with open(self.data_dir / "subunit_ids.json", "r") as f:
                self.subunit_ids = json.load(f)

            self.subunit_embeddings = np.load(self.data_dir / "subunit_embeddings.npy")
            print(f"Loaded L0 embeddings: {self.subunit_embeddings.shape}")

        except FileNotFoundError as e:
            print(f"Error loading L0 embeddings: {e}")

    def load_model(self):
        """
        Load model and compute embeddings on the fly.
        """
        print(f"Loading embedding model: {MODEL_NAME}...")
        try:
            model_path_env = os.environ.get("SKILLSBENCH_EMBED_MODEL_PATH")
            load_errors = []

            # Strategy:
            # 1) Explicit local path from env
            # 2) Auto-detected local HF snapshots
            # 3) Model ID online (last resort)
            if model_path_env:
                model_path = Path(model_path_env).expanduser()
                self.model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
            else:
                self.model = None
                for local_path in _local_snapshot_candidates(MODEL_NAME):
                    try:
                        self.model = SentenceTransformer(str(local_path), device="cpu", local_files_only=True)
                        print(f"Loaded local embedding snapshot: {local_path}")
                        break
                    except Exception as e:
                        load_errors.append(f"{local_path}: {e}")

                if self.model is None:
                    try:
                        self.model = SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
                        print("Loaded from local HF cache by model id.")
                    except Exception as e:
                        load_errors.append(f"{MODEL_NAME} (local cache): {e}")
                        self.model = SentenceTransformer(MODEL_NAME, device="cpu")

            # 1. Build L1 Skill Texts
            self.skill_texts = []
            self.skill_name_texts = []
            self.skill_ids_in_order = []

            sorted_sids = sorted(self.skill_representations.keys())

            for sid in sorted_sids:
                name = self.skill_nodes[sid]
                rep = self.skill_representations[sid]

                self.skill_texts.append(rep)
                self.skill_name_texts.append(name)
                self.skill_ids_in_order.append(sid)

            if self.skill_texts:
                print(f"Encoding {len(self.skill_texts)} skills (L1)...")
                self.skill_embeddings_l1 = self.model.encode(
                    self.skill_texts,
                    batch_size=32,
                    show_progress_bar=True,
                    normalize_embeddings=True
                )

                print(f"Encoding {len(self.skill_name_texts)} skills (Prior)...")
                self.skill_name_embeddings = self.model.encode(
                    self.skill_name_texts,
                    batch_size=32,
                    show_progress_bar=True,
                    normalize_embeddings=True
                )

            # 2. Build L2 Capability Texts (Enhanced)
            self.l2_ids_in_order = []
            self.l2_texts = []

            sorted_l2_ids = sorted(self.l2_clusters.keys())

            for l2_id in sorted_l2_ids:
                label = self.l2_clusters[l2_id]["label"]
                # Get example skills (up to 5)
                example_sids = self.l2_clusters[l2_id]["skills"][:5]
                example_names = [self.skill_nodes.get(sid, sid) for sid in example_sids]
                example_str = ", ".join(example_names)

                # Construct enhanced L2 text
                l2_text = f"{label}. Example skills: {example_str}"

                self.l2_ids_in_order.append(l2_id)
                self.l2_texts.append(l2_text)

            if self.l2_texts:
                print(f"Encoding {len(self.l2_texts)} L2 clusters (Enhanced)...")
                self.l2_embeddings = self.model.encode(
                    self.l2_texts,
                    batch_size=32,
                    show_progress_bar=True,
                    normalize_embeddings=True
                )

        except Exception as e:
            print(f"Error loading model: {e}")
            if "load_errors" in locals() and load_errors:
                print("Local loading attempts:")
                for err in load_errors:
                    print(f"- {err}")

    def _normalize_scores(self, score_dict):
        """
        Min-Max normalize a dictionary of scores to [0, 1].
        """
        if not score_dict:
            return {}

        vals = list(score_dict.values())
        min_v, max_v = min(vals), max(vals)

        if max_v == min_v:
            return {k: 0.5 for k in score_dict}

        return {k: (v - min_v) / (max_v - min_v) for k, v in score_dict.items()}

    def retrieve_with_metadata(self, task_description, k=5):
        """Hierarchical retrieval with exportable selected-skill metadata."""
        if self.model is None:
            return [], {}

        task_embedding = self.model.encode([task_description], normalize_embeddings=True)[0]

        # ---------------------------------------------------------
        # Step 1: L2 Cluster Selection (Soft Boost)
        # ---------------------------------------------------------
        selected_l2_labels = []
        boosted_skills = set()

        if self.l2_embeddings is not None:
            # (N_l2, D) @ (D,) -> (N_l2,)
            l2_sims = self.l2_embeddings @ task_embedding

            # Select Top-K Clusters
            top_l2_indices = np.argsort(l2_sims)[-TOP_N_L2_CLUSTERS:][::-1]

            for idx in top_l2_indices:
                l2_id = self.l2_ids_in_order[idx]
                cluster_data = self.l2_clusters[l2_id]

                selected_l2_labels.append(cluster_data["label"])
                boosted_skills.update(cluster_data["skills"])

        # ---------------------------------------------------------
        # Channel B (L1 Path): Representation Matching
        # ---------------------------------------------------------
        l1_norm = {}
        if self.skill_embeddings_l1 is not None:
            l1_raw_scores = self.skill_embeddings_l1 @ task_embedding
            l1_scores_map = {}
            for sid, score in zip(self.skill_ids_in_order, l1_raw_scores):
                l1_scores_map[sid] = float(score)
            l1_norm = self._normalize_scores(l1_scores_map)

        # ---------------------------------------------------------
        # Semantic Prior Boost (Name Matching)
        # ---------------------------------------------------------
        prior_norm = {}
        if self.skill_name_embeddings is not None:
            prior_raw_scores = self.skill_name_embeddings @ task_embedding
            prior_scores_map = {}
            for sid, score in zip(self.skill_ids_in_order, prior_raw_scores):
                prior_scores_map[sid] = float(score)
            prior_norm = self._normalize_scores(prior_scores_map)

        # ---------------------------------------------------------
        # Channel A (L0 Path): Subunit Graph Projection
        # ---------------------------------------------------------
        l0_norm = {}
        top_subunits_by_skill = collections.defaultdict(list)
        if self.subunit_embeddings is not None:
            l0_scores_map = collections.defaultdict(float)
            subunit_sims = self.subunit_embeddings @ task_embedding
            top_indices = np.argsort(subunit_sims)[-TOP_N_SUBUNITS:][::-1]

            for idx in top_indices:
                uid = self.subunit_ids[idx]
                sim = subunit_sims[idx]

                connected_skills = self.subunit_to_skills.get(uid, [])
                degree = len(connected_skills)

                if degree > HUB_THRESHOLD:
                    continue

                inc = sim / degree if degree > 0 else 0
                for sid in connected_skills:
                    l0_scores_map[sid] += inc
                    top_subunits_by_skill[sid].append(
                        {
                            "subunit_id": uid,
                            "graph_skill_id": sid,
                            "source_skill_id": self.skill_nodes.get(sid, sid),
                            "subunit_text": self.subunit_nodes.get(uid, ""),
                            "subunit_score": round(float(inc), 4),
                            "subunit_similarity": round(float(sim), 4),
                        }
                    )

            l0_norm = self._normalize_scores(l0_scores_map)

        # ---------------------------------------------------------
        # Fusion (All Skills) with Soft L2 Boost
        # ---------------------------------------------------------
        final_scores = []

        for sid in self.skill_ids_in_order:
            s1 = l1_norm.get(sid, 0.0)
            s0 = l0_norm.get(sid, 0.0)
            s_prior = prior_norm.get(sid, 0.0)

            base_score = 0.6 * s1 + 0.4 * s0 + 0.15 * s_prior

            # Apply L2 Boost
            is_boosted = False
            if sid in boosted_skills:
                base_score *= 1.10
                is_boosted = True

            final_scores.append((sid, base_score, is_boosted))

        ranked = sorted(final_scores, key=lambda x: x[1], reverse=True)

        results = []
        selected_skill_records = []
        rescue_candidate_skill_records = []
        for sid, score, boosted in ranked[:k]:
            source_skill_id = self.skill_nodes.get(sid, "Unknown")
            deduped_subunits = []
            seen_subunits = set()
            for subunit in sorted(
                top_subunits_by_skill.get(sid, []),
                key=lambda item: (item["subunit_score"], item["subunit_similarity"]),
                reverse=True,
            ):
                subunit_id = subunit["subunit_id"]
                if subunit_id in seen_subunits:
                    continue
                seen_subunits.add(subunit_id)
                deduped_subunits.append(subunit)
                if len(deduped_subunits) >= TOP_SELECTED_SUBUNITS_PER_SKILL:
                    break
            results.append({
                "skill_name": source_skill_id,
                "score": round(score, 4),
                "l2_clusters": selected_l2_labels,
                "debug": f"(L1: {l1_norm.get(sid,0):.2f}, L0: {l0_norm.get(sid,0):.2f}, Prior: {prior_norm.get(sid,0):.2f}, L2_boost: {'Yes' if boosted else 'No'})"
            })
            selected_skill_records.append(
                {
                    "skill_id": source_skill_id,
                    "graph_skill_id": sid,
                    "final_score": round(float(score), 4),
                    "l1_score": round(float(l1_norm.get(sid, 0.0)), 4),
                    "l0_score": round(float(l0_norm.get(sid, 0.0)), 4),
                    "prior_score": round(float(prior_norm.get(sid, 0.0)), 4),
                    "l2_boosted": boosted,
                    "l2_clusters": list(selected_l2_labels),
                    "top_subunits": deduped_subunits,
                }
            )

        rescue_candidate_pool_size = min(
            len(ranked),
            max(MIN_RESCUE_CANDIDATE_POOL, k * RESCUE_CANDIDATE_POOL_MULTIPLIER),
        )
        selected_skill_id_set = {record["skill_id"] for record in selected_skill_records}
        for sid, score, boosted in ranked[:rescue_candidate_pool_size]:
            source_skill_id = self.skill_nodes.get(sid, "Unknown")
            if source_skill_id in selected_skill_id_set:
                continue
            deduped_subunits = []
            seen_subunits = set()
            for subunit in sorted(
                top_subunits_by_skill.get(sid, []),
                key=lambda item: (item["subunit_score"], item["subunit_similarity"]),
                reverse=True,
            ):
                subunit_id = subunit["subunit_id"]
                if subunit_id in seen_subunits:
                    continue
                seen_subunits.add(subunit_id)
                deduped_subunits.append(subunit)
                if len(deduped_subunits) >= TOP_SELECTED_SUBUNITS_PER_SKILL:
                    break
            if not deduped_subunits:
                continue
            rescue_candidate_skill_records.append(
                {
                    "skill_id": source_skill_id,
                    "graph_skill_id": sid,
                    "final_score": round(float(score), 4),
                    "l1_score": round(float(l1_norm.get(sid, 0.0)), 4),
                    "l0_score": round(float(l0_norm.get(sid, 0.0)), 4),
                    "prior_score": round(float(prior_norm.get(sid, 0.0)), 4),
                    "l2_boosted": boosted,
                    "l2_clusters": list(selected_l2_labels),
                    "top_subunits": deduped_subunits,
                }
            )

        metadata = {
            "retrieval_mode": "topk",
            "selected_skill_records": selected_skill_records,
            "rescue_candidate_skill_records": rescue_candidate_skill_records,
            "selected_skill_ids": [record["skill_id"] for record in selected_skill_records],
            "top_n_subunits_considered": TOP_N_SUBUNITS,
            "top_selected_subunits_per_skill": TOP_SELECTED_SUBUNITS_PER_SKILL,
            "rescue_candidate_pool_size": rescue_candidate_pool_size,
        }
        return results, metadata

    def retrieve(self, task_description, k=5):
        """
        Hierarchical Retrieval: L2 Soft Boost + L1/L0/Prior Fusion.
        """
        rows, _ = self.retrieve_with_metadata(task_description, k=k)
        return rows

def main():
    retriever = SkillRetriever()

    # Mock Tasks
    tasks = [
        "I need to analyze a large CSV file using pandas and visualize the results with matplotlib.",
        "Please help me set up a git repository and commit my changes.",
        "I need to extract text from a PDF document and perform OCR on images."
    ]

    print("\n" + "="*40)
    print("Hierarchical Retrieval (Soft Boost) Sanity Check")
    print("="*40)

    for i, task in enumerate(tasks):
        print(f"\nTask {i+1}: {task}")
        print("-" * 20)

        results = retriever.retrieve(task, k=5)

        if not results:
            print("No relevant skills found.")
        else:
            # Print L2 info once
            if results:
                print(f"Selected L2 Clusters: {results[0]['l2_clusters']}")

            for rank, res in enumerate(results):
                print(f"{rank+1}. {res['skill_name']} (Score: {res['score']})")
                # print(f"    Debug: {res['debug']}")

if __name__ == "__main__":
    main()
