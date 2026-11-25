"""Configurable costs and learned state-action utility controller for Adaptive QA."""
import math
import numpy as np
from adaptive_qa.evaluation.answers import normalize
from adaptive_qa.llm.model import Usage
from adaptive_qa.retrieval.evidence import evidence_features

FEATURE_GROUPS = {
    "confidence": ("calibrated_prob", "raw_confidence", "valid_entity"),
    "consistency": ("self_consistency", "repeat_agreement", "answer_word_len"),
    "evidence": ("doc_support", "top_doc_agreement", "bm25_score_norm", "bm25_margin_norm"),
    "cost": ("action_token_cost", "lambda_scaled_cost"),
    "progress": ("fraction_revealed", "words_revealed_norm", "has_giveaway"),
}


class CostModel:
    def __init__(self, config, rewards):
        self.config, self.rewards = config, rewards

    def dollars(self, usage):
        ip, op = self.config["input_usd_per_million"], self.config["output_usd_per_million"]
        if ip is None or op is None:
            return None
        return (usage.input_tokens * ip + usage.output_tokens * op) / 1_000_000

    def cost(self, usage):
        result = sum(getattr(usage, field) * self.config[field + "_weight"] for field in
                     ("input_tokens", "output_tokens", "llm_calls", "retrieval_calls", "latency_seconds"))
        if self.config["dollars_weight"]:
            dollars = self.dollars(usage)
            if dollars is None:
                raise ValueError("Missing prices")
            result += dollars * self.config["dollars_weight"]
        return float(result)

    def reward(self, correct=None, answered=True):
        return self.rewards["abstain"] if not answered else self.rewards["correct" if correct else "incorrect"]

    def utility(self, correct, answered, usage, lambda_):
        return self.reward(correct, answered) - lambda_ * self.cost(usage)


def is_placeholder_or_echo(answer):
    """Detect non-entity demonstrative echoes (e.g. 'this novel', 'these events') from prediction text alone."""
    a = str(answer or "").strip().casefold()
    if not a:
        return True
    if a in {"short answer", "your answer", "answer here", "candidate name", "unknown", "n/a", "none"}:
        return True
    return a.startswith(("this ", "these ", "what ", "which ", "name this", "identify this"))


def valid_entity_flag(node):
    return float(bool(node.get("parse_valid")) and not is_placeholder_or_echo(node.get("prediction", "")))


def has_giveaway_clue(text):
    t = str(text or "").casefold()
    return float("for 10 points" in t or "for ten points" in t or " ftp," in t or " ftp " in t)


def enrich_root_and_child(root, child):
    """Compute pre-retrieval lexical support of root candidate against BM25 passages without extra LLM calls."""
    docs = child.get("documents", []) if child else []
    ev = evidence_features(root.get("prediction", ""), docs)
    enriched = dict(root)
    enriched["root_doc_support"] = ev["evidence_support"]
    enriched["root_top_agreement"] = ev["top_answer_agreement"]
    enriched["bm25_score"] = ev["retrieval_score"]
    enriched["bm25_margin"] = ev["retrieval_margin"]
    enriched["valid_entity"] = valid_entity_flag(root)
    return enriched


class StateCalibrator:
    """Calibrates P(correct | state) using observable state features."""
    def __init__(self, include_evidence=True):
        self.include_evidence = include_evidence
        self.weights = None
        self.mean = None
        self.scale = None

    def _features(self, node, is_retrieved=False):
        valid = valid_entity_flag(node)
        sc = float(node.get("self_consistency") or 0.0) * valid
        rep = float(node.get("repeat_agreement") if node.get("repeat_agreement") is not None else sc) * valid
        frac = float(node.get("fraction_revealed") or 0.5)
        giveaway = has_giveaway_clue(node.get("question_text", ""))
        words = len(normalize(node.get("prediction", "")).split())
        good_len = float(1 <= words <= 5) * valid
        if not self.include_evidence:
            return np.array([valid, sc, rep, frac, giveaway, good_len], dtype=float)
        if is_retrieved:
            doc_sup = float(node.get("evidence_support") or 0.0) * valid
            top_ag = float(node.get("top_answer_agreement") or 0.0) * valid
            score = min(float(node.get("retrieval_score") or 0.0) / 100.0, 1.5)
            margin = min(float(node.get("retrieval_margin") or 0.0) / 20.0, 1.5)
        else:
            doc_sup = float(node.get("root_doc_support") or 0.0) * valid
            top_ag = float(node.get("root_top_agreement") or 0.0) * valid
            score = min(float(node.get("bm25_score") or 0.0) / 100.0, 1.5)
            margin = min(float(node.get("bm25_margin") or 0.0) / 20.0, 1.5)
        return np.array([valid, sc, rep, frac, giveaway, good_len, doc_sup, top_ag, score, margin], dtype=float)

    def fit(self, nodes, is_retrieved=False, c_reg=1.0):
        X = np.vstack([self._features(n, is_retrieved=is_retrieved) for n in nodes])
        y = np.array([int(bool(n["correct"])) for n in nodes], dtype=float)
        self.mean = X.mean(axis=0)
        self.scale = np.where(X.std(axis=0) > 1e-6, X.std(axis=0), 1.0)
        Xs = (X - self.mean) / self.scale
        Xb = np.column_stack([np.ones(len(Xs)), Xs])
        d = Xb.shape[1]
        w = np.zeros(d, dtype=float)
        reg = np.diag([0.0] + [1.0 / c_reg] * (d - 1))
        for _ in range(300):
            z = np.clip(Xb @ w, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-z))
            grad = Xb.T @ (p - y) + reg @ w
            w_diag = np.maximum(p * (1.0 - p), 1e-6)
            hess = (Xb.T * w_diag) @ Xb + reg
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.max(np.abs(step)) < 1e-7:
                break
        self.weights = w
        return self

    def predict_proba(self, node, is_retrieved=False):
        if not valid_entity_flag(node):
            return 0.01
        x = self._features(node, is_retrieved=is_retrieved)
        xs = (x - self.mean) / self.scale
        xb = np.concatenate([[1.0], xs])
        z = float(np.clip(xb @ self.weights, -30.0, 30.0))
        return float(1.0 / (1.0 + math.exp(-z)))


def raw_confidence_proxy(node):
    """Uncalibrated confidence proxy when verbal_confidence is omitted (consistency + prefix fraction)."""
    if node.get("verbal_confidence") is not None:
        return float(node["verbal_confidence"])
    if not node.get("parse_valid"):
        return 0.0
    sc = float(node.get("self_consistency") or 0.0)
    frac = float(node.get("fraction_revealed") or 0.5)
    return float(np.clip(0.65 * sc + 0.35 * frac, 0.0, 1.0))


class LearnedUtilityController:
    """Supervised state-action utility predictor U(s, a) = R(s, a) - lambda * C(s, a) with ablation support."""
    def __init__(self, dropped_group=None, ridge_alpha=2.0):
        if dropped_group is not None and dropped_group not in FEATURE_GROUPS:
            raise ValueError(f"Unknown feature group: {dropped_group}")
        self.dropped_group = dropped_group
        self.ridge_alpha = ridge_alpha
        self.root_cal = StateCalibrator(include_evidence=(dropped_group != "evidence"))
        self.ret_cal = StateCalibrator(include_evidence=(dropped_group != "evidence"))
        self.action_models = {}
        self.avg_ret_cost = 0.75

    def _state_feature_dict(self, root, lambda_, action, is_post_retrieve=False, ret_node=None):
        node = ret_node if is_post_retrieve else root
        valid = valid_entity_flag(node)
        raw_conf = raw_confidence_proxy(node)
        cal_prob = (
            self.ret_cal.predict_proba(ret_node, is_retrieved=True)
            if is_post_retrieve
            else self.root_cal.predict_proba(root, is_retrieved=False)
        )
        sc = float(node.get("self_consistency") or 0.0)
        rep = float(node.get("repeat_agreement") if node.get("repeat_agreement") is not None else sc)
        words = len(normalize(node.get("prediction", "")).split())
        word_len = float(1 <= words <= 5)
        if is_post_retrieve:
            doc_sup = float(ret_node.get("evidence_support") or 0.0)
            top_ag = float(ret_node.get("top_answer_agreement") or 0.0)
            b_score = min(float(ret_node.get("retrieval_score") or 0.0) / 100.0, 1.5)
            b_margin = min(float(ret_node.get("retrieval_margin") or 0.0) / 20.0, 1.5)
        else:
            doc_sup = float(root.get("root_doc_support") or 0.0)
            top_ag = float(root.get("root_top_agreement") or 0.0)
            b_score = min(float(root.get("bm25_score") or 0.0) / 100.0, 1.5)
            b_margin = min(float(root.get("bm25_margin") or 0.0) / 20.0, 1.5)
        add_cost = self.avg_ret_cost if action == "RETRIEVE" else 0.0
        feats = {
            "calibrated_prob": cal_prob,
            "raw_confidence": raw_conf,
            "valid_entity": valid,
            "self_consistency": sc,
            "repeat_agreement": rep,
            "answer_word_len": word_len,
            "doc_support": doc_sup,
            "top_doc_agreement": top_ag,
            "bm25_score_norm": b_score,
            "bm25_margin_norm": b_margin,
            "action_token_cost": add_cost,
            "lambda_scaled_cost": lambda_ * add_cost,
            "fraction_revealed": float(root.get("fraction_revealed") or 0.5),
            "words_revealed_norm": min(float(root.get("words_revealed") or 50) / 150.0, 1.5),
            "has_giveaway": has_giveaway_clue(root.get("question_text", "")),
        }
        if self.dropped_group:
            for k in FEATURE_GROUPS[self.dropped_group]:
                feats[k] = 0.0
        return feats

    def _vector(self, feats):
        keys = [k for group in FEATURE_GROUPS.values() for k in group]
        return np.array([feats[k] for k in keys], dtype=float)

    def fit(self, root_pairs, cost_model, lambdas=(0.0, 0.05, 0.1, 0.2, 0.4, 0.8)):
        """Fit calibrators and state-action utility regressors on training/calibration pairs."""
        enriched_roots = [enrich_root_and_child(r, c) for r, c in root_pairs]
        ret_nodes = [c for _, c in root_pairs]
        self.root_cal.fit(enriched_roots, is_retrieved=False)
        self.ret_cal.fit(ret_nodes, is_retrieved=True)
        costs = [cost_model.cost(Usage(**c["incremental_usage"])) for _, c in root_pairs]
        self.avg_ret_cost = float(np.mean(costs)) if costs else 0.75

        for action in ("ANSWER", "RETRIEVE"):
            X_rows, y_rows = [], []
            for r, c in zip(enriched_roots, ret_nodes):
                for lam in lambdas:
                    feats = self._state_feature_dict(r, lam, action, is_post_retrieve=False)
                    X_rows.append(self._vector(feats))
                    if action == "ANSWER":
                        u = cost_model.reward(r["correct"], answered=True) if valid_entity_flag(r) else cost_model.reward(False, answered=True)
                    else:
                        u_ans = cost_model.utility(c["correct"], True, Usage(**c["incremental_usage"]), lam)
                        u_abs = cost_model.utility(False, False, Usage(**c["incremental_usage"]), lam)
                        u = u_ans if valid_entity_flag(c) and c.get("evidence_support", 0) > 0 else max(u_ans, u_abs)
                    y_rows.append(u)
            X = np.vstack(X_rows)
            y = np.array(y_rows, dtype=float)
            mean = X.mean(axis=0)
            scale = np.where(X.std(axis=0) > 1e-6, X.std(axis=0), 1.0)
            Xs = (X - mean) / scale
            Xb = np.column_stack([np.ones(len(Xs)), Xs])
            reg = np.diag([0.0] + [self.ridge_alpha] * Xs.shape[1])
            w = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
            self.action_models[action] = (w, mean, scale)
        return self

    def predict_utilities(self, root, child, cost_model, lambda_):
        r = enrich_root_and_child(root, child)
        r_valid = bool(valid_entity_flag(r))
        r_prob = (
            self.root_cal.predict_proba(r, is_retrieved=False)
            if self.dropped_group != "confidence"
            else raw_confidence_proxy(r)
        )
        r_correct = cost_model.reward(True, answered=True)
        r_wrong = cost_model.reward(False, answered=True)
        r_abstain = cost_model.reward(None, answered=False)

        analytical_ans = (r_prob * r_correct + (1.0 - r_prob) * r_wrong) if r_valid else r_wrong
        w_a, m_a, s_a = self.action_models["ANSWER"]
        v_a = (self._vector(self._state_feature_dict(r, lambda_, "ANSWER")) - m_a) / s_a
        ridge_ans = float(np.concatenate([[1.0], v_a]) @ w_a)
        u_answer = 0.65 * analytical_ans + 0.35 * ridge_ans if r_valid else r_wrong

        w_r, m_r, s_r = self.action_models["RETRIEVE"]
        v_r = (self._vector(self._state_feature_dict(r, lambda_, "RETRIEVE")) - m_r) / s_r
        u_retrieve = float(np.concatenate([[1.0], v_r]) @ w_r)
        if self.dropped_group != "cost":
            u_retrieve = u_retrieve - 0.35 * lambda_ * self.avg_ret_cost

        if self.dropped_group != "evidence" and r["root_doc_support"] > 0 and r.get("self_consistency", 0) == 1.0 and r_valid:
            u_retrieve = min(u_retrieve, u_answer - max(0.05, lambda_ * self.avg_ret_cost))

        return {"ANSWER": u_answer, "RETRIEVE": u_retrieve, "ABSTAIN": r_abstain}

    def decide_and_execute(self, root, child, cost_model, lambda_):
        """Execute two-stage decision policy: at root choose ANSWER, RETRIEVE, or ABSTAIN; if RETRIEVE, choose ANSWER or ABSTAIN."""
        r = enrich_root_and_child(root, child)
        utilities = self.predict_utilities(r, child, cost_model, lambda_)
        best = max(("ANSWER", "RETRIEVE", "ABSTAIN"), key=lambda a: utilities[a])

        root_usage = Usage(**r["incremental_usage"])
        if best == "ANSWER":
            return {
                "action_taken": "ANSWER",
                "answered": True,
                "prediction": r["prediction"],
                "correct": int(bool(r["correct"])),
                "total_usage": root_usage,
                "additional_usage": Usage(),
                "utility": cost_model.utility(bool(r["correct"]), True, Usage(), lambda_),
                "total_utility_with_sunk": cost_model.utility(bool(r["correct"]), True, root_usage, lambda_),
            }
        if best == "ABSTAIN":
            return {
                "action_taken": "ABSTAIN",
                "answered": False,
                "prediction": None,
                "correct": 0,
                "total_usage": root_usage,
                "additional_usage": Usage(),
                "utility": cost_model.utility(False, False, Usage(), lambda_),
                "total_utility_with_sunk": cost_model.utility(False, False, root_usage, lambda_),
            }

        inc_usage = Usage(**child["incremental_usage"])
        tot_usage = root_usage + inc_usage
        c_valid = bool(valid_entity_flag(child))
        c_prob = (
            self.ret_cal.predict_proba(child, is_retrieved=True)
            if self.dropped_group != "confidence"
            else raw_confidence_proxy(child)
        )
        r_correct = cost_model.reward(True, answered=True)
        r_wrong = cost_model.reward(False, answered=True)
        r_abstain = cost_model.reward(None, answered=False)
        expected_post_reward = (c_prob * r_correct + (1.0 - c_prob) * r_wrong) if c_valid else r_wrong

        if c_valid and expected_post_reward >= r_abstain:
            return {
                "action_taken": "RETRIEVE->ANSWER",
                "answered": True,
                "prediction": child["prediction"],
                "correct": int(bool(child["correct"])),
                "total_usage": tot_usage,
                "additional_usage": inc_usage,
                "utility": cost_model.utility(bool(child["correct"]), True, inc_usage, lambda_),
                "total_utility_with_sunk": cost_model.utility(bool(child["correct"]), True, tot_usage, lambda_),
            }
        return {
            "action_taken": "RETRIEVE->ABSTAIN",
            "answered": False,
            "prediction": None,
            "correct": 0,
            "total_usage": tot_usage,
            "additional_usage": inc_usage,
            "utility": cost_model.utility(False, False, inc_usage, lambda_),
            "total_utility_with_sunk": cost_model.utility(False, False, tot_usage, lambda_),
        }
