"""Original public-data contextual-UPOS qualification, not a normalizer.

The exact four-file allowlist excludes test/protected inputs. NLTK supplies the
documented tagger; all parsing, grouping, paired interventions and audits here
are original. Run with Python hash seed 0 and -B. No remote code/model loading.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
import time
import unicodedata

PROJECT = Path(__file__).resolve().parent
BASE = Path(os.environ.get("UPOS_OUTPUT", str(PROJECT / "run"))).resolve()
DATA = Path(os.environ.get("UPOS_INPUTS", str(PROJECT / "inputs"))).resolve()
WORK = BASE / "work"
FIELDS = ("CorrectForm", "FullForm")
SEEDS = (7, 23, 47)
UPOS = frozenset("ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X".split())
INPUTS = {
    "news_train": ("pt_porttinari-ud-train.conllu", "b685d552f23dd0155072b11b1282d6e52dc7f4b57f335cc3486be23ef68c4ead"),
    "news_dev": ("pt_porttinari-ud-dev.conllu", "208483e5f5a29a0f6b6e0fc1f6e6499a10a5f5cbde53fdad20b30bf146a9e316"),
    "tweets_train": ("pt_dantestocks-ud-train.conllu", "43a4791e6502d3b1bcdefc36c6ad5a133c0d4569dd20d34c0a9bf4913d38f07d"),
    "tweets_dev": ("pt_dantestocks-ud-dev.conllu", "729bafef6f73d21b3c6115fee354d2a978e553cd2253ba4b6cb93bfee3aebfbc"),
}


def digest(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def frozen_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def save_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def attrs(value):
    if value == "_":
        return {}
    result = {}
    for item in value.split("|"):
        key, separator, val = item.partition("=")
        if not key or not separator or key in result:
            raise ValueError("Invalid/duplicate exact annotation key")
        result[key] = val
    return result


@dataclass
class Token:
    number: int
    form: str
    lemma: str
    upos: str
    features: dict
    misc: dict
    component: bool


@dataclass
class Document:
    dataset: str
    sent_id: str
    block_number: int
    raw_block_sha256: str
    tokens: list[Token]
    noninteger_rows: list[list[str]]

    @property
    def key(self):
        return self.dataset + ":" + self.sent_id

    @property
    def words(self):
        return [token.form for token in self.tokens]

    @property
    def gold(self):
        return [token.upos for token in self.tokens]


def parse_conllu(text, dataset):
    """Parse supplied integer words; never repair an annotation silently."""
    docs = []
    seen_ids = set()
    if not text.strip():
        return docs
    for block_number, block in enumerate(re.split(r"\r?\n\s*\r?\n", text.strip()), 1):
        lines = block.splitlines()
        ids = [x.removeprefix("# sent_id = ") for x in lines if x.startswith("# sent_id = ")]
        if len(ids) != 1 or ids[0] in seen_ids:
            raise ValueError("Missing/repeated document identifier")
        seen_ids.add(ids[0])
        rows = [x.split("\t") for x in lines if x and not x.startswith("#")]
        if not rows or any(len(row) != 10 for row in rows):
            raise ValueError("Missing words or non-ten-column CoNLL-U row")
        spans = []
        for row in rows:
            if re.fullmatch(r"[1-9][0-9]*-[1-9][0-9]*", row[0]):
                a, b = map(int, row[0].split("-"))
                if b <= a:
                    raise ValueError("Invalid multiword range")
                spans.append((a, b))
            elif not re.fullmatch(r"[1-9][0-9]*(?:\.[1-9][0-9]*)?", row[0]):
                raise ValueError("Invalid CoNLL-U identifier")
        words = []
        noninteger = []
        for row in rows:
            if not row[0].isdigit():
                noninteger.append(row)
                continue
            number = int(row[0])
            if row[3] not in UPOS or not row[1]:
                raise ValueError("Integer word lacks usable FORM/UPOS")
            words.append(Token(number, row[1], row[2], row[3], attrs(row[5]), attrs(row[9]), any(a <= number <= b for a, b in spans)))
        if [word.number for word in words] != list(range(1, len(words) + 1)):
            raise ValueError("Integer word IDs are not contiguous and ordered")
        if any(b > len(words) for a, b in spans):
            raise ValueError("Multiword span exceeds integer words")
        docs.append(Document(dataset, ids[0], block_number, digest(block), words, noninteger))
    return docs


def load_inputs():
    loaded = {}
    for name, (relative, expected) in INPUTS.items():
        path = DATA / relative
        if path.is_symlink() or not path.name.endswith(("-ud-train.conllu", "-ud-dev.conllu")) or not path.resolve().is_relative_to(DATA):
            raise ValueError("Disallowed input")
        data = path.read_bytes()
        if digest(data) != expected:
            raise ValueError(f"Changed pinned public input: {name}")
        loaded[name] = parse_conllu(data.decode("utf-8"), name)
    return loaded


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def normalized_words(doc):
    return tuple(unicodedata.normalize("NFC", word).lower() for word in doc.words)


def group_documents(loaded):
    docs = [doc for name in INPUTS for doc in loaded.get(name, [])]
    uf = UnionFind(len(docs))
    exact, article = {}, {}
    counts = Counter()
    sets = []
    index = defaultdict(list)
    for i, doc in enumerate(docs):
        norm = normalized_words(doc)
        if norm in exact:
            uf.union(i, exact[norm]); counts["exact_normalized_sequence_edges"] += 1
        else:
            exact[norm] = i
        match = re.fullmatch(r"(FOLHA_DOC[0-9]+)_SENT[0-9]+", doc.sent_id)
        if match and doc.dataset.startswith("news_"):
            source_id = match.group(1)
            if source_id in article:
                uf.union(i, article[source_id]); counts["same_news_article_edges"] += 1
            else:
                article[source_id] = i
        shingles = {norm[k:k + 5] for k in range(len(norm) - 4)} if len(norm) >= 10 else set()
        candidates = Counter(j for shingle in shingles for j in index[shingle])
        for j, overlap in candidates.items():
            union = len(shingles) + len(sets[j]) - overlap
            # Exact integer inequality avoids threshold floating-point ambiguity.
            if 5 * overlap >= 4 * union:
                uf.union(i, j); counts["near_duplicate_edges"] += 1
        sets.append(shingles)
        for shingle in shingles:
            index[shingle].append(i)
    components = defaultdict(list)
    for i, doc in enumerate(docs):
        components[uf.find(i)].append(doc)
    mapping, excluded, ledger = {}, set(), []
    for members in components.values():
        keys = sorted(doc.key for doc in members)
        component = digest(frozen_json(keys))
        dev_present = any(doc.dataset.endswith("_dev") for doc in members)
        tweets_dev_present = any(doc.dataset == "tweets_dev" for doc in members)
        training_present = any(doc.dataset.endswith("_train") for doc in members)
        for doc in members:
            mapping[doc.key] = component
            if tweets_dev_present and doc.dataset.endswith("_train"):
                excluded.add(doc.key)
            if training_present and doc.dataset == "news_dev":
                excluded.add(doc.key)
        if dev_present:
            ledger.append({"component_sha256": component, "members": keys, "excluded_training_blocks": sum(doc.key in excluded and doc.dataset.endswith("_train") for doc in members), "excluded_optional_news_dev_blocks": sum(doc.key in excluded and doc.dataset == "news_dev" for doc in members)})
    return mapping, excluded, {"edge_counts": dict(counts), "components": len(components), "dev_components": ledger}


def relation(source, target):
    a, b = (unicodedata.normalize("NFC", x) for x in (source, target))
    if a == b:
        return "identity"
    if a.lower() == b.lower():
        return "case_only"
    def strip_marks(value):
        return "".join(c for c in unicodedata.normalize("NFD", value.lower()) if unicodedata.category(c) != "Mn")
    if strip_marks(a) == strip_marks(b):
        return "accent_only_after_lowercase"
    return "other"


def focal_events(docs):
    events = []
    audit = {field: Counter() for field in FIELDS}
    similar_keys = Counter()
    for doc in docs:
        for row in doc.noninteger_rows:
            misc = attrs(row[9])
            for field in FIELDS:
                if field in misc:
                    audit[field]["explicit_noninteger_row"] += 1
        for i, token in enumerate(doc.tokens):
            for key in token.misc:
                if key not in FIELDS and (key.lower().startswith("correct") or key.lower().startswith("full")):
                    similar_keys[key] += 1
            for field in FIELDS:
                if field not in token.misc:
                    continue
                audit[field]["explicit_integer_row"] += 1
                target = token.misc[field]
                if token.component:
                    reason = "excluded_multiword_component"
                elif token.form == "_":
                    reason = "excluded_placeholder_source"
                elif not target or target == "_" or not target.strip():
                    reason = "excluded_empty_or_placeholder_target"
                elif any(c.isspace() for c in token.form + target):
                    reason = "excluded_whitespace_split_or_merge"
                else:
                    audit[field]["eligible_including_identity"] += 1
                    if relation(token.form, target) == "identity":
                        reason = "identity_target"
                    else:
                        reason = "eligible_changed_target"
                        events.append((doc, i, field, target))
                audit[field][reason] += 1
    return events, {"fields": {field: dict(value) for field, value in audit.items()}, "unrepaired_similar_keys": dict(similar_keys)}


def paired_tags(predict, words, index, target, gold):
    """The prediction API receives words only; scoring metadata stays outside."""
    raw = list(predict(list(words)))
    identity_words = list(words)
    identity_words[index] = words[index]
    identity = list(predict(identity_words))
    if raw != identity:
        raise AssertionError("Identity intervention changed predictions")
    edited_words = list(words)
    edited_words[index] = target
    edited = list(predict(edited_words))
    if len(raw) != len(words) or len(edited) != len(words):
        raise AssertionError("Predictor changed supplied alignment")
    raw_correct, edited_correct = raw[index] == gold[index], edited[index] == gold[index]
    label = { (False, True): "gain", (True, False): "harm", (True, True): "both_correct", (False, False): "both_wrong" }[(raw_correct, edited_correct)]
    context = Counter()
    changes = []
    for j, (a, b, g) in enumerate(zip(raw, edited, gold)):
        if j == index:
            continue
        context["exposures"] += 1
        context["raw_correct"] += a == g
        context["edited_correct"] += b == g
        if a != b:
            context["changed_predictions"] += 1
            context["gain"] += a != g and b == g
            context["harm"] += a == g and b != g
            changes.append({"integer_id": j + 1, "distance": j - index, "gold": g, "raw": a, "edited": b})
    return {"outcome": label, "raw": raw[index], "edited": edited[index], "gold": gold[index], "prediction_changed": raw[index] != edited[index], "context": dict(context), "context_changes": changes}


class LexicalMajority:
    def __init__(self, docs):
        self.counts = defaultdict(Counter)
        total = Counter()
        for doc in docs:
            for token in doc.tokens:
                self.counts[token.form][token.upos] += 1
                total[token.upos] += 1
        if not total:
            raise ValueError("Cannot train empty baseline")
        self.fallback = max(total, key=lambda tag: (total[tag], tag))
        self.lookup = {form: max(count, key=lambda tag: (count[tag], tag)) for form, count in self.counts.items()}

    def predict(self, words):
        return [self.lookup.get(word, self.fallback) for word in words]


def coverage(lexical, source, target, gold):
    source_counts, target_counts = lexical.counts.get(source, {}), lexical.counts.get(target, {})
    a, b = sum(source_counts.values()), sum(target_counts.values())
    return {"raw_frequency": a, "target_frequency": b, "raw_gold_frequency": source_counts.get(gold, 0), "target_gold_frequency": target_counts.get(gold, 0), "transition": ("seen" if a else "unseen") + "_to_" + ("seen" if b else "unseen")}


def raw_accuracy(predict, docs):
    total, correct = 0, 0
    classes = {tag: Counter() for tag in sorted(UPOS)}
    for doc in docs:
        tags = list(predict(doc.words))
        if len(tags) != len(doc.tokens):
            raise AssertionError("Bad raw alignment")
        for pred, token in zip(tags, doc.tokens):
            total += 1; correct += pred == token.upos
            classes[token.upos]["gold_count"] += 1
            classes[token.upos]["correct"] += pred == token.upos
            classes[pred]["predicted_count"] += 1
    return {"words": total, "correct": correct, "accuracy": correct / total if total else None, "upos": {k: dict(v) for k, v in classes.items()}}


def event_results(predict, events, lexical, group_map, model_family, regime, seed):
    output = []
    for doc, i, field, target in events:
        token = doc.tokens[i]
        pair = paired_tags(predict, doc.words, i, target, doc.gold)
        if any(change["distance"] < -2 for change in pair["context_changes"]):
            raise AssertionError("Unexpected effect before greedy two-word future window")
        row = {"model_family": model_family, "regime": regime, "seed": seed, "sent_id": doc.sent_id, "block_number": doc.block_number, "raw_block_sha256": doc.raw_block_sha256, "component_sha256": group_map[doc.key], "integer_id": token.number, "field": field, "source_sha256": digest(token.form), "target_sha256": digest(target), "lemma_sha256": digest(token.lemma), "lemma_missing": token.lemma == "_", "features": token.features, "relation": relation(token.form, target), "coverage": coverage(lexical, token.form, target, token.upos), **pair}
        output.append(row)
    return output


def summarize_events(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model_family"], row["regime"], row["seed"], row["field"])].append(row)
    output = []
    for (family, regime, seed, field), panel in groups.items():
        outcomes = Counter(row["outcome"] for row in panel)
        context = Counter()
        strata = {}
        for row in panel:
            context.update(row["context"])
        for variable in ("relation", "gold", "coverage_transition"):
            cells = defaultdict(Counter)
            for row in panel:
                value = row["coverage"]["transition"] if variable == "coverage_transition" else row[variable]
                cells[value]["n"] += 1
                cells[value][row["outcome"]] += 1
            strata[variable] = {key: dict(value) for key, value in sorted(cells.items())}
        concentrations = {}
        for label, key in (("source", "source_sha256"), ("lemma", "lemma_sha256"), ("dev_component", "component_sha256")):
            counts = Counter(row[key] for row in panel)
            effects = Counter()
            for row in panel:
                effects[row[key]] += (row["outcome"] == "gain") - (row["outcome"] == "harm")
            concentrations[label] = {"types_or_groups": len(counts), "largest_count": max(counts.values()), "top_three_count": sum(n for _, n in counts.most_common(3)), "group_counts_and_net": [{"sha256": k, "n": counts[k], "net": effects[k]} for k in sorted(counts)]}
        output.append({"model_family": family, "regime": regime, "seed": seed, "field": field, "n": len(panel), "raw_correct": outcomes["both_correct"] + outcomes["harm"], "oracle_correct": outcomes["both_correct"] + outcomes["gain"], "net": outcomes["gain"] - outcomes["harm"], "outcomes": dict(outcomes), "prediction_changes": sum(row["prediction_changed"] for row in panel), "nonfocal_exposures": dict(context), "strata": strata, "concentration": concentrations})
    return output


def prepare():
    loaded = load_inputs()
    groups, excluded, group_audit = group_documents(loaded)
    kept = {name: [doc for doc in docs if doc.key not in excluded] for name, docs in loaded.items()}
    events, eligibility = focal_events(loaded["tweets_dev"])
    counts = {}
    for name, docs in loaded.items():
        forms = Counter(tuple(doc.words) for doc in docs)
        counts[name] = {"original_blocks": len(docs), "original_words": sum(len(doc.tokens) for doc in docs), "retained_blocks": len(kept[name]), "retained_words": sum(len(doc.tokens) for doc in kept[name]), "excluded_blocks": len(docs) - len(kept[name]), "excluded_words": sum(len(doc.tokens) for doc in docs if doc.key in excluded), "exact_raw_sequence_duplicate_excess_blocks": sum(n - 1 for n in forms.values()), "noninteger_rows": sum(len(doc.noninteger_rows) for doc in docs)}
    report = {"input_files": {name: {"path": path, "sha256": sha} for name, (path, sha) in INPUTS.items()}, "counts": counts, "eligibility": eligibility, "groups": group_audit, "excluded_locators": sorted(excluded), "events": len(events)}
    return kept, events, groups, report


def fit_model(docs, seed):
    import nltk
    from nltk.tag.perceptron import PerceptronTagger
    if nltk.__version__ != "3.9.2":
        raise ValueError("Unpinned NLTK version")
    import nltk.tag.perceptron as implementation
    if digest(Path(implementation.__file__).read_bytes()) != "2332a63191688f70c64d7f609d82b8f61267dca6cb9f5b19314fd8efe6d3dcd8":
        raise ValueError("NLTK source differs from fully read module")
    random.seed(seed)
    model = PerceptronTagger(load=False)
    start = time.monotonic()
    # Nothing except FORM/UPOS from retained training enters this constructor.
    model.train([list(zip(doc.words, doc.gold)) for doc in docs], nr_iter=5)
    return model, time.monotonic() - start


def model_predict(model):
    def predict(words):
        tagged = model.tag(words, use_tagdict=True)
        if [word for word, tag in tagged] != words:
            raise AssertionError("Library altered supplied words")
        return [tag for word, tag in tagged]
    return predict


def run():
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("Set PYTHONHASHSEED=0 before launching Python")
    kept, events, groups, preparation = prepare()
    frozen = {"status": "REPRODUCTION_RUN_AFTER_PUBLICATION_PREPARATION", "export_implementation_sha256": digest(Path(__file__).read_bytes()), "original_implementation_sha256": "a2c876fa1efb5ad101c29727b4567a97ba87d1a15036c4d3ab7735093b603dd4"}
    if (BASE / "results.json").exists() or (BASE / "events.jsonl").exists():
        raise ValueError("Refusing to overwrite prior corpus predictions")
    save_new(BASE / "preparation_at_run.json", preparation)
    (WORK / "models").mkdir(parents=True, exist_ok=True)
    rows, models = [], []
    for regime in ("news", "news_plus_tweets"):
        training = kept["news_train"] + (kept["tweets_train"] if regime == "news_plus_tweets" else [])
        lexical = LexicalMajority(training)
        rows.extend(event_results(lexical.predict, events, lexical, groups, "literal_majority", regime, None))
        models.append({"family": "literal_majority", "regime": regime, "seed": None, "raw_dev": {name: raw_accuracy(lexical.predict, kept[name]) for name in ("news_dev", "tweets_dev")}})
        for seed in SEEDS:
            print(f"Training {regime}, seed {seed}, {len(training)} blocks", flush=True)
            model, seconds = fit_model(training, seed)
            parameters = {"weights": model.model.weights, "tagdict": model.tagdict, "classes": sorted(model.classes)}
            path = WORK / "models" / f"{regime}_seed{seed}.json"
            save_new(path, parameters)
            # JSON reconstruction, without pickle or a downloaded model.
            import nltk.tag.perceptron as p
            replay = p.PerceptronTagger(load=False)
            saved = json.loads(path.read_text())
            replay.model.weights = saved["weights"]
            replay.tagdict = saved["tagdict"]
            replay.classes = set(saved["classes"])
            replay.model.classes = replay.classes
            synthetic = ["Hoje", "uma", "ideia", "cresce", "."]
            if model_predict(model)(synthetic) != model_predict(replay)(synthetic):
                raise AssertionError("JSON model persistence mismatch")
            predict = model_predict(model)
            rows.extend(event_results(predict, events, lexical, groups, "greedy_contextual_perceptron", regime, seed))
            meta = {"family": "greedy_contextual_perceptron", "regime": regime, "seed": seed, "epochs": 5, "training_blocks": len(training), "training_words": sum(len(doc.tokens) for doc in training), "training_seconds": seconds, "model_file": str(path.relative_to(BASE)), "model_sha256": digest(path.read_bytes()), "model_bytes": path.stat().st_size, "classes": sorted(model.classes), "tag_dictionary_entries": len(model.tagdict), "raw_dev": {name: raw_accuracy(predict, kept[name]) for name in ("news_dev", "tweets_dev")}}
            models.append(meta)
            print(f"Finished {regime}, seed {seed}, {seconds:.2f} seconds; events scored", flush=True)
    with (BASE / "events.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(frozen_json(row) + "\n")
    result = {"status": "DESCRIPTIVE_DEVELOPMENT_PANEL_REPRODUCED", "finished_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "python_hash_seed": os.environ["PYTHONHASHSEED"], "freeze": frozen, "preparation_sha256": digest((BASE / "preparation_at_run.json").read_bytes()), "events_sha256": digest((BASE / "events.jsonl").read_bytes()), "models": models, "paired_summaries": summarize_events(rows), "checks": {"identity_repeat": "PASS all events/models", "pre_focal_future_window": "PASS all events/models", "json_persistence": "PASS six models"}}
    save_new(BASE / "results.json", result)
    print(f"Completed {len(rows)} model-event pairs; results.json and events.jsonl saved", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        kept, events, groups, report = prepare()
        save_new(BASE / "prepared_inputs.json", report)
        print(json.dumps({"counts": report["counts"], "eligibility": report["eligibility"], "group_edges": report["groups"]["edge_counts"], "events": len(events)}, indent=2))
    else:
        run()


if __name__ == "__main__":
    main()
