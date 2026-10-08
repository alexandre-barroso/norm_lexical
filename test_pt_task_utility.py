"""Synthetic boundary tests; no public/private corpus bodies are opened here."""
import unittest
from unittest.mock import patch

import pt_task_utility as q


def doc(name, words, gold=None, dataset="news_train"):
    gold = gold or ["NOUN"] * len(words)
    tokens = [q.Token(i + 1, word, word, tag, {}, {}, False) for i, (word, tag) in enumerate(zip(words, gold))]
    return q.Document(dataset, name, 1, q.digest(name), tokens, [])


class QualificationTests(unittest.TestCase):
    def test_parser_multiword_empty_nodes_and_exact_fields(self):
        text = "\n".join([
            "# sent_id = synthetic",
            "1-2\tfused\t_\t_\t_\t_\t_\t_\t_\tFullForm=two words",
            "1\ta\ta\tADP\t_\t_\t0\troot\t_\tCorrectForm=A",
            "2\tb\tb\tDET\t_\t_\t1\tdep\t_\t_",
            "2.1\tempty\t_\tX\t_\t_\t_\t_\t1:dep\t_",
            "3\tx\tx\tNOUN\t_\tTypo=Yes\t1\tdep\t_\tCorrectForm=x|FullForm=x y",
            "4\tt\tt\tVERB\t_\tVerbForm=Fin\t1\tdep\t_\tCorrectForm=T|FullForm=long|FullFrom=unrepaired",
        ])
        docs = q.parse_conllu(text, "tweets_dev")
        self.assertEqual(docs[0].words, ["a", "b", "x", "t"])
        self.assertEqual(len(docs[0].noninteger_rows), 2)
        events, audit = q.focal_events(docs)
        self.assertEqual([(i, field, target) for _, i, field, target in events], [(3, "CorrectForm", "T"), (3, "FullForm", "long")])
        self.assertEqual(audit["fields"]["CorrectForm"]["excluded_multiword_component"], 1)
        self.assertEqual(audit["fields"]["CorrectForm"]["identity_target"], 1)
        self.assertEqual(audit["fields"]["FullForm"]["excluded_whitespace_split_or_merge"], 1)
        self.assertEqual(audit["unrepaired_similar_keys"], {"FullFrom": 1})

    def test_malformed_and_repeated_keys_are_not_repaired(self):
        for value in ("CorrectForm=a|CorrectForm=b", "broken", "=x"):
            with self.assertRaises(ValueError):
                q.attrs(value)
        malformed = "# sent_id = synthetic\n2\tx\tx\tNOUN\t_\t_\t0\troot\t_\t_"
        with self.assertRaises(ValueError):
            q.parse_conllu(malformed, "tweets_dev")
        self.assertEqual(q.parse_conllu("", "tweets_dev"), [])

    def test_empty_training_rejected(self):
        with self.assertRaises(ValueError):
            q.LexicalMajority([])

    def test_supplied_underscore_punctuation_is_not_reconstructed(self):
        text = "# sent_id = synthetic\n1\t_\t_\tPUNCT\t_\t_\t0\troot\t_\tCorrectForm=x"
        docs = q.parse_conllu(text, "tweets_dev")
        self.assertEqual(docs[0].words, ["_"])
        events, audit = q.focal_events(docs)
        self.assertEqual(events, [])
        self.assertEqual(audit["fields"]["CorrectForm"]["excluded_placeholder_source"], 1)

    def test_article_and_near_duplicate_exclusion_is_transitive(self):
        words = [f"synthetic{i}" for i in range(35)]
        altered = ["changed"] + words[1:]
        train = [doc("FOLHA_DOC1_SENT1", words), doc("FOLHA_DOC1_SENT2", ["unrelated"]), doc("safe", ["safe"])]
        dev = [doc("dante_synthetic", altered, dataset="tweets_dev")]
        groups, excluded, report = q.group_documents({"news_train": train, "tweets_dev": dev})
        self.assertEqual(excluded, {x.key for x in train[:2]})
        self.assertEqual(groups[train[1].key], groups[dev[0].key])
        self.assertNotEqual(groups[train[2].key], groups[dev[0].key])
        self.assertEqual(report["edge_counts"]["near_duplicate_edges"], 1)

    def test_exact_case_duplicate_small_blocks_are_grouped(self):
        train = doc("a", ["ABC", "é"])
        dev = doc("b", ["abc", "e\u0301"], dataset="tweets_dev")
        groups, excluded, _ = q.group_documents({"news_train": [train], "tweets_dev": [dev]})
        self.assertEqual(groups[train.key], groups[dev.key])
        self.assertEqual(excluded, {train.key})

    def test_optional_news_check_does_not_discard_standard_training(self):
        train = doc("FOLHA_DOC1_SENT1", ["training"])
        dev = doc("FOLHA_DOC1_SENT2", ["evaluation"], dataset="news_dev")
        groups, excluded, _ = q.group_documents({"news_train": [train], "news_dev": [dev]})
        self.assertEqual(groups[train.key], groups[dev.key])
        self.assertEqual(excluded, {dev.key})

    def test_near_duplicate_does_not_conflate_mostly_different_sequences(self):
        a = doc("a", [str(i) for i in range(20)])
        b = doc("b", [str(i) for i in range(10)] + ["x"] * 10, dataset="tweets_dev")
        groups, excluded, _ = q.group_documents({"news_train": [a], "tweets_dev": [b]})
        self.assertFalse(excluded)
        self.assertNotEqual(groups[a.key], groups[b.key])

    def test_pair_is_exactly_one_edit_and_gold_is_never_passed(self):
        received = []
        def predict(words):
            self.assertTrue(all(type(x) is str for x in words))
            received.append(list(words))
            return ["NOUN", "VERB" if words[1] == "expanded" else "NOUN", "ADJ" if words[1] == "expanded" else "NOUN"]
        words = ["first", "short", "last"]
        result = q.paired_tags(predict, words, 1, "expanded", ["NOUN", "VERB", "NOUN"])
        self.assertEqual(received, [words, words, ["first", "expanded", "last"]])
        self.assertEqual(words, ["first", "short", "last"])
        self.assertEqual(result["outcome"], "gain")
        self.assertEqual(result["context"]["harm"], 1)
        self.assertEqual(result["context"]["raw_correct"], 2)
        self.assertEqual(result["context_changes"][0]["distance"], 1)

    def test_identity_or_alignment_failure_is_fatal(self):
        answers = iter([["NOUN"], ["VERB"]])
        with self.assertRaises(AssertionError):
            q.paired_tags(lambda words: next(answers), ["x"], 0, "y", ["NOUN"])
        with self.assertRaises(AssertionError):
            q.paired_tags(lambda words: [], ["x"], 0, "y", ["NOUN"])

    def test_lexical_tie_fallback_and_label_support_are_separate(self):
        train = [doc("a", ["same", "same", "other"], ["NOUN", "VERB", "VERB"])]
        model = q.LexicalMajority(train)
        self.assertEqual(model.predict(["same", "unknown"]), ["VERB", "VERB"])
        cov = q.coverage(model, "unknown", "same", "NOUN")
        self.assertEqual(cov, {"raw_frequency": 0, "target_frequency": 2, "raw_gold_frequency": 0, "target_gold_frequency": 1, "transition": "unseen_to_seen"})

    def test_string_strata_are_mutually_exclusive(self):
        self.assertEqual(q.relation("é", "e\u0301"), "identity")
        self.assertEqual(q.relation("É", "é"), "case_only")
        self.assertEqual(q.relation("Café", "cafe"), "accent_only_after_lowercase")
        self.assertEqual(q.relation("p", "para"), "other")

    def test_allowlist_contains_no_test_or_protected_file(self):
        self.assertEqual(set(q.INPUTS), {"news_train", "news_dev", "tweets_train", "tweets_dev"})
        for path, sha in q.INPUTS.values():
            self.assertNotIn("-test", path)
            self.assertTrue(path.endswith(("-ud-train.conllu", "-ud-dev.conllu")))
            self.assertNotIn("source_material", path)
            self.assertNotIn("master_references", path)
            self.assertEqual(len(sha), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
