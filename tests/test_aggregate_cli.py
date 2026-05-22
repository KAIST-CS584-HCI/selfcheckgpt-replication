import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path


def dataset_item(idx: int) -> dict:
    return {
        "wiki_bio_test_idx": 100 + idx,
        "wiki_bio_text": f"wiki {idx}",
        "main_passage": f"main {idx}",
        "main_sentences": [f"sentence {idx}"],
        "annotation": ["accurate"],
        "sample_passages": [f"sample {idx}"],
    }


def slice_entry(idx: int, *, variant: str, scores: list[float], responses=None) -> dict:
    scores_block = {variant: scores}
    responses_block: dict = {}
    if responses is not None and variant == "prompt":
        responses_block = {"prompt": responses}
    return {
        **dataset_item(idx),
        "dataset_idx": idx,
        "scores": scores_block,
        "responses": responses_block,
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def touch_mtime(path: Path, when: float) -> None:
    os.utime(path, (when, when))


class CompactRangesTest(unittest.TestCase):
    def test_groups_consecutive_runs_and_keeps_singletons(self) -> None:
        from replication.aggregate import compact_ranges

        self.assertEqual(compact_ranges([0, 1, 2, 4, 6, 7]), "0-2,4,6-7")

    def test_empty_input_returns_empty_string(self) -> None:
        from replication.aggregate import compact_ranges

        self.assertEqual(compact_ranges([]), "")

    def test_deduplicates_and_sorts(self) -> None:
        from replication.aggregate import compact_ranges

        self.assertEqual(compact_ranges([5, 1, 2, 5, 1]), "1-2,5")


class AggregateRunnerTest(unittest.TestCase):
    def _run(self, tmp: Path, dataset_size: int = 5):
        from replication.aggregate import AggregateConfig, AggregateRunner

        dataset_path = tmp / "dataset.json"
        dataset_path.write_text(json.dumps([dataset_item(i) for i in range(dataset_size)]))
        output_path = tmp / "out" / "aggregated.json"
        config = AggregateConfig(
            dataset_path=dataset_path,
            inputs_dir=tmp / "slices",
            output_path=output_path,
        )
        report = AggregateRunner().run(config)
        return report, output_path

    def test_merges_different_variants_into_same_passage(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            slices_dir = tmp / "slices"
            write_json(slices_dir / "bert-0-to-2.json", [
                slice_entry(0, variant="bert", scores=[0.1]),
                slice_entry(1, variant="bert", scores=[0.2]),
            ])
            write_json(slices_dir / "nli-0-to-2.json", [
                slice_entry(0, variant="nli", scores=[0.3]),
                slice_entry(1, variant="nli", scores=[0.4]),
            ])

            _, output_path = self._run(tmp)
            saved = json.loads(output_path.read_text())

            self.assertEqual([entry["dataset_idx"] for entry in saved], [0, 1])
            self.assertEqual(saved[0]["scores"], {"bert": [0.1], "nli": [0.3]})
            self.assertEqual(saved[1]["scores"], {"bert": [0.2], "nli": [0.4]})

    def test_unions_disjoint_slices(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            slices_dir = tmp / "slices"
            write_json(slices_dir / "bert-0-to-1.json", [slice_entry(0, variant="bert", scores=[0.1])])
            write_json(slices_dir / "bert-5-to-6.json", [slice_entry(5, variant="bert", scores=[0.5])])

            _, output_path = self._run(tmp, dataset_size=6)
            saved = json.loads(output_path.read_text())

            self.assertEqual([entry["dataset_idx"] for entry in saved], [0, 5])

    def test_reuses_existing_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            slices_dir = tmp / "slices"
            slices_dir.mkdir()
            output_path = tmp / "out" / "aggregated.json"
            write_json(output_path, [slice_entry(100, variant="bert", scores=[0.9])])

            write_json(slices_dir / "bert-101-to-102.json", [slice_entry(101, variant="bert", scores=[0.7])])

            from replication.aggregate import AggregateConfig, AggregateRunner

            dataset_path = tmp / "dataset.json"
            dataset_path.write_text(json.dumps([dataset_item(i) for i in range(200)]))
            AggregateRunner().run(AggregateConfig(
                dataset_path=dataset_path,
                inputs_dir=slices_dir,
                output_path=output_path,
            ))
            saved = json.loads(output_path.read_text())
            self.assertEqual([entry["dataset_idx"] for entry in saved], [100, 101])
            self.assertEqual(saved[0]["scores"], {"bert": [0.9]})
            self.assertEqual(saved[1]["scores"], {"bert": [0.7]})

    def test_last_write_wins_on_conflict_with_warning(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            slices_dir = tmp / "slices"
            old_path = slices_dir / "bert-0-to-1.json"
            new_path = slices_dir / "bert-0-to-2.json"
            write_json(old_path, [slice_entry(0, variant="bert", scores=[0.1])])
            write_json(new_path, [slice_entry(0, variant="bert", scores=[0.2])])
            # Force mtime ordering: old < new.
            touch_mtime(old_path, time.time() - 100)
            touch_mtime(new_path, time.time())

            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                _, output_path = self._run(tmp)
            saved = json.loads(output_path.read_text())

            self.assertEqual(saved[0]["scores"], {"bert": [0.2]})
            self.assertIn("WARN", captured.getvalue())
            self.assertIn("dataset_idx=0", captured.getvalue())

    def test_coverage_report_lists_missing_indices_per_variant(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            slices_dir = tmp / "slices"
            write_json(slices_dir / "bert-0-to-3.json", [
                slice_entry(0, variant="bert", scores=[0.1]),
                slice_entry(1, variant="bert", scores=[0.2]),
                slice_entry(2, variant="bert", scores=[0.3]),
            ])
            write_json(slices_dir / "nli-0-to-5.json", [
                slice_entry(0, variant="nli", scores=[0.4]),
                slice_entry(4, variant="nli", scores=[0.5]),
            ])

            report, _ = self._run(tmp, dataset_size=5)
            text = report.format()

            self.assertIn("bert: 3/5 done, missing 3-4", text)
            self.assertIn("nli: 2/5 done, missing 1-3", text)
            self.assertIn("prompt: 0/5 done, missing 0-4", text)

    def test_ignores_non_matching_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            slices_dir = tmp / "slices"
            slices_dir.mkdir()
            (slices_dir / "notes.txt").write_text("hello")
            (slices_dir / "random-0-to-1.json").write_text("[]")
            write_json(slices_dir / "bert-0-to-1.json", [slice_entry(0, variant="bert", scores=[0.1])])

            _, output_path = self._run(tmp)
            saved = json.loads(output_path.read_text())

            self.assertEqual([entry["dataset_idx"] for entry in saved], [0])

    def test_skips_output_file_during_discovery(self) -> None:
        # Place the output inside the inputs dir; aggregator must not consume itself.
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            slices_dir = tmp / "slices"
            slices_dir.mkdir()
            output_path = slices_dir / "aggregated.json"
            write_json(output_path, [slice_entry(0, variant="bert", scores=[0.1])])
            write_json(slices_dir / "bert-1-to-2.json", [slice_entry(1, variant="bert", scores=[0.2])])

            from replication.aggregate import AggregateConfig, AggregateRunner

            dataset_path = tmp / "dataset.json"
            dataset_path.write_text(json.dumps([dataset_item(i) for i in range(5)]))
            AggregateRunner().run(AggregateConfig(
                dataset_path=dataset_path,
                inputs_dir=slices_dir,
                output_path=output_path,
            ))
            saved = json.loads(output_path.read_text())
            self.assertEqual([entry["dataset_idx"] for entry in saved], [0, 1])


class AggregateCliTest(unittest.TestCase):
    def test_parser_requires_dataset(self) -> None:
        from aggregate import build_parser

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args([])

    def test_parser_accepts_inputs_dir_and_output(self) -> None:
        from aggregate import build_parser

        args = build_parser().parse_args([
            "--dataset", "data/dataset.json",
            "--inputs-dir", "output/",
            "--output", "output/aggregated.json",
        ])
        self.assertEqual(args.dataset, "data/dataset.json")
        self.assertEqual(args.inputs_dir, "output/")
        self.assertEqual(args.output, "output/aggregated.json")

    def test_main_runs_end_to_end(self) -> None:
        import aggregate as aggregate_module

        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            dataset_path = tmp / "dataset.json"
            dataset_path.write_text(json.dumps([dataset_item(i) for i in range(2)]))
            slices_dir = tmp / "slices"
            write_json(slices_dir / "bert-0-to-1.json", [slice_entry(0, variant="bert", scores=[0.1])])
            output_path = tmp / "agg.json"

            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                aggregate_module.main([
                    "--dataset", str(dataset_path),
                    "--inputs-dir", str(slices_dir),
                    "--output", str(output_path),
                ])

            self.assertTrue(output_path.exists())
            self.assertIn("bert: 1/2 done", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
