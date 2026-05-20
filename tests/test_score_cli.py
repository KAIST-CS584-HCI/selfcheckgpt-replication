import json
import tempfile
import unittest
from pathlib import Path

from replication.entity import PassageInstance, PassageResponses, PassageResult, PassageScores


class FakeScorer:
    method_name = "fake"
    default_dataset_path = "unused.json"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def score(self, dataset_idx: int, passage: PassageInstance) -> PassageResult:
        self.calls.append(dataset_idx)
        return PassageResult(
            dataset_idx=dataset_idx,
            wiki_bio_test_idx=passage.wiki_bio_test_idx,
            wiki_bio_text=passage.wiki_bio_text,
            main_passage=passage.main_passage,
            main_sentences=passage.main_sentences,
            annotation=passage.annotation,
            sample_passages=passage.sample_passages,
            scores=PassageScores(prompt=[float(dataset_idx)]),
            responses=PassageResponses(prompt=[["Yes"]]),
        )


def dataset_item(idx: int) -> dict:
    return {
        "wiki_bio_test_idx": 100 + idx,
        "wiki_bio_text": f"wiki {idx}",
        "main_passage": f"main {idx}",
        "main_sentences": [f"sentence {idx}"],
        "annotation": ["accurate"],
        "sample_passages": [f"sample {idx}"],
    }


class ScoreRunnerTest(unittest.TestCase):
    def test_runner_skips_existing_outputs_by_default(self) -> None:
        from replication.score.base import ScoreRunner

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "dataset.json"
            output_dir = tmp_path / "out"
            output_dir.mkdir()
            dataset_path.write_text(json.dumps([dataset_item(0), dataset_item(1)]))
            (output_dir / "0.json").write_text('{"existing": true}')

            scorer = FakeScorer()
            written = ScoreRunner(scorer).run(
                start=0,
                end=2,
                dataset_path=dataset_path,
                output_dir=output_dir,
                overwrite=False,
            )

            self.assertEqual(written, [1])
            self.assertEqual(scorer.calls, [1])
            self.assertEqual(json.loads((output_dir / "0.json").read_text()), {"existing": True})
            self.assertEqual(json.loads((output_dir / "1.json").read_text())["scores"]["prompt"], [1.0])

    def test_runner_overwrites_existing_outputs_when_requested(self) -> None:
        from replication.score.base import ScoreRunner

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "dataset.json"
            output_dir = tmp_path / "out"
            output_dir.mkdir()
            dataset_path.write_text(json.dumps([dataset_item(0)]))
            (output_dir / "0.json").write_text('{"existing": true}')

            scorer = FakeScorer()
            written = ScoreRunner(scorer).run(
                start=0,
                end=1,
                dataset_path=dataset_path,
                output_dir=output_dir,
                overwrite=True,
            )

            self.assertEqual(written, [0])
            self.assertEqual(scorer.calls, [0])
            self.assertEqual(json.loads((output_dir / "0.json").read_text())["scores"]["prompt"], [0.0])


class ScoreIOTest(unittest.TestCase):
    def test_score_io_loads_dataset_and_saves_result_json(self) -> None:
        from replication.score.base import ScoreIO

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "dataset.json"
            output_dir = tmp_path / "out"
            dataset_path.write_text(json.dumps([dataset_item(0)]))

            score_io = ScoreIO(dataset_path=dataset_path, output_dir=output_dir)
            dataset = score_io.load_dataset()
            result = FakeScorer().score(0, dataset[0])

            self.assertFalse(score_io.result_exists(0))
            score_io.save_result(0, result)

            self.assertTrue(score_io.result_exists(0))
            self.assertEqual(score_io.result_path(0), output_dir / "0.json")
            saved = json.loads((output_dir / "0.json").read_text())
            self.assertEqual(saved["wiki_bio_test_idx"], 100)
            self.assertEqual(saved["scores"]["prompt"], [0.0])


class ScoreCliTest(unittest.TestCase):
    def test_parser_accepts_method_subcommands_with_shared_options(self) -> None:
        from replication.score.main import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "prompt",
            "--start",
            "2",
            "--end",
            "3",
            "--dataset",
            "dataset.json",
            "--output-dir",
            "out",
            "--overwrite",
            "--think",
        ])

        self.assertEqual(args.method, "prompt")
        self.assertEqual(args.start, 2)
        self.assertEqual(args.end, 3)
        self.assertEqual(args.dataset, "dataset.json")
        self.assertEqual(args.output_dir, "out")
        self.assertTrue(args.overwrite)
        self.assertTrue(args.think)


if __name__ == "__main__":
    unittest.main()
