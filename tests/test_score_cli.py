import json
import os
import tempfile
import unittest
import contextlib
import io
from pathlib import Path
from unittest import mock

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
    def test_runner_writes_one_aggregate_output_file(self) -> None:
        from replication.score.base import ScoreIO, ScoreRunner

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "dataset.json"
            output_path = tmp_path / "results" / "fake.json"
            dataset_path.write_text(json.dumps([dataset_item(0), dataset_item(1)]))

            scorer = FakeScorer()
            score_io = ScoreIO(dataset_path=dataset_path, output_path=output_path)
            results = ScoreRunner(scorer, score_io).run(start=0, end=2)

            self.assertEqual([result.dataset_idx for result in results], [0, 1])
            self.assertEqual(scorer.calls, [0, 1])
            self.assertFalse((output_path.parent / "0.json").exists())
            self.assertFalse((output_path.parent / "1.json").exists())
            saved = json.loads(output_path.read_text())
            self.assertEqual([item["dataset_idx"] for item in saved], [0, 1])
            self.assertEqual(saved[1]["scores"]["prompt"], [1.0])

    def test_runner_skips_existing_aggregate_output_by_default(self) -> None:
        from replication.score.base import ScoreIO, ScoreRunner

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "dataset.json"
            output_path = tmp_path / "fake.json"
            dataset_path.write_text(json.dumps([dataset_item(0), dataset_item(1)]))
            output_path.write_text('[{"existing": true}]')

            scorer = FakeScorer()
            score_io = ScoreIO(dataset_path=dataset_path, output_path=output_path)
            results = ScoreRunner(scorer, score_io).run(start=0, end=2)

            self.assertEqual(results, [])
            self.assertEqual(scorer.calls, [])
            self.assertEqual(json.loads(output_path.read_text()), [{"existing": True}])


class ScoreIOTest(unittest.TestCase):
    def test_score_io_loads_dataset_and_saves_aggregate_result_json(self) -> None:
        from replication.score.base import ScoreIO

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset_path = tmp_path / "dataset.json"
            output_path = tmp_path / "out" / "fake.json"
            dataset_path.write_text(json.dumps([dataset_item(0)]))

            score_io = ScoreIO(dataset_path=dataset_path, output_path=output_path)
            dataset = score_io.load_dataset()
            result = FakeScorer().score(0, dataset[0])

            self.assertFalse(score_io.output_exists())
            self.assertFalse(score_io.should_skip())
            score_io.save_results([result])

            self.assertTrue(score_io.output_exists())
            self.assertTrue(score_io.should_skip())
            saved = json.loads(output_path.read_text())
            self.assertEqual(saved[0]["wiki_bio_test_idx"], 100)
            self.assertEqual(saved[0]["scores"]["prompt"], [0.0])

class ScoreCliTest(unittest.TestCase):
    def test_parser_accepts_method_subcommands_with_shared_options(self) -> None:
        from score import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "prompt",
            "--start",
            "2",
            "--end",
            "3",
            "--dataset",
            "dataset.json",
            "--output",
            "out.json",
            "--think",
        ])

        self.assertEqual(args.method, "prompt")
        self.assertEqual(args.start, 2)
        self.assertEqual(args.end, 3)
        self.assertEqual(args.dataset, "dataset.json")
        self.assertEqual(args.output, "out.json")
        self.assertTrue(args.think)

    def test_parser_rejects_overwrite_option(self) -> None:
        from score import build_parser

        parser = build_parser()

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["bert", "--start", "0", "--end", "1", "--overwrite"])

    def test_default_score_output_path_uses_method_and_range(self) -> None:
        from replication.score.base import REPO_ROOT
        from score import build_parser, build_score_io

        parser = build_parser()
        args = parser.parse_args(["bert", "--start", "0", "--end", "5"])
        score_io = build_score_io(args)

        self.assertEqual(score_io.output_path, REPO_ROOT / "output" / "bert-0-to-5.json")


class ScoreEnvironmentTest(unittest.TestCase):
    def test_load_environment_reads_env_file_without_overriding_existing_values(self) -> None:
        from score import load_environment

        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "OPENROUTER_API_KEY=from-file\n"
                "OPENROUTER_MODEL=from-file-model\n"
                "OPENROUTER_BASE_URL='https://example.test/api'\n"
            )

            with mock.patch.dict(os.environ, {"OPENROUTER_MODEL": "already-set"}, clear=False):
                for key in ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"):
                    os.environ.pop(key, None)

                load_environment(env_path)

                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "from-file")
                self.assertEqual(os.environ["OPENROUTER_MODEL"], "already-set")
                self.assertEqual(os.environ["OPENROUTER_BASE_URL"], "https://example.test/api")


class PromptConfigTest(unittest.TestCase):
    def test_openrouter_config_uses_env_values(self) -> None:
        from replication.score.prompt import get_openrouter_config

        with mock.patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "key",
                "OPENROUTER_MODEL": "model",
                "OPENROUTER_BASE_URL": "https://example.test/api",
            },
            clear=True,
        ):
            config = get_openrouter_config()

        self.assertEqual(config.api_key, "key")
        self.assertEqual(config.model, "model")
        self.assertEqual(config.base_url, "https://example.test/api")

    def test_openrouter_config_defaults_optional_values(self) -> None:
        from replication.score.prompt import DEFAULT_OPENROUTER_BASE_URL, DEFAULT_OPENROUTER_MODEL, get_openrouter_config

        with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "key"}, clear=True):
            config = get_openrouter_config()

        self.assertEqual(config.api_key, "key")
        self.assertEqual(config.model, DEFAULT_OPENROUTER_MODEL)
        self.assertEqual(config.base_url, DEFAULT_OPENROUTER_BASE_URL)

    def test_openrouter_config_requires_api_key(self) -> None:
        from replication.score.prompt import get_openrouter_config

        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENROUTER_API_KEY"):
                get_openrouter_config()


if __name__ == "__main__":
    unittest.main()
