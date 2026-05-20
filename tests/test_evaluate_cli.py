import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from replication.entity import PassageResponses, PassageResult, PassageScores


def result_item(
    idx: int,
    annotation: list[str],
    scores: PassageScores,
    responses: PassageResponses | None = None,
) -> PassageResult:
    return PassageResult(
        dataset_idx=idx,
        wiki_bio_test_idx=100 + idx,
        wiki_bio_text=f"wiki {idx}",
        main_passage=f"main {idx}",
        main_sentences=[f"sentence {idx}-{sent_idx}" for sent_idx in range(len(annotation))],
        annotation=annotation,
        sample_passages=[f"sample {idx}"],
        scores=scores,
        responses=responses or PassageResponses(),
    )


class EvaluationDatasetTest(unittest.TestCase):
    def test_active_variants_only_returns_variants_with_scores(self) -> None:
        from replication.evaluation.dataset import EvaluationDataset

        dataset = EvaluationDataset([
            result_item(0, ["accurate"], PassageScores(prompt=[0.1])),
            result_item(1, ["major_inaccurate"], PassageScores(bert=[0.9])),
        ])

        self.assertEqual(dataset.active_variants(["prompt", "bert", "nli"]), ["prompt", "bert"])

    def test_slice_uses_inclusive_end_index(self) -> None:
        from replication.evaluation.dataset import EvaluationDataset

        dataset = EvaluationDataset([
            result_item(0, ["accurate"], PassageScores(prompt=[0.1])),
            result_item(1, ["accurate"], PassageScores(prompt=[0.2])),
            result_item(2, ["accurate"], PassageScores(prompt=[0.3])),
        ])

        sliced = dataset.slice(start=1, end=2)

        self.assertEqual([result.dataset_idx for result in sliced.results], [1, 2])


class MetricSuiteTest(unittest.TestCase):
    def test_perfect_scores_produce_expected_percent_metrics(self) -> None:
        from replication.evaluation.dataset import EvaluationDataset
        from replication.evaluation.metrics import MetricSuite

        dataset = EvaluationDataset([
            result_item(0, ["accurate", "minor_inaccurate"], PassageScores(prompt=[0.1, 0.8])),
            result_item(1, ["accurate", "major_inaccurate"], PassageScores(prompt=[0.2, 0.9])),
        ])

        metrics = MetricSuite().evaluate_variant(dataset, "prompt")

        self.assertEqual(metrics.variant, "prompt")
        self.assertEqual(metrics.n_passages, 2)
        self.assertEqual(metrics.n_sentences, 4)
        self.assertAlmostEqual(metrics.nonfact, 100.0)
        self.assertAlmostEqual(metrics.nonfact_star, 100.0)
        self.assertAlmostEqual(metrics.factual, 100.0)
        self.assertAlmostEqual(metrics.pearson, 100.0)
        self.assertAlmostEqual(metrics.spearman, 100.0)


class EvaluationReportTest(unittest.TestCase):
    def test_response_distribution_normalizes_prompt_responses(self) -> None:
        from replication.evaluation.dataset import EvaluationDataset
        from replication.evaluation.report import ResponseDistribution

        dataset = EvaluationDataset([
            result_item(
                0,
                ["accurate"],
                PassageScores(prompt=[0.1]),
                PassageResponses(prompt=[["Yes, supported", " no", None, "", "maybe"]]),
            )
        ])

        dist = ResponseDistribution.from_dataset(dataset)

        self.assertEqual(dist.total, 5)
        self.assertEqual(dist.counts["Yes"], 1)
        self.assertEqual(dist.counts["No"], 1)
        self.assertEqual(dist.counts["null"], 1)
        self.assertEqual(dist.counts["(empty)"], 1)
        self.assertEqual(dist.counts["maybe"], 1)

    def test_renderer_matches_existing_summary_sections(self) -> None:
        from replication.evaluation.report import EvaluationReport, EvaluationReportRenderer

        report = EvaluationReport(
            metrics=[
                {
                    "variant": "prompt",
                    "n_passages": 1,
                    "n_sentences": 2,
                    "nonfact": 100.0,
                    "nonfact_star": 100.0,
                    "factual": 100.0,
                    "pearson": 100.0,
                    "spearman": 100.0,
                }
            ],
            variants=["prompt"],
            response_distribution={"total": 2, "counts": {"Yes": 1, "No": 1}},
        )

        output = EvaluationReportRenderer().render(report)

        self.assertIn("Response distribution (N=2 total API calls):", output)
        self.assertIn("Passages: 1 | Sentences: 2", output)
        self.assertIn("SelfCk-Prompt", output)
        self.assertIn("Paper (Table 2, N=20):", output)

    def test_renderer_reports_missing_score_data(self) -> None:
        from replication.evaluation.report import EvaluationReport, EvaluationReportRenderer

        output = EvaluationReportRenderer().render(EvaluationReport(metrics=[], variants=[]))

        self.assertEqual(output, "No score data found for the requested variant(s).\n")


class EvaluationCliTest(unittest.TestCase):
    def test_root_main_prints_rendered_report_for_requested_slice(self) -> None:
        from evaluate import main

        with tempfile.TemporaryDirectory() as tmp:
            results_path = Path(tmp) / "results.json"
            results = [
                result_item(0, ["accurate"], PassageScores(prompt=[0.1])),
                result_item(
                    1,
                    ["accurate", "major_inaccurate"],
                    PassageScores(prompt=[0.1, 0.9]),
                    PassageResponses(prompt=[["Yes", "No"]]),
                ),
                result_item(2, ["major_inaccurate"], PassageScores(prompt=[0.9])),
            ]
            results_path.write_text(json.dumps([result.to_dict() for result in results]))

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                main([
                    "--output",
                    str(results_path),
                    "--variant",
                    "prompt",
                    "--start",
                    "1",
                    "--end",
                    "2",
                ])

        output = stdout.getvalue()
        self.assertIn("Response distribution (N=2 total API calls):", output)
        self.assertIn("Passages: 2 | Sentences: 3", output)

    def test_parser_uses_output_option_for_result_file_path(self) -> None:
        from evaluate import build_parser

        parser = build_parser()
        args = parser.parse_args(["--output", "results.json", "--variant", "bert"])

        self.assertEqual(args.output, "results.json")
        self.assertFalse(hasattr(args, "results"))

    def test_evaluation_package_does_not_include_cli_module(self) -> None:
        self.assertIsNone(importlib.util.find_spec("replication.evaluation.cli"))


if __name__ == "__main__":
    unittest.main()
