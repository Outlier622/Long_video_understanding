"""Client metric tests kept outside the frozen serving source tree."""
import unittest

from scripts.benchmark_http_serving import percentile, summarize


class BenchmarkMetricsTests(unittest.TestCase):
    def test_rejections_do_not_make_success_latency_look_fast(self):
        rows = [{'status': 200, 'client_seconds': 20, 'body': {'output_tokens': 16}},
                {'status': 429, 'client_seconds': .01, 'body': {}}]
        report = summarize(rows, 20)
        self.assertEqual(report['success_latency_seconds']['p95'], 20)
        self.assertEqual(report['error_rate'], .5)
        self.assertEqual(report['successful_requests_per_second'], .05)
        self.assertEqual(report['returned_output_tokens_per_second'], .8)

    def test_all_failures_have_no_success_percentile(self):
        report = summarize([{'status': 502, 'client_seconds': 400, 'body': {}}], 400)
        self.assertEqual(report['successes'], 0)
        self.assertIsNone(report['success_latency_seconds']['p99'])
        self.assertEqual(report['status_counts'], {'502': 1})
        self.assertEqual(report['returned_output_tokens_per_second'], 0)

    def test_percentiles_interpolate_and_handle_empty_or_single_sample(self):
        self.assertIsNone(percentile([], .99))
        self.assertEqual(percentile([7], .99), 7)
        self.assertEqual(percentile([30, 10, 20], .95), 29)


if __name__ == '__main__':
    unittest.main()
