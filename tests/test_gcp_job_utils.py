import unittest
from unittest.mock import patch, MagicMock

from gcp_job_utils import (
    build_candidate_regions,
    _is_retryable_error,
    _cancel_job,
    DEFAULT_FALLBACK_REGIONS,
)

class TestGCPJobUtils(unittest.TestCase):
    def test_build_candidate_regions(self):
        primary = "us-central1"
        candidates = build_candidate_regions(primary)
        self.assertEqual(candidates[0], "us-central1")
        self.assertIn("us-west1", candidates)
        self.assertIn("europe-west4", candidates)
        self.assertIn("asia-east1", candidates)
        self.assertEqual(len(candidates), len(set(candidates)))

    def test_build_candidate_regions_timezone_americas_day(self):
        # At 15:00 UTC (Daytime in Americas, off-peak in Europe & Asia),
        # European and Asian regions should be prioritized before other US regions
        primary = "us-central1"
        candidates = build_candidate_regions(primary, current_utc_hour=15)
        self.assertEqual(candidates[0], "us-central1")
        # europe-west4 should come before us-west1 in daytime fallback
        eu_idx = candidates.index("europe-west4")
        us_idx = candidates.index("us-west1")
        self.assertLess(eu_idx, us_idx)

    def test_build_candidate_regions_timezone_americas_night(self):
        # At 04:00 UTC (Night in Americas, off-peak in Americas),
        # North American regions should come before European regions
        primary = "us-central1"
        candidates = build_candidate_regions(primary, current_utc_hour=4)
        self.assertEqual(candidates[0], "us-central1")
        us_idx = candidates.index("us-west1")
        eu_idx = candidates.index("europe-west4")
        self.assertLess(us_idx, eu_idx)

    def test_build_candidate_regions_custom(self):
        candidates = build_candidate_regions("us-east1", ["us-central1", "us-east1", "us-west1"])
        self.assertEqual(candidates, ["us-east1", "us-central1", "us-west1"])

    def test_is_retryable_error(self):
        self.assertTrue(_is_retryable_error("Resources are insufficient in region: us-central1"))
        self.assertTrue(_is_retryable_error("Internal error occurred for the current attempt"))
        self.assertFalse(_is_retryable_error("Permission_denied: SA lacks Vertex AI User role"))
        self.assertFalse(_is_retryable_error("Invalid_argument: Machine type g2-standard-4 invalid"))

    @patch("subprocess.run")
    def test_cancel_job(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        _cancel_job("dev-gemini-ai", "us-central1", "123456789")
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        self.assertIn("gcloud ai custom-jobs cancel 123456789", cmd)
        self.assertIn("--region=us-central1", cmd)

if __name__ == "__main__":
    unittest.main()
