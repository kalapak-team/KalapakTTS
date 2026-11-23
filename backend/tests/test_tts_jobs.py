import unittest
from uuid import uuid4

from app.core.tts_jobs import _jobs, create_job


class GenerationQueueTests(unittest.TestCase):
    def test_queued_jobs_report_position_and_update_when_worker_starts(self):
        user_id = uuid4()
        first = create_job(user_id, 2)
        second = create_job(user_id, 3)
        self.addCleanup(_jobs.pop, first.job_id, None)
        self.addCleanup(_jobs.pop, second.job_id, None)

        self.assertEqual(first.as_response()["queued_position"], 1)
        self.assertEqual(second.as_response()["queued_position"], 2)

        first.status = "running"
        self.assertIsNone(first.as_response()["queued_position"])
        self.assertEqual(second.as_response()["queued_position"], 1)
