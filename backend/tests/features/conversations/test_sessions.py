from tests import _isolate  # noqa: F401  (must come before any app import)
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.shared import storage
from app.main import app
from app.shared.models import AskRequest
from app.features.ask import router as qa
from app.features.conversations import sessions

CLIENT = "browser-1234-abcd"


class SessionTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        patcher = patch.object(storage, "DATA_DIR", Path(self.directory.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.directory.cleanup)


class StoreTest(SessionTestCase):
    def test_turns_are_kept_per_browser_and_titled_by_the_first_question(self):
        conversation = sessions.create(CLIENT, "v1")
        sessions.append_turn(conversation["id"], CLIENT, {"question": "Liên hợp quốc ra đời khi nào?", "answer": "1945"})
        sessions.append_turn(conversation["id"], CLIENT, {"question": "Còn ASEAN?", "answer": "1967"})
        listed = sessions.list_for(CLIENT)
        self.assertEqual([item["title"] for item in listed], ["Liên hợp quốc ra đời khi nào?"])
        self.assertEqual(listed[0]["turns"], 2)
        self.assertEqual(sessions.list_for("someone-else-999"), [])
        self.assertIsNone(sessions.get(conversation["id"], "someone-else-999"))
        self.assertEqual([t["question"] for t in sessions.recent(sessions.get(conversation["id"], CLIENT), 1)],
                         ["Còn ASEAN?"])

    def test_empty_conversations_stay_out_of_the_list_and_bad_ids_are_rejected(self):
        sessions.create(CLIENT, "v1")
        self.assertEqual(sessions.list_for(CLIENT), [])
        self.assertIsNone(sessions.get("../../etc/passwd", CLIENT))
        self.assertFalse(sessions.valid_client("short"))
        self.assertTrue(sessions.valid_client(CLIENT))


class PrepareTest(SessionTestCase):
    def ask(self, question, conversation_id=None):
        return AskRequest(video_id="v1", question=question, conversation_id=conversation_id)

    def started(self):
        conversation = sessions.create(CLIENT, "v1")
        sessions.append_turn(conversation["id"], CLIENT, {
            "question": "Liên hợp quốc ra đời trong hoàn cảnh nào?", "answer": "Sau Hội nghị Yalta [02:10].",
            "sources": [{"segment_id": "v1:seg_00005"}]})
        return conversation["id"]

    def test_a_follow_up_is_rewritten_and_prefers_the_slides_just_cited(self):
        conversation_id = self.started()
        with patch.object(qa.llm, "rewrite_question", return_value="ASEAN ra đời trong hoàn cảnh nào?") as rewrite:
            plan = qa.prepare(self.ask("Còn ASEAN thì sao?", conversation_id), CLIENT)
        rewrite.assert_called_once()
        self.assertEqual(plan["query"], "ASEAN ra đời trong hoàn cảnh nào?")
        self.assertEqual(plan["prefer"], frozenset({"v1:seg_00005"}))
        self.assertEqual(len(plan["history"]), 1)
        self.assertIsNone(plan["previous"])   # the history block replaces the one-line hint

    def test_a_failed_rewrite_falls_back_to_the_previous_question(self):
        conversation_id = self.started()
        with patch.object(qa.llm, "rewrite_question", return_value=None):
            plan = qa.prepare(self.ask("Còn ASEAN thì sao?", conversation_id), CLIENT)
        self.assertEqual(plan["query"], "Liên hợp quốc ra đời trong hoàn cảnh nào? Còn ASEAN thì sao?")

    def test_a_standalone_question_is_searched_as_it_is(self):
        conversation_id = self.started()
        with patch.object(qa.llm, "rewrite_question") as rewrite:
            plan = qa.prepare(self.ask("Mặt trận Việt Minh thành lập năm nào?", conversation_id), CLIENT)
        rewrite.assert_not_called()
        self.assertEqual(plan["query"], "Mặt trận Việt Minh thành lập năm nào?")
        self.assertEqual(plan["prefer"], frozenset())

    def test_no_browser_id_means_no_saved_conversation(self):
        plan = qa.prepare(self.ask("Còn ASEAN thì sao?"), None)
        self.assertIsNone(plan["conversation"])

    def test_the_summary_runs_once_enough_turns_left_the_history_window(self):
        conversation_id = self.started()
        plan = {"conversation": {"id": conversation_id}, "rewritten": None, "query": "q"}
        with patch.object(qa.llm, "summarize", return_value="Đã hỏi về LHQ và ASEAN.") as summarize, \
                patch.object(qa.threading, "Thread") as thread:
            thread.side_effect = lambda target, daemon: type("T", (), {"start": staticmethod(target)})()
            for n in range(qa.HISTORY_TURNS + qa.SUMMARY_EVERY - 1):
                qa.remember(self.ask(f"Câu {n}", conversation_id), CLIENT, plan, f"Đáp {n}", [], 1.0)
        summarize.assert_called_once()
        saved = sessions.get(conversation_id, CLIENT)
        self.assertEqual(saved["summary"], "Đã hỏi về LHQ và ASEAN.")
        self.assertEqual(saved["summarized_turns"], qa.SUMMARY_EVERY)


class RoutesTest(SessionTestCase):
    def test_list_open_rename_delete(self):
        conversation = sessions.create(CLIENT, "v1")
        sessions.append_turn(conversation["id"], CLIENT, {"question": "Xác suất là gì?", "answer": "P(A) = n(A)/n(Ω)"})
        api = TestClient(app)
        headers = {"X-Client-Id": CLIENT}
        self.assertEqual(api.get("/api/conversations").status_code, 400)
        self.assertEqual(len(api.get("/api/conversations", headers=headers).json()), 1)
        opened = api.get(f"/api/conversations/{conversation['id']}", headers=headers).json()
        self.assertEqual(opened["turns"][0]["answer"], "P(A) = n(A)/n(Ω)")
        renamed = api.patch(f"/api/conversations/{conversation['id']}", json={"title": "Xác suất"}, headers=headers)
        self.assertEqual(renamed.json()["title"], "Xác suất")
        self.assertEqual(api.delete(f"/api/conversations/{conversation['id']}", headers=headers).status_code, 204)
        self.assertEqual(api.get(f"/api/conversations/{conversation['id']}", headers=headers).status_code, 404)


if __name__ == "__main__":
    unittest.main()
