import json
from unittest.mock import AsyncMock, MagicMock, patch

from app.modules.interview.service import generate_and_process_feedback_background


def _qa_row(q_num, q_text):
    # (q_num, q_text, ans_text, sent_score, mand_skills, vid_sent, vid_dem,
    #  code_sub, code_succ, code_run, q_type)
    return (q_num, q_text, "some answer", 0.8, "python,sql", None, None, None, None, None, "standard")


def _run_feedback_flow(session_id, qa_history, company_name, rag_side_effect):
    """Patch all collaborators and drive generate_and_process_feedback_background,
    returning the render_template call kwargs plus the mocks for assertions."""
    mock_repo = MagicMock()
    mock_repo.get_pending_video_count.return_value = (0, 0)
    mock_repo.get_qa_history_for_feedback.return_value = (
        qa_history, "Backend Engineer", "Software", company_name, "standard",
    )
    mock_repo.save_detailed_feedback.return_value = True

    mock_render_template = MagicMock(return_value="rendered-prompt")

    mock_gemini_response = MagicMock()
    mock_gemini_response.text = json.dumps({"questions": []})
    mock_generate_content = AsyncMock(return_value=mock_gemini_response)

    mock_retrieve = AsyncMock(side_effect=rag_side_effect)

    with patch("app.modules.interview.service.InterviewRepository", mock_repo), \
         patch("app.modules.interview.service.render_template", mock_render_template), \
         patch("app.modules.interview.service.generate_content_with_fallback", mock_generate_content), \
         patch("app.modules.interview.service.retrieve_company_context", mock_retrieve), \
         patch("app.modules.interview.service.update_scores_from_feedback", MagicMock(return_value=True)):
        import asyncio
        asyncio.run(generate_and_process_feedback_background(session_id))

    assert mock_repo.mark_feedback_failed.called is False, (
        f"feedback marked failed unexpectedly: {mock_repo.mark_feedback_failed.call_args}"
    )
    assert mock_repo.mark_feedback_completed.called is True

    return mock_render_template.call_args.kwargs, mock_retrieve


def test_reference_answers_populated_when_rag_match_found():
    qa_history = [_qa_row(1, "Tell me about a time you led a project."), _qa_row(2, "Explain REST vs RPC.")]

    async def rag_side_effect(company_name, query, top_k=5):
        if "led a project" in query:
            return ["Acme values ownership: describe a project you drove end to end."]
        return []

    kwargs, mock_retrieve = _run_feedback_flow(
        "session-1", qa_history, company_name="Acme", rag_side_effect=rag_side_effect,
    )

    assert kwargs["reference_answers"] == [
        {"number": 1, "answer": "Acme values ownership: describe a project you drove end to end."}
    ]
    assert mock_retrieve.await_count == 2


def test_reference_answers_empty_when_no_rag_match():
    qa_history = [_qa_row(1, "Tell me about a time you led a project."), _qa_row(2, "Explain REST vs RPC.")]

    async def rag_side_effect(company_name, query, top_k=5):
        return []

    kwargs, mock_retrieve = _run_feedback_flow(
        "session-2", qa_history, company_name="Acme", rag_side_effect=rag_side_effect,
    )

    assert kwargs["reference_answers"] == []
    assert mock_retrieve.await_count == 2


def test_rag_skipped_entirely_when_no_company_name():
    qa_history = [_qa_row(1, "Tell me about a time you led a project.")]

    mock_retrieve_should_not_be_called = AsyncMock(side_effect=AssertionError("RAG must not be called"))

    kwargs, mock_retrieve = _run_feedback_flow(
        "session-3", qa_history, company_name=None, rag_side_effect=mock_retrieve_should_not_be_called,
    )

    assert kwargs["reference_answers"] == []
    assert mock_retrieve.await_count == 0
