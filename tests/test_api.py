import pytest
from palimpsest.api.app import app, chat, get_memory, pin_memory_item, get_versions, ChatRequest, PinRequest

def test_api_endpoints_directly():
    # 1. Test chat endpoint
    chat_req = ChatRequest(
        session_id="api_direct_test",
        message="Remember: Always format outputs as bullet points.",
        create_snapshot=True,
        snapshot_label="Format Instruction Established"
    )
    result = chat(chat_req)
    assert result.session_id == "api_direct_test"
    assert result.active_memory_count >= 1
    assert result.reconstructed_context.total_tokens > 0

    # 2. Test get_memory endpoint
    items = get_memory(session_id="api_direct_test", include_discarded=False)
    assert len(items) >= 1
    user_items = [it for it in items if "format outputs" in it.content]
    assert len(user_items) >= 1
    target_id = user_items[0].id

    # 3. Test pin endpoint
    pin_res = pin_memory_item(target_id, PinRequest(is_pinned=False))
    assert pin_res["status"] == "success"
    assert pin_res["is_pinned"] is False

    # 4. Test get_versions endpoint
    versions = get_versions()
    assert len(versions) >= 1

