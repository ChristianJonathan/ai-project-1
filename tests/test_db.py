def test_messages_table_exists(db):
    cols = {row[1] for row in db.execute("PRAGMA table_info(messages)")}
    assert {"id", "session_id", "content"}.issubset(cols)
