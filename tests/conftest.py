import pytest, sqlite3, os, app.main as main  # adjust import path if needed

@pytest.fixture(scope="session")
def db():
    path = "/tmp/test_chat.db"
    os.environ["DB_PATH"] = path      # point main.init_db() to temp file
    yield main.init_db()
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
