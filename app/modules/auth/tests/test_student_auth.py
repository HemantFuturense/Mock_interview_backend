from app.modules.auth.service import hash_password, verify_password


def test_hash_and_verify_password_roundtrip():
    hashed = hash_password("demo123")

    assert hashed.startswith("pbkdf2_sha256$")
    assert verify_password("demo123", hashed) is True
    assert verify_password("wrong-password", hashed) is False
