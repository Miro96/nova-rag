"""Sample Python file for testing the chunker."""


class UserService:
    """Handles user-related operations."""

    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: int):
        """Fetch a user by ID."""
        return self.db.query(f"SELECT * FROM users WHERE id = {user_id}")

    def create_user(self, name: str, email: str):
        """Create a new user."""
        return self.db.execute(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (name, email),
        )

    def delete_user(self, user_id: int):
        """Delete a user by ID."""
        return self.db.execute("DELETE FROM users WHERE id = ?", (user_id,))


def authenticate(username: str, password: str) -> bool:
    """Check user credentials against the database."""
    # This is a standalone function for authentication
    return username == "admin" and password == "secret"


def handle_error(error: Exception) -> dict:
    """Convert an exception to an error response."""
    return {
        "error": True,
        "message": str(error),
        "type": type(error).__name__,
    }
