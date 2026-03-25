import java.util.Date

class UserService(private val db: Database) {
    fun getUser(id: Int): User {
        val result = db.query(id)
        return formatUser(result)
    }

    fun deleteUser(id: Int) {
        db.execute("DELETE FROM users WHERE id = ?", id)
    }
}

fun formatUser(data: Any): User {
    return User(data.toString())
}
