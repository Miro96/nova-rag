require 'json'
require_relative './helper'

class UserService
  def initialize(db)
    @db = db
  end

  def get_user(id)
    result = @db.query(id)
    format_user(result)
  end

  def delete_user(id)
    @db.execute("DELETE FROM users WHERE id = ?", id)
  end
end

def format_user(data)
  data.to_s.upcase
end
