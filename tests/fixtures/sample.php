<?php

use App\Models\User;

class UserService {
    private $db;

    public function __construct($db) {
        $this->db = $db;
    }

    public function getUser(int $id): User {
        $result = $this->db->query($id);
        return $this->formatUser($result);
    }

    private function formatUser($data): User {
        return new User($data);
    }
}

function helper(): string {
    return "helper";
}
