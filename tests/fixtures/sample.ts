interface User {
  id: number;
  name: string;
  email: string;
}

class AuthService {
  private secret: string;

  constructor(secret: string) {
    this.secret = secret;
  }

  validateToken(token: string): boolean {
    return token.startsWith(this.secret);
  }

  generateToken(user: User): string {
    return `${this.secret}_${user.id}_${Date.now()}`;
  }
}

function formatError(error: Error): { message: string; stack?: string } {
  return {
    message: error.message,
    stack: error.stack,
  };
}

export { AuthService, formatError };
export type { User };
