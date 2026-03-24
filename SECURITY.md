# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do NOT** open a public GitHub issue
2. Email the details to the maintainer or use [GitHub Security Advisories](https://github.com/Miro96/nova-rag/security/advisories/new)
3. Include steps to reproduce and potential impact

We will respond within 48 hours and provide a fix as soon as possible.

## Security Design

nova-rag runs 100% locally:
- No data is sent to external servers
- Embedding model runs on your machine
- All indexes are stored locally in `~/.nova-rag/`
- No API keys required
- No network connections after initial model download
