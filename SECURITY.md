# Security Policy

## Supported Versions

We actively maintain and provide security updates for the `main` branch.

| Version / Branch | Supported          |
| ---------------- | ------------------ |
| `main`           | :white_check_mark: |

## Reporting a Vulnerability

We take the security of our infrastructure, code, and user data seriously. If you discover a security vulnerability or potential credential leak, please **DO NOT** open a public GitHub issue.

Instead, please report it privately:

- **Email:** Send an email to `sebaeze@gmail.com` (or contact the repository maintainers directly).
- **Details:** Include a detailed description of the vulnerability, proof-of-concept steps, and potential impact.

## Security Practices

- **Zero Hardcoded Secrets:** All infrastructure identifiers, Service Accounts, and GCS buckets must be supplied via local `.env` files or environment variables.
- **Automated Secret Scanning:** All Pull Requests undergo automated Gitleaks secret scanning and GitHub Push Protection checks.
- **Containerized Isolation:** Training pipelines execute inside ephemeral, isolated containers on GCP Vertex AI with minimal IAM permissions (`roles/storage.objectUser`).
