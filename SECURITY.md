# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of this project seriously. If you discover a security vulnerability, please follow these steps:

### 1. **Do Not** Open a Public Issue

Security vulnerabilities should not be reported through public GitHub issues to prevent exploitation before a fix is available.

### 2. Report Privately

Send your report via one of these channels:

- **GitHub Security Advisories**: Use the "Security" tab in this repository to report a vulnerability privately
- **Discord**: Contact a maintainer directly on the [huggingscience Discord server](https://discord.com/invite/VYkdEVjJ5J)
- **Email**: [Include project maintainer email if available]

### 3. Include in Your Report

To help us assess and address the issue quickly, include:

- Description of the vulnerability and its potential impact
- Steps to reproduce the issue
- Affected versions
- Any potential fixes or mitigations you've identified
- Your contact information for follow-up questions

### 4. What to Expect

- **Acknowledgment**: We'll confirm receipt of your report within 48 hours
- **Assessment**: We'll investigate and provide an initial assessment within 5 business days
- **Updates**: We'll keep you informed of our progress
- **Resolution**: Once fixed, we'll coordinate disclosure timing with you
- **Credit**: We'll acknowledge your contribution in the security advisory (unless you prefer to remain anonymous)

## Security Considerations for This Project

### Data Privacy

This project processes sensitive health and microbiome data:

- **Never commit real patient data** to the repository
- Use synthetic or properly anonymized datasets for examples
- Follow HIPAA, GDPR, and other relevant data protection regulations
- Implement proper access controls when deploying models

### Model Security

- **Adversarial robustness**: Be aware that ML models can be vulnerable to adversarial inputs
- **Model poisoning**: Validate data sources to prevent training on malicious data
- **Checkpoint integrity**: Verify checksums of downloaded model checkpoints
- **Dependency scanning**: Regularly update dependencies to patch known vulnerabilities

### Deployment Best Practices

If deploying this model in production:

- Use encrypted connections (HTTPS/TLS) for all data transmission
- Implement authentication and authorization controls
- Log access appropriately while respecting privacy
- Regularly audit and update security measures
- Follow responsible AI practices for healthcare applications

## Dependencies

We use automated tools to monitor dependencies for known vulnerabilities:

- Regular dependency updates via Dependabot
- Security scanning in CI/CD pipelines

You can check for vulnerabilities in your local environment:

```bash
pip install safety
safety check
```

## Scope

This security policy applies to:

- Core codebase in this repository
- Official releases and distributions
- Documentation and examples

It does not cover:

- Third-party dependencies (report to their respective projects)
- Forks or derivative works
- User-specific deployments and configurations

## Questions?

If you have questions about this security policy or general security concerns, open a discussion in the repository or reach out on Discord.

---

**Remember**: Security is a shared responsibility. Follow best practices when using this code, especially with sensitive health data.
