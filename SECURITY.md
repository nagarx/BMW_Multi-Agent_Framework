# Security Policy

## Supported Versions

Currently, we provide security updates for the following versions of BMW Agents:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue in the BMW Agents framework, please follow these steps:

1. **Do not disclose the vulnerability publicly** until it has been addressed by the maintainers.
2. Submit a report to the project maintainers by opening an issue labeled "[SECURITY]" with a clear description of the issue.
3. Include relevant information such as:
   - Type of vulnerability
   - Affected components or functionality
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Process

When a security vulnerability is reported, the maintainers will:

1. Acknowledge receipt of the vulnerability report within 48 hours
2. Assess the vulnerability and determine its scope and severity
3. Develop and test a fix
4. Release a patched version as soon as possible

## Security Best Practices

When using BMW Agents, consider these security best practices:

1. **API Keys**: Store API keys and credentials securely. Never hardcode them in your application code.
2. **Tool Permissions**: Be mindful of the permissions granted to tools used with agents. Tools should follow the principle of least privilege.
3. **Input Validation**: Validate and sanitize all inputs, especially user-provided content that might be used in prompts.
4. **Content Filtering**: Implement appropriate content filtering for agent outputs when deployed in public-facing applications.
5. **Regular Updates**: Keep your BMW Agents installation up to date with the latest security patches.

## Security-related Dependencies

Be aware of security implications in dependencies:

- **LLM Providers**: Review the security policies of any LLM provider you're using with BMW Agents
- **Third-party Tools**: Ensure any third-party tools integrated with your agents are properly vetted for security issues

## Disclosure Policy

We follow a coordinated disclosure approach. Once a vulnerability is fixed, we will publish security advisories with appropriate credits to the reporter. 