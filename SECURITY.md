# Security Policy

## Supported versions

Security fixes are made on the latest development branch and the latest published release. Older
development snapshots are not supported.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's **Report a vulnerability**
control in this repository's Security tab. If private vulnerability reporting is unavailable,
contact the maintainer privately through their GitHub profile.

Include the affected revision, impact, and the smallest safe reproduction you can provide. Do not
include credentials, access tokens, private endpoints, customer data, or other secrets.

Maintainers aim to acknowledge reports within five business days and will coordinate remediation
and disclosure with the reporter.

## Secret handling

Never commit credentials, private keys, tokens, production data, or a real `.env` file. If a
secret is committed, revoke or rotate it immediately; removing it in a later commit is not enough.
