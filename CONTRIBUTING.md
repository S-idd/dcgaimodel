# Contributing Guidelines

## Development Philosophy

This project is built for learning and production-quality engineering.

Every feature should be:

- Well documented
- Unit tested
- Idiomatic Rust
- Modular
- Easy to understand

---

# Development Workflow

1. Design
2. Implementation
3. Unit Tests
4. cargo fmt
5. cargo check
6. cargo test
7. Git Commit
8. Git Push

---

# Commit Style

Examples:

```
Implement vector addition

Implement matrix multiplication

Add SGD optimizer

Fix dot product overflow
```

---

# Coding Standards

- Avoid duplicated code.
- Prefer composition over duplication.
- Document public APIs.
- Handle errors using Result.
- Never panic unless absolutely necessary.
- Keep modules focused on a single responsibility.

---

# Testing

Every public function should have at least one unit test.

New functionality must include tests before merging.

---

# Formatting

Always run

cargo fmt

before committing.

Always verify

cargo check

cargo test

before pushing.

---

Happy Coding 🚀
