# Contributing

## Commit messages

This project follows [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary).
Each commit message has the form:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

- **type** — one of `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, or `chore`.
- **scope** *(optional)* — the area touched, e.g. `grid_tensor`, `forest`, `tsl-py`, `dashboard`.
- **description** — short, imperative, lower-case, no trailing period.
- **body** *(optional)* — what changed and why, wrapped at ~72 columns.
- **breaking changes** — append `!` after the type/scope (e.g. `feat(forest)!:`) **and/or** add a `BREAKING CHANGE:` footer describing the break.

Examples:

```
feat(forest): add orthogonal-greedy OLS refit across stages
fix(grid_tensor): apply OLS scaling exactly once in predict
docs: restyle README header with floated logo and badge row
perf(grid_tensor): use binned prefix sums for split candidates
```

### Examples and other non-shipped code

`feat`, `fix`, and `perf` map to SemVer bumps of the published Rust crate and Python
package, so reserve them for changes to shipped code (`src/`, `tsl-py/src/`,
`tsl-py/python/`). Changes that don't affect the published packages take a non-release
type with a scope:

- `docs(examples):` — example scripts (`tsl-py/examples/*.py`) and their README.
- `chore(examples):` — regenerated figures (`tsl-py/examples/figures/`) or pretrained
  model binaries (`tsl-py/examples/models/`).

(This replaces the bare `example:` type used in earlier history.)

See the [Conventional Commits summary](https://www.conventionalcommits.org/en/v1.0.0/#summary)
for the full specification.
