# Code Norms — Python

## Docstrings

All functions and methods must have a docstring. Use this format:

```python
def fun_name(arg1: type, arg2: type) -> type:
    """ Short one-line description.

    Args:
        * arg1: description
        * arg2: description

    Returns:
        * type: description

    """
```

Rules:
- The short description goes on the **same line** as the opening `"""`
- One blank line before `Args:` and before `Returns:`
- Use `*` bullets for each argument and return value
- Argument names must match the function signature exactly
- If a function returns `None`, omit the `Returns:` block entirely
- For simple one-liner functions (under 5 lines, obvious purpose), a single-line docstring is acceptable:
  ```python
  def is_active(user: User) -> bool:
      """ Returns True if the user account is active. """
  ```

## File Structure

Every `.py` file must follow this order — no exceptions:

```python
# 1. Standard library imports
import os
import json

# 2. Third-party imports
from fastapi import HTTPException
from sqlalchemy.orm import Session

# 3. Internal imports
from app.core.config import settings
from app.models.user import User

# 4. Constants (if any)
MAX_RETRIES = 3

# 5. Code (classes, functions)
```

**Do NOT add module-level docstrings** (`"""..."""`) at the top of files or before the imports block.
Use a `# comment` for file-level notes if truly necessary — and only if truly necessary.

## Naming Conventions

| Element | Convention | Example |
|---|---|---|
| Functions | `snake_case` | `get_user_by_id` |
| Variables | `snake_case` | `user_count` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_RETRIES` |
| Classes | `PascalCase` | `UserService` |
| Private methods | `_leading_underscore` | `_validate_token` |
| Files / modules | `snake_case` | `user_service.py` |

## Type Hints

All function signatures must have type hints — arguments and return type:

```python
# Good
def get_user(user_id: int) -> User:

# Bad
def get_user(user_id):
```

- Use `Optional[X]` (or `X | None` in Python 3.10+) for nullable values
- Use `list[X]`, `dict[K, V]` (lowercase, Python 3.9+) — not `List`, `Dict` from `typing`
- If a function can raise, document it in the docstring but do not add a `Raises:` block unless the exception is non-obvious

## General Rules

- Maximum line length: **100 characters**
- No unused imports — remove them, don't comment them out
- No `print()` in application code — use the logger (`from app.core.logger import get_logger`)
- Avoid inline comments that just restate the code:
  ```python
  # Bad
  user_count += 1  # increment user count

  # Good — only comment the non-obvious
  user_count += 1  # includes soft-deleted users per business rule
  ```
- One blank line between methods inside a class; two blank lines between top-level definitions

## Example: complete function

```python
def calculate_enrichment_score(
    attributes: list[dict],
    weight_map: dict[str, float],
) -> float:
    """ Calculates a weighted enrichment score for a set of attributes.

    Args:
        * attributes: list of attribute dicts with 'key' and 'value' fields
        * weight_map: mapping of attribute keys to their weight (0.0 to 1.0)

    Returns:
        * float: normalized score between 0.0 and 1.0

    """
    if not attributes:
        return 0.0

    total = sum(
        weight_map.get(attr["key"], 0.0)
        for attr in attributes
        if attr.get("value") is not None
    )
    return min(total, 1.0)
```
