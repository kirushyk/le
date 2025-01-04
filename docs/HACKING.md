# Code Style Guidelines

To keep the code consistent and ensure high quality, it would be nice to use a workflow that includes automatic formatting and style checks wherever possible.

## C Code Style

For C code, we follow the GStreamer code style guidelines. Here are the key points:

- **Indentation**: Use 2 spaces for indentation. Do not use tabs.
- **Braces**: 
    - Place the opening brace on the same line as the control statement.
    - The closing brace should be on its own line, aligned with the start of the control statement.

Example:
```c
if (condition) {
  do_something ();
} else {
  do_something_else ();
}
```

## C++ Code Style
For C++ code, including Metal and CUDA, we follow a style inspired by Qt guidelines. Here are the key points:

- **Indentation**: Use 4 spaces per indentation level. Do not use tabs.
- **Braces**:
    - Place the opening brace on the same line as the control statement or function definition.
    - The closing brace should be on its own line, aligned with the start of the control statement or function definition.
- **Naming Conventions**: 
  - Classes: CamelCase starting with an uppercase letter.
  - Functions: camelCase starting with a lowercase letter.
  - Variables: camelCase starting with a lowercase letter.
  - Constants: ALL_CAPS with underscores separating words.
- **Line Length**: Maximum of 100 characters per line.
- **Comments**: Use C++ style (//) for single-line comments and C style (/* */) for multi-line comments.
- **Spaces**: 
  - After keywords (e.g., `if`, `for`, `while`).
  - Around operators (e.g., `=`, `+`, `-`, `*`, `/`).
- **Pointer and Reference Alignment**: Attach the `*` or `&` to the variable name (e.g., `int *ptr`).
- **File Naming**: Use lowercase with underscores separating words (e.g., `my_class.cpp`).

Example:
```cpp
if (condition) {
    doSomething();
} else {
    doSomethingElse();
}

void MyClass::myFunction()
{
    function body
}
```

By following these guidelines, we ensure that our code is consistent and easy to read across the entire project.