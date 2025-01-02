# Code Style Guidelines

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

For C++ code (also Metal and CUDA C++), we adhere to the Qt code style guidelines. Here are the key points:

- **Indentation**: Use 4 spaces for indentation. Do not use tabs.
- **Braces**:
    - Place the opening brace on the same line as the control statement or function definition.
    - The closing brace should be on its own line, aligned with the start of the control statement or function definition.

Example:
```cpp
if (condition) {
    doSomething();
} else {
    doSomethingElse();
}

void MyClass::myFunction()
{
    // function body
}
```

By following these guidelines, we ensure that our code is consistent and easy to read across the entire project.