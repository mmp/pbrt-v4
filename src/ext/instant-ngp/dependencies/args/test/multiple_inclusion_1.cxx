#include <args.hxx>

int foo() { return 42; }
int bar();

int main() { return foo() - bar(); }
