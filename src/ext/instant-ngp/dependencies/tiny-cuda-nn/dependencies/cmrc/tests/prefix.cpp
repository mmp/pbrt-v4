#include <cmrc/cmrc.hpp>

#include <iostream>

CMRC_DECLARE(prefix);

int main() {
    auto fs = cmrc::prefix::get_filesystem();
    auto data = fs.open("some-prefix/hello.txt");
    std::cout << std::string(data.begin(), data.end()) << '\n';
}
