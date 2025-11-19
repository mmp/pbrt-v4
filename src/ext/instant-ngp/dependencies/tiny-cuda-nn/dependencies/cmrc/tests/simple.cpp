#include <cmrc/cmrc.hpp>

#include <iostream>

CMRC_DECLARE(simple);

int main() {
    auto fs = cmrc::simple::get_filesystem();
    auto data = fs.open("hello.txt");
    std::cout << std::string(data.begin(), data.end()) << '\n';
}
