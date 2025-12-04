#include <cmrc/cmrc.hpp>

#include <iostream>

CMRC_DECLARE(whence);

int main() {
    auto fs = cmrc::whence::get_filesystem();
    auto data = fs.open("subdir_b/file_a.txt");
    std::cout << std::string(data.begin(), data.end()) << '\n';
}
