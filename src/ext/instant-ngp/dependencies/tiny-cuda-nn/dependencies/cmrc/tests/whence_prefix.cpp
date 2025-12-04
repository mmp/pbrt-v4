#include <cmrc/cmrc.hpp>

#include <iostream>

CMRC_DECLARE(whence_prefix);

int main() {
    auto fs = cmrc::whence_prefix::get_filesystem();
    auto data = fs.open("imaginary-prefix/subdir_b/file_b.txt");
    std::cout << std::string(data.begin(), data.end()) << '\n';
}
