#include <cmrc/cmrc.hpp>

#include <iostream>

CMRC_DECLARE(iterate);

int main() {
    auto fs = cmrc::iterate::get_filesystem();
    for (auto&& entry : fs.iterate_directory("")) {
        std::cout << entry.filename() << '\n';
    }
    for (auto&& entry : fs.iterate_directory("subdir_a/subdir_b")) {
        std::cout << entry.filename() << '\n';
    }
}