#include <cmrc/cmrc.hpp>

#include <iostream>

CMRC_DECLARE(enoent);

int main() {
    auto fs = cmrc::enoent::get_filesystem();
    try {
        auto data = fs.open("hello.txt");
    } catch (std::system_error e) {
        if (e.code() == std::errc::no_such_file_or_directory) {
            return 1;
        }
    }
    return 0;
}
