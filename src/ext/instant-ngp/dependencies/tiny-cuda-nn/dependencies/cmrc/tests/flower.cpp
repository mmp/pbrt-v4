#include <cmrc/cmrc.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>

CMRC_DECLARE(flower);

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Invalid arguments passed to flower\n";
        return 2;
    }
    std::cout << "Reading flower from " << argv[1] << '\n';
    std::ifstream flower_fs{argv[1], std::ios_base::binary};
    if (!flower_fs) {
        std::cerr << "Invalid filename passed to flower: " << argv[1] << '\n';
        return 2;
    }

    using iter         = std::istreambuf_iterator<char>;
    const auto fs_size = std::distance(iter(flower_fs), iter());
    flower_fs.seekg(0);

    auto       fs        = cmrc::flower::get_filesystem();
    auto       flower_rc = fs.open("flower.jpg");
    const auto rc_size   = std::distance(flower_rc.begin(), flower_rc.end());
    if (rc_size != fs_size) {
        std::cerr << "Flower file sizes do not match: FS == " << fs_size << ", RC == " << rc_size
                  << "\n";
        return 1;
    }
    if (!std::equal(flower_rc.begin(), flower_rc.end(), iter(flower_fs))) {
        std::cerr << "Flower file contents do not match\n";
        return 1;
    }
}