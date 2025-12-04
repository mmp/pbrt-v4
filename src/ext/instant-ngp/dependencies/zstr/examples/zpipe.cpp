#include <iostream>
#include <string>
#include "zstr.hpp"

int main()
{
    //
    // Create explicit zstr::streambuf feeding off the streambuf of std::cin.
    // This syntax allows for setting the buffer size and the auto-detect option.
    //
    zstr::istreambuf zsbuf(std::cin.rdbuf(), 1<<16, true);
    //
    // Create an std::istream wrapper for the zstr::streambuf.
    // NOTE: A zstr::istream constructed with a zstr::streambuf parameter would decompress twice.
    //
    std::istream is(&zsbuf);
    //
    // Turn on error reporting (otherwise, zstream exceptions are hidden).
    //
    is.exceptions(std::ios_base::badbit);
    //
    // Main loop
    //
    const std::streamsize buff_size = 1 << 16;
    char * buff = new char [buff_size];
    while (true)
    {
        is.read(buff, buff_size);
        std::streamsize cnt = is.gcount();
        if (cnt == 0) break;
        std::cout.write(buff, cnt);
    }
    delete [] buff;
}
