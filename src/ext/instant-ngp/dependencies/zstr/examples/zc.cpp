#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "zstr.hpp"

void usage(std::ostream& os, const std::string& prog_name)
{
    os << "Use: " << prog_name << " [-c] [-o output_file] files..." << std::endl
       << "Synopsis:" << std::endl
       << "  Decompress (with `-c`, compress) files to stdout (with `-o`, to output_file)." << std::endl;
}

void cat_stream(std::istream& is, std::ostream& os)
{
    const std::streamsize buff_size = 1 << 16;
    char * buff = new char [buff_size];
    while (true)
    {
        is.read(buff, buff_size);
        std::streamsize cnt = is.gcount();
        if (cnt == 0) break;
        os.write(buff, cnt);
    }
    delete [] buff;
} // cat_stream

void decompress_files(const std::vector< std::string >& file_v, const std::string& output_file)
{
    //
    // Set up sink ostream
    //
    std::unique_ptr< std::ofstream > ofs_p;
    std::ostream * os_p = &std::cout;
    if (not output_file.empty())
    {
        ofs_p = std::unique_ptr< std::ofstream >(new strict_fstream::ofstream(output_file));
        os_p = ofs_p.get();
    }
    //
    // Process files
    //
    for (const auto& f : file_v)
    {
        //
        // If `f` is a file, create a zstr::ifstream, else (it is stdin) create a zstr::istream wrapper
        //
        std::unique_ptr< std::istream > is_p =
            (f != "-"
             ? std::unique_ptr< std::istream >(new zstr::ifstream(f))
             : std::unique_ptr< std::istream >(new zstr::istream(std::cin)));
        //
        // Cat stream
        //
        cat_stream(*is_p, *os_p);
    }
} // decompress_files

void compress_files(const std::vector< std::string >& file_v, const std::string& output_file)
{
    //
    // Set up compression sink ostream
    //
    std::unique_ptr< std::ostream > os_p =
        (not output_file.empty()
         ? std::unique_ptr< std::ostream >(new zstr::ofstream(output_file))
         : std::unique_ptr< std::ostream >(new zstr::ostream(std::cout)));
    //
    // Process files
    //
    for (const auto& f : file_v)
    {
        //
        // If `f` is a file, create an ifstream, else read stdin
        //
        std::unique_ptr< std::ifstream > ifs_p;
        std::istream * is_p = &std::cin;
        if (f != "-")
        {
            ifs_p = std::unique_ptr< std::ifstream >(new strict_fstream::ifstream(f));
            is_p = ifs_p.get();
        }
        //
        // Cat stream
        //
        cat_stream(*is_p, *os_p);
    }
} // compress_files

int main(int argc, char * argv[])
{
    bool compress = false;
    std::string output_file;
    int c;
    while ((c = getopt(argc, argv, "co:h?")) != -1)
    {
        switch (c)
        {
        case 'c':
            compress = true;
            break;
        case 'o':
            if (std::string("-") != optarg)
            {
                output_file = optarg;
            }
            break;
        case '?':
        case 'h':
            usage(std::cout, argv[0]);
            std::exit(EXIT_SUCCESS);
            break;
        default:
            usage(std::cerr, argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    //
    // Gather files to process
    //
    std::vector< std::string > file_v(&argv[optind], &argv[argc]);
    //
    // With no other arguments, process stdin
    //
    if (file_v.empty()) file_v.push_back("-");
    //
    // Perform compression/decompression
    //
    if (compress)
    {
        compress_files(file_v, output_file);
    }
    else
    {
        decompress_files(file_v, output_file);
    }
}
