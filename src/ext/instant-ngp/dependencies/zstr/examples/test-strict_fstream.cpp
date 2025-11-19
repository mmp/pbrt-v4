#include <iostream>
#include <sstream>
#include <vector>
#include "strict_fstream.hpp"

template < typename Stream_Type >
void test_open(const std::string& stream_class, const std::string& stream_type,
               const std::string& filename, int mode, bool set_fail_bit)
{
    Stream_Type * s_p = new Stream_Type();
    if (set_fail_bit)
    {
        s_p->exceptions(std::ios_base::failbit);
    }
    bool exception_thrown = true;
    try
    {
        s_p->open(filename, static_cast< std::ios_base::openmode >(mode));
        exception_thrown = false;
    }
    catch (const std::exception &) {}
    std::cout << stream_class << " " << stream_type << " " << (set_fail_bit? "failbit" : "nofailbit") << " "
              << strict_fstream::detail::static_method_holder::mode_to_string(
                  static_cast< std::ios_base::openmode >(mode))
              << " " << (exception_thrown? "1" : "0") << std::endl;
    delete s_p;
}

int main(int argc, char * argv[])
{
    if (argc != 2)
    {
        std::cerr
            << "Use: " << argv[0] << " file" << std::endl
            << "Synopsis: Open `file` as a file stream object" << std::endl
            << "Stream Classes:" << std::endl
            << "  std" << std::endl
            << "  std_mask" << std::endl
            << "  strict_fstream" << std::endl
            << "Stream Types:" << std::endl
            << "  ifstream" << std::endl
            << "  ofstream" << std::endl
            << "  fstream" << std::endl
            << "Modes:" << std::endl
            << "  in=" << std::ios_base::in << std::endl
            << "  out=" << std::ios_base::out << std::endl
            << "  app=" << std::ios_base::app << std::endl
            << "  ate=" << std::ios_base::ate << std::endl
            << "  trunc=" << std::ios_base::trunc << std::endl
            << "  binary=" << std::ios_base::binary << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::vector< int > in_mode_v = {
        0,
        std::ios_base::in
    };
    std::vector< int > out_mode_v = {
        0,
        std::ios_base::out,
        std::ios_base::out | std::ios_base::app,
        std::ios_base::out | std::ios_base::trunc
    };
    std::vector< int > alt_mode_v = {
        0,
        std::ios_base::binary,
        std::ios_base::ate,
        std::ios_base::binary | std::ios_base::ate
    };
    for (const auto& in_mode : in_mode_v)
        for (const auto& out_mode : out_mode_v)
            //for (const auto& alt_mode : alt_mode_v)
            {
                int mode = in_mode | out_mode; // | alt_mode;
                //test_open< std::ifstream >("std", "ifstream", argv[1], mode, false);
                //test_open< std::ofstream >("std", "ofstream", argv[1], mode, false);
                //test_open< std::fstream  >("std", "fstream",  argv[1], mode, false);
                test_open< std::ifstream >("std", "ifstream", argv[1], mode, true);
                test_open< std::ofstream >("std", "ofstream", argv[1], mode, true);
                test_open< std::fstream  >("std", "fstream",  argv[1], mode, true);
                test_open< strict_fstream::ifstream >("strict_fstream", "ifstream", argv[1], mode, false);
                test_open< strict_fstream::ofstream >("strict_fstream", "ofstream", argv[1], mode, false);
                test_open< strict_fstream::fstream  >("strict_fstream", "fstream",  argv[1], mode, false);
            }
}
