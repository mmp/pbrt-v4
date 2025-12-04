#ifdef _WIN32
#include "windows.h"
#else
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
#endif

#include <iostream>
#include <args.hxx>

int main(int argc, char** argv)
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::CompletionFlag completion(parser, {"complete"});
    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch (args::Completion& e)
    {
        std::cout << e.what();
        return 0;
    }
    catch (args::Help&)
    {
        std::cout << parser;
        return 0;
    }
    catch (args::ParseError& e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    return 0;
}
