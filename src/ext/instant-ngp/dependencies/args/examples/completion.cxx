/* Copyright © 2016-2017 Taylor C. Richberger <taywee@gmx.com> and Pavel Belikov
 * This code is released under the license described in the LICENSE file
 */

#include "args.hxx"
#include <iostream>

int main(int argc, const char **argv)
{
    args::ArgumentParser p("parser");
    args::CompletionFlag c(p, {"complete"});
    args::ValueFlag<std::string> f(p, "name", "description", {'f', "foo"}, "abc");
    args::ValueFlag<std::string> b(p, "name", "description", {'b', "bar"}, "abc");
    args::MapFlag<std::string, int> m(p, "mappos", "mappos", {'m', "map"}, {{"1",1}, {"2", 2}});
    args::Positional<std::string> pos(p, "name", "desc");

    try
    {
        p.ParseCLI(argc, argv);
    }
    catch (args::Completion &e)
    {
        std::cout << e.what();
    }

    return 0;
}
