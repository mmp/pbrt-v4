/* Copyright © 2016 Taylor C. Richberger <taywee@gmx.com>
 * This code is released under the license described in the LICENSE file
 */

#include <iostream>
#include <unordered_map>
#include <functional>
#include <args.hxx>

void Init(const std::string &progname, std::vector<std::string>::const_iterator beginargs, std::vector<std::string>::const_iterator endargs);
void Add(const std::string &progname, std::vector<std::string>::const_iterator beginargs, std::vector<std::string>::const_iterator endargs);

using commandtype = std::function<void(const std::string &, std::vector<std::string>::const_iterator, std::vector<std::string>::const_iterator)>;

int main(int argc, char **argv)
{
    std::unordered_map<std::string, commandtype> map{
        {"init", Init},
        {"add", Add}};

    const std::vector<std::string> args(argv + 1, argv + argc);
    args::ArgumentParser parser("This is a git-like program", "Valid commands are init and add");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    parser.Prog(argv[0]);
    parser.ProglinePostfix("{command options}");
    args::Flag version(parser, "version", "Show the version of this program", {"version"});
    args::ValueFlag<std::string> htmlpath(parser, "html-path", "Specify the html path", {"html-path"});
    args::MapPositional<std::string, commandtype> command(parser, "command", "Command to execute", map);
    command.KickOut(true);
    try
    {
        auto next = parser.ParseArgs(args);
        std::cout << std::boolalpha;
        std::cout << "Before command options:" << std::endl;
        std::cout << "Version called: " << bool{version} << std::endl;
        std::cout << "html-path called: " << bool{htmlpath} << ", value: " << args::get(htmlpath) << std::endl;
        if (command)
        {
            args::get(command)(argv[0], next, std::end(args));
        } else
        {
            std::cout << parser;
        }
    }
    catch (args::Help)
    {
        std::cout << parser;
        return 0;
    }
    catch (args::Error e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }
    return 0;
}

void Init(const std::string &progname, std::vector<std::string>::const_iterator beginargs, std::vector<std::string>::const_iterator endargs)
{
    std::cout << "In Init" << std::endl;
    args::ArgumentParser parser("");
    parser.Prog(progname + " init");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::ValueFlag<std::string> templatedir(parser, "template-directory", "directory from which templates will be used", {"template"});
    args::Flag bare(parser, "bare", "create a bare repository", {"bare"});
    args::Flag quiet(parser, "quiet", "be quiet", {'q', "quiet"});
    args::Positional<std::string> directory(parser, "directory", "The directory to create in", ".");
    try
    {
        parser.ParseArgs(beginargs, endargs);
        std::cout << std::boolalpha;
        std::cout << "templatedir: " << bool{templatedir} << ", value: " << args::get(templatedir) << std::endl;
        std::cout << "bare: " << bool{bare} << std::endl;
        std::cout << "quiet: " << bool{quiet} << std::endl;
        std::cout << "directory: " << bool{directory} << ", value: " << args::get(directory) << std::endl;
    }
    catch (args::Help)
    {
        std::cout << parser;
        return;
    }
    catch (args::ParseError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return;
    }
}

void Add(const std::string &progname, std::vector<std::string>::const_iterator beginargs, std::vector<std::string>::const_iterator endargs)
{
    std::cout << "In Add" << std::endl;
    args::ArgumentParser parser("");
    parser.Prog(progname + " add");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    args::Flag dryrun(parser, "dryrun", "dry run", {'n', "dry-run"});
    args::Flag verbose(parser, "verbose", "be verbose", {'v', "verbose"});
    args::Flag refresh(parser, "refresh", "Don't add, only refresh the index", {"refresh"});
    args::PositionalList<std::string> pathspec(parser, "pathspec", "pathspecs");
    try
    {
        parser.ParseArgs(beginargs, endargs);
        std::cout << std::boolalpha;
        std::cout << "dryrun: " << bool{dryrun} << std::endl;;
        std::cout << "verbose: " << bool{verbose} << std::endl;
        std::cout << "refresh: " << bool{refresh} << std::endl;
        std::cout << "pathspec: " << bool{pathspec} << std::endl;
        if (pathspec)
        {
            std::cout << "values: " << std::endl;
            for (const auto &spec: args::get(pathspec))
            {
                std::cout << " - " << spec << std::endl;
            }
        }
    }
    catch (args::Help)
    {
        std::cout << parser;
        return;
    }
    catch (args::ParseError e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return;
    }
}
