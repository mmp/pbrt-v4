/* Copyright © 2016-2017 Taylor C. Richberger <taywee@gmx.com> and Pavel Belikov
 * This code is released under the license described in the LICENSE file
 */

#include <tuple>
#include <iostream>

std::istream& operator>>(std::istream& is, std::tuple<int, int>& ints)
{
    is >> std::get<0>(ints);
    is.get();
    is >> std::get<1>(ints);
    return is;
}

#include <args.hxx>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("Help flag throws Help exception", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{}));
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"--help"}), args::Help);
}

TEST_CASE("Unknown flags throw exceptions", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{}));
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"--Help"}), args::ParseError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-H"}), args::ParseError);
}

TEST_CASE("Boolean flags work as expected, with clustering", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::Flag foo(parser, "FOO", "test flag", {'f', "foo"});
    args::Flag bar(parser, "BAR", "test flag", {'b', "bar"});
    args::Flag baz(parser, "BAZ", "test flag", {'a', "baz"});
    args::Flag bix(parser, "BAZ", "test flag", {'x', "bix"});
    parser.ParseArgs(std::vector<std::string>{"--baz", "-fb"});
    REQUIRE(foo);
    REQUIRE(bar);
    REQUIRE(baz);
    REQUIRE_FALSE(bix);
}

TEST_CASE("Count flag works as expected", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::CounterFlag foo(parser, "FOO", "test flag", {'f', "foo"});
    args::CounterFlag bar(parser, "BAR", "test flag", {'b', "bar"}, 7);
    args::CounterFlag baz(parser, "BAZ", "test flag", {'z', "baz"}, 7);
    parser.ParseArgs(std::vector<std::string>{"--foo", "-fb", "--bar", "-b", "-f", "--foo"});
    REQUIRE(foo);
    REQUIRE(bar);
    REQUIRE_FALSE(baz);
    REQUIRE(*foo == 4);
    REQUIRE(*bar == 10);
    REQUIRE(*baz == 7);
}

TEST_CASE("Argument flags work as expected, with clustering", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::ValueFlag<std::string> foo(parser, "FOO", "test flag", {'f', "foo"});
    args::Flag bar(parser, "BAR", "test flag", {'b', "bar"});
    args::ValueFlag<double> baz(parser, "BAZ", "test flag", {'a', "baz"});
    args::ValueFlag<char> bim(parser, "BAZ", "test flag", {'B', "bim"});
    args::Flag bix(parser, "BAZ", "test flag", {'x', "bix"});
    parser.ParseArgs(std::vector<std::string>{"-bftest", "--baz=7.555e2", "--bim", "c"});
    REQUIRE(foo);
    REQUIRE(*foo == "test");
    REQUIRE(bar);
    REQUIRE(baz);
    REQUIRE((*baz > 755.49 && *baz < 755.51));
    REQUIRE(bim);
    REQUIRE(*bim == 'c');
    REQUIRE_FALSE(bix);
}

TEST_CASE("Passing an argument to a non-argument flag throws an error", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::Flag bar(parser, "BAR", "test flag", {'b', "bar"});
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"--bar=test"}), args::ParseError);
}

TEST_CASE("Unified argument lists for match work", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::ValueFlag<std::string> foo(parser, "FOO", "test flag", {'f', "foo"});
    args::Flag bar(parser, "BAR", "test flag", {"bar", 'b'});
    args::ValueFlag<double> baz(parser, "BAZ", "test flag", {'a', "baz"});
    args::ValueFlag<char> bim(parser, "BAZ", "test flag", {'B', "bim"});
    args::Flag bix(parser, "BAZ", "test flag", {"bix"});
    parser.ParseArgs(std::vector<std::string>{"-bftest", "--baz=7.555e2", "--bim", "c"});
    REQUIRE(foo);
    REQUIRE(*foo == "test");
    REQUIRE(bar);
    REQUIRE(baz);
    REQUIRE((*baz > 755.49 && *baz < 755.51));
    REQUIRE(bim);
    REQUIRE(*bim == 'c');
    REQUIRE_FALSE(bix);
}

TEST_CASE("Get can be assigned to for non-reference types", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::ValueFlag<std::string> foo(parser, "FOO", "test flag", {'f', "foo"});
    parser.ParseArgs(std::vector<std::string>{"--foo=test"});
    REQUIRE(foo);
    REQUIRE(*foo == "test");
    *foo = "bar";
    REQUIRE(*foo == "bar");
}

TEST_CASE("Invalid argument parsing throws parsing exceptions", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::ValueFlag<int> foo(parser, "FOO", "test flag", {'f', "foo"});
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"--foo=7.5"}), args::ParseError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"--foo", "7a"}), args::ParseError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"--foo", "7e4"}), args::ParseError);
}

TEST_CASE("Argument flag lists work as expected", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::ValueFlagList<int> foo(parser, "FOO", "test flag", {'f', "foo"});
    parser.ParseArgs(std::vector<std::string>{"--foo=7", "-f2", "-f", "9", "--foo", "42"});
    REQUIRE((*foo == std::vector<int>{7, 2, 9, 42}));
}

TEST_CASE("Argument flag lists use default values", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::ValueFlagList<int> foo(parser, "FOO", "test flag", {'f', "foo"}, {9, 7, 5});
    parser.ParseArgs(std::vector<std::string>());
    REQUIRE((*foo == std::vector<int>{9, 7, 5}));
}

TEST_CASE("Argument flag lists replace default values", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::ValueFlagList<int> foo(parser, "FOO", "test flag", {'f', "foo"}, {9, 7, 5});
    parser.ParseArgs(std::vector<std::string>{"--foo=7", "-f2", "-f", "9", "--foo", "42"});
    REQUIRE((*foo == std::vector<int>{7, 2, 9, 42}));
}

TEST_CASE("Positional lists work as expected", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::PositionalList<int> foo(parser, "FOO", "test flag");
    parser.ParseArgs(std::vector<std::string>{"7", "2", "9", "42"});
    REQUIRE((*foo == std::vector<int>{7, 2, 9, 42}));
}

TEST_CASE("Positional lists use default values", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::PositionalList<int> foo(parser, "FOO", "test flag", {9, 7, 5});
    parser.ParseArgs(std::vector<std::string>());
    REQUIRE((*foo == std::vector<int>{9, 7, 5}));
}

TEST_CASE("Positional lists replace default values", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::PositionalList<int> foo(parser, "FOO", "test flag", {9, 7, 5});
    parser.ParseArgs(std::vector<std::string>{"7", "2", "9", "42"});
    REQUIRE((*foo == std::vector<int>{7, 2, 9, 42}));
}

#include <unordered_set>

TEST_CASE("Argument flag lists work with sets", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::ValueFlagList<std::string, std::unordered_set> foo(parser, "FOO", "test flag", {'f', "foo"});
    parser.ParseArgs(std::vector<std::string>{"--foo=7", "-fblah", "-f", "9", "--foo", "blah"});
    REQUIRE((*foo == std::unordered_set<std::string>{"7", "9", "blah"}));
}

TEST_CASE("Positional arguments and positional argument lists work as expected", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::Positional<std::string> foo(parser, "FOO", "test flag");
    args::Positional<bool> bar(parser, "BAR", "test flag");
    args::PositionalList<char> baz(parser, "BAZ", "test flag");
    parser.ParseArgs(std::vector<std::string>{"this is a test flag", "0", "a", "b", "c", "x", "y", "z"});
    REQUIRE(foo);
    REQUIRE((*foo == "this is a test flag"));
    REQUIRE(bar);
    REQUIRE(!*bar);
    REQUIRE(baz);
    REQUIRE((*baz == std::vector<char>{'a', 'b', 'c', 'x', 'y', 'z'}));
}

TEST_CASE("The option terminator works as expected", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::Positional<std::string> foo(parser, "FOO", "test flag");
    args::Positional<bool> bar(parser, "BAR", "test flag");
    args::PositionalList<std::string> baz(parser, "BAZ", "test flag");
    args::Flag ofoo(parser, "FOO", "test flag", {'f', "foo"});
    args::Flag obar(parser, "BAR", "test flag", {"bar", 'b'});
    args::ValueFlag<double> obaz(parser, "BAZ", "test flag", {'a', "baz"});
    parser.ParseArgs(std::vector<std::string>{"--foo", "this is a test flag", "0", "a", "b", "--baz", "7.0", "c", "x", "y", "z"});
    REQUIRE(foo);
    REQUIRE((*foo == "this is a test flag"));
    REQUIRE(bar);
    REQUIRE(!*bar);
    REQUIRE(baz);
    REQUIRE((*baz == std::vector<std::string>{"a", "b", "c", "x", "y", "z"}));
    REQUIRE(ofoo);
    REQUIRE(!obar);
    REQUIRE(obaz);
    parser.ParseArgs(std::vector<std::string>{"--foo", "this is a test flag", "0", "a", "--", "b", "--baz", "7.0", "c", "x", "y", "z"});
    REQUIRE(foo);
    REQUIRE((*foo == "this is a test flag"));
    REQUIRE(bar);
    REQUIRE(!*bar);
    REQUIRE(baz);
    REQUIRE((*baz == std::vector<std::string>{"a", "b", "--baz", "7.0", "c", "x", "y", "z"}));
    REQUIRE(ofoo);
    REQUIRE(!obar);
    REQUIRE(!obaz);
    parser.ParseArgs(std::vector<std::string>{"--foo", "--", "this is a test flag", "0", "a", "b", "--baz", "7.0", "c", "x", "y", "z"});
    REQUIRE(foo);
    REQUIRE((*foo == "this is a test flag"));
    REQUIRE(bar);
    REQUIRE(!*bar);
    REQUIRE(baz);
    REQUIRE((*baz == std::vector<std::string>{"a", "b", "--baz", "7.0", "c", "x", "y", "z"}));
    REQUIRE(ofoo);
    REQUIRE(!obar);
    REQUIRE(!obaz);
}

TEST_CASE("Positional lists work with sets", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::PositionalList<std::string, std::unordered_set> foo(parser, "FOO", "test positional");
    parser.ParseArgs(std::vector<std::string>{"foo", "FoO", "bar", "baz", "foo", "9", "baz"});
    REQUIRE((*foo == std::unordered_set<std::string>{"foo", "FoO", "bar", "baz", "9"}));
}


TEST_CASE("Positionals that are unspecified evaluate false", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::Positional<std::string> foo(parser, "FOO", "test flag");
    args::Positional<bool> bar(parser, "BAR", "test flag");
    args::PositionalList<char> baz(parser, "BAZ", "test flag");
    parser.ParseArgs(std::vector<std::string>{"this is a test flag again"});
    REQUIRE(foo);
    REQUIRE((*foo == "this is a test flag again"));
    REQUIRE_FALSE(bar);
    REQUIRE_FALSE(baz);
}

TEST_CASE("Additional positionals throw an exception", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::Positional<std::string> foo(parser, "FOO", "test flag");
    args::Positional<bool> bar(parser, "BAR", "test flag");
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"this is a test flag again", "1", "this has no positional available"}), args::ParseError);
}

TEST_CASE("Argument groups should throw when validation fails", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::Group xorgroup(parser, "this group provides xor validation", args::Group::Validators::Xor);
    args::Flag a(xorgroup, "a", "test flag", {'a'});
    args::Flag b(xorgroup, "b", "test flag", {'b'});
    args::Flag c(xorgroup, "c", "test flag", {'c'});
    args::Group nxor(parser, "this group provides all-or-none (nxor) validation", args::Group::Validators::AllOrNone);
    args::Flag d(nxor, "d", "test flag", {'d'});
    args::Flag e(nxor, "e", "test flag", {'e'});
    args::Flag f(nxor, "f", "test flag", {'f'});
    args::Group atleastone(parser, "this group provides at-least-one validation", args::Group::Validators::AtLeastOne);
    args::Flag g(atleastone, "g", "test flag", {'g'});
    args::Flag h(atleastone, "h", "test flag", {'h'});
    // Needs g or h
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-a"}), args::ValidationError);
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-g", "-a"}));
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-h", "-a"}));
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-gh", "-a"}));
    // Xor stuff
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-g"}), args::ValidationError);
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-h", "-b"}));
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-g", "-ab"}), args::ValidationError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-g", "-ac"}), args::ValidationError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-g", "-abc"}), args::ValidationError);
    // Nxor stuff
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-h", "-a"}));
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-h", "-adef"}));
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-g", "-ad"}), args::ValidationError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-g", "-adf"}), args::ValidationError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-g", "-aef"}), args::ValidationError);
}

TEST_CASE("Argument groups should nest", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::Group xorgroup(parser, "this group provides xor validation", args::Group::Validators::Xor);
    args::Flag a(xorgroup, "a", "test flag", {'a'});
    args::Flag b(xorgroup, "b", "test flag", {'b'});
    args::Flag c(xorgroup, "c", "test flag", {'c'});
    args::Group nxor(xorgroup, "this group provides all-or-none (nxor) validation", args::Group::Validators::AllOrNone);
    args::Flag d(nxor, "d", "test flag", {'d'});
    args::Flag e(nxor, "e", "test flag", {'e'});
    args::Flag f(nxor, "f", "test flag", {'f'});
    args::Group atleastone(xorgroup, "this group provides at-least-one validation", args::Group::Validators::AtLeastOne);
    args::Flag g(atleastone, "g", "test flag", {'g'});
    args::Flag h(atleastone, "h", "test flag", {'h'});
    // Nothing actually matches, because nxor validates properly when it's empty, 
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{}));
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-a", "-d"}));
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-c", "-f"}));
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-de", "-f"}));
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-gh", "-f"}));
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-g"}), args::ValidationError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-a"}), args::ValidationError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-b"}), args::ValidationError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-a", "-dg"}), args::ValidationError);
}

struct DoublesReader
{
    void operator()(const std::string &, const std::string &value, std::tuple<double, double> &destination)
    {
        size_t commapos = 0;
        std::get<0>(destination) = std::stod(value, &commapos);
        std::get<1>(destination) = std::stod(std::string(value, commapos + 1));
    }
};

TEST_CASE("Custom types work", "[args]")
{
    {
        args::ArgumentParser parser("This is a test program.");
        args::Positional<std::tuple<int, int>> ints(parser, "INTS", "This takes a pair of integers.");
        args::Positional<std::tuple<double, double>, DoublesReader> doubles(parser, "DOUBLES", "This takes a pair of doubles.");
        REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"1.2,2", "3.8,4"}), args::ParseError);
    }
    args::ArgumentParser parser("This is a test program.");
    args::Positional<std::tuple<int, int>> ints(parser, "INTS", "This takes a pair of integers.");
    args::Positional<std::tuple<double, double>, DoublesReader> doubles(parser, "DOUBLES", "This takes a pair of doubles.");
    parser.ParseArgs(std::vector<std::string>{"1,2", "3.8,4"});
    REQUIRE(std::get<0>(*ints) == 1);
    REQUIRE(std::get<1>(*ints) == 2);
    REQUIRE((std::get<0>(*doubles) > 3.79 && std::get<0>(*doubles) < 3.81));
    REQUIRE((std::get<1>(*doubles) > 3.99 && std::get<1>(*doubles) < 4.01));
}

TEST_CASE("Custom parser prefixes (dd-style)", "[args]")
{
    args::ArgumentParser parser("This command likes to break your disks");
    parser.LongPrefix("");
    parser.LongSeparator("=");
    args::HelpFlag help(parser, "HELP", "Show this help menu.", {"help"});
    args::ValueFlag<long> bs(parser, "BYTES", "Block size", {"bs"}, 512);
    args::ValueFlag<long> skip(parser, "BYTES", "Bytes to skip", {"skip"}, 0);
    args::ValueFlag<std::string> input(parser, "BLOCK SIZE", "Block size", {"if"});
    args::ValueFlag<std::string> output(parser, "BLOCK SIZE", "Block size", {"of"});
    parser.ParseArgs(std::vector<std::string>{"skip=8", "if=/dev/null"});
    REQUIRE_FALSE(bs);
    REQUIRE(*bs == 512);
    REQUIRE(skip);
    REQUIRE(*skip == 8);
    REQUIRE(input);
    REQUIRE(*input == "/dev/null");
    REQUIRE_FALSE(output);
}

TEST_CASE("Custom parser prefixes (Some Windows styles)", "[args]")
{
    args::ArgumentParser parser("This command likes to break your disks");
    parser.LongPrefix("/");
    parser.LongSeparator(":");
    args::HelpFlag help(parser, "HELP", "Show this help menu.", {"help"});
    args::ValueFlag<long> bs(parser, "BYTES", "Block size", {"bs"}, 512);
    args::ValueFlag<long> skip(parser, "BYTES", "Bytes to skip", {"skip"}, 0);
    args::ValueFlag<std::string> input(parser, "BLOCK SIZE", "Block size", {"if"});
    args::ValueFlag<std::string> output(parser, "BLOCK SIZE", "Block size", {"of"});
    parser.ParseArgs(std::vector<std::string>{"/skip:8", "/if:/dev/null"});
    REQUIRE_FALSE(bs);
    REQUIRE(*bs == 512);
    REQUIRE(skip);
    REQUIRE(*skip == 8);
    REQUIRE(input);
    REQUIRE(*input == "/dev/null");
    REQUIRE_FALSE(output);
}

TEST_CASE("Help menu can be grabbed as a string, passed into a stream, or by using the overloaded stream operator", "[args]")
{
    std::ostream null(nullptr);
    args::ArgumentParser parser("This command likes to break your disks");
    args::HelpFlag help(parser, "HELP", "Show this help menu.", {"help"});
    args::ValueFlag<long> bs(parser, "BYTES", "Block size", {"bs"}, 512);
    args::ValueFlag<long> skip(parser, "BYTES", "Bytes to skip", {"skip"}, 0);
    args::ValueFlag<std::string> input(parser, "BLOCK SIZE", "Block size", {"if"});
    args::ValueFlag<std::string> output(parser, "BLOCK SIZE", "Block size", {"of"});
    parser.ParseArgs(std::vector<std::string>{"--skip=8", "--if=/dev/null"});
    null << parser.Help();
    parser.Help(null);
    null << parser;
}

TEST_CASE("Required argument separation being violated throws an error", "[args]")
{
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::ValueFlag<std::string> bar(parser, "BAR", "test flag", {'b', "bar"});
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-btest"}));
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"--bar=test"}));
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-b", "test"}));
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"--bar", "test"}));
    parser.SetArgumentSeparations(true, false, false, false);
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-btest"}));
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"--bar=test"}), args::ParseError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-b", "test"}), args::ParseError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"--bar", "test"}), args::ParseError);
    parser.SetArgumentSeparations(false, true, false, false);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-btest"}), args::ParseError);
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"--bar=test"}));
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-b", "test"}), args::ParseError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"--bar", "test"}), args::ParseError);
    parser.SetArgumentSeparations(false, false, true, false);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-btest"}), args::ParseError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"--bar=test"}), args::ParseError);
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-b", "test"}));
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"--bar", "test"}), args::ParseError);
    parser.SetArgumentSeparations(false, false, false, true);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-btest"}), args::ParseError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"--bar=test"}), args::ParseError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-b", "test"}), args::ParseError);
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"--bar", "test"}));
}

enum class MappingEnum
{
    def,
    foo,
    bar,
    red,
    yellow,
    green
};

#include <unordered_map>
#include <algorithm>
#include <string>

struct ToLowerReader
{
    void operator()(const std::string &, const std::string &value, std::string &destination)
    {
        destination = value;
        std::transform(destination.begin(), destination.end(), destination.begin(), [](char c) -> char { return static_cast<char>(tolower(c)); });
    }
};

TEST_CASE("Mapping types work as needed", "[args]")
{
    std::unordered_map<std::string, MappingEnum> map{
        {"default", MappingEnum::def},
        {"foo", MappingEnum::foo},
        {"bar", MappingEnum::bar},
        {"red", MappingEnum::red},
        {"yellow", MappingEnum::yellow},
        {"green", MappingEnum::green}};
    args::ArgumentParser parser("This is a test program.", "This goes after the options.");
    args::MapFlag<std::string, MappingEnum> dmf(parser, "DMF", "Maps string to an enum", {"dmf"}, map);
    args::MapFlag<std::string, MappingEnum> mf(parser, "MF", "Maps string to an enum", {"mf"}, map);
    args::MapFlag<std::string, MappingEnum, ToLowerReader> cimf(parser, "CIMF", "Maps string to an enum case-insensitively", {"cimf"}, map);
    args::MapFlagList<std::string, MappingEnum> mfl(parser, "MFL", "Maps string to an enum list", {"mfl"}, map);
    args::MapPositional<std::string, MappingEnum> mp(parser, "MP", "Maps string to an enum", map);
    args::MapPositionalList<std::string, MappingEnum> mpl(parser, "MPL", "Maps string to an enum list", map);
    parser.ParseArgs(std::vector<std::string>{"--mf=red", "--cimf=YeLLoW", "--mfl=bar", "foo", "--mfl=green", "red", "--mfl", "bar", "default"});
    REQUIRE_FALSE(dmf);
    REQUIRE(*dmf == MappingEnum::def);
    REQUIRE(mf);
    REQUIRE(*mf == MappingEnum::red);
    REQUIRE(cimf);
    REQUIRE(*cimf == MappingEnum::yellow);
    REQUIRE(mfl);
    REQUIRE((*mfl == std::vector<MappingEnum>{MappingEnum::bar, MappingEnum::green, MappingEnum::bar}));
    REQUIRE(mp);
    REQUIRE((*mp == MappingEnum::foo));
    REQUIRE(mpl);
    REQUIRE((*mpl == std::vector<MappingEnum>{MappingEnum::red, MappingEnum::def}));
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"--mf=YeLLoW"}), args::MapError);
}

TEST_CASE("An exception should be thrown when a single-argument flag is matched multiple times and the constructor option is specified", "[args]")
{
    std::unordered_map<std::string, MappingEnum> map{
        {"default", MappingEnum::def},
        {"foo", MappingEnum::foo},
        {"bar", MappingEnum::bar},
        {"red", MappingEnum::red},
        {"yellow", MappingEnum::yellow},
        {"green", MappingEnum::green}};

    args::ArgumentParser parser("Test command");
    args::Flag foo(parser, "Foo", "Foo", {'f', "foo"}, true);
    args::ValueFlag<std::string> bar(parser, "Bar", "Bar", {'b', "bar"}, "", true);
    args::Flag bix(parser, "Bix", "Bix", {'x', "bix"});
    args::MapFlag<std::string, MappingEnum> baz(parser, "Baz", "Baz", {'B', "baz"}, map, MappingEnum::def, true);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"--foo", "-f", "-bblah"}), args::ExtraError);
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"--foo", "-xxx", "--bix", "-bblah", "--bix"}));
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"--foo", "-bblah", "-blah"}), args::ExtraError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"--foo", "-bblah", "--bar", "blah"}), args::ExtraError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"--baz=red", "-B", "yellow"}), args::ExtraError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"--baz", "red", "-Byellow"}), args::ExtraError);
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"--foo", "-Bgreen"}));
    REQUIRE(foo);
    REQUIRE_FALSE(bar);
    REQUIRE_FALSE(bix);
    REQUIRE(baz);
    REQUIRE(*baz == MappingEnum::green);
}

TEST_CASE("Sub-parsers should work through kick-out", "[args]")
{
    std::unordered_map<std::string, MappingEnum> map{
        {"default", MappingEnum::def},
        {"foo", MappingEnum::foo},
        {"bar", MappingEnum::bar},
        {"red", MappingEnum::red},
        {"yellow", MappingEnum::yellow},
        {"green", MappingEnum::green}};

    const std::vector<std::string> args{"--foo", "green", "--bar"};

    args::ArgumentParser parser1("Test command");
    args::Flag foo1(parser1, "Foo", "Foo", {'f', "foo"});
    args::Flag bar1(parser1, "Bar", "Bar", {'b', "bar"});
    args::MapPositional<std::string, MappingEnum> sub(parser1, "sub", "sub", map);
    sub.KickOut(true);

    auto next = parser1.ParseArgs(args);

    args::ArgumentParser parser2("Test command");
    args::Flag foo2(parser2, "Foo", "Foo", {'f', "foo"});
    args::Flag bar2(parser2, "Bar", "Bar", {'b', "bar"});

    parser2.ParseArgs(next, std::end(args));

    REQUIRE(foo1);
    REQUIRE_FALSE(bar1);
    REQUIRE(sub);
    REQUIRE(*sub == MappingEnum::green);
    REQUIRE_FALSE(foo2);
    REQUIRE(bar2);
}

TEST_CASE("Kick-out should work via all flags and value flags", "[args]")
{
    const std::vector<std::string> args{"-a", "-b", "--foo", "-ca", "--bar", "barvalue", "-db"};

    args::ArgumentParser parser1("Test command");
    args::Flag a1(parser1, "a", "a", {'a'});
    args::Flag b1(parser1, "b", "b", {'b'});
    args::Flag c1(parser1, "c", "c", {'c'});
    args::Flag d1(parser1, "d", "d", {'d'});
    args::Flag foo(parser1, "foo", "foo", {'f', "foo"});
    foo.KickOut(true);

    args::ArgumentParser parser2("Test command");
    args::Flag a2(parser2, "a", "a", {'a'});
    args::Flag b2(parser2, "b", "b", {'b'});
    args::Flag c2(parser2, "c", "c", {'c'});
    args::Flag d2(parser2, "d", "d", {'d'});
    args::ValueFlag<std::string> bar(parser2, "bar", "bar", {'B', "bar"});
    bar.KickOut(true);

    args::ArgumentParser parser3("Test command");
    args::Flag a3(parser3, "a", "a", {'a'});
    args::Flag b3(parser3, "b", "b", {'b'});
    args::Flag c3(parser3, "c", "c", {'c'});
    args::Flag d3(parser3, "d", "d", {'d'});

    auto next = parser1.ParseArgs(args);
    next = parser2.ParseArgs(next, std::end(args));
    next = parser3.ParseArgs(next, std::end(args));
    REQUIRE(next == std::end(args));
    REQUIRE(a1);
    REQUIRE(b1);
    REQUIRE_FALSE(c1);
    REQUIRE_FALSE(d1);
    REQUIRE(foo);
    REQUIRE(a2);
    REQUIRE_FALSE(b2);
    REQUIRE(c2);
    REQUIRE_FALSE(d2);
    REQUIRE(bar);
    REQUIRE(*bar == "barvalue");
    REQUIRE_FALSE(a3);
    REQUIRE(b3);
    REQUIRE_FALSE(c3);
    REQUIRE(d3);
}

TEST_CASE("Required flags work as expected", "[args]")
{
    args::ArgumentParser parser1("Test command");
    args::ValueFlag<int> foo(parser1, "foo", "foo", {'f', "foo"}, args::Options::Required);
    args::ValueFlag<int> bar(parser1, "bar", "bar", {'b', "bar"});

    parser1.ParseArgs(std::vector<std::string>{"-f", "42"});
    REQUIRE(*foo == 42);

    REQUIRE_THROWS_AS(parser1.ParseArgs(std::vector<std::string>{"-b4"}), args::RequiredError);

    args::ArgumentParser parser2("Test command");
    args::Positional<int> pos1(parser2, "a", "a");
    REQUIRE_NOTHROW(parser2.ParseArgs(std::vector<std::string>{}));

    args::ArgumentParser parser3("Test command");
    args::Positional<int> pos2(parser3, "a", "a", args::Options::Required);
    REQUIRE_THROWS_AS(parser3.ParseArgs(std::vector<std::string>{}), args::RequiredError);
}

TEST_CASE("Hidden options are excluded from help", "[args]")
{
    args::ArgumentParser parser1("");
    args::ValueFlag<int> foo(parser1, "foo", "foo", {'f', "foo"}, args::Options::HiddenFromDescription);
    args::ValueFlag<int> bar(parser1, "bar", "bar", {'b'}, args::Options::HiddenFromUsage);
    args::Group group(parser1, "group");
    args::ValueFlag<int> foo1(group, "foo", "foo", {'f', "foo"}, args::Options::Hidden);
    args::ValueFlag<int> bar2(group, "bar", "bar", {'b'});

    auto desc = parser1.GetDescription(parser1.helpParams, 0);
    REQUIRE(desc.size() == 3);
    REQUIRE(std::get<0>(desc[0]) == "-b[bar]");
    REQUIRE(std::get<0>(desc[1]) == "group");
    REQUIRE(std::get<0>(desc[2]) == "-b[bar]");

    parser1.helpParams.proglineShowFlags = true;
    parser1.helpParams.proglinePreferShortFlags = true;
    REQUIRE((parser1.GetProgramLine(parser1.helpParams) == std::vector<std::string>{"[-f <foo>]", "[-b <bar>]"}));
}

TEST_CASE("Implicit values work as expected", "[args]")
{
    args::ArgumentParser parser("Test command");
    args::ImplicitValueFlag<int> j(parser, "parallel", "parallel", {'j', "parallel"}, 0, 1);
    args::Flag foo(parser, "FOO", "test flag", {'f', "foo"});
    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-j"}));
    REQUIRE(*j == 0);

    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-j4"}));
    REQUIRE(*j == 4);

    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-j", "4"}));
    REQUIRE(*j == 4);

    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-j", "-f"}));
    REQUIRE(*j == 0);
    REQUIRE(foo);

    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-f"}));
    REQUIRE(*j == 1);
    REQUIRE_FALSE(j);
}

TEST_CASE("Nargs work as expected", "[args]")
{
    args::ArgumentParser parser("Test command");
    args::NargsValueFlag<int> a(parser, "", "", {'a'}, 2);
    args::NargsValueFlag<int> b(parser, "", "", {'b'}, {2, 3});
    args::NargsValueFlag<std::string> c(parser, "", "", {'c'}, {0, 2});
    args::NargsValueFlag<int> d(parser, "", "", {'d'}, {1, 3});
    args::Flag f(parser, "", "", {'f'});

    REQUIRE_THROWS_AS(args::Nargs(3, 2), args::UsageError);

    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-a", "1", "2"}));
    REQUIRE((*a == std::vector<int>{1, 2}));

    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-a", "1", "2", "-f"}));
    REQUIRE((*a == std::vector<int>{1, 2}));
    REQUIRE(f);

    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-a", "1"}), args::ParseError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-a1"}), args::ParseError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-a1", "2"}), args::ParseError);

    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-b", "1", "-2", "-f"}));
    REQUIRE((*b == std::vector<int>{1, -2}));
    REQUIRE(f);

    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-b", "1", "2", "3"}));
    REQUIRE((*b == std::vector<int>{1, 2, 3}));
    REQUIRE(!f);

    std::vector<int> vec;
    for (int be : b)
    {
        vec.push_back(be);
    }

    REQUIRE((vec == std::vector<int>{1, 2, 3}));
    vec.assign(std::begin(b), std::end(b));
    REQUIRE((vec == std::vector<int>{1, 2, 3}));

    parser.SetArgumentSeparations(true, true, false, false);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-a", "1", "2"}), args::ParseError);

    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-c", "-f"}));
    REQUIRE(c->empty());
    REQUIRE(f);

    REQUIRE_NOTHROW(parser.ParseArgs(std::vector<std::string>{"-cf"}));
    REQUIRE((*c == std::vector<std::string>{"f"}));
    REQUIRE(!f);

    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-d"}), args::ParseError);
    REQUIRE_THROWS_AS(parser.ParseArgs(std::vector<std::string>{"-b"}), args::ParseError);
}

TEST_CASE("Simple commands work as expected", "[args]")
{
    args::ArgumentParser p("git-like parser");
    args::ValueFlag<std::string> gitdir(p, "path", "", {"git-dir"}, args::Options::Global);
    args::HelpFlag h(p, "help", "help", {"help"}, args::Options::Global);
    args::PositionalList<std::string> pathsList(p, "paths", "files to commit", args::Options::Global);
    args::Command add(p, "add", "Add file contents to the index");
    args::Command commit(p, "commit", "record changes to the repository");

    p.RequireCommand(true);
    p.ParseArgs(std::vector<std::string>{"add", "--git-dir", "A", "B", "C", "D"});
    REQUIRE(add);
    REQUIRE(!commit);
    REQUIRE((*pathsList == std::vector<std::string>{"B", "C", "D"}));
    REQUIRE(*gitdir == "A");
}

TEST_CASE("Subparser commands work as expected", "[args]")
{
    args::Group globals;
    args::ValueFlag<std::string> gitdir(globals, "path", "", {"git-dir"});
    args::HelpFlag h(globals, "help", "help", {"help"});

    args::ArgumentParser p("git-like parser");
    args::GlobalOptions g(p, globals);

    std::vector<std::string> paths;

    args::Command add(p, "add", "Add file contents to the index", [&](args::Subparser &c)
    {
        args::PositionalList<std::string> pathsList(c, "paths", "files to add");
        c.Parse();
        paths.assign(std::begin(pathsList), std::end(pathsList));
    });

    args::Command commit(p, "commit", "record changes to the repository", [&](args::Subparser &c)
    {
        args::PositionalList<std::string> pathsList(c, "paths", "files to commit");
        c.Parse();
        paths.assign(std::begin(pathsList), std::end(pathsList));
    });

    p.RequireCommand(true);
    p.ParseArgs(std::vector<std::string>{"add", "--git-dir", "A", "B", "C", "D"});
    REQUIRE(add);
    REQUIRE(!commit);
    REQUIRE((paths == std::vector<std::string>{"B", "C", "D"}));
    REQUIRE(*gitdir == "A");
}

TEST_CASE("Subparser commands with kick-out flags work as expected", "[args]")
{
    args::ArgumentParser p("git-like parser");

    std::vector<std::string> kickedOut;
    args::Command add(p, "add", "Add file contents to the index", [&](args::Subparser &c)
    {
        args::Flag kickoutFlag(c, "kick-out", "kick-out flag", {'k'}, args::Options::KickOut);
        c.Parse();
        REQUIRE(kickoutFlag);
        kickedOut = c.KickedOut();
    });

    p.ParseArgs(std::vector<std::string>{"add", "-k", "A", "B", "C", "D"});
    REQUIRE(add);
    REQUIRE((kickedOut == std::vector<std::string>{"A", "B", "C", "D"}));
}

TEST_CASE("Subparser help works as expected", "[args]")
{
    args::ArgumentParser p("git-like parser");
    args::Flag g(p, "GLOBAL", "global flag", {'g'}, args::Options::Global);

    args::Command add(p, "add", "add file contents to the index", [&](args::Subparser &c)
    {
        args::Flag flag(c, "FLAG", "flag", {'f'});
        c.Parse();
    });

    args::Command commit(p, "commit", "record changes to the repository", [&](args::Subparser &c)
    {
        args::Flag flag(c, "FLAG", "flag", {'f'});
        c.Parse();
    });

    p.Prog("git");
    p.RequireCommand(false);

    std::ostringstream s;

    auto d = p.GetDescription(p.helpParams, 0);
    s << p;
    REQUIRE(s.str() == R"(  git [COMMAND] {OPTIONS}

    git-like parser

  OPTIONS:

      -g                                global flag
      add                               add file contents to the index
      commit                            record changes to the repository

)");

    p.ParseArgs(std::vector<std::string>{"add"});
    s.str("");
    s << p;
    REQUIRE(s.str() == R"(  git add {OPTIONS}

    add file contents to the index

  OPTIONS:

      -f                                flag

)");

    p.ParseArgs(std::vector<std::string>{});
    s.str("");
    s << p;
    REQUIRE(s.str() == R"(  git [COMMAND] {OPTIONS}

    git-like parser

  OPTIONS:

      -g                                global flag
      add                               add file contents to the index
      commit                            record changes to the repository

)");

    p.helpParams.showCommandChildren = true;
    p.ParseArgs(std::vector<std::string>{});
    s.str("");
    s << p;
    REQUIRE(s.str() == R"(  git [COMMAND] {OPTIONS}

    git-like parser

  OPTIONS:

      -g                                global flag
      add                               add file contents to the index
        -f                                flag
      commit                            record changes to the repository
        -f                                flag

)");

    commit.Epilog("epilog");
    p.helpParams.showCommandFullHelp = true;
    p.ParseArgs(std::vector<std::string>{});
    s.str("");
    s << p;
    REQUIRE(s.str() == R"(  git [COMMAND] {OPTIONS}

    git-like parser

  OPTIONS:

      -g                                global flag
      add {OPTIONS}

        add file contents to the index

        -f                                flag

      commit {OPTIONS}

        record changes to the repository

        -f                                flag

        epilog

)");

}

TEST_CASE("Subparser validation works as expected", "[args]")
{
    args::ArgumentParser p("parser");
    args::Command a(p, "a", "command a", [](args::Subparser &s)
    {
        args::ValueFlag<std::string> f(s, "", "", {'f'}, args::Options::Required);
        s.Parse();
    });

    args::Command b(p, "b", "command b");
    args::ValueFlag<std::string> f(b, "", "", {'f'}, args::Options::Required);

    args::Command c(p, "c", "command c", [](args::Subparser&){});

    REQUIRE_THROWS_AS(p.ParseArgs(std::vector<std::string>{}), args::ValidationError);
    REQUIRE_THROWS_AS(p.ParseArgs(std::vector<std::string>{"a"}), args::RequiredError);
    REQUIRE_NOTHROW(p.ParseArgs(std::vector<std::string>{"a", "-f", "F"}));
    REQUIRE_THROWS_AS(p.ParseArgs(std::vector<std::string>{"b"}), args::RequiredError);
    REQUIRE_NOTHROW(p.ParseArgs(std::vector<std::string>{"b", "-f", "F"}));

    p.RequireCommand(false);
    REQUIRE_NOTHROW(p.ParseArgs(std::vector<std::string>{}));

    REQUIRE_THROWS_AS(p.ParseArgs(std::vector<std::string>{"c"}), args::UsageError);

    REQUIRE_THROWS_AS(p.ParseArgs(std::vector<std::string>{"unknown-command"}), args::ParseError);
}

TEST_CASE("Subparser group validation works as expected", "[args]")
{
    int x = 0;
    args::ArgumentParser p("parser");
    args::Command a(p, "a", "command a", [&](args::Subparser &s)
    {
        args::Group required(s, "", args::Group::Validators::All);
        args::ValueFlag<std::string> f(required, "", "", {'f'});
        s.Parse();
        ++x;
    });

    p.RequireCommand(false);
    REQUIRE_NOTHROW(p.ParseArgs(std::vector<std::string>{}));
    REQUIRE_NOTHROW(p.ParseArgs(std::vector<std::string>{"a", "-f", "F"}));
    REQUIRE_THROWS_AS(p.ParseArgs(std::vector<std::string>{"a"}), args::ValidationError);
    REQUIRE(x == 1);
}

TEST_CASE("Global options work as expected", "[args]")
{
    args::Group globals;
    args::Flag f(globals, "f", "f", {'f'});

    args::ArgumentParser p("parser");
    args::GlobalOptions g(p, globals);
    args::Command a(p, "a", "command a");
    args::Command b(p, "b", "command b");

    p.RequireCommand(false);

    REQUIRE_NOTHROW(p.ParseArgs(std::vector<std::string>{"-f"}));
    REQUIRE_NOTHROW(p.ParseArgs(std::vector<std::string>{"a", "-f"}));
    REQUIRE_NOTHROW(p.ParseArgs(std::vector<std::string>{"b", "-f"}));
}

TEST_CASE("GetProgramLine works as expected", "[args]")
{
    args::ArgumentParser p("parser");
    args::Flag g(p, "g", "g", {'g'}, args::Options::Global);
    args::Flag hidden(p, "hidden", "hidden flag", {'h'}, args::Options::Hidden);
    args::Command a(p, "a", "command a", [](args::Subparser &s)
    {
        args::ValueFlag<std::string> f(s, "STRING", "my f flag", {'f', "f-long"}, args::Options::Required);
        args::Positional<std::string> pos(s, "positional", "positional", args::Options::Required);
        s.Parse();
    });

    args::Command b(p, "b", "command b");
    args::ValueFlag<std::string> f(b, "STRING", "my f flag", {'f'}, args::Options::Required);
    args::Positional<std::string> pos(b, "positional", "positional");

    auto line = [&](args::Command &element)
    {
        p.Reset();
        auto strings = element.GetCommandProgramLine(p.helpParams);
        std::string res;
        for (const std::string &s: strings)
        {
            if (!res.empty())
            {
                res += ' ';
            }

            res += s;
        }

        return res;
    };

    REQUIRE(line(p) == "COMMAND {OPTIONS}");
    REQUIRE(line(a) == "a positional {OPTIONS}");
    REQUIRE(line(b) == "b [positional] {OPTIONS}");

    p.helpParams.proglineShowFlags = true;
    REQUIRE(line(p) == "COMMAND [-g]");
    REQUIRE(line(a) == "a --f-long <STRING> positional");
    REQUIRE(line(b) == "b -f <STRING> [positional]");

    p.helpParams.proglinePreferShortFlags = true;
    REQUIRE(line(p) == "COMMAND [-g]");
    REQUIRE(line(a) == "a -f <STRING> positional");
    REQUIRE(line(b) == "b -f <STRING> [positional]");
}

TEST_CASE("Program line wrapping works as expected", "[args]")
{
    args::ArgumentParser p("parser");
    args::ValueFlag<std::string> f(p, "foo_name", "f", {"foo"});
    args::ValueFlag<std::string> g(p, "bar_name", "b", {"bar"});
    args::ValueFlag<std::string> z(p, "baz_name", "z", {"baz"});

    p.helpParams.proglineShowFlags = true;
    p.helpParams.width = 42;
    p.Prog("parser");
    p.ProglinePostfix("\na\nliiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiine line2 line2tail");

    REQUIRE((p.GetProgramLine(p.helpParams) == std::vector<std::string>{
             "[--foo <foo_name>]",
             "[--bar <bar_name>]",
             "[--baz <baz_name>]",
             "\n",
             "a",
             "\n",
             "liiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiine",
             "line2",
             "line2tail",
             }));

    std::ostringstream s;
    s << p;
    REQUIRE(s.str() == R"(  parser [--foo <foo_name>]
    [--bar <bar_name>]
    [--baz <baz_name>]
    a
    liiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiine
    line2 line2tail

    parser

  OPTIONS:

      --foo=[foo_name]                  f
      --bar=[bar_name]                  b
      --baz=[baz_name]                  z

)");
}

TEST_CASE("Matcher validation works as expected", "[args]")
{
    args::ArgumentParser parser("Test command");
    REQUIRE_THROWS_AS(args::ValueFlag<int>(parser, "", "", {}), args::UsageError);
}

TEST_CASE("HelpParams work as expected", "[args]")
{
    args::ArgumentParser p("parser");
    args::ValueFlag<std::string> f(p, "name", "description", {'f', "foo"});
    args::ValueFlag<std::string> g(p, "name", "description\n  d1\n  d2", {'g'});
    p.Prog("prog");

    REQUIRE(p.Help() == R"(  prog {OPTIONS}

    parser

  OPTIONS:

      -f[name], --foo=[name]            description
      -g[name]                          description
                                          d1
                                          d2

)");

    p.helpParams.usageString = "usage:";
    p.helpParams.optionsString = "Options";
    p.helpParams.useValueNameOnce = true;
    REQUIRE(p.Help() == R"(  usage: prog {OPTIONS}

    parser

  Options

      -f, --foo=[name]                  description
      -g[name]                          description
                                          d1
                                          d2

)");

    p.helpParams.showValueName = false;
    p.helpParams.optionsString = {};
    REQUIRE(p.Help() == R"(  usage: prog {OPTIONS}

    parser

      -f, --foo                         description
      -g                                description
                                          d1
                                          d2

)");

    p.helpParams.helpindent = 12;
    p.helpParams.optionsString = "Options";
    REQUIRE(p.Help() == R"(  usage: prog {OPTIONS}

    parser

  Options

      -f, --foo
            description
      -g    description
              d1
              d2

)");

    p.helpParams.addNewlineBeforeDescription = true;
    REQUIRE(p.Help() == R"(  usage: prog {OPTIONS}

    parser

  Options

      -f, --foo
            description
      -g
            description
              d1
              d2

)");

    args::ValueFlag<std::string> e(p, "name", "some reaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaally loooooooooooooooooooooooooooong description", {'e'});
    REQUIRE(p.Help() == R"(  usage: prog {OPTIONS}

    parser

  Options

      -f, --foo
            description
      -g
            description
              d1
              d2
      -e
            some reaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaally
            loooooooooooooooooooooooooooong description

)");

}

struct StringAssignable
{
public:
    StringAssignable() = default;
    StringAssignable(const std::string &p) : path(p) {}
    std::string path;

    friend std::istream &operator >> (std::istream &s, StringAssignable &a)
    { return s >> a.path; }
};

TEST_CASE("ValueParser works as expected", "[args]")
{
    static_assert(std::is_assignable<StringAssignable, std::string>::value, "StringAssignable must be assignable to std::string");

    args::ArgumentParser p("parser");
    args::ValueFlag<std::string> f(p, "name", "description", {'f'});
    args::ValueFlag<StringAssignable> b(p, "name", "description", {'b'});
    args::ValueFlag<int> i(p, "name", "description", {'i'});
    args::ValueFlag<int> d(p, "name", "description", {'d'});
    args::PositionalList<double> ds(p, "name", "description");

    REQUIRE_NOTHROW(p.ParseArgs(std::vector<std::string>{"-f", "a b"}));
    REQUIRE(*f == "a b");

    REQUIRE_NOTHROW(p.ParseArgs(std::vector<std::string>{"-b", "a b"}));
    REQUIRE(b->path == "a b");

    REQUIRE_NOTHROW(p.ParseArgs(std::vector<std::string>{"-i", "42 "}));
    REQUIRE(*i == 42);

    REQUIRE_NOTHROW(p.ParseArgs(std::vector<std::string>{"-i", " 12"}));
    REQUIRE(*i == 12);

    REQUIRE_THROWS_AS(p.ParseArgs(std::vector<std::string>{"-i", "a"}), args::ParseError);
    REQUIRE_THROWS_AS(p.ParseArgs(std::vector<std::string>{"-d", "b"}), args::ParseError);
    REQUIRE_THROWS_AS(p.ParseArgs(std::vector<std::string>{"c"}), args::ParseError);
    REQUIRE_THROWS_AS(p.ParseArgs(std::vector<std::string>{"s"}), args::ParseError);
}

TEST_CASE("ActionFlag works as expected", "[args]")
{
    args::ArgumentParser p("parser");
    std::string s;

    args::ActionFlag action0(p, "name", "description", {'x'}, [&]() { s = "flag"; });
    args::ActionFlag action1(p, "name", "description", {'y'}, [&](const std::string &arg) { s = arg; });
    args::ActionFlag actionN(p, "name", "description", {'z'}, 2, [&](const std::vector<std::string> &arg) { s = arg[0] + arg[1]; });
    args::ActionFlag actionThrow(p, "name", "description", {'v'}, [&]() { throw std::runtime_error(""); });

    p.ParseArgs(std::vector<std::string>{"-x"});
    REQUIRE(s == "flag");

    p.ParseArgs(std::vector<std::string>{"-y", "a"});
    REQUIRE(s == "a");

    p.ParseArgs(std::vector<std::string>{"-z", "a", "b"});
    REQUIRE(s == "ab");

    REQUIRE_THROWS_AS(p.ParseArgs(std::vector<std::string>{"-v"}), std::runtime_error);
}

TEST_CASE("Default values work as expected", "[args]")
{
    args::ArgumentParser p("parser");
    args::ValueFlag<std::string> f(p, "name", "description", {'f', "foo"}, "abc");
    args::MapFlag<std::string, int> b(p, "name", "description", {'b', "bar"}, {{"a", 1}, {"b", 2}, {"c", 3}});
    p.Prog("prog");
    REQUIRE(p.Help() == R"(  prog {OPTIONS}

    parser

  OPTIONS:

      -f[name], --foo=[name]            description
      -b[name], --bar=[name]            description

)");

    p.helpParams.addDefault = true;
    p.helpParams.addChoices = true;

    REQUIRE(p.Help() == R"(  prog {OPTIONS}

    parser

  OPTIONS:

      -f[name], --foo=[name]            description
                                        Default: abc
      -b[name], --bar=[name]            description
                                        One of: a, b, c

)");

    f.HelpDefault("123");
    b.HelpChoices({"1", "2", "3"});
    REQUIRE(p.Help() == R"(  prog {OPTIONS}

    parser

  OPTIONS:

      -f[name], --foo=[name]            description
                                        Default: 123
      -b[name], --bar=[name]            description
                                        One of: 1, 2, 3

)");

    f.HelpDefault({});
    b.HelpChoices({});
    REQUIRE(p.Help() == R"(  prog {OPTIONS}

    parser

  OPTIONS:

      -f[name], --foo=[name]            description
      -b[name], --bar=[name]            description

)");
}

TEST_CASE("Choices description works as expected", "[args]")
{
    args::ArgumentParser p("parser");
    args::MapFlag<int, int> map(p, "map", "map", {"map"}, {{1,1}, {2, 2}});
    args::MapFlagList<char, int> maplist(p, "maplist", "maplist", {"maplist"}, {{'1',1}, {'2', 2}});
    args::MapPositional<std::string, int, args::ValueReader, std::map> mappos(p, "mappos", "mappos", {{"1",1}, {"2", 2}});
    args::MapPositionalList<char, int, std::vector, args::ValueReader, std::map> mapposlist(p, "mapposlist", "mapposlist", {{'1',1}, {'2', 2}});

    REQUIRE(map.HelpChoices(p.helpParams) == std::vector<std::string>{"1", "2"});
    REQUIRE(maplist.HelpChoices(p.helpParams) == std::vector<std::string>{"1", "2"});
    REQUIRE(mappos.HelpChoices(p.helpParams) == std::vector<std::string>{"1", "2"});
    REQUIRE(mapposlist.HelpChoices(p.helpParams) == std::vector<std::string>{"1", "2"});
}

TEST_CASE("Completion works as expected", "[args]")
{
    using namespace Catch::Matchers;

    args::ArgumentParser p("parser");
    args::CompletionFlag c(p, {"completion"});
    args::Group g(p);
    args::ValueFlag<std::string> f(g, "name", "description", {'f', "foo"}, "abc");
    args::ValueFlag<std::string> b(g, "name", "description", {'b', "bar"}, "abc");

    REQUIRE_THROWS_WITH(p.ParseArgs(std::vector<std::string>{"--completion", "bash", "1", "test", "-"}), Equals("-f\n-b"));
    REQUIRE_THROWS_WITH(p.ParseArgs(std::vector<std::string>{"--completion", "bash", "1", "test", "-f"}), Equals("-f"));
    REQUIRE_THROWS_WITH(p.ParseArgs(std::vector<std::string>{"--completion", "bash", "1", "test", "--"}), Equals("--foo\n--bar"));

    args::MapFlag<std::string, int> m(p, "mappos", "mappos", {'m', "map"}, {{"1",1}, {"2", 2}});
    REQUIRE_THROWS_WITH(p.ParseArgs(std::vector<std::string>{"--completion", "bash", "2", "test", "-m", ""}), Equals("1\n2"));
    REQUIRE_THROWS_WITH(p.ParseArgs(std::vector<std::string>{"--completion", "bash", "1", "test", "--map="}), Equals("1\n2"));
    REQUIRE_THROWS_WITH(p.ParseArgs(std::vector<std::string>{"--completion", "bash", "2", "test", "--map", "="}), Equals("1\n2"));
    REQUIRE_THROWS_WITH(p.ParseArgs(std::vector<std::string>{"--completion", "bash", "1", "test", "-m1"}), Equals("-m1"));

    args::Positional<std::string> pos(p, "name", "desc");
    REQUIRE_THROWS_WITH(p.ParseArgs(std::vector<std::string>{"--completion", "bash", "1", "test", ""}), Equals(""));
    REQUIRE_THROWS_WITH(p.ParseArgs(std::vector<std::string>{"--completion", "bash", "1", "test", "-"}), Equals("-f\n-b\n-m"));
    REQUIRE_THROWS_WITH(p.ParseArgs(std::vector<std::string>{"--completion", "bash", "1", "test", "--"}), Equals("--foo\n--bar\n--map"));

    args::ArgumentParser p2("parser");
    args::CompletionFlag complete2(p2, {"completion"});

    args::Command c1(p2, "command1", "desc", [](args::Subparser &sp)
    {
        args::ValueFlag<std::string> f1(sp, "name", "description", {'f', "foo"}, "abc");
        f1.KickOut();
        sp.Parse();
    });

    args::Command c2(p2, "command2", "desc", [](args::Subparser &sp)
    {
        args::ValueFlag<std::string> f1(sp, "name", "description", {'b', "bar"}, "abc");
        sp.Parse();
    });

    REQUIRE_THROWS_WITH(p2.ParseArgs(std::vector<std::string>{"--completion", "bash", "1", "test", "-"}), Equals(""));
    REQUIRE_THROWS_WITH(p2.ParseArgs(std::vector<std::string>{"--completion", "bash", "1", "test", ""}), Equals("command1\ncommand2"));
    REQUIRE_THROWS_WITH(p2.ParseArgs(std::vector<std::string>{"--completion", "bash", "2", "test", "command1", ""}), Equals("-f"));
    REQUIRE_THROWS_WITH(p2.ParseArgs(std::vector<std::string>{"--completion", "bash", "2", "test", "command2", ""}), Equals("-b"));
    REQUIRE_THROWS_WITH(p2.ParseArgs(std::vector<std::string>{"--completion", "bash", "2", "test", "command3", ""}), Equals(""));
    REQUIRE_THROWS_WITH(p2.ParseArgs(std::vector<std::string>{"--completion", "bash", "3", "test", "command1", "-f", "-"}), Equals(""));
}

#undef ARGS_HXX
#define ARGS_TESTNAMESPACE
#define ARGS_NOEXCEPT
#include <args.hxx>

TEST_CASE("Noexcept mode works as expected", "[args]")
{
    std::unordered_map<std::string, MappingEnum> map{
        {"default", MappingEnum::def},
        {"foo", MappingEnum::foo},
        {"bar", MappingEnum::bar},
        {"red", MappingEnum::red},
        {"yellow", MappingEnum::yellow},
        {"green", MappingEnum::green}};

    argstest::ArgumentParser parser("This is a test program.", "This goes after the options.");
    argstest::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
    argstest::Flag bar(parser, "BAR", "test flag", {'b', "bar"}, true);
    argstest::ValueFlag<int> foo(parser, "FOO", "test flag", {'f', "foo"});
    argstest::Group nandgroup(parser, "this group provides nand validation", argstest::Group::Validators::AtMostOne);
    argstest::Flag x(nandgroup, "x", "test flag", {'x'});
    argstest::Flag y(nandgroup, "y", "test flag", {'y'});
    argstest::Flag z(nandgroup, "z", "test flag", {'z'});
    argstest::MapFlag<std::string, MappingEnum> mf(parser, "MF", "Maps string to an enum", {"mf"}, map);
    parser.ParseArgs(std::vector<std::string>{"-h"});
    REQUIRE(parser.GetError() == argstest::Error::Help);
    parser.ParseArgs(std::vector<std::string>{"--Help"});
    REQUIRE(parser.GetError() == argstest::Error::Parse);
    parser.ParseArgs(std::vector<std::string>{"--bar=test"});
    REQUIRE(parser.GetError() == argstest::Error::Parse);
    parser.ParseArgs(std::vector<std::string>{"--bar"});
    REQUIRE(parser.GetError() == argstest::Error::None);
    parser.ParseArgs(std::vector<std::string>{"--bar", "-b"});
    REQUIRE(parser.GetError() == argstest::Error::Extra);

    parser.ParseArgs(std::vector<std::string>{"--foo=7.5"});
    REQUIRE(parser.GetError() == argstest::Error::Parse);
    parser.ParseArgs(std::vector<std::string>{"--foo", "7a"});
    REQUIRE(parser.GetError() == argstest::Error::Parse);
    parser.ParseArgs(std::vector<std::string>{"--foo", "7e4"});
    REQUIRE(parser.GetError() == argstest::Error::Parse);
    parser.ParseArgs(std::vector<std::string>{"--foo"});
    REQUIRE(parser.GetError() == argstest::Error::Parse);

    parser.ParseArgs(std::vector<std::string>{"--foo=85"});
    REQUIRE(parser.GetError() == argstest::Error::None);

    parser.ParseArgs(std::vector<std::string>{"this is a test flag again", "1", "this has no positional available"});
    REQUIRE(parser.GetError() == argstest::Error::Parse);

    parser.ParseArgs(std::vector<std::string>{"-x"});
    REQUIRE(parser.GetError() == argstest::Error::None);
    parser.ParseArgs(std::vector<std::string>{"-xz"});
    REQUIRE(parser.GetError() == argstest::Error::Validation);
    parser.ParseArgs(std::vector<std::string>{"-y"});
    REQUIRE(parser.GetError() == argstest::Error::None);
    parser.ParseArgs(std::vector<std::string>{"-y", "-xz"});
    REQUIRE(parser.GetError() == argstest::Error::Validation);
    parser.ParseArgs(std::vector<std::string>{"--mf", "YeLLoW"});
    REQUIRE(parser.GetError() == argstest::Error::Map);
    parser.ParseArgs(std::vector<std::string>{"--mf", "yellow"});
    REQUIRE(parser.GetError() == argstest::Error::None);
}

TEST_CASE("Required flags work as expected in noexcept mode", "[args]")
{
    argstest::ArgumentParser parser1("Test command");
    argstest::ValueFlag<int> foo(parser1, "foo", "foo", {'f', "foo"}, argstest::Options::Required);
    argstest::ValueFlag<int> bar(parser1, "bar", "bar", {'b', "bar"});

    parser1.ParseArgs(std::vector<std::string>{"-f", "42"});
    REQUIRE(*foo == 42);
    REQUIRE(parser1.GetError() == argstest::Error::None);

    parser1.ParseArgs(std::vector<std::string>{"-b4"});
    REQUIRE(parser1.GetError() == argstest::Error::Required);

    argstest::ArgumentParser parser2("Test command");
    argstest::Positional<int> pos1(parser2, "a", "a");
    parser2.ParseArgs(std::vector<std::string>{});
    REQUIRE(parser2.GetError() == argstest::Error::None);

    argstest::ArgumentParser parser3("Test command");
    argstest::Positional<int> pos2(parser3, "a", "a", argstest::Options::Required);
    parser3.ParseArgs(std::vector<std::string>{});
    REQUIRE(parser3.GetError() == argstest::Error::Required);
}

TEST_CASE("Subparser validation works as expected in noexcept mode", "[args]")
{
    argstest::ArgumentParser p("parser");
    argstest::Command a(p, "a", "command a", [](argstest::Subparser &s)
    {
        argstest::ValueFlag<std::string> f(s, "", "", {'f'}, argstest::Options::Required);
        s.Parse();
    });

    argstest::Command b(p, "b", "command b");
    argstest::ValueFlag<std::string> f(b, "", "", {'f'}, argstest::Options::Required);

    argstest::Command c(p, "c", "command c", [](argstest::Subparser&){});

    p.ParseArgs(std::vector<std::string>{});
    REQUIRE(p.GetError() == argstest::Error::Validation);

    p.ParseArgs(std::vector<std::string>{"a"});
    REQUIRE((size_t)p.GetError() == (size_t)argstest::Error::Required);

    p.ParseArgs(std::vector<std::string>{"a", "-f", "F"});
    REQUIRE(p.GetError() == argstest::Error::None);

    p.ParseArgs(std::vector<std::string>{"b"});
    REQUIRE(p.GetError() == argstest::Error::Required);

    p.ParseArgs(std::vector<std::string>{"b", "-f", "F"});
    REQUIRE(p.GetError() == argstest::Error::None);

    p.RequireCommand(false);
    p.ParseArgs(std::vector<std::string>{});
    REQUIRE(p.GetError() == argstest::Error::None);

    p.ParseArgs(std::vector<std::string>{"c"});
    REQUIRE(p.GetError() == argstest::Error::Usage);
}

TEST_CASE("Nargs work as expected in noexcept mode", "[args]")
{
    argstest::ArgumentParser parser("Test command");
    argstest::NargsValueFlag<int> a(parser, "", "", {'a'}, {3, 2});

    REQUIRE(parser.GetError() == argstest::Error::Usage);
    parser.ParseArgs(std::vector<std::string>{"-a", "1", "2"});
    REQUIRE(parser.GetError() == argstest::Error::Usage);
}

TEST_CASE("Matcher validation works as expected in noexcept mode", "[args]")
{
    argstest::ArgumentParser parser("Test command");
    argstest::ValueFlag<int> a(parser, "", "", {});

    REQUIRE(parser.GetError() == argstest::Error::Usage);
    parser.ParseArgs(std::vector<std::string>{"-a", "1", "2"});
    REQUIRE(parser.GetError() == argstest::Error::Usage);
}

TEST_CASE("Completion works as expected in noexcept mode", "[args]")
{
    using namespace Catch::Matchers;

    argstest::ArgumentParser p("parser");
    argstest::CompletionFlag c(p, {"completion"});
    argstest::Group g(p);
    argstest::ValueFlag<std::string> f(g, "name", "description", {'f', "foo"}, "abc");
    argstest::ValueFlag<std::string> b(g, "name", "description", {'b', "bar"}, "abc");

    p.ParseArgs(std::vector<std::string>{"--completion", "bash", "1", "test", "-"});
    REQUIRE(p.GetError() == argstest::Error::Completion);
    REQUIRE(argstest::get(c) == "-f\n-b");
}
