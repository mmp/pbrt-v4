#ifndef _TESTING_H_
#define _TESTING_H_ 1

// Standard Dependencies
#include <iostream>
#include <iomanip>

#ifdef ASSERT
# undef ASSERT
#endif

/** Run the test with the given name. */
#define RUN_TEST(TestName) {                                                     \
    bool __test_result = true;                                                   \
    std::cout << "Executing test " << std::left << std::setw(40) << #TestName;   \
    TestName(__test_result);                                                     \
    std::cout << "=> " << (__test_result ? "Success" : "Fail") << std::endl;     \
}

/** Define a test with the given name. */
#define TEST(TestName)                 \
    void TestName(bool& __test_result)

/** Ensures the given condition passes for the test to pass. */
#define ASSERT(condition) {            \
    if (!(condition)) {                \
        __test_result = false;         \
        return;                        \
    }                                  \
}

#endif // _TESTING_H_

