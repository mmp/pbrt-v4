#include "tinylogger/tinylogger.h"

#include <chrono>
#include <thread>

int main() {
    tlog::none()
        << "===========================================\n"
        << "===== tinylogger demo =====================\n"
        << "===========================================";

    tlog::info()
        << "This is an info message meant to let you know "
        << 42 <<
        " is the Answer to the Ultimate Question of Life, the Universe, and Everything";

    tlog::debug() << "You can only see this message if this demo has been compiled in debug mode.";

    tlog::warning() << "This " << "is " << "a " << "warning!";

    tlog::error()
        << "This is an error message that is so long, it likely does not fit within a single line. "
        << "The purpose of this message is to make sure line breaks do not break anything (get it?). "
        << "This sentence only exists to make this message even longer.";

    tlog::info() << "Demonstrating a progress bar...";

    static const int N = 100;

    auto progress = tlog::progress(N);
    for (int i = 1; i <= N; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
        progress.update(i);
    }

    tlog::success() << "Progress finished after " << tlog::durationToString(progress.duration());

    auto scopedLogger = tlog::Logger("inner scope");
    scopedLogger.info("This info message is written by a scoped logger.");
    progress = scopedLogger.progress(N);
    for (int i = 1; i <= N; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds{10});
        progress.update(i);
    }
    scopedLogger.success("Now you know how to use scoped loggers!");

    tlog::success() << "The demo application terminated successfully. Hooray!";
}
