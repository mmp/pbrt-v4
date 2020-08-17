// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/parallel.h>

#include <pbrt/util/check.h>
#include <pbrt/util/print.h>

#include <iterator>
#include <list>
#include <thread>
#include <vector>

namespace pbrt {

std::string AtomicFloat::ToString() const {
    return StringPrintf("%f", float(*this));
}

std::string AtomicDouble::ToString() const {
    return StringPrintf("%f", double(*this));
}

// Barrier Method Definitions
bool Barrier::Block() {
    std::unique_lock<std::mutex> lock(mutex);

    --numToBlock;
    CHECK_GE(numToBlock, 0);

    if (numToBlock > 0) {
        cv.wait(lock, [this]() { return numToBlock == 0; });
    } else
        cv.notify_all();

    return --numToExit == 0;
}

// ParallelJob Definition
class ParallelJob {
  public:
    virtual ~ParallelJob() { DCHECK(removed); }

    // *lock should be locked going in and and unlocked coming out.
    virtual void RunStep(std::unique_lock<std::mutex> *lock) = 0;
    virtual bool HaveWork() const = 0;

    bool Finished() const { return !HaveWork() && activeWorkers == 0; }

    virtual std::string ToString() const = 0;

  protected:
    std::string BaseToString() const {
        return StringPrintf("activeWorkers: %d removed: %s", activeWorkers, removed);
    }

  private:
    friend class ThreadPool;

    ParallelJob *prev = nullptr, *next = nullptr;
    int activeWorkers = 0;
    bool removed = false;
};

// ThreadPool Definition
class ThreadPool {
  public:
    explicit ThreadPool(int nThreads);
    ~ThreadPool();

    size_t size() const { return threads.size(); }

    std::unique_lock<std::mutex> AddToJobList(ParallelJob *job);
    void RemoveFromJobList(ParallelJob *job);

    void WorkOrWait(std::unique_lock<std::mutex> *lock);

    void ForEachThread(std::function<void(void)> func);

    std::string ToString() const;

  private:
    void workerFunc(int tIndex);

    ParallelJob *jobList = nullptr;
    // Protects jobList
    mutable std::mutex jobListMutex;
    // Signaled both when a new job is added to the list and when a job has
    // finished.
    std::condition_variable jobListCondition;

    std::vector<std::thread> threads;
    bool shutdownThreads = false;
};

thread_local int ThreadIndex;

static std::unique_ptr<ThreadPool> threadPool;
static bool maxThreadIndexCalled = false;

// ThreadPool Method Definitions
ThreadPool::ThreadPool(int nThreads) {
    ThreadIndex = 0;

    // Launch one fewer worker thread than the total number we want doing
    // work, since the main thread helps out, too.
    for (int i = 0; i < nThreads - 1; ++i)
        threads.push_back(std::thread(&ThreadPool::workerFunc, this, i + 1));
}

std::unique_lock<std::mutex> ThreadPool::AddToJobList(ParallelJob *job) {
    std::unique_lock<std::mutex> lock(jobListMutex);
    if (jobList != nullptr)
        jobList->prev = job;
    job->next = jobList;
    jobList = job;
    jobListCondition.notify_all();
    return lock;
}

void ThreadPool::RemoveFromJobList(ParallelJob *job) {
    DCHECK(!job->removed);

    if (job->prev != nullptr) {
        job->prev->next = job->next;
    } else {
        DCHECK(jobList == job);
        jobList = job->next;
    }
    if (job->next != nullptr)
        job->next->prev = job->prev;

    job->removed = true;
}

void ThreadPool::workerFunc(int tIndex) {
    LOG_VERBOSE("Started execution in worker thread %d", tIndex);
    ThreadIndex = tIndex;

    std::unique_lock<std::mutex> lock(jobListMutex);
    while (!shutdownThreads)
        WorkOrWait(&lock);

    LOG_VERBOSE("Exiting worker thread %d", tIndex);
}

void ThreadPool::WorkOrWait(std::unique_lock<std::mutex> *lock) {
    DCHECK(lock->owns_lock());

    ParallelJob *job = jobList;
    while ((job != nullptr) && !job->HaveWork())
        job = job->next;
    if (job != nullptr) {
        // Run a chunk of loop iterations for _loop_
        job->activeWorkers++;

        job->RunStep(lock);

        DCHECK(!lock->owns_lock());
        lock->lock();

        // Update _loop_ to reflect completion of iterations
        job->activeWorkers--;

        if (job->Finished())
            jobListCondition.notify_all();
    } else
        // Wait for something to change (new work, or this loop being
        // finished).
        jobListCondition.wait(*lock);
}

void ThreadPool::ForEachThread(std::function<void(void)> func) {
    Barrier *barrier = new Barrier(threads.size() + 1);

    ParallelFor(0, threads.size() + 1, [barrier, &func](int64_t) {
        func();
        if (barrier->Block())
            delete barrier;
    });
}

ThreadPool::~ThreadPool() {
    if (threads.empty())
        return;

    {
        std::lock_guard<std::mutex> lock(jobListMutex);
        shutdownThreads = true;
        jobListCondition.notify_all();
    }

    for (std::thread &thread : threads)
        thread.join();
}

std::string ThreadPool::ToString() const {
    std::string s = StringPrintf("[ ThreadPool threads.size(): %d shutdownThreads: %s ",
                                 threads.size(), shutdownThreads);
    if (jobListMutex.try_lock()) {
        s += "jobList: [ ";
        ParallelJob *job = jobList;
        while (job) {
            s += job->ToString() + " ";
            job = job->next;
        }
        s += "] ";
        jobListMutex.unlock();
    } else
        s += "(job list mutex locked) ";
    return s + "]";
}

// ParallelForLoop1D Definition
class ParallelForLoop1D : public ParallelJob {
  public:
    ParallelForLoop1D(int64_t start, int64_t end, int chunkSize,
                      std::function<void(int64_t, int64_t)> func)
        : func(std::move(func)), nextIndex(start), maxIndex(end), chunkSize(chunkSize) {}

    bool HaveWork() const { return nextIndex < maxIndex; }
    void RunStep(std::unique_lock<std::mutex> *lock);

    std::string ToString() const {
        return StringPrintf("[ ParallelForLoop1D nextIndex: %d maxIndex: %d "
                            "chunkSize: %d ]",
                            nextIndex, maxIndex, chunkSize);
    }

  private:
    std::function<void(int64_t, int64_t)> func;
    int64_t nextIndex;
    int64_t maxIndex;
    int chunkSize;
};

class ParallelForLoop2D : public ParallelJob {
  public:
    ParallelForLoop2D(const Bounds2i &extent, int chunkSize,
                      std::function<void(Bounds2i)> func)
        : func(std::move(func)),
          extent(extent),
          nextStart(extent.pMin),
          chunkSize(chunkSize) {}

    bool HaveWork() const { return nextStart.y < extent.pMax.y; }
    void RunStep(std::unique_lock<std::mutex> *lock);

    std::string ToString() const {
        return StringPrintf("[ ParallelForLoop2D extent: %s nextStart: %s "
                            "chunkSize: %d ]",
                            extent, nextStart, chunkSize);
    }

  private:
    std::function<void(Bounds2i)> func;
    const Bounds2i extent;
    Point2i nextStart;
    int chunkSize;
};

// ParallelForLoop1D Method Definitions
void ParallelForLoop1D::RunStep(std::unique_lock<std::mutex> *lock) {
    // Find the set of loop iterations to run next
    int64_t indexStart = nextIndex;
    int64_t indexEnd = std::min(indexStart + chunkSize, maxIndex);

    // Update _loop_ to reflect iterations this thread will run
    nextIndex = indexEnd;

    if (!HaveWork())
        threadPool->RemoveFromJobList(this);

    lock->unlock();

    // Run loop indices in _[indexStart, indexEnd)_
    func(indexStart, indexEnd);
}

void ParallelForLoop2D::RunStep(std::unique_lock<std::mutex> *lock) {
    // Compute extent for this step
    Point2i end = nextStart + Vector2i(chunkSize, chunkSize);
    Bounds2i b = Intersect(Bounds2i(nextStart, end), extent);
    CHECK(!b.IsEmpty());

    // Advance to be ready for the next extent.
    nextStart.x += chunkSize;
    if (nextStart.x >= extent.pMax.x) {
        nextStart.x = extent.pMin.x;
        nextStart.y += chunkSize;
    }

    if (!HaveWork())
        threadPool->RemoveFromJobList(this);

    lock->unlock();

    // Run the loop iteration
    func(b);
}

// Parallel Function Defintions
void ParallelFor(int64_t start, int64_t end, std::function<void(int64_t, int64_t)> func) {
    CHECK(threadPool);
    int64_t chunkSize = std::max<int64_t>(1, (end - start) / (8 * RunningThreads()));
    if (end - start < chunkSize) {
        func(start, end);
        return;
    }

    // Create and enqueue _ParallelJob_ for this loop
    ParallelForLoop1D loop(start, end, chunkSize, std::move(func));
    std::unique_lock<std::mutex> lock = threadPool->AddToJobList(&loop);

    // Help out with parallel loop iterations in the current thread
    while (!loop.Finished())
        threadPool->WorkOrWait(&lock);
}

int MaxThreadIndex() {
    maxThreadIndexCalled = true;
    return threadPool ? (1 + threadPool->size()) : 1;
}

void ParallelFor2D(const Bounds2i &extent, std::function<void(Bounds2i)> func) {
    CHECK(threadPool);

    if (extent.IsEmpty())
        return;
    if (extent.Area() == 1) {
        func(extent);
        return;
    }

    // Want at least 8 tiles per thread, subject to not too big and not too
    // small.
    // TODO: should we do non-square?
    int tileSize = Clamp(int(std::sqrt(extent.Diagonal().x * extent.Diagonal().y /
                                       (8 * RunningThreads()))),
                         1, 32);

    ParallelForLoop2D loop(extent, tileSize, std::move(func));
    std::unique_lock<std::mutex> lock = threadPool->AddToJobList(&loop);

    // Help out with parallel loop iterations in the current thread
    while (!loop.Finished())
        threadPool->WorkOrWait(&lock);
}

///////////////////////////////////////////////////////////////////////////

int AvailableCores() {
    return std::max<int>(1, std::thread::hardware_concurrency());
}

int RunningThreads() {
    return threadPool ? (1 + threadPool->size()) : 1;
}

void ParallelInit(int nThreads) {
    // This is risky: if the caller has allocated per-thread data
    // structures before calling ParallelInit(), then we may end up having
    // them accessed with a higher ThreadIndex than the caller expects.
    CHECK(!maxThreadIndexCalled);

    CHECK(!threadPool);
    if (nThreads <= 0)
        nThreads = AvailableCores();
    threadPool = std::make_unique<ThreadPool>(nThreads);
}

void ParallelCleanup() {
    threadPool.reset();
    maxThreadIndexCalled = false;
}

void ForEachThread(std::function<void(void)> func) {
    if (threadPool)
        threadPool->ForEachThread(std::move(func));
}

}  // namespace pbrt
