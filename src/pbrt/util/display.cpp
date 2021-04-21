// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/display.h>

#include <pbrt/util/error.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/image.h>
#include <pbrt/util/print.h>
#include <pbrt/util/string.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#ifdef PBRT_IS_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Ws2tcpip.h>
#include <winsock2.h>
#undef NOMINMAX
using socket_t = SOCKET;
#else
using socket_t = int;
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#define SOCKET_ERROR (-1)
#define INVALID_SOCKET (-1)
#endif

namespace pbrt {

enum SocketError : int {
#ifdef PBRT_IS_WINDOWS
    Again = EAGAIN,
    ConnRefused = WSAECONNREFUSED,
    WouldBlock = WSAEWOULDBLOCK,
#else
    Again = EAGAIN,
    ConnRefused = ECONNREFUSED,
    WouldBlock = EWOULDBLOCK,
#endif
};

static int closeSocket(socket_t socket) {
#ifdef PBRT_IS_WINDOWS
    return closesocket(socket);
#else
    return close(socket);
#endif
}

static std::atomic<int> numActiveChannels{0};

class IPCChannel {
  public:
    IPCChannel(const std::string &host);
    ~IPCChannel();

    IPCChannel(const IPCChannel &) = delete;
    IPCChannel &operator=(const IPCChannel &) = delete;

    bool Send(pstd::span<const uint8_t> message);

    bool Connected() const { return socketFd != INVALID_SOCKET; }

  private:
    void Connect();
    void Disconnect();

    int numFailures = 0;
    std::string address, port;
    socket_t socketFd = INVALID_SOCKET;
};

IPCChannel::IPCChannel(const std::string &hostname) {
    if (numActiveChannels++ == 0) {
#ifdef PBRT_IS_WINDOWS
        WSADATA wsaData;
        int err = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (err != NO_ERROR)
            LOG_FATAL("Unable to initialize WinSock: %s", ErrorString(err));
#else
        // We don't care about getting a SIGPIPE if the display server goes
        // away...
        signal(SIGPIPE, SIG_IGN);
#endif
    }

    size_t split = hostname.find_last_of(':');
    if (split == std::string::npos)
        ErrorExit("Expected \"host:port\" for display server address. Given \"%s\".",
                  hostname);
    address = std::string(hostname.begin(), hostname.begin() + split);
    port = std::string(hostname.begin() + split + 1, hostname.end());

    Connect();
}

void IPCChannel::Connect() {
    CHECK_EQ(socketFd, INVALID_SOCKET);

    LOG_VERBOSE("Trying to connect to display server");

    struct addrinfo hints = {}, *addrinfo;
    hints.ai_family = PF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    int err = getaddrinfo(address.c_str(), port.c_str(), &hints, &addrinfo);
    if (err)
        ErrorExit("%s", gai_strerror(err));

    socketFd = INVALID_SOCKET;
    for (struct addrinfo *ptr = addrinfo; ptr; ptr = ptr->ai_next) {
        socketFd = socket(ptr->ai_family, ptr->ai_socktype, ptr->ai_protocol);
        if (socketFd == INVALID_SOCKET) {
            LOG_VERBOSE("socket() failed: %s", ErrorString());
            continue;
        }

#ifdef PBRT_IS_LINUX
        struct timeval timeout;
        timeout.tv_sec = 3;
        timeout.tv_usec = 0;
        if (setsockopt(socketFd, SOL_SOCKET, SO_SNDTIMEO, &timeout,
                       sizeof(timeout)) == SOCKET_ERROR) {
            LOG_VERBOSE("setsockopt() failed: %s", ErrorString());
        }
#endif // PBRT_IS_LINUX

        if (connect(socketFd, ptr->ai_addr, ptr->ai_addrlen) == SOCKET_ERROR) {
#ifdef PBRT_IS_WINDOWS
            int err = WSAGetLastError();
#else
            int err = errno;
#endif
            if (err == SocketError::ConnRefused)
                LOG_VERBOSE("Connection refused. Will try again...");
            else
                LOG_VERBOSE("connect() failed: %s", ErrorString(err));

            closeSocket(socketFd);
            socketFd = INVALID_SOCKET;
            continue;
        }

        break;  // success
    }

    freeaddrinfo(addrinfo);
    if (socketFd != INVALID_SOCKET)
        LOG_VERBOSE("Connected to display server");
}

IPCChannel::~IPCChannel() {
    if (Connected())
        Disconnect();

    if (--numActiveChannels == 0) {
#ifdef PBRT_IS_WINDOWS
        WSACleanup();
#endif
    }
}

void IPCChannel::Disconnect() {
    CHECK(Connected());

    closeSocket(socketFd);
    socketFd = INVALID_SOCKET;
}

bool IPCChannel::Send(pstd::span<const uint8_t> message) {
    if (!Connected()) {
        Connect();
        if (!Connected())
            return false;
    }

    // Start with the length of the message.
    // FIXME: annoying coupling w/sending code's message buffer layout...
    int *startPtr = (int *)message.data();
    *startPtr = message.size();

    int bytesSent = send(socketFd, (const char *)message.data(), message.size(), 0 /* flags */);
    if (bytesSent == message.size())
        return true;

    LOG_ERROR("send to display server failed: %s", ErrorString());
    Disconnect();
    return false;
}

namespace {

enum DisplayDirective : uint8_t {
    OpenImage = 0,
    ReloadImage = 1,
    CloseImage = 2,
    UpdateImage = 3,
    CreateImage = 4,
};

void Serialize(uint8_t **ptr, const std::string &s) {
    for (size_t i = 0; i < s.size(); ++i, *ptr += 1)
        **ptr = s[i];
    **ptr = '\0';
    *ptr += 1;
}

template <typename T>
void Serialize(uint8_t **ptr, T value) {
    memcpy(*ptr, &value, sizeof(T));
    *ptr += sizeof(T);
}

constexpr int tileSize = 128;

}  // namespace

class DisplayItem {
  public:
    DisplayItem(
        const std::string &title, Point2i resolution,
        std::vector<std::string> channelNames,
        std::function<void(Bounds2i b, pstd::span<pstd::span<Float>>)> getTileValues);

    bool Display(IPCChannel &channel);

  private:
    bool SendOpenImage(IPCChannel &channel);

    bool openedImage = false;
    std::string title;
    Point2i resolution;
    std::function<void(Bounds2i b, pstd::span<pstd::span<Float>>)> getTileValues;
    std::vector<std::string> channelNames;

    struct ImageChannelBuffer {
        ImageChannelBuffer(const std::string &channelName, int nTiles,
                           const std::string &title);

        void SetTileBounds(int x, int y, int width, int height);
        bool SendIfChanged(IPCChannel &channel, int tileIndex);

        std::vector<uint8_t> buffer;
        int tileBoundsOffset = 0, channelValuesOffset = 0;
        std::vector<uint64_t> tileHashes;

        int setCount, tileIndex;
    };
    std::vector<ImageChannelBuffer> channelBuffers;
};

DisplayItem::DisplayItem(
    const std::string &baseTitle, Point2i resolution,
    std::vector<std::string> channelNames,
    std::function<void(Bounds2i b, pstd::span<pstd::span<Float>>)> getTileValues)
    : resolution(resolution), getTileValues(getTileValues), channelNames(channelNames) {
#ifdef PBRT_IS_WINDOWS
    title = StringPrintf("%s (%d)", baseTitle, GetCurrentThreadId());
#else
    title = StringPrintf("%s (%d)", baseTitle, getpid());
#endif

    int nTiles = ((resolution.x + tileSize - 1) / tileSize) *
                 ((resolution.y + tileSize - 1) / tileSize);

    for (const std::string &channelName : channelNames)
        channelBuffers.push_back(ImageChannelBuffer(channelName, nTiles, title));
}

DisplayItem::ImageChannelBuffer::ImageChannelBuffer(const std::string &channelName,
                                                    int nTiles,
                                                    const std::string &title) {
    int bufferAlloc = tileSize * tileSize * sizeof(float) + title.size() + 32;

    buffer.resize(bufferAlloc);

    uint8_t *ptr = buffer.data();
    Serialize(&ptr, int(0));  // reserve space for message length
    Serialize(&ptr, DisplayDirective::UpdateImage);
    uint8_t grabFocus = 0;
    Serialize(&ptr, grabFocus);
    Serialize(&ptr, title);
    Serialize(&ptr, channelName);

    tileBoundsOffset = ptr - buffer.data();
    // Note: may not be float-aligned, but that's not a problem on x86...
    // TODO: fix this. The problem is that it breaks the whole idea of
    // passing a span<Float> to the callback function...
    channelValuesOffset = tileBoundsOffset + 4 * sizeof(int);

    // Zero-initialize the buffer color contents before computing the hash
    // for a fully-zero tile (which corresponds to the initial state on the
    // viewer side.)
    memset(buffer.data() + channelValuesOffset, 0, tileSize * tileSize * sizeof(float));
    uint64_t zeroHash = HashBuffer(buffer.data() + channelValuesOffset,
                                   tileSize * tileSize * sizeof(float));
    tileHashes.assign(nTiles, zeroHash);
}

void DisplayItem::ImageChannelBuffer::SetTileBounds(int x, int y, int width, int height) {
    uint8_t *ptr = buffer.data() + tileBoundsOffset;

    Serialize(&ptr, x);
    Serialize(&ptr, y);
    Serialize(&ptr, width);
    Serialize(&ptr, height);

    setCount = width * height;
}

bool DisplayItem::ImageChannelBuffer::SendIfChanged(IPCChannel &ipcChannel,
                                                    int tileIndex) {
    int excess = setCount - tileSize * tileSize;
    if (excess > 0)
        memset(buffer.data() + channelValuesOffset + setCount * sizeof(float), 0,
               excess * sizeof(float));

    uint64_t hash = HashBuffer(buffer.data() + channelValuesOffset,
                               tileSize * tileSize * sizeof(float));
    if (hash == tileHashes[tileIndex])
        return true;

    if (!ipcChannel.Send(pstd::MakeSpan(buffer.data(),
                                        channelValuesOffset + setCount * sizeof(float))))
        return false;

    tileHashes[tileIndex] = hash;
    return true;
}

bool DisplayItem::Display(IPCChannel &ipcChannel) {
    if (!openedImage) {
        if (!SendOpenImage(ipcChannel))
            // maybe next time
            return false;
        openedImage = true;
    }

    std::vector<pstd::span<Float>> displayValues(channelBuffers.size());
    for (int c = 0; c < channelBuffers.size(); ++c) {
        Float *ptr = (Float *)(channelBuffers[c].buffer.data() +
                               channelBuffers[c].channelValuesOffset);
        displayValues[c] = pstd::MakeSpan(ptr, tileSize * tileSize);
    }

    int tileIndex = 0;
    for (int y = 0; y < resolution.y; y += tileSize)
        for (int x = 0; x < resolution.x; x += tileSize, ++tileIndex) {
            int height = std::min(y + tileSize, resolution.y) - y;
            int width = std::min(x + tileSize, resolution.x) - x;

            for (int c = 0; c < channelBuffers.size(); ++c)
                channelBuffers[c].SetTileBounds(x, y, width, height);

            Bounds2i b(Point2i(x, y), Point2i(x + width, y + height));
            getTileValues(b, pstd::MakeSpan(displayValues));

            // Send the RGB buffers only if they're different than
            // the last version sent.
            for (int c = 0; c < channelBuffers.size(); ++c)
                if (!channelBuffers[c].SendIfChanged(ipcChannel, tileIndex)) {
                    // Welp. Stop for now...
                    openedImage = false;
                    return false;
                }
        }

    return true;
}

bool DisplayItem::SendOpenImage(IPCChannel &ipcChannel) {
    // Initial "open the image" message
    uint8_t buffer[1024];
    uint8_t *ptr = buffer;

    Serialize(&ptr, int(0));  // reserve space for message length
    Serialize(&ptr, DisplayDirective::CreateImage);
    uint8_t grabFocus = 1;
    Serialize(&ptr, grabFocus);
    Serialize(&ptr, title);

    int nChannels = channelNames.size();
    Serialize(&ptr, resolution.x);
    Serialize(&ptr, resolution.y);
    Serialize(&ptr, nChannels);
    for (int c = 0; c < nChannels; ++c)
        Serialize(&ptr, channelNames[c]);

    return ipcChannel.Send(pstd::MakeSpan(buffer, ptr - buffer));
}

static std::atomic<bool> exitThread{false};
static std::mutex mutex;
static std::thread updateThread;
static std::vector<DisplayItem> dynamicItems;

static IPCChannel *channel;

static void updateDynamicItems() {
    while (!exitThread) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));

        std::lock_guard<std::mutex> lock(mutex);
        for (auto &item : dynamicItems)
            item.Display(*channel);
    }

    // One last time to get the last bits
    std::lock_guard<std::mutex> lock(mutex);
    for (auto &item : dynamicItems)
        item.Display(*channel);

    dynamicItems.clear();
    delete channel;
    channel = nullptr;
}

void ConnectToDisplayServer(const std::string &host) {
    CHECK(channel == nullptr);
    channel = new IPCChannel(host);

    updateThread = std::thread(updateDynamicItems);
}

void DisconnectFromDisplayServer() {
    if (updateThread.get_id() != std::thread::id()) {
        exitThread = true;
        updateThread.join();
        updateThread = std::thread();
        exitThread = false;
    }
}

static DisplayItem GetImageDisplayItem(const std::string &title, const Image &image,
                                       pstd::optional<ImageChannelDesc> channelDesc) {
    auto getValues = [=](Bounds2i b, pstd::span<pstd::span<Float>> displayValues) {
        int offset = 0;
        for (Point2i p : b) {
            ImageChannelValues v =
                channelDesc ? image.GetChannels(p, *channelDesc) : image.GetChannels(p);
            for (int i = 0; i < v.size(); ++i)
                displayValues[i][offset] = v[i];
            ++offset;
        }
    };

    std::vector<std::string> channelNames;
    if (channelDesc)
        channelNames = image.ChannelNames(*channelDesc);
    else {
        if (image.NChannels() == 3)
            channelNames = {"R", "G", "B"};
        else
            for (int i = 0; i < image.NChannels(); ++i)
                channelNames.push_back(StringPrintf("channel %d", i));
    }

    return DisplayItem(title, image.Resolution(), channelNames, getValues);
}

void DisplayStatic(const std::string &title, const Image &image,
                   pstd::optional<ImageChannelDesc> channelDesc) {
    DisplayItem item = GetImageDisplayItem(title, image, channelDesc);

    std::lock_guard<std::mutex> lock(mutex);
    if (!item.Display(*channel))
        LOG_ERROR("Unable to display static content \"%s\".", title);
}

void DisplayDynamic(const std::string &title, const Image &image,
                    pstd::optional<ImageChannelDesc> channelDesc) {
    std::lock_guard<std::mutex> lock(mutex);
    dynamicItems.push_back(GetImageDisplayItem(title, image, channelDesc));
}

void DisplayStatic(
    const std::string &title, const Point2i &resolution,
    std::vector<std::string> channelNames,
    std::function<void(Bounds2i b, pstd::span<pstd::span<Float>>)> getTileValues) {
    DisplayItem item(title, resolution, channelNames, getTileValues);

    std::lock_guard<std::mutex> lock(mutex);
    if (!item.Display(*channel))
        LOG_ERROR("Unable to display static content \"%s\".", title);
}

void DisplayDynamic(
    const std::string &title, const Point2i &resolution,
    std::vector<std::string> channelNames,
    std::function<void(Bounds2i b, pstd::span<pstd::span<Float>>)> getTileValues) {
    std::lock_guard<std::mutex> lock(mutex);
    dynamicItems.push_back(DisplayItem(title, resolution, channelNames, getTileValues));
}

}  // namespace pbrt
