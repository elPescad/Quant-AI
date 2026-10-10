#ifndef ALPACA_CLIENT_HPP
#define ALPACA_CLIENT_HPP

// HTTPS GET and WebSocket client on libcurl (>= 7.86 built with WebSocket support, e.g.
// Debian trixie's libcurl 8.14). Blocking calls with timeouts; nothing here spins.

#include <curl/curl.h>
#include <poll.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

struct HttpResponse {
    long status = 0;
    std::string body;
    std::string error; // transport error, empty on success
};

class CurlGlobal {
public:
    CurlGlobal() { curl_global_init(CURL_GLOBAL_DEFAULT); }
    ~CurlGlobal() { curl_global_cleanup(); }
};

// method: GET, POST (body sent as JSON) or DELETE
inline HttpResponse http_request(const std::string& method, const std::string& url, const std::vector<std::string>& headers,
                                 const std::string& body = "", long timeout_s = 20) {
    HttpResponse r;
    CURL* c = curl_easy_init();
    if (!c) {
        r.error = "curl_easy_init failed";
        return r;
    }
    curl_slist* hl = nullptr;
    for (const auto& h : headers) hl = curl_slist_append(hl, h.c_str());
    auto sink = +[](char* p, size_t size, size_t n, void* out) {
        static_cast<std::string*>(out)->append(p, size * n);
        return size * n;
    };
    if (method == "POST") {
        hl = curl_slist_append(hl, "Content-Type: application/json");
        curl_easy_setopt(c, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(c, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    } else if (method != "GET") {
        curl_easy_setopt(c, CURLOPT_CUSTOMREQUEST, method.c_str());
    }
    curl_easy_setopt(c, CURLOPT_URL, url.c_str());
    curl_easy_setopt(c, CURLOPT_HTTPHEADER, hl);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, sink);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &r.body);
    curl_easy_setopt(c, CURLOPT_TIMEOUT, timeout_s);
    curl_easy_setopt(c, CURLOPT_NOSIGNAL, 1L);
    const CURLcode rc = curl_easy_perform(c);
    if (rc != CURLE_OK) r.error = curl_easy_strerror(rc);
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &r.status);
    curl_slist_free_all(hl);
    curl_easy_cleanup(c);
    return r;
}

inline HttpResponse http_get(const std::string& url, const std::vector<std::string>& headers, long timeout_s = 20) {
    return http_request("GET", url, headers, "", timeout_s);
}

class WebSocket {
public:
    WebSocket() = default;
    WebSocket(const WebSocket&) = delete;
    WebSocket& operator=(const WebSocket&) = delete;
    ~WebSocket() { close(); }

    bool connect(const std::string& url, std::string& error, long timeout_s = 20) {
        close();
        curl_ = curl_easy_init();
        if (!curl_) {
            error = "curl_easy_init failed";
            return false;
        }
        curl_easy_setopt(curl_, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl_, CURLOPT_CONNECT_ONLY, 2L); // WebSocket upgrade, then hand the socket to us
        curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT, timeout_s);
        curl_easy_setopt(curl_, CURLOPT_NOSIGNAL, 1L);
        const CURLcode rc = curl_easy_perform(curl_);
        if (rc != CURLE_OK) {
            error = curl_easy_strerror(rc);
            close();
            return false;
        }
        curl_easy_getinfo(curl_, CURLINFO_ACTIVESOCKET, &sock_);
        return true;
    }

    bool connected() const { return curl_ != nullptr; }

    bool send_text(const std::string& msg) {
        size_t off = 0;
        while (curl_ && off < msg.size()) {
            size_t sent = 0;
            const CURLcode rc = curl_ws_send(curl_, msg.data() + off, msg.size() - off, &sent, 0, CURLWS_TEXT);
            if (rc == CURLE_AGAIN) {
                if (!wait(POLLOUT, 5000)) return false;
                continue;
            }
            if (rc != CURLE_OK) return false;
            off += sent;
        }
        return curl_ != nullptr;
    }

    // 1: a complete text/binary message in `out`; 0: nothing within timeout_ms; -1: closed or failed
    int recv(std::string& out, int timeout_ms) {
        out.clear();
        char buf[16384];
        bool started = false;
        while (curl_) {
            size_t n = 0;
            const curl_ws_frame* meta = nullptr;
            const CURLcode rc = curl_ws_recv(curl_, buf, sizeof(buf), &n, &meta);
            if (rc == CURLE_AGAIN) {
                // TLS may have decrypted data buffered, so poll only after curl reports AGAIN
                if (!wait(POLLIN, started ? 5000 : timeout_ms)) {
                    if (!started) return 0;
                    return -1; // stalled mid-message
                }
                continue;
            }
            if (rc != CURLE_OK || !meta) return -1;
            if (meta->flags & CURLWS_CLOSE) return -1;
            if (meta->flags & (CURLWS_PING | CURLWS_PONG)) continue; // libcurl answers pings itself
            started = true;
            out.append(buf, n);
            if (meta->bytesleft == 0 && !(meta->flags & CURLWS_CONT)) return 1;
        }
        return -1;
    }

    void close() {
        if (curl_) {
            size_t sent = 0;
            curl_ws_send(curl_, "", 0, &sent, 0, CURLWS_CLOSE);
            curl_easy_cleanup(curl_);
        }
        curl_ = nullptr;
        sock_ = CURL_SOCKET_BAD;
    }

private:
    bool wait(short events, int timeout_ms) {
        if (sock_ == CURL_SOCKET_BAD) return false;
        pollfd p{static_cast<int>(sock_), events, 0};
        const int r = ::poll(&p, 1, timeout_ms);
        return r > 0 && !(p.revents & (POLLERR | POLLNVAL));
    }

    CURL* curl_ = nullptr;
    curl_socket_t sock_ = CURL_SOCKET_BAD;
};

#endif
