#ifndef LIVE_FEED_HPP
#define LIVE_FEED_HPP

// Live market data from Alpaca for the engine's producer thread.
//
//   loop:  market clock (trading API /v2/clock)
//          closed -> sleep until the next open (interruptible, no polling of the data API)
//          open   -> REST catch-up: complete 5-minute bars since the last one delivered
//                    (the first time: --warmup-days of history, which warms the features)
//                 -> WebSocket stream: subscribe to 1-minute bars and quotes, build
//                    5-minute bars, deliver each one as soon as it is complete
//                 -> at the close, or on a dropped connection, back to the top
// Every delivered bar goes through the same feature code as the training CSV
// (live_bars.hpp), followed by an end-of-bar marker so the engine acts on it at once.
// A bar is never delivered twice: everything at or before the last delivered bar start
// is skipped, whether it comes from REST or from the stream.

#include <atomic>
#include <chrono>
#include <ctime>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "alpaca_client.hpp"
#include "json_lite.hpp"
#include "live_bars.hpp"

struct LiveConfig {
    std::vector<std::string> symbols{"SPY", "QQQ", "AAPL", "NVDA", "MSFT", "AMD"};
    std::string feed = "iex"; // iex (free plan) or sip (paid real-time plan)
    std::string key, secret;
    std::string data_url = "https://data.alpaca.markets";
    std::string trading_url = "https://paper-api.alpaca.markets";
    std::string stream_url; // default wss://stream.data.alpaca.markets/v2/<feed>
    int warmup_days = 30;   // calendar days of 5-minute history to warm the features
    int bar_grace_s = 20;   // deliver an incomplete 5-minute bar this long after it ends
    int close_grace_s = 90; // keep streaming this long after the close for the last bar
};

class LiveFeed {
public:
    using EmitTick = std::function<void(const MarketTick&)>;
    using EmitBarEnd = std::function<void(int64_t bar_start)>;

    LiveFeed(LiveConfig cfg, EmitTick tick, EmitBarEnd bar_end)
        : cfg_(std::move(cfg)), emit_tick_(std::move(tick)), emit_bar_end_(std::move(bar_end)), agg_(cfg_.symbols),
          stream_overridden_(!cfg_.stream_url.empty()) {
        if (cfg_.stream_url.empty()) cfg_.stream_url = "wss://stream.data.alpaca.markets/v2/" + cfg_.feed;
    }

    void run(const std::atomic<bool>& stop) {
        int backoff_s = 1;
        while (!stop.load()) {
            auto clock = market_clock();
            if (!clock) {
                log("market clock unavailable, retrying in " + std::to_string(backoff_s) + " s");
                sleep_for(backoff_s, stop);
                backoff_s = std::min(backoff_s * 2, 60);
                continue;
            }
            if (!clock->is_open) {
                log("market closed; next open " + live::to_rfc3339(clock->next_open) + ", sleeping");
                sleep_until(clock->next_open - 60, stop);
                continue;
            }
            if (!catch_up() && delivered_ == kNone) {
                log("history unavailable, retrying in " + std::to_string(backoff_s) + " s");
                sleep_for(backoff_s, stop);
                backoff_s = std::min(backoff_s * 2, 60);
                continue;
            }
            if (stream_session(clock->next_close, stop)) {
                backoff_s = 1;
            } else if (!stop.load()) {
                log("stream dropped, reconnecting in " + std::to_string(backoff_s) + " s");
                sleep_for(backoff_s, stop);
                backoff_s = std::min(backoff_s * 2, 60);
            }
        }
    }

    // Connect, authenticate and subscribe, print what arrives for `seconds`, then return.
    // Uses Alpaca's always-on test stream (symbol FAKEPACA) unless a stream URL was given.
    bool check(int seconds, std::ostream& out) {
        auto clock = market_clock();
        out << "[check] market clock: " << (clock ? (clock->is_open ? "open" : "closed, next open " + live::to_rfc3339(clock->next_open)) : "UNAVAILABLE") << "\n";
        auto bars = history(now() - 7 * 86400, now());
        out << "[check] REST 5-minute bars (" << cfg_.feed << ", last 7 days): " << (bars ? std::to_string(bars->size()) + " bars" : "FAILED") << "\n";
        WebSocket ws;
        std::string err;
        const std::string url = stream_overridden_ ? cfg_.stream_url : "wss://stream.data.alpaca.markets/v2/test";
        if (!open_stream(ws, err, url, {"FAKEPACA"})) {
            out << "[check] stream: FAILED (" << err << ")\n";
            return false;
        }
        out << "[check] stream " << url << ": authenticated and subscribed\n";
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
        int shown = 0;
        std::string msg;
        while (std::chrono::steady_clock::now() < until && shown < 5) {
            const int r = ws.recv(msg, 1000);
            if (r < 0) break;
            if (r == 1) {
                out << "[check]   " << msg.substr(0, 200) << "\n";
                shown++;
            }
        }
        return clock.has_value() && bars.has_value();
    }

private:
    static constexpr int64_t kNone = INT64_MIN;

    struct Clock {
        bool is_open = false;
        int64_t next_open = 0, next_close = 0;
    };
    struct Quote {
        double spread = -1.0;
        int64_t time = 0;
    };

    static int64_t now() { return static_cast<int64_t>(std::time(nullptr)); }

    static void log(const std::string& s) { std::cerr << "[live " << live::to_rfc3339(now()) << "] " << s << std::endl; }

    static void sleep_for(int seconds, const std::atomic<bool>& stop) {
        for (int i = 0; i < seconds && !stop.load(); ++i) std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // 1-second steps so a stop request (docker stop, Ctrl-C) is honoured within a second
    static void sleep_until(int64_t t, const std::atomic<bool>& stop) {
        while (!stop.load() && now() < t) std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::vector<std::string> auth_headers() const {
        return {"APCA-API-KEY-ID: " + cfg_.key, "APCA-API-SECRET-KEY: " + cfg_.secret, "Accept: application/json"};
    }

    std::optional<Clock> market_clock() {
        const auto r = http_get(cfg_.trading_url + "/v2/clock", auth_headers());
        if (!r.error.empty() || r.status != 200) {
            log("clock request failed: " + (r.error.empty() ? "HTTP " + std::to_string(r.status) + " " + r.body.substr(0, 200) : r.error));
            return std::nullopt;
        }
        try {
            const Json j = Json::parse(r.body);
            auto open = live::parse_rfc3339(j.text("next_open"));
            auto close = live::parse_rfc3339(j.text("next_close"));
            if (!open || !close) return std::nullopt;
            return Clock{j["is_open"].boolean, *open, *close};
        } catch (const std::exception& e) {
            log(std::string("bad clock response: ") + e.what());
            return std::nullopt;
        }
    }

    // Complete 5-minute bars in [start, end), all symbols, oldest first
    std::optional<std::vector<live::Bar>> history(int64_t start, int64_t end) {
        std::string syms;
        for (const auto& s : cfg_.symbols) syms += (syms.empty() ? "" : ",") + s;
        std::vector<live::Bar> out;
        std::string page;
        do {
            std::string url = cfg_.data_url + "/v2/stocks/bars?symbols=" + syms + "&timeframe=5Min&start=" + live::to_rfc3339(start) +
                              "&end=" + live::to_rfc3339(end) + "&limit=10000&adjustment=raw&sort=asc&feed=" + cfg_.feed;
            if (!page.empty()) url += "&page_token=" + page;
            const auto r = http_get(url, auth_headers(), 60);
            if (!r.error.empty() || r.status != 200) {
                log("bars request failed: " + (r.error.empty() ? "HTTP " + std::to_string(r.status) + " " + r.body.substr(0, 200) : r.error));
                return std::nullopt;
            }
            try {
                const Json j = Json::parse(r.body);
                for (const auto& [sym, list] : j["bars"].fields) {
                    for (const auto& b : list.items) {
                        auto t = live::parse_rfc3339(b.text("t"));
                        if (t) out.push_back({sym, *t, b.num("o"), b.num("h"), b.num("l"), b.num("c"), b.num("v")});
                    }
                }
                page = j.text("next_page_token");
            } catch (const std::exception& e) {
                log(std::string("bad bars response: ") + e.what());
                return std::nullopt;
            }
        } while (!page.empty());
        return out;
    }

    // Deliver every complete bar after the last delivered one (or warm-up history)
    bool catch_up() {
        const int64_t t = now();
        const int64_t from = delivered_ == kNone ? t - int64_t{cfg_.warmup_days} * 86400 : delivered_ + live::kBarSeconds;
        auto bars = history(from, t);
        if (!bars) return false;
        std::map<int64_t, std::vector<live::Bar>> by_start;
        for (auto& b : *bars)
            if (b.start + live::kBarSeconds <= t) by_start[b.start].push_back(std::move(b));
        size_t n = 0;
        for (auto& [start, group] : by_start) n += deliver(group);
        if (n) log("caught up " + std::to_string(n) + " bars from REST, through " + live::to_rfc3339(delivered_));
        agg_.skip_through(delivered_);
        return true;
    }

    bool open_stream(WebSocket& ws, std::string& err, const std::string& url, const std::vector<std::string>& symbols) {
        if (!ws.connect(url, err)) return false;
        std::string msg;
        auto expect = [&](const std::string& what) {
            const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(15);
            while (std::chrono::steady_clock::now() < until) {
                const int r = ws.recv(msg, 1000);
                if (r < 0) { err = "connection closed waiting for " + what; return false; }
                if (r == 0) continue;
                try {
                    for (const auto& m : Json::parse(msg).items) {
                        const std::string type = m.text("T");
                        if (type == "error") { err = "Alpaca error " + std::to_string(static_cast<int>(m.num("code"))) + ": " + m.text("msg"); return false; }
                        if ((what == "subscription" && type == "subscription") || (type == "success" && m.text("msg") == what)) return true;
                    }
                } catch (const std::exception& e) { err = std::string("bad message: ") + e.what(); return false; }
            }
            err = "timed out waiting for " + what;
            return false;
        };
        if (!expect("connected")) return false;
        if (!ws.send_text(R"({"action":"auth","key":")" + cfg_.key + R"(","secret":")" + cfg_.secret + R"("})")) { err = "send failed"; return false; }
        if (!expect("authenticated")) return false;
        std::string list;
        for (const auto& s : symbols) list += (list.empty() ? "\"" : ",\"") + s + "\"";
        if (!ws.send_text(R"({"action":"subscribe","bars":[)" + list + R"(],"quotes":[)" + list + "]}")) { err = "send failed"; return false; }
        return expect("subscription");
    }

    // Returns true when the session ended normally (market close or stop)
    bool stream_session(int64_t next_close, const std::atomic<bool>& stop) {
        WebSocket ws;
        std::string err, msg;
        if (!open_stream(ws, err, cfg_.stream_url, cfg_.symbols)) {
            log("stream: " + err);
            return false;
        }
        log("streaming " + cfg_.feed + " bars and quotes until " + live::to_rfc3339(next_close));
        while (!stop.load()) {
            const int r = ws.recv(msg, 1000);
            if (r < 0) return false;
            if (r == 1) handle(msg);
            deliver(agg_.on_clock(now(), cfg_.bar_grace_s));
            if (now() >= next_close + cfg_.close_grace_s) {
                deliver(agg_.flush());
                log("market closed");
                return true;
            }
        }
        return true;
    }

    void handle(const std::string& msg) {
        Json j;
        try { j = Json::parse(msg); } catch (const std::exception& e) { log(std::string("bad message: ") + e.what()); return; }
        for (const auto& m : j.items) {
            const std::string type = m.text("T");
            if (type == "b") {
                auto t = live::parse_rfc3339(m.text("t"));
                if (t) deliver(agg_.add_minute({m.text("S"), *t, m.num("o"), m.num("h"), m.num("l"), m.num("c"), m.num("v")}));
            } else if (type == "q") {
                const double bp = m.num("bp"), ap = m.num("ap");
                auto t = live::parse_rfc3339(m.text("t"));
                if (bp > 0 && ap > bp && t) quotes_[m.text("S")] = {ap - bp, *t};
            } else if (type == "error") {
                log("Alpaca error " + std::to_string(static_cast<int>(m.num("code"))) + ": " + m.text("msg"));
            }
        }
    }

    // One cross-section (bars sharing a start time); returns how many ticks went out
    size_t deliver(const std::vector<live::Bar>& bars) {
        if (bars.empty() || bars.front().start <= delivered_ || !live::in_regular_session(bars.front().start)) return 0;
        const int64_t start = bars.front().start;
        size_t n = 0;
        for (const auto& sym : cfg_.symbols) {
            for (const auto& b : bars) {
                if (b.symbol != sym) continue;
                emit_tick_(features_.make_tick(b, spread_at(sym, start + live::kBarSeconds)));
                n++;
            }
        }
        delivered_ = start;
        emit_bar_end_(start);
        return n;
    }

    // Latest quoted spread if it is from this bar, else unknown (the engine then uses its proxy)
    float spread_at(const std::string& sym, int64_t bar_end) const {
        auto it = quotes_.find(sym);
        if (it == quotes_.end() || it->second.time < bar_end - live::kBarSeconds || it->second.time > bar_end + 60) return -1.0f;
        return static_cast<float>(it->second.spread);
    }

    LiveConfig cfg_;
    EmitTick emit_tick_;
    EmitBarEnd emit_bar_end_;
    live::FiveMinuteAggregator agg_;
    live::FeatureState features_;
    std::map<std::string, Quote> quotes_;
    int64_t delivered_ = kNone;
    bool stream_overridden_ = false;
    CurlGlobal curl_global_;
};

#endif
