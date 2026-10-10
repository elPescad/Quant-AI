#ifndef PAPER_BROKER_HPP
#define PAPER_BROKER_HPP

// Mirrors the engine's simulated positions into an Alpaca *paper* account.
//
// After every bar the engine hands over its target position per ticker (whole shares). A
// separate thread, so the engine never waits on HTTP, then reconciles the account:
//   read positions -> cancel our open orders -> market orders for the differences.
// A position never flips long <-> short in one order (Alpaca rejects that): it is closed
// first and the new side opened on the next pass a moment later. The account is the source
// of truth, so restarts, partial fills and rejected orders all converge on the next bar.
// Only the configured symbols are touched, only while the market is open, and only on the
// paper endpoint (or a local mock): this class refuses any other trading URL.

#include <chrono>
#include <condition_variable>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "alpaca_client.hpp"
#include "json_lite.hpp"
#include "live_bars.hpp"

class PaperBroker {
public:
    using Targets = std::map<std::string, long>;

    PaperBroker(std::string trading_url, std::string key, std::string secret, std::vector<std::string> symbols,
                const std::string& out_dir)
        : url_(std::move(trading_url)), key_(std::move(key)), secret_(std::move(secret)), symbols_(std::move(symbols)) {
        if (!allowed(url_))
            throw std::runtime_error("paper orders only go to https://paper-api.alpaca.markets (or a local mock), not " + url_);
        orders_.open(out_dir + "/orders.csv", std::ios::app);
        account_.open(out_dir + "/account.csv", std::ios::app);
        if (orders_.tellp() == 0) orders_ << "time,bar,symbol,action,qty,http_status,result" << std::endl;
        if (account_.tellp() == 0) account_ << "time,bar,equity,cash,positions,targets" << std::endl;
        thread_ = std::thread([this] { loop(); });
    }

    PaperBroker(const PaperBroker&) = delete;
    PaperBroker& operator=(const PaperBroker&) = delete;
    ~PaperBroker() { finish(); }

    static bool allowed(const std::string& url) {
        for (const char* ok : {"https://paper-api.alpaca.markets", "http://127.0.0.1", "http://localhost"})
            if (url.rfind(ok, 0) == 0) return true;
        return false;
    }

    // Latest wins: a newer bar's targets replace ones not yet worked on
    void submit(Targets targets, int64_t bar) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            pending_ = std::move(targets);
            pending_bar_ = bar;
            has_pending_ = true;
        }
        cv_.notify_one();
    }

    // Work off what is pending, then stop the thread
    void finish() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stopping_) return;
            stopping_ = true;
        }
        cv_.notify_one();
        if (thread_.joinable()) thread_.join();
    }

private:
    std::vector<std::string> headers() const {
        return {"APCA-API-KEY-ID: " + key_, "APCA-API-SECRET-KEY: " + secret_, "Accept: application/json"};
    }

    static std::string now_str() { return live::to_rfc3339(static_cast<int64_t>(std::time(nullptr))); }

    void loop() {
        while (true) {
            Targets targets;
            int64_t bar = 0;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this] { return has_pending_ || stopping_; });
                if (!has_pending_) return;
                targets = std::move(pending_);
                bar = pending_bar_;
                has_pending_ = false;
            }
            reconcile(targets, bar);
        }
    }

    std::optional<bool> market_open() {
        const auto r = http_request("GET", url_ + "/v2/clock", headers());
        if (!r.error.empty() || r.status != 200) return std::nullopt;
        try { return Json::parse(r.body)["is_open"].boolean; } catch (const std::exception&) { return std::nullopt; }
    }

    std::optional<std::map<std::string, long>> positions() {
        const auto r = http_request("GET", url_ + "/v2/positions", headers());
        if (!r.error.empty() || r.status != 200) {
            log_line(0, "-", "positions", 0, r.status, r.error.empty() ? r.body.substr(0, 200) : r.error);
            return std::nullopt;
        }
        std::map<std::string, long> out;
        try {
            for (const auto& p : Json::parse(r.body).items) out[p.text("symbol")] = std::lround(std::stod(p.text("qty")));
        } catch (const std::exception& e) {
            log_line(0, "-", "positions", 0, r.status, e.what());
            return std::nullopt;
        }
        return out;
    }

    void cancel_open_orders(int64_t bar) {
        std::string syms;
        for (const auto& s : symbols_) syms += (syms.empty() ? "" : ",") + s;
        const auto r = http_request("GET", url_ + "/v2/orders?status=open&symbols=" + syms, headers());
        if (!r.error.empty() || r.status != 200) return;
        try {
            for (const auto& o : Json::parse(r.body).items) {
                const auto d = http_request("DELETE", url_ + "/v2/orders/" + o.text("id"), headers());
                log_line(bar, o.text("symbol"), "cancel", 0, d.status, o.text("id"));
            }
        } catch (const std::exception&) {
        }
    }

    void order(int64_t bar, const std::string& sym, long qty, int attempt) {
        const std::string side = qty > 0 ? "buy" : "sell";
        const long n = qty > 0 ? qty : -qty;
        const std::string coid = "qai-" + std::to_string(bar) + "-" + sym + "-" + std::to_string(attempt);
        const std::string body = R"({"symbol":")" + sym + R"(","qty":")" + std::to_string(n) + R"(","side":")" + side +
                                 R"(","type":"market","time_in_force":"day","client_order_id":")" + coid + R"("})";
        const auto r = http_request("POST", url_ + "/v2/orders", headers(), body);
        std::string result = r.error;
        if (result.empty()) {
            try {
                const Json j = Json::parse(r.body);
                result = r.status < 300 ? j.text("id") + " " + j.text("status") : j.text("message");
            } catch (const std::exception&) {
                result = r.body.substr(0, 200);
            }
        }
        log_line(bar, sym, side, n, r.status, result);
    }

    void close_position(int64_t bar, const std::string& sym, long have) {
        const auto r = http_request("DELETE", url_ + "/v2/positions/" + sym, headers());
        log_line(bar, sym, "close", have, r.status, r.error.empty() ? (r.status < 300 ? "ok" : r.body.substr(0, 200)) : r.error);
    }

    void reconcile(const Targets& targets, int64_t bar) {
        const auto open = market_open();
        if (!open.value_or(false)) {
            log_line(bar, "-", "skip", 0, 0, open ? "market closed" : "clock unavailable");
            return;
        }
        std::map<std::string, long> have;
        for (int attempt = 0; attempt < 5; ++attempt) {
            auto pos = positions();
            if (!pos) break;
            have = *pos;
            if (attempt == 0) cancel_open_orders(bar);
            bool done = true;
            for (const auto& sym : symbols_) {
                const long want = targets.count(sym) ? targets.at(sym) : 0;
                const long cur = have.count(sym) ? have.at(sym) : 0;
                if (want == cur) continue;
                done = false;
                if (cur != 0 && (want == 0 || (want > 0) != (cur > 0))) close_position(bar, sym, cur);
                else order(bar, sym, want - cur, attempt);
            }
            if (done) break;
            std::this_thread::sleep_for(std::chrono::seconds(1)); // let market orders fill
        }
        snapshot(bar, have, targets);
    }

    void snapshot(int64_t bar, const std::map<std::string, long>& have, const Targets& targets) {
        auto fmt = [&](const std::map<std::string, long>& m) {
            std::string s;
            for (const auto& sym : symbols_) {
                auto it = m.find(sym);
                if (it != m.end() && it->second != 0) s += (s.empty() ? "" : ";") + sym + ":" + std::to_string(it->second);
            }
            return s;
        };
        std::string equity = "", cash = "";
        const auto r = http_request("GET", url_ + "/v2/account", headers());
        if (r.error.empty() && r.status == 200) {
            try {
                const Json j = Json::parse(r.body);
                equity = j.text("equity");
                cash = j.text("cash");
            } catch (const std::exception&) {
            }
        }
        account_ << now_str() << "," << live::to_rfc3339(bar) << "," << equity << "," << cash << "," << fmt(have) << ","
                 << fmt(targets) << std::endl;
    }

    void log_line(int64_t bar, const std::string& sym, const std::string& action, long qty, long status, const std::string& result) {
        std::string clean = result;
        for (char& c : clean)
            if (c == ',' || c == '\n' || c == '\r') c = ' ';
        orders_ << now_str() << "," << (bar ? live::to_rfc3339(bar) : "-") << "," << sym << "," << action << "," << qty << ","
                << status << "," << clean << std::endl;
        if (status >= 300) std::cerr << "[paper] " << action << " " << sym << " failed: HTTP " << status << " " << clean << std::endl;
    }

    std::string url_, key_, secret_;
    std::vector<std::string> symbols_;
    std::ofstream orders_, account_;
    std::mutex mu_;
    std::condition_variable cv_;
    Targets pending_;
    int64_t pending_bar_ = 0;
    bool has_pending_ = false;
    bool stopping_ = false;
    std::thread thread_;
};

#endif
