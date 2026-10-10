#ifndef PORTFOLIO_HPP
#define PORTFOLIO_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

struct Position {
    double units = 0.0; // Positive for LONG, negative for SHORT
    double avg_price = 0.0;
    double current_mid_price = 0.0;
    double half_spread_pct = 0.0; // Capped half spread we expect to pay per fill
    double highest_price_since_entry = 0.0;
    double lowest_price_since_entry = std::numeric_limits<double>::max();

    int direction() const { return units > 1e-9 ? 1 : (units < -1e-9 ? -1 : 0); }
};

struct PortfolioParams {
    double starting_cash = 10000.0;
    // Per-fill fees as a fraction of notional. Alpaca charges no commission; the regulatory fees
    // on sales (SEC Section 31, FINRA TAF) are a fraction of a basis point.
    double fee_rate = 0.00002;
    double position_weight = 0.20;     // Fraction of equity per position
    double max_gross_leverage = 1.0;   // Cap on sum(|position value|) / equity
    double take_profit_pct = 0.0160;
    double stop_loss_pct = 0.0100;
    double trailing_stop_pct = 0.0050;
    double trailing_activation_pct = 0.0020; // Trailing stop arms after this open profit
    double max_spread_cost_pct = 0.0005;     // Without quotes, raw_spread (the bar's high-low range) is a cost proxy; cap it
    double max_raw_spread_pct = 0.01;        // Larger spreads are treated as bad data
    double bars_per_year = 252.0 * 78.0;     // 5-minute bars
    std::string trade_log = "trades.csv";
};

class PortfolioManager {
public:
    explicit PortfolioManager(PortfolioParams p = {})
        : p_(p), cash_balance_(p.starting_cash), peak_equity_(p.starting_cash) {
        equity_curve_.reserve(100000);
        if (!p_.trade_log.empty()) {
            log_file_.open(p_.trade_log);
            if (log_file_.is_open()) log_file_ << "tick_id,ticker,action,fill_price,units_traded,cash,total_equity\n";
        }
    }

    // Updates the mark price and runs take-profit / stop-loss / trailing-stop exits.
    // Returns true when a risk exit closed the position.
    // quoted_spread: measured bid-ask spread in $ (<= 0: unknown, use the capped raw_spread proxy)
    bool mark(int tick_id, const std::string& ticker, float raw_price, float raw_spread, float quoted_spread = -1.0f) {
        // Reject malformed ticks before they can turn into absurd position sizes
        if (!std::isfinite(raw_price) || raw_price <= 0.0f || !std::isfinite(raw_spread) || raw_spread < 0.0f) {
            return false;
        }
        Position& pos = positions_[ticker];
        pos.current_mid_price = raw_price;
        const bool quoted = std::isfinite(quoted_spread) && quoted_spread > 0.0f;
        const double spread_rel = static_cast<double>(quoted ? quoted_spread : raw_spread) / pos.current_mid_price;
        spread_ok_[ticker] = spread_rel <= p_.max_raw_spread_pct;
        pos.half_spread_pct = 0.5 * (quoted ? spread_rel : std::min(spread_rel, p_.max_spread_cost_pct));

        if (pos.direction() > 0) {
            const double fill = bid(pos);
            pos.highest_price_since_entry = std::max(pos.highest_price_since_entry, fill);
            const double ret = (fill - pos.avg_price) / pos.avg_price;
            const double dd = (pos.highest_price_since_entry - fill) / pos.highest_price_since_entry;
            if (ret >= p_.take_profit_pct) return close(tick_id, ticker, pos, "TAKE_PROFIT_LONG");
            if (ret <= -p_.stop_loss_pct) return close(tick_id, ticker, pos, "HARD_STOP_LOSS_LONG");
            if (ret > p_.trailing_activation_pct && dd >= p_.trailing_stop_pct) return close(tick_id, ticker, pos, "TRAILING_STOP_LONG");
        } else if (pos.direction() < 0) {
            const double fill = ask(pos);
            pos.lowest_price_since_entry = std::min(pos.lowest_price_since_entry, fill);
            const double ret = (pos.avg_price - fill) / pos.avg_price;
            const double dd = (fill - pos.lowest_price_since_entry) / pos.lowest_price_since_entry;
            if (ret >= p_.take_profit_pct) return close(tick_id, ticker, pos, "TAKE_PROFIT_SHORT");
            if (ret <= -p_.stop_loss_pct) return close(tick_id, ticker, pos, "HARD_STOP_LOSS_SHORT");
            if (ret > p_.trailing_activation_pct && dd >= p_.trailing_stop_pct) return close(tick_id, ticker, pos, "TRAILING_STOP_SHORT");
        }
        return false;
    }

    // Moves every listed ticker to its target direction (-1, 0, +1).
    // All closes run before any opens so freed cash can fund the new positions.
    void rebalance(int tick_id, const std::vector<std::pair<std::string, int>>& targets) {
        for (const auto& [ticker, dir] : targets) {
            auto it = positions_.find(ticker);
            if (it == positions_.end()) continue;
            Position& pos = it->second;
            if (pos.direction() != 0 && pos.direction() != dir) {
                close(tick_id, ticker, pos, pos.direction() > 0 ? "SIGNAL_SELL_LONG" : "SIGNAL_COVER_SHORT");
            }
        }
        for (const auto& [ticker, dir] : targets) {
            auto it = positions_.find(ticker);
            if (it == positions_.end() || dir == 0 || it->second.direction() == dir) continue;
            if (!spread_ok_[ticker]) continue;
            open(tick_id, ticker, it->second, dir);
        }
    }

    // Close every open position at the last mark +/- capped half spread
    void liquidate_all(int tick_id) {
        for (auto& [ticker, pos] : positions_) {
            if (pos.direction() > 0) close(tick_id, ticker, pos, "END_OF_DATA_SELL");
            else if (pos.direction() < 0) close(tick_id, ticker, pos, "END_OF_DATA_COVER");
        }
    }

    // Call once per bar so the Sharpe ratio is per-bar
    void record_equity() {
        const double eq = get_total_equity();
        equity_curve_.push_back(eq);
        gross_exposure_sum_ += get_gross_exposure() / std::max(eq, 1e-9);
        if (eq > peak_equity_) peak_equity_ = eq;
        else if (peak_equity_ > 0.0) max_drawdown_ = std::max(max_drawdown_, (peak_equity_ - eq) / peak_equity_);
    }

    int direction(const std::string& ticker) const {
        auto it = positions_.find(ticker);
        return it == positions_.end() ? 0 : it->second.direction();
    }

    // Expected one-way cost of trading this ticker, as a fraction of notional
    double one_way_cost(const std::string& ticker) const {
        auto it = positions_.find(ticker);
        return (it == positions_.end() ? 0.5 * p_.max_spread_cost_pct : it->second.half_spread_pct) + p_.fee_rate;
    }

    double get_total_equity() const {
        double position_value = 0.0;
        for (const auto& [ticker, pos] : positions_) position_value += pos.units * pos.current_mid_price;
        return cash_balance_ + position_value;
    }

    double get_gross_exposure() const {
        double gross = 0.0;
        for (const auto& [ticker, pos] : positions_) gross += std::abs(pos.units * pos.current_mid_price);
        return gross;
    }

    double get_cash() const { return cash_balance_; }
    double get_pnl() const { return get_total_equity() - p_.starting_cash; }
    double get_return_pct() const { return 100.0 * get_pnl() / p_.starting_cash; }
    double get_max_drawdown() const { return max_drawdown_ * 100.0; }
    int get_total_trades() const { return total_trades_; }
    int get_risk_exits() const { return risk_exits_; }
    double get_fees_paid() const { return fees_paid_; }
    double get_avg_gross_exposure() const {
        return equity_curve_.empty() ? 0.0 : 100.0 * gross_exposure_sum_ / equity_curve_.size();
    }

    double get_win_rate() const {
        return total_trades_ == 0 ? 0.0 : 100.0 * winning_trades_ / total_trades_;
    }

    double calculate_sharpe_ratio() const {
        if (equity_curve_.size() < 2) return 0.0;
        std::vector<double> r;
        r.reserve(equity_curve_.size() - 1);
        for (size_t i = 1; i < equity_curve_.size(); ++i) {
            if (equity_curve_[i - 1] > 0) r.push_back(equity_curve_[i] / equity_curve_[i - 1] - 1.0);
        }
        if (r.size() < 2) return 0.0;
        const double mean = std::accumulate(r.begin(), r.end(), 0.0) / r.size();
        double sq = 0.0;
        for (double x : r) sq += (x - mean) * (x - mean);
        const double sd = std::sqrt(sq / (r.size() - 1));
        return sd == 0.0 ? 0.0 : (mean / sd) * std::sqrt(p_.bars_per_year);
    }

private:
    static double bid(const Position& p) { return p.current_mid_price * (1.0 - p.half_spread_pct); }
    static double ask(const Position& p) { return p.current_mid_price * (1.0 + p.half_spread_pct); }

    double get_short_liability() const {
        double liability = 0.0;
        for (const auto& [ticker, pos] : positions_) if (pos.units < 0.0) liability -= pos.units * pos.current_mid_price;
        return liability;
    }

    // Dollars a new position may use without breaching the size, leverage or cash limits
    double available_trade_usd() const {
        const double equity = get_total_equity();
        if (equity <= 0.0) return 0.0;
        const double target = equity * p_.position_weight;
        // Fees reduce equity, so solve gross + usd <= lev * (equity - usd * fee) for usd
        const double exposure_room = (equity * p_.max_gross_leverage - get_gross_exposure()) /
                                     (1.0 + p_.max_gross_leverage * p_.fee_rate);
        // Short proceeds sit in cash but are owed back, so they are not buying power
        const double free_cash = std::max(0.0, cash_balance_ - get_short_liability()) / (1.0 + p_.fee_rate);
        return std::max(0.0, std::min({target, exposure_room, free_cash}));
    }

    void open(int tick_id, const std::string& ticker, Position& pos, int dir) {
        const double usd = available_trade_usd();
        if (usd < 10.0) return;
        const double fee = usd * p_.fee_rate;
        fees_paid_ += fee;
        if (dir > 0) {
            const double fill = ask(pos);
            pos.units = usd / fill;
            pos.avg_price = fill;
            pos.highest_price_since_entry = fill;
            cash_balance_ -= usd + fee;
            log_trade(tick_id, ticker, "OPEN_LONG", pos.units, fill);
        } else {
            const double fill = bid(pos);
            pos.units = -usd / fill;
            pos.avg_price = fill;
            pos.lowest_price_since_entry = fill;
            cash_balance_ += usd - fee;
            log_trade(tick_id, ticker, "OPEN_SHORT", -pos.units, fill);
        }
    }

    // Always returns true so risk exits can `return close(...)`
    bool close(int tick_id, const std::string& ticker, Position& pos, const char* reason) {
        const double units = std::abs(pos.units);
        double pnl;
        double fill;
        if (pos.units > 0.0) {
            fill = bid(pos);
            const double gross = units * fill;
            const double fee = gross * p_.fee_rate;
            fees_paid_ += fee;
            cash_balance_ += gross - fee;
            pnl = gross - fee - units * pos.avg_price;
        } else {
            fill = ask(pos);
            const double gross = units * fill;
            const double fee = gross * p_.fee_rate;
            fees_paid_ += fee;
            cash_balance_ -= gross + fee;
            pnl = units * pos.avg_price - gross - fee;
        }
        pos.units = 0.0;
        pos.avg_price = 0.0;
        pos.highest_price_since_entry = 0.0;
        pos.lowest_price_since_entry = std::numeric_limits<double>::max();

        total_trades_++;
        if (pnl > 0.0) winning_trades_++;
        if (std::string(reason).find("STOP") != std::string::npos || std::string(reason).find("TAKE_PROFIT") != std::string::npos) {
            risk_exits_++;
        }
        log_trade(tick_id, ticker, reason, units, fill);
        return true;
    }

    void log_trade(int tick_id, const std::string& ticker, const char* action, double units, double price) {
        if (log_file_.is_open()) {
            log_file_ << tick_id << "," << ticker << "," << action << "," << price << "," << units << ","
                      << cash_balance_ << "," << get_total_equity() << "\n";
        }
    }

    PortfolioParams p_;
    double cash_balance_;
    std::unordered_map<std::string, Position> positions_;
    std::unordered_map<std::string, bool> spread_ok_;

    double peak_equity_;
    double max_drawdown_ = 0.0;
    double gross_exposure_sum_ = 0.0;
    double fees_paid_ = 0.0;
    int total_trades_ = 0;
    int winning_trades_ = 0;
    int risk_exits_ = 0;
    std::vector<double> equity_curve_;
    std::ofstream log_file_;
};

#endif // PORTFOLIO_HPP
