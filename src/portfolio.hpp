#ifndef PORTFOLIO_HPP
#define PORTFOLIO_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <limits>

struct Position {
    double units = 0.0; // Positive for LONG, Negative for SHORT
    double avg_price = 0.0;
    double current_mid_price = 100.0;
    double highest_price_since_entry = 0.0;
    double lowest_price_since_entry = std::numeric_limits<double>::max();
};

class PortfolioManager {
private:
    double cash_balance_;
    double initial_capital_;
    double fee_rate_;
    double max_allocation_pct_;

    double take_profit_pct_;      
    double initial_stop_loss_pct_;
    double trailing_stop_pct_;    

    std::unordered_map<std::string, Position> positions_;

    double peak_equity_;
    double max_drawdown_;
    int total_trades_;
    int winning_trades_;
    std::vector<double> equity_curve_;

    std::ofstream log_file_;

public:
    PortfolioManager(double starting_cash = 10000.0, 
                    double fee_rate = 0.0001, 
                    double max_alloc_pct = 0.20,
                    double tp_pct = 0.0160,     
                    double sl_pct = 0.0100,     
                    double trail_pct = 0.0050)  
        : cash_balance_(starting_cash),
          initial_capital_(starting_cash),
          fee_rate_(fee_rate),
          max_allocation_pct_(max_alloc_pct),
          take_profit_pct_(tp_pct),
          initial_stop_loss_pct_(sl_pct),
          trailing_stop_pct_(trail_pct),
          peak_equity_(starting_cash),
          max_drawdown_(0.0),
          total_trades_(0),
          winning_trades_(0)
    {
        equity_curve_.reserve(500000);
        log_file_.open("trades.csv");
        if (log_file_.is_open()) {
            log_file_ << "tick_id,ticker,action,fill_price,units_traded,cash,total_equity\n";
        }
    }

    ~PortfolioManager() {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }

    void process_signal(int tick_id, const std::string& ticker, int action, float raw_price, float raw_spread) {
        Position& pos = positions_[ticker];
        pos.current_mid_price = static_cast<double>(raw_price);
        double half_spread = static_cast<double>(raw_spread) * 0.5;

        double current_equity = get_total_equity();
        if (current_equity <= 0.0) return;

        // ====================================================================
        // 1. RISK MANAGEMENT: Trailing Stop & Take Profit Checks (Long & Short)
        // ====================================================================
        if (pos.units > 0.0001) { // LONG POSITION ACTIVE
            double sell_fill_price = pos.current_mid_price - half_spread; // Sell at Bid

            if (sell_fill_price > pos.highest_price_since_entry) {
                pos.highest_price_since_entry = sell_fill_price;
            }

            double total_return = (sell_fill_price - pos.avg_price) / pos.avg_price;
            double drawdown_from_peak = (pos.highest_price_since_entry - sell_fill_price) / pos.highest_price_since_entry;

            if (total_return >= take_profit_pct_) {
                execute_sell(tick_id, ticker, pos, sell_fill_price, "TAKE_PROFIT_LONG");
                return;
            }
            if (total_return <= -initial_stop_loss_pct_) {
                execute_sell(tick_id, ticker, pos, sell_fill_price, "HARD_STOP_LOSS_LONG");
                return;
            }
            if (total_return > 0.0020 && drawdown_from_peak >= trailing_stop_pct_) {
                execute_sell(tick_id, ticker, pos, sell_fill_price, "TRAILING_STOP_LONG");
                return;
            }
        }
        else if (pos.units < -0.0001) { // SHORT POSITION ACTIVE
            double cover_fill_price = pos.current_mid_price + half_spread; // Cover at Ask

            if (cover_fill_price < pos.lowest_price_since_entry) {
                pos.lowest_price_since_entry = cover_fill_price;
            }

            double total_return = (pos.avg_price - cover_fill_price) / pos.avg_price;
            double drawdown_from_peak = (cover_fill_price - pos.lowest_price_since_entry) / pos.lowest_price_since_entry;

            if (total_return >= take_profit_pct_) {
                execute_cover(tick_id, ticker, pos, cover_fill_price, "TAKE_PROFIT_SHORT");
                return;
            }
            if (total_return <= -initial_stop_loss_pct_) {
                execute_cover(tick_id, ticker, pos, cover_fill_price, "HARD_STOP_LOSS_SHORT");
                return;
            }
            if (total_return > 0.0030 && drawdown_from_peak >= trailing_stop_pct_) {
                execute_cover(tick_id, ticker, pos, cover_fill_price, "TRAILING_STOP_SHORT");
                return;
            }
        }

        // ====================================================================
        // 2. MODEL SIGNAL EXECUTION WITH SPREAD-HURDLE FILTER
        // ====================================================================
        double spread_relative = static_cast<double>(raw_spread) / pos.current_mid_price;

        if (spread_relative > 0.0025 && std::abs(pos.units) < 0.0001) {
            return;
        }

        double target_trade_usd = current_equity * max_allocation_pct_;

        if (action == 2) { // SIGNAL: BUY / LONG
            double buy_fill_price = pos.current_mid_price + half_spread; // Pay Ask
            
            if (pos.units < -0.0001) { 
                execute_cover(tick_id, ticker, pos, buy_fill_price, "SIGNAL_COVER_SHORT"); 
            }
            
            if (std::abs(pos.units) < 0.0001) { // Flat, open LONG
                double max_spendable_cash = cash_balance_ / (1.0 + fee_rate_);
                double actual_trade_usd = std::min(target_trade_usd, max_spendable_cash);

                if (actual_trade_usd >= 10.0) {
                    double fee = actual_trade_usd * fee_rate_;
                    double units_to_buy = actual_trade_usd / buy_fill_price;

                    cash_balance_ -= (actual_trade_usd + fee);
                    pos.units = units_to_buy;
                    pos.avg_price = buy_fill_price;
                    pos.highest_price_since_entry = buy_fill_price;
                    pos.lowest_price_since_entry = std::numeric_limits<double>::max();

                    log_trade(tick_id, ticker, "OPEN_LONG", units_to_buy, buy_fill_price);
                }
            }
        } 
        else if (action == 0) { // SIGNAL: SELL / SHORT
            double sell_fill_price = pos.current_mid_price - half_spread; // Sell to Bid

            if (pos.units > 0.0001) {
                execute_sell(tick_id, ticker, pos, sell_fill_price, "SIGNAL_SELL_LONG");
            }

            if (std::abs(pos.units) < 0.0001) { // Flat, open SHORT
                double max_spendable_cash = cash_balance_ / (1.0 + fee_rate_);
                double actual_trade_usd = std::min(target_trade_usd, max_spendable_cash);

                if (actual_trade_usd >= 10.0) {
                    double fee = actual_trade_usd * fee_rate_;
                    double units_to_short = actual_trade_usd / sell_fill_price;

                    cash_balance_ += (actual_trade_usd - fee);
                    pos.units = -units_to_short;
                    pos.avg_price = sell_fill_price;
                    pos.lowest_price_since_entry = sell_fill_price;
                    pos.highest_price_since_entry = 0.0;

                    log_trade(tick_id, ticker, "OPEN_SHORT", units_to_short, sell_fill_price);
                }
            }
        }

        // ====================================================================
        // 3. METRIC TRACKING
        // ====================================================================
        current_equity = get_total_equity();
        equity_curve_.push_back(current_equity);

        if (current_equity > peak_equity_) {
            peak_equity_ = current_equity;
        } else if (peak_equity_ > 0.0) {
            double drawdown = (peak_equity_ - current_equity) / peak_equity_;
            if (drawdown > max_drawdown_) max_drawdown_ = drawdown;
        }
    }

    double get_total_equity() const {
        double position_value = 0.0;
        for (const auto& [ticker, pos] : positions_) {
            position_value += (pos.units * pos.current_mid_price);
        }
        return cash_balance_ + position_value;
    }

    double get_pnl() const { return get_total_equity() - initial_capital_; }
    double get_max_drawdown() const { return max_drawdown_ * 100.0; }

    double get_win_rate() const {
        if (total_trades_ == 0) return 0.0;
        return (static_cast<double>(winning_trades_) / total_trades_) * 100.0;
    }

    double calculate_sharpe_ratio() const {
        if (equity_curve_.size() < 2) return 0.0;
        std::vector<double> returns;
        returns.reserve(equity_curve_.size() - 1);

        for (size_t i = 1; i < equity_curve_.size(); ++i) {
            if (equity_curve_[i - 1] > 0) {
                returns.push_back((equity_curve_[i] - equity_curve_[i - 1]) / equity_curve_[i - 1]);
            }
        }
        if (returns.empty()) return 0.0;

        double sum = std::accumulate(returns.begin(), returns.end(), 0.0);
        double mean = sum / returns.size();

        double sq_sum = 0.0;
        for (double r : returns) sq_sum += (r - mean) * (r - mean);
        double stdev = std::sqrt(sq_sum / returns.size());

        if (stdev == 0.0) return 0.0;
        // Adjusted for tick data (~10,000 ticks per day annualized across 252 days)
        return (mean / stdev) * std::sqrt(252.0 * 10000.0);
    }

private:
    void execute_sell(int tick_id, const std::string& ticker, Position& pos, double fill_price, const std::string& reason) {
        double units_to_sell = pos.units;
        double gross_proceeds = units_to_sell * fill_price;
        double fee = gross_proceeds * fee_rate_;
        double net_proceeds = gross_proceeds - fee;

        double cost_basis = units_to_sell * pos.avg_price;
        double trade_pnl = net_proceeds - cost_basis;

        cash_balance_ += net_proceeds;
        pos.units = 0.0;
        pos.avg_price = 0.0;
        pos.highest_price_since_entry = 0.0;
        pos.lowest_price_since_entry = std::numeric_limits<double>::max();

        total_trades_++;
        if (trade_pnl > 0.0) winning_trades_++;

        log_trade(tick_id, ticker, reason, units_to_sell, fill_price);
    }

    void execute_cover(int tick_id, const std::string& ticker, Position& pos, double fill_price, const std::string& reason) {
        double units_to_cover = std::abs(pos.units);
        double gross_cost = units_to_cover * fill_price;
        double fee = gross_cost * fee_rate_;
        double net_cost = gross_cost + fee;

        double initial_proceeds = units_to_cover * pos.avg_price;
        double trade_pnl = initial_proceeds - net_cost;

        cash_balance_ -= net_cost;
        pos.units = 0.0;
        pos.avg_price = 0.0;
        pos.highest_price_since_entry = 0.0;
        pos.lowest_price_since_entry = std::numeric_limits<double>::max();

        total_trades_++;
        if (trade_pnl > 0.0) winning_trades_++;

        log_trade(tick_id, ticker, reason, units_to_cover, fill_price);
    }

    void log_trade(int tick_id, const std::string& ticker, const std::string& action_str, double units, double price) {
        if (log_file_.is_open()) {
            log_file_ << tick_id << "," << ticker << "," << action_str << ","
                      << price << "," << units << "," << cash_balance_ << ","
                      << get_total_equity() << "\n";
        }
    }
};

#endif // PORTFOLIO_HPP