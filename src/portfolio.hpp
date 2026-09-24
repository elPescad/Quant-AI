#ifndef PORTFOLIO_HPP
#define PORTFOLIO_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>

class PortfolioManager {
private:
    double cash_balance_;
    double position_units_;
    double current_asset_price_;
    double initial_capital_;
    double fee_rate_;

    double peak_equity_;
    double max_drawdown_;
    int total_trades_;
    int winning_trades_;
    std::vector<double> equity_curve_;

    std::ofstream log_file_;

public:
    PortfolioManager(double starting_cash = 10000.0, double fee_rate = 0.0005)
        : cash_balance_(starting_cash),
          position_units_(0.0),
          current_asset_price_(100.0),
          initial_capital_(starting_cash),
          fee_rate_(fee_rate),
          peak_equity_(starting_cash),
          max_drawdown_(0.0),
          total_trades_(0),
          winning_trades_(0)
    {
        equity_curve_.reserve(100000);
        log_file_.open("trades.csv");
        if (log_file_.is_open()) {
            log_file_ << "tick_id,action,price,units_traded,cash,position_units,total_equity\n";
        }
    }

    ~PortfolioManager() {
        if (log_file_.is_open()) {
            log_file_.close();
        }
    }

    void process_signal(int tick_id, int action, float price_delta) {
        current_asset_price_ += price_delta;
        if (current_asset_price_ <= 0.01) current_asset_price_ = 0.01;

        double trade_size_usd = 1000.0;
        double units_to_trade = trade_size_usd / current_asset_price_;

        double prev_equity = get_total_equity();

        if (action == 2) { // BUY
            if (cash_balance_ >= trade_size_usd) {
                double fee = trade_size_usd * fee_rate_;
                cash_balance_ -= (trade_size_usd + fee);
                position_units_ += units_to_trade;
                total_trades_++;
                log_trade(tick_id, "BUY", units_to_trade);
            }
        } 
        else if (action == 0) { // SELL
            if (position_units_ >= units_to_trade) {
                double gross_proceeds = units_to_trade * current_asset_price_;
                double fee = gross_proceeds * fee_rate_;
                cash_balance_ += (gross_proceeds - fee);
                position_units_ -= units_to_trade;
                total_trades_++;
                log_trade(tick_id, "SELL", units_to_trade);
            }
        }

        double current_equity = get_total_equity();
        equity_curve_.push_back(current_equity);

        if (current_equity > prev_equity && action != 1) {
            winning_trades_++;
        }

        if (current_equity > peak_equity_) {
            peak_equity_ = current_equity;
        } else {
            double drawdown = (peak_equity_ - current_equity) / peak_equity_;
            if (drawdown > max_drawdown_) {
                max_drawdown_ = drawdown;
            }
        }
    }

    double get_total_equity() const {
        return cash_balance_ + (position_units_ * current_asset_price_);
    }

    double get_pnl() const {
        return get_total_equity() - initial_capital_;
    }

    double get_max_drawdown() const {
        return max_drawdown_ * 100.0; // Percentage
    }

    double get_win_rate() const {
        if (total_trades_ == 0) return 0.0;
        return (static_cast<double>(winning_trades_) / total_trades_) * 100.0;
    }

    double calculate_sharpe_ratio() const {
        if (equity_curve_.size() < 2) return 0.0;

        std::vector<double> returns;
        returns.reserve(equity_curve_.size() - 1);

        for (size_t i = 1; i < equity_curve_.size(); ++i) {
            double ret = (equity_curve_[i] - equity_curve_[i - 1]) / equity_curve_[i - 1];
            returns.push_back(ret);
        }

        double sum = std::accumulate(returns.begin(), returns.end(), 0.0);
        double mean = sum / returns.size();

        double sq_sum = 0.0;
        for (double r : returns) {
            sq_sum += (r - mean) * (r - mean);
        }
        double stdev = std::sqrt(sq_sum / returns.size());

        if (stdev == 0.0) return 0.0;
        return (mean / stdev) * std::sqrt(252.0 * 390.0);
    }

private:
    void log_trade(int tick_id, const std::string& action_str, double units) {
        if (log_file_.is_open()) {
            log_file_ << tick_id << ","
                      << action_str << ","
                      << current_asset_price_ << ","
                      << units << ","
                      << cash_balance_ << ","
                      << position_units_ << ","
                      << get_total_equity() << "\n";
        }
    }
};

#endif // PORTFOLIO_HPP