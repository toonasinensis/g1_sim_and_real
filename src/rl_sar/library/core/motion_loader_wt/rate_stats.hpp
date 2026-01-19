#pragma once

#include <deque>
#include <mutex>
#include <chrono>
#include <string>
#include <iostream>
#include <cmath>
#include <limits>

class RateStats
{
public:
    struct Stats
    {
        double dt_mean;
        double dt_std;
        double dt_min;
        double dt_max;

        double fps_mean;
        double fps_std;
        double fps_min;
        double fps_max;

        size_t samples;
    };

    RateStats(
        const std::string& name = "rate",
        size_t window = 300,
        double print_interval = 1.0,
        bool auto_print = true)
        : name_(name),
          window_(window),
          print_interval_(print_interval),
          auto_print_(auto_print),
          last_ts_(-1.0),
          last_print_time_(now_sec())
    {}

    /// Register a new timestamp (seconds)
    void tick(double ts)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);

            if (last_ts_ >= 0.0)
            {
                double dt = ts - last_ts_;
                if (dt > 0.0)
                {
                    if (dt_buf_.size() >= window_)
                        dt_buf_.pop_front();
                    dt_buf_.push_back(dt);
                }
            }
            last_ts_ = ts;
        }

        if (auto_print_)
            maybe_print();
    }

    bool get_stats(Stats& out) const
    {
        std::lock_guard<std::mutex> lock(mutex_);

        if (dt_buf_.size() < 5)
            return false;

        double sum = 0.0;
        double sum_sq = 0.0;
        double dt_min = std::numeric_limits<double>::max();
        double dt_max = 0.0;

        double fps_sum = 0.0;
        double fps_sum_sq = 0.0;
        double fps_min = std::numeric_limits<double>::max();
        double fps_max = 0.0;

        for (double dt : dt_buf_)
        {
            sum += dt;
            sum_sq += dt * dt;
            dt_min = std::min(dt_min, dt);
            dt_max = std::max(dt_max, dt);

            double fps = 1.0 / dt;
            fps_sum += fps;
            fps_sum_sq += fps * fps;
            fps_min = std::min(fps_min, fps);
            fps_max = std::max(fps_max, fps);
        }

        size_t n = dt_buf_.size();

        double dt_mean = sum / n;
        double dt_var = sum_sq / n - dt_mean * dt_mean;

        double fps_mean = fps_sum / n;
        double fps_var = fps_sum_sq / n - fps_mean * fps_mean;

        out = {
            dt_mean,
            std::sqrt(std::max(0.0, dt_var)),
            dt_min,
            dt_max,
            fps_mean,
            std::sqrt(std::max(0.0, fps_var)),
            fps_min,
            fps_max,
            n
        };

        return true;
    }

    void reset()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        dt_buf_.clear();
        last_ts_ = -1.0;
        last_print_time_ = now_sec();
    }

private:
    static double now_sec()
    {
        using clock = std::chrono::steady_clock;
        return std::chrono::duration<double>(
            clock::now().time_since_epoch()).count();
    }

    void maybe_print()
    {
        double now = now_sec();
        if (now - last_print_time_ < print_interval_)
            return;

        Stats stats;
        if (get_stats(stats))
        {
            std::cout
                << "[" << name_ << "] fps stats: "
                << "mean=" << stats.fps_mean
                << ", std=" << stats.fps_std
                << ", min=" << stats.fps_min
                << ", max=" << stats.fps_max
                << ", n=" << stats.samples
                << std::endl;
        }

        last_print_time_ = now;
    }

private:
    std::string name_;
    size_t window_;
    double print_interval_;
    bool auto_print_;

    mutable std::mutex mutex_;
    std::deque<double> dt_buf_;

    double last_ts_;
    double last_print_time_;
};