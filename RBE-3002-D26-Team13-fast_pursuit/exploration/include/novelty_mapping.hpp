#pragma once

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/map_meta_data.hpp>
#include <rclcpp/callback_group.hpp>

#include <vector>
#include <tuple>
#include <cmath>

class NoveltyMapping : public rclcpp::Node
{
public:
    NoveltyMapping();

private:
    // Callbacks
    void mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr map);

    // Core logic
    std::vector<std::vector<int>>   findFrontier(const std::vector<std::vector<int>>& map);
    std::vector<std::vector<int8_t>> noveltyMapGenerator();
    std::vector<std::vector<int8_t>> visibleCells(int ox, int oy);

    // Helpers
    std::vector<std::pair<int,int>> neighborsOf8(
        int row, int col,
        const std::vector<std::vector<int>>& map);

    // ROS interfaces
    rclcpp::CallbackGroup::SharedPtr cb_group_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr subscriber_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr novelty_map_publisher_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr frontier_map_publisher_;

    // State
    std::vector<std::vector<int>>    map_;
    nav_msgs::msg::MapMetaData       map_info_;

    std::vector<std::vector<int>>    frontier_map_;
    std::vector<std::vector<int8_t>> novelty_map_;

    // Ray-cast constants
    static constexpr int    NUM_RAY_CASTS = 36;
    static constexpr double MAX_RANGE     = 2.5;

    std::vector<double> cos_dirs_;
    std::vector<double> sin_dirs_;
};
