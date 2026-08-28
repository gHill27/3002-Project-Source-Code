#include "novelty_mapping.hpp"

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
NoveltyMapping::NoveltyMapping()
: Node("Novelty_Node")
{
    cb_group_ = create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive);

    rclcpp::SubscriptionOptions sub_opts;
    sub_opts.callback_group = cb_group_;

    subscriber_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        "/map", 10,
        std::bind(&NoveltyMapping::mapCallback, this, std::placeholders::_1),
        sub_opts);

    novelty_map_publisher_  = create_publisher<nav_msgs::msg::OccupancyGrid>("/novelty",   10);
    frontier_map_publisher_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/frontier",  10);

    // Pre-compute ray directions
    cos_dirs_.resize(NUM_RAY_CASTS);
    sin_dirs_.resize(NUM_RAY_CASTS);
    for (int i = 0; i < NUM_RAY_CASTS; ++i) {
        double angle = 2.0 * M_PI * i / NUM_RAY_CASTS;
        cos_dirs_[i] = std::cos(angle);
        sin_dirs_[i] = std::sin(angle);
    }
}

// ---------------------------------------------------------------------------
// /map callback
// ---------------------------------------------------------------------------
void NoveltyMapping::mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr map)
{
    const int height = static_cast<int>(map->info.height);
    const int width  = static_cast<int>(map->info.width);

    // Reshape flat data → 2-D row-major array (row = y, col = x — same as Python)
    map_.assign(height, std::vector<int>(width));
    for (int r = 0; r < height; ++r)
        for (int c = 0; c < width; ++c)
            map_[r][c] = static_cast<int>(map->data[r * width + c]);

    map_info_ = map->info;

    // --- Frontier ---
    frontier_map_ = findFrontier(map_);

    {
        nav_msgs::msg::OccupancyGrid frontier_grid;
        frontier_grid.header = map->header;
        frontier_grid.info   = map->info;
        frontier_grid.data.resize(height * width);
        for (int r = 0; r < height; ++r)
            for (int c = 0; c < width; ++c)
                frontier_grid.data[r * width + c] =
                    static_cast<int8_t>(std::clamp(frontier_map_[r][c], 0, 100));
        frontier_map_publisher_->publish(frontier_grid);
    }

    // --- Novelty ---
    novelty_map_ = noveltyMapGenerator();

    {
        nav_msgs::msg::OccupancyGrid novelty_grid;
        novelty_grid.header = map->header;
        novelty_grid.info   = map->info;
        novelty_grid.data.resize(height * width);
        for (int r = 0; r < height; ++r)
            for (int c = 0; c < width; ++c)
                novelty_grid.data[r * width + c] =
                    static_cast<int8_t>(std::clamp(static_cast<int>(novelty_map_[r][c]), 0, 100));
        novelty_map_publisher_->publish(novelty_grid);
    }
}

// ---------------------------------------------------------------------------
// neighborsOf8
// ---------------------------------------------------------------------------
std::vector<std::pair<int,int>> NoveltyMapping::neighborsOf8(
    int row, int col,
    const std::vector<std::vector<int>>& map)
{
    const int rows = static_cast<int>(map_info_.height);
    const int cols = static_cast<int>(map_info_.width);

    const int dr[] = { 1, -1,  0,  0,  1, -1, -1,  1};
    const int dc[] = { 0,  0,  1, -1,  1, -1,  1, -1};

    std::vector<std::pair<int,int>> safe;
    for (int i = 0; i < 8; ++i) {
        int nr = row + dr[i];
        int nc = col + dc[i];
        if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && map[nr][nc] == 0)
            safe.emplace_back(nr, nc);
    }
    return safe;
}

// ---------------------------------------------------------------------------
// findFrontier
// ---------------------------------------------------------------------------
std::vector<std::vector<int>> NoveltyMapping::findFrontier(
    const std::vector<std::vector<int>>& map)
{
    const int rows = static_cast<int>(map_info_.height);
    const int cols = static_cast<int>(map_info_.width);

    std::vector<std::vector<int>> frontier(rows, std::vector<int>(cols, 0));

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (map[r][c] == -1) {
                if (!neighborsOf8(r, c, map).empty())
                    frontier[r][c] = 100;
            }
        }
    }
    return frontier;
}

// ---------------------------------------------------------------------------
// noveltyMapGenerator
// ---------------------------------------------------------------------------
std::vector<std::vector<int8_t>> NoveltyMapping::noveltyMapGenerator()
{
    const int rows = static_cast<int>(map_info_.height);
    const int cols = static_cast<int>(map_info_.width);

    // Accumulate as int to avoid overflow before final clamp
    std::vector<std::vector<int>> accum(rows, std::vector<int>(cols, 0));

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (frontier_map_[r][c] > 0) {
                auto visible = visibleCells(r, c);
                for (int vr = 0; vr < rows; ++vr)
                    for (int vc = 0; vc < cols; ++vc)
                        accum[vr][vc] += static_cast<int>(visible[vr][vc]);
            }
        }
    }

    std::vector<std::vector<int8_t>> novelty(rows, std::vector<int8_t>(cols, 0));
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            novelty[r][c] = static_cast<int8_t>(std::clamp(accum[r][c], 0, 127));

    return novelty;
}

// ---------------------------------------------------------------------------
// visibleCells  (Bresenham-style ray casting)
// ---------------------------------------------------------------------------
std::vector<std::vector<int8_t>> NoveltyMapping::visibleCells(int ox, int oy)
{
    constexpr int EMPTY    =  0;
    constexpr int UNKNOWN  = 50;   // -1 raw → treated as unknown inside ray cast
    constexpr int OBSTACLE = 100;

    const int rows = static_cast<int>(map_info_.height);
    const int cols = static_cast<int>(map_info_.width);

    const double resolution    = map_info_.resolution;
    const int    max_range_cells = static_cast<int>(MAX_RANGE / resolution);

    if (ox < 0 || ox >= rows || oy < 0 || oy >= cols)
        throw std::out_of_range("Origin outside grid bounds");
    if (map_[ox][oy] == OBSTACLE)
        throw std::invalid_argument("Origin is inside an obstacle");

    std::vector<std::vector<int8_t>> visible(rows, std::vector<int8_t>(cols, 0));

    for (int i = 0; i < NUM_RAY_CASTS; ++i) {
        const double dx = cos_dirs_[i];
        const double dy = sin_dirs_[i];

        for (int step = 1; step <= max_range_cells; ++step) {
            int rx = static_cast<int>(std::round(ox + step * dx));
            int ry = static_cast<int>(std::round(oy + step * dy));

            if (rx < 0 || rx >= rows || ry < 0 || ry >= cols)
                break;  // Out of bounds

            int cell = map_[rx][ry];

            if (cell == OBSTACLE) {
                break;  // Blocked — stop ray, obstacle not counted
            } else if (cell == -1 && frontier_map_[rx][ry] < 50) {
                break;  // Unknown cell not on frontier — can't see past it
            } else if (cell == EMPTY) {
                visible[rx][ry] = 1;
            }
            // If it's a frontier cell (unknown but bordering free space), the
            // ray is NOT broken — matching the Python `elif cell == UNKNOWN and
            // frontier_map[rx,ry] < 50: break` logic (frontier cells pass through).
        }
    }

    return visible;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<NoveltyMapping>();

    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);

    try {
        executor.spin();
    } catch (const std::exception & e) {
        RCLCPP_ERROR(node->get_logger(), "Exception: %s", e.what());
    }

    rclcpp::shutdown();
    return 0;
}
