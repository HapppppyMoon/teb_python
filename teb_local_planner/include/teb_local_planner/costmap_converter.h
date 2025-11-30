/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2016,
 *  TU Dortmund - Institute of Control Theory and Systems Engineering.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the institute nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Christoph Rösmann, Otniel Rinaldo
 * Modified: Standalone version without ROS dependencies
 *********************************************************************/

#ifndef TEB_LOCAL_PLANNER_COSTMAP_CONVERTER_H_
#define TEB_LOCAL_PLANNER_COSTMAP_CONVERTER_H_

#include <teb_local_planner/obstacles.h>
#include <teb_local_planner/distance_calculations.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/StdVector>

namespace teb_local_planner
{

/**
 * @struct CostmapConverterConfig
 * @brief Configuration parameters for CostmapConverter
 */
struct CostmapConverterConfig
{
    double max_distance = 0.4;          //!< DBSCAN: maximum distance to neighbors [m]
    int min_pts = 2;                    //!< DBSCAN: minimum number of points that define a cluster
    int max_pts = 30;                   //!< DBSCAN: maximum number of points per cluster
    double min_keypoint_separation = 0.1; //!< Douglas-Peucker: simplification threshold [m]
};

/**
 * @class CostmapConverter
 * @brief Converts a costmap (2D occupancy grid) into polygon obstacles
 *
 * The conversion is performed in two stages:
 * 1. Clusters in the costmap are collected using the DBSCAN Algorithm
 * 2. Clusters are converted into convex polygons using the monotone chain algorithm
 *
 * Based on costmap_converter ROS package by TU Dortmund.
 */
class CostmapConverter
{
public:
    /**
     * @brief Constructor
     * @param config Configuration parameters
     */
    CostmapConverter(const CostmapConverterConfig& config = CostmapConverterConfig());

    /**
     * @brief Destructor
     */
    ~CostmapConverter() = default;

    /**
     * @brief Set configuration parameters
     * @param config Configuration parameters
     */
    void setConfig(const CostmapConverterConfig& config) { config_ = config; }

    /**
     * @brief Get current configuration
     * @return Current configuration parameters
     */
    const CostmapConverterConfig& getConfig() const { return config_; }

    /**
     * @brief Process costmap and extract polygon obstacles
     * @param data Row-major 2D array (data[y*width + x])
     * @param width Costmap width [cells]
     * @param height Costmap height [cells]
     * @param resolution Cell size [m/cell]
     * @param origin_x Costmap origin x [m]
     * @param origin_y Costmap origin y [m]
     * @param threshold Obstacle threshold (default 200, lethal=254)
     */
    void compute(const uint8_t* data, int width, int height,
                 double resolution, double origin_x, double origin_y,
                 uint8_t threshold = 200);

    /**
     * @brief Add extracted obstacles to an ObstContainer
     * @param obstacles Target obstacle container
     */
    void addToObstacles(ObstContainer& obstacles);

    /**
     * @brief Get number of extracted polygons
     */
    size_t numPolygons() const { return polygons_.size(); }

    /**
     * @brief Get number of noise points (isolated points)
     */
    size_t numNoisePoints() const { return noise_points_.size(); }

    /**
     * @brief Get extracted polygons
     */
    const std::vector<Point2dContainer>& getPolygons() const { return polygons_; }

    /**
     * @brief Get noise points
     */
    const std::vector<Eigen::Vector2d>& getNoisePoints() const { return noise_points_; }

    /**
     * @struct KeyPoint
     * @brief Internal 2D point structure (public for helper functions)
     */
    struct KeyPoint
    {
        double x = 0;
        double y = 0;
        KeyPoint() = default;
        KeyPoint(double x_, double y_) : x(x_), y(y_) {}
    };

    /**
     * @brief Compute squared distance from point to line segment (public for Douglas-Peucker)
     */
    static double computeSquaredDistanceToLineSegment(
        const Eigen::Vector2d& point,
        const Eigen::Vector2d& line_start,
        const Eigen::Vector2d& line_end);

protected:
    /**
     * @brief DBSCAN clustering algorithm
     * @param[out] clusters Output clusters (clusters[0] contains noise points)
     */
    void dbScan(std::vector<std::vector<KeyPoint>>& clusters);

    /**
     * @brief Find neighbors within max_distance
     * @param curr_index Index of current point
     * @param[out] neighbor_indices Indices of neighboring points
     */
    void regionQuery(int curr_index, std::vector<int>& neighbor_indices);

    /**
     * @brief Add a point to the lookup data structures
     */
    void addPoint(double x, double y);

    /**
     * @brief Compute convex hull using monotone chain algorithm
     * @param cluster Input cluster points
     * @param[out] polygon Output polygon vertices
     */
    void convexHull(std::vector<KeyPoint>& cluster, Point2dContainer& polygon);

    /**
     * @brief Simplify polygon using Douglas-Peucker algorithm
     * @param polygon Polygon to simplify (modified in place)
     */
    void simplifyPolygon(Point2dContainer& polygon);

    /**
     * @brief 2D cross product
     */
    template <typename P1, typename P2, typename P3>
    double cross(const P1& O, const P2& A, const P3& B)
    {
        return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
    }

    /**
     * @brief Cross product for Eigen vectors
     */
    double crossEigen(const Eigen::Vector2d& O, const Eigen::Vector2d& A, const Eigen::Vector2d& B)
    {
        return (A.x() - O.x()) * (B.y() - O.y()) - (A.y() - O.y()) * (B.x() - O.x());
    }

    /**
     * @brief Convert 2D cell coordinate to 1D index
     */
    int neighborCellsToIndex(int cx, int cy)
    {
        if (cx < 0 || cx >= neighbor_size_x_ || cy < 0 || cy >= neighbor_size_y_)
            return -1;
        return cy * neighbor_size_x_ + cx;
    }

    /**
     * @brief Compute cell indices for a keypoint
     */
    void pointToNeighborCells(const KeyPoint& kp, int& cx, int& cy)
    {
        cx = static_cast<int>((kp.x - offset_x_) / config_.max_distance);
        cy = static_cast<int>((kp.y - offset_y_) / config_.max_distance);
    }

private:
    CostmapConverterConfig config_;

    // Occupied cells from costmap
    std::vector<KeyPoint> occupied_cells_;

    // Neighbor lookup for efficient DBSCAN
    std::vector<std::vector<int>> neighbor_lookup_;
    int neighbor_size_x_ = 0;
    int neighbor_size_y_ = 0;
    double offset_x_ = 0;
    double offset_y_ = 0;

    // Output
    std::vector<Point2dContainer> polygons_;
    std::vector<Eigen::Vector2d> noise_points_;
};

} // namespace teb_local_planner

#endif // TEB_LOCAL_PLANNER_COSTMAP_CONVERTER_H_
