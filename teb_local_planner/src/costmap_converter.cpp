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

#include <teb_local_planner/costmap_converter.h>
#include <limits>

namespace teb_local_planner
{

namespace
{

/**
 * @brief Douglas-Peucker Algorithm for simplifying polygon
 */
Point2dContainer douglasPeucker(Point2dContainer::iterator begin,
                                Point2dContainer::iterator end,
                                double epsilon)
{
    if (std::distance(begin, end) <= 2)
    {
        return Point2dContainer(begin, end);
    }

    // Find the point with the maximum distance from the line [begin, end)
    double dmax = std::numeric_limits<double>::lowest();
    Point2dContainer::iterator max_dist_it;
    Point2dContainer::iterator last = std::prev(end);

    for (auto it = std::next(begin); it != last; ++it)
    {
        double d = CostmapConverter::computeSquaredDistanceToLineSegment(*it, *begin, *last);
        if (d > dmax)
        {
            max_dist_it = it;
            dmax = d;
        }
    }

    if (dmax < epsilon * epsilon)
    {
        // termination criterion reached, line is good enough
        Point2dContainer result;
        result.push_back(*begin);
        result.push_back(*last);
        return result;
    }

    // Recursive calls for the two split parts
    auto firstLineSimplified = douglasPeucker(begin, std::next(max_dist_it), epsilon);
    auto secondLineSimplified = douglasPeucker(max_dist_it, end, epsilon);

    // Combine the two lines into one line and return the merged line.
    firstLineSimplified.insert(firstLineSimplified.end(),
                               std::make_move_iterator(std::next(secondLineSimplified.begin())),
                               std::make_move_iterator(secondLineSimplified.end()));
    return firstLineSimplified;
}

bool isXCoordinateSmaller(const CostmapConverter::KeyPoint& p1,
                          const CostmapConverter::KeyPoint& p2)
{
    return p1.x < p2.x || (p1.x == p2.x && p1.y < p2.y);
}

} // end anonymous namespace


CostmapConverter::CostmapConverter(const CostmapConverterConfig& config)
    : config_(config)
{
}

void CostmapConverter::compute(const uint8_t* data, int width, int height,
                               double resolution, double origin_x, double origin_y,
                               uint8_t threshold)
{
    // Clear previous data
    occupied_cells_.clear();
    polygons_.clear();
    noise_points_.clear();

    // Calculate costmap size in meters
    double size_x = width * resolution;
    double size_y = height * resolution;

    // Allocate neighbor lookup
    int cells_x = static_cast<int>(size_x / config_.max_distance) + 1;
    int cells_y = static_cast<int>(size_y / config_.max_distance) + 1;

    if (cells_x != neighbor_size_x_ || cells_y != neighbor_size_y_)
    {
        neighbor_size_x_ = cells_x;
        neighbor_size_y_ = cells_y;
        neighbor_lookup_.resize(neighbor_size_x_ * neighbor_size_y_);
    }

    offset_x_ = origin_x;
    offset_y_ = origin_y;

    for (auto& n : neighbor_lookup_)
        n.clear();

    // Extract occupied cells
    for (int j = 0; j < height; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            uint8_t value = data[j * width + i];
            if (value >= threshold)
            {
                double x = origin_x + (i + 0.5) * resolution;
                double y = origin_y + (j + 0.5) * resolution;
                addPoint(x, y);
            }
        }
    }

    // Run DBSCAN clustering
    std::vector<std::vector<KeyPoint>> clusters;
    dbScan(clusters);

    // Convert clusters to polygons (skip first cluster which is noise)
    for (size_t i = 1; i < clusters.size(); ++i)
    {
        Point2dContainer polygon;
        convexHull(clusters[i], polygon);
        if (!polygon.empty())
        {
            polygons_.push_back(std::move(polygon));
        }
    }

    // Store noise points
    if (!clusters.empty())
    {
        for (const auto& kp : clusters[0])
        {
            noise_points_.emplace_back(kp.x, kp.y);
        }
    }
}

void CostmapConverter::addToObstacles(ObstContainer& obstacles)
{
    // Add polygons
    for (const auto& polygon : polygons_)
    {
        if (polygon.size() >= 2)
        {
            obstacles.push_back(ObstaclePtr(new PolygonObstacle(polygon)));
        }
    }

    // Add noise points as point obstacles
    for (const auto& pt : noise_points_)
    {
        obstacles.push_back(ObstaclePtr(new PointObstacle(pt)));
    }
}

void CostmapConverter::addPoint(double x, double y)
{
    int idx = static_cast<int>(occupied_cells_.size());
    occupied_cells_.emplace_back(x, y);

    int cx, cy;
    pointToNeighborCells(occupied_cells_.back(), cx, cy);
    int nidx = neighborCellsToIndex(cx, cy);
    if (nidx >= 0)
        neighbor_lookup_[nidx].push_back(idx);
}

void CostmapConverter::dbScan(std::vector<std::vector<KeyPoint>>& clusters)
{
    std::vector<bool> visited(occupied_cells_.size(), false);
    clusters.clear();

    // DB Scan Algorithm
    int cluster_id = 0;
    clusters.push_back(std::vector<KeyPoint>()); // clusters[0] for noise

    for (int i = 0; i < static_cast<int>(occupied_cells_.size()); ++i)
    {
        if (!visited[i])
        {
            visited[i] = true;
            std::vector<int> neighbors;
            regionQuery(i, neighbors);

            if (static_cast<int>(neighbors.size()) < config_.min_pts)
            {
                // Mark as noise
                clusters[0].push_back(occupied_cells_[i]);
            }
            else
            {
                // Start new cluster
                ++cluster_id;
                clusters.push_back(std::vector<KeyPoint>());

                // Expand the cluster
                clusters[cluster_id].push_back(occupied_cells_[i]);

                for (int j = 0; j < static_cast<int>(neighbors.size()); ++j)
                {
                    if (static_cast<int>(clusters[cluster_id].size()) == config_.max_pts)
                        break;

                    if (!visited[neighbors[j]])
                    {
                        visited[neighbors[j]] = true;
                        std::vector<int> further_neighbors;
                        regionQuery(neighbors[j], further_neighbors);

                        if (static_cast<int>(further_neighbors.size()) >= config_.min_pts)
                        {
                            neighbors.insert(neighbors.end(),
                                           further_neighbors.begin(),
                                           further_neighbors.end());
                            clusters[cluster_id].push_back(occupied_cells_[neighbors[j]]);
                        }
                    }
                }
            }
        }
    }
}

void CostmapConverter::regionQuery(int curr_index, std::vector<int>& neighbors)
{
    neighbors.clear();

    double dist_sqr_threshold = config_.max_distance * config_.max_distance;
    const KeyPoint& kp = occupied_cells_[curr_index];
    int cx, cy;
    pointToNeighborCells(kp, cx, cy);

    // Loop over neighboring cells
    const int offsets[9][2] = {{-1, -1}, {0, -1}, {1, -1},
                               {-1,  0}, {0,  0}, {1,  0},
                               {-1,  1}, {0,  1}, {1,  1}};

    for (int i = 0; i < 9; ++i)
    {
        int idx = neighborCellsToIndex(cx + offsets[i][0], cy + offsets[i][1]);
        if (idx < 0 || idx >= static_cast<int>(neighbor_lookup_.size()))
            continue;

        const std::vector<int>& pointIndicesToCheck = neighbor_lookup_[idx];
        for (int point_idx : pointIndicesToCheck)
        {
            if (point_idx == curr_index)
                continue;

            const KeyPoint& other = occupied_cells_[point_idx];
            double dx = other.x - kp.x;
            double dy = other.y - kp.y;
            double dist_sqr = dx * dx + dy * dy;

            if (dist_sqr <= dist_sqr_threshold)
                neighbors.push_back(point_idx);
        }
    }
}

void CostmapConverter::convexHull(std::vector<KeyPoint>& cluster, Point2dContainer& polygon)
{
    if (cluster.empty())
        return;

    if (cluster.size() == 1)
    {
        polygon.emplace_back(cluster[0].x, cluster[0].y);
        return;
    }

    if (cluster.size() == 2)
    {
        polygon.emplace_back(cluster[0].x, cluster[0].y);
        polygon.emplace_back(cluster[1].x, cluster[1].y);
        return;
    }

    std::vector<KeyPoint>& P = cluster;

    // Sort by x and y
    std::sort(P.begin(), P.end(), isXCoordinateSmaller);

    int i;
    int minmin = 0, minmax;
    double xmin = P[0].x;

    for (i = 1; i < static_cast<int>(P.size()); ++i)
        if (P[i].x != xmin) break;
    minmax = i - 1;

    if (minmax == static_cast<int>(P.size()) - 1)
    {
        // Degenerate case: all x-coords == xmin
        polygon.emplace_back(P[minmin].x, P[minmin].y);
        if (P[minmax].y != P[minmin].y)
        {
            polygon.emplace_back(P[minmax].x, P[minmax].y);
        }
        polygon.emplace_back(P[minmin].x, P[minmin].y);
        simplifyPolygon(polygon);
        return;
    }

    // Get indices of points with max x-coord
    int maxmin, maxmax = static_cast<int>(P.size()) - 1;
    double xmax = P.back().x;
    for (i = static_cast<int>(P.size()) - 2; i >= 0; --i)
        if (P[i].x != xmax) break;
    maxmin = i + 1;

    // Compute lower hull
    polygon.emplace_back(P[minmin].x, P[minmin].y);
    i = minmax;
    while (++i <= maxmin)
    {
        if (cross(P[minmin], P[maxmin], P[i]) >= 0 && i < maxmin)
            continue;

        while (polygon.size() > 1)
        {
            if (crossEigen(polygon[polygon.size() - 2], polygon.back(),
                          Eigen::Vector2d(P[i].x, P[i].y)) > 0)
                break;
            polygon.pop_back();
        }
        polygon.emplace_back(P[i].x, P[i].y);
    }

    // Compute upper hull
    if (maxmax != maxmin)
    {
        polygon.emplace_back(P[maxmax].x, P[maxmax].y);
    }
    int bot = static_cast<int>(polygon.size());
    i = maxmin;
    while (--i >= minmax)
    {
        if (cross(P[maxmax], P[minmax], P[i]) >= 0 && i > minmax)
            continue;

        while (static_cast<int>(polygon.size()) > bot)
        {
            if (crossEigen(polygon[polygon.size() - 2], polygon.back(),
                          Eigen::Vector2d(P[i].x, P[i].y)) > 0)
                break;
            polygon.pop_back();
        }
        polygon.emplace_back(P[i].x, P[i].y);
    }

    if (minmax != minmin)
    {
        polygon.emplace_back(P[minmin].x, P[minmin].y);
    }

    simplifyPolygon(polygon);
}

void CostmapConverter::simplifyPolygon(Point2dContainer& polygon)
{
    size_t triangleThreshold = 3;

    // Check if first and last point are the same
    if (polygon.size() > 1 &&
        std::abs(polygon.front().x() - polygon.back().x()) < 1e-5 &&
        std::abs(polygon.front().y() - polygon.back().y()) < 1e-5)
    {
        triangleThreshold = 4;
    }

    if (polygon.size() <= triangleThreshold)
        return;

    polygon = douglasPeucker(polygon.begin(), polygon.end(), config_.min_keypoint_separation);
}

double CostmapConverter::computeSquaredDistanceToLineSegment(
    const Eigen::Vector2d& point,
    const Eigen::Vector2d& line_start,
    const Eigen::Vector2d& line_end)
{
    double dx = line_end.x() - line_start.x();
    double dy = line_end.y() - line_start.y();
    double length_sqr = dx * dx + dy * dy;

    double u = 0;
    if (length_sqr > 0)
    {
        u = ((point.x() - line_start.x()) * dx +
             (point.y() - line_start.y()) * dy) / length_sqr;
    }

    if (u <= 0)
    {
        return std::pow(point.x() - line_start.x(), 2) +
               std::pow(point.y() - line_start.y(), 2);
    }

    if (u >= 1)
    {
        return std::pow(point.x() - line_end.x(), 2) +
               std::pow(point.y() - line_end.y(), 2);
    }

    return std::pow(point.x() - (line_start.x() + u * dx), 2) +
           std::pow(point.y() - (line_start.y() + u * dy), 2);
}

} // namespace teb_local_planner
