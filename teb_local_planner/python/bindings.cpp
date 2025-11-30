/*********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 *  TEB Local Planner - Python Bindings
 *
 *********************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include "teb_local_planner/teb_config.h"
#include "teb_local_planner/pose_se2.h"
#include "teb_local_planner/planner_interface.h"
#include "teb_local_planner/obstacles.h"
#include "teb_local_planner/robot_footprint_model.h"
#include "teb_local_planner/timed_elastic_band.h"
#include "teb_local_planner/optimal_planner.h"
#include "teb_local_planner/homotopy_class_planner.h"
#include "teb_local_planner/costmap_converter.h"

#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace teb_local_planner;

// Wrapper class for ObstContainer to provide Python-friendly interface
class ObstacleContainer {
public:
    ObstacleContainer() = default;

    void add(const ObstaclePtr& obs) {
        obstacles_.push_back(obs);
    }

    void clear() {
        obstacles_.clear();
    }

    size_t size() const {
        return obstacles_.size();
    }

    ObstaclePtr& operator[](size_t idx) {
        return obstacles_[idx];
    }

    const ObstaclePtr& operator[](size_t idx) const {
        return obstacles_[idx];
    }

    ObstContainer* ptr() {
        return &obstacles_;
    }

    const ObstContainer& get() const {
        return obstacles_;
    }

private:
    ObstContainer obstacles_;
};

// Wrapper class for ViaPointContainer
class ViaPointContainerWrapper {
public:
    ViaPointContainerWrapper() = default;

    void add(const Eigen::Vector2d& point) {
        via_points_.push_back(point);
    }

    void clear() {
        via_points_.clear();
    }

    size_t size() const {
        return via_points_.size();
    }

    const ViaPointContainer* ptr() const {
        return &via_points_;
    }

private:
    ViaPointContainer via_points_;
};

PYBIND11_MODULE(pyteb, m) {
    m.doc() = "TEB Local Planner - Python bindings for trajectory optimization";

    // ==================== Velocity2D ====================
    py::class_<Velocity2D>(m, "Velocity2D")
        .def(py::init<>())
        .def(py::init([](double vx, double vy, double omega) {
            Velocity2D v;
            v.vx = vx;
            v.vy = vy;
            v.omega = omega;
            return v;
        }), py::arg("vx") = 0.0, py::arg("vy") = 0.0, py::arg("omega") = 0.0)
        .def_readwrite("vx", &Velocity2D::vx)
        .def_readwrite("vy", &Velocity2D::vy)
        .def_readwrite("omega", &Velocity2D::omega)
        .def("__repr__", [](const Velocity2D& v) {
            return "Velocity2D(vx=" + std::to_string(v.vx) +
                   ", vy=" + std::to_string(v.vy) +
                   ", omega=" + std::to_string(v.omega) + ")";
        });

    // ==================== PoseSE2 ====================
    py::class_<PoseSE2>(m, "PoseSE2")
        .def(py::init<>())
        .def(py::init<double, double, double>(), py::arg("x"), py::arg("y"), py::arg("theta"))
        .def(py::init<const Eigen::Vector2d&, double>(), py::arg("position"), py::arg("theta"))
        .def_property("x",
            [](const PoseSE2& p) { return p.x(); },
            [](PoseSE2& p, double val) { p.x() = val; })
        .def_property("y",
            [](const PoseSE2& p) { return p.y(); },
            [](PoseSE2& p, double val) { p.y() = val; })
        .def_property("theta",
            [](const PoseSE2& p) { return p.theta(); },
            [](PoseSE2& p, double val) { p.theta() = val; })
        .def("position", [](const PoseSE2& p) { return p.position(); })
        .def("orientation_unit_vec", &PoseSE2::orientationUnitVec)
        .def("set_zero", &PoseSE2::setZero)
        .def("__repr__", [](const PoseSE2& p) {
            return "PoseSE2(x=" + std::to_string(p.x()) +
                   ", y=" + std::to_string(p.y()) +
                   ", theta=" + std::to_string(p.theta()) + ")";
        });

    // ==================== RotType ====================
    py::enum_<RotType>(m, "RotType")
        .value("left", RotType::left)
        .value("none", RotType::none)
        .value("right", RotType::right);

    // ==================== Obstacle Classes ====================
    py::class_<Obstacle, std::shared_ptr<Obstacle>>(m, "Obstacle")
        .def("get_centroid", &Obstacle::getCentroid)
        .def("is_dynamic", &Obstacle::isDynamic)
        .def("set_centroid_velocity", &Obstacle::setCentroidVelocity)
        .def("get_centroid_velocity", &Obstacle::getCentroidVelocity)
        .def("check_collision", &Obstacle::checkCollision)
        .def("get_minimum_distance",
            static_cast<double (Obstacle::*)(const Eigen::Vector2d&) const>(&Obstacle::getMinimumDistance));

    py::class_<PointObstacle, Obstacle, std::shared_ptr<PointObstacle>>(m, "PointObstacle")
        .def(py::init<>())
        .def(py::init<double, double>(), py::arg("x"), py::arg("y"))
        .def(py::init<const Eigen::Vector2d&>(), py::arg("position"))
        .def_property("x",
            [](const PointObstacle& o) { return o.x(); },
            [](PointObstacle& o, double val) { o.x() = val; })
        .def_property("y",
            [](const PointObstacle& o) { return o.y(); },
            [](PointObstacle& o, double val) { o.y() = val; })
        .def("position", [](const PointObstacle& o) { return o.position(); });

    py::class_<CircularObstacle, Obstacle, std::shared_ptr<CircularObstacle>>(m, "CircularObstacle")
        .def(py::init<>())
        .def(py::init<double, double, double>(), py::arg("x"), py::arg("y"), py::arg("radius"))
        .def(py::init<const Eigen::Vector2d&, double>(), py::arg("position"), py::arg("radius"))
        .def_property("x",
            [](const CircularObstacle& o) { return o.x(); },
            [](CircularObstacle& o, double val) { o.x() = val; })
        .def_property("y",
            [](const CircularObstacle& o) { return o.y(); },
            [](CircularObstacle& o, double val) { o.y() = val; })
        .def_property("radius",
            [](const CircularObstacle& o) { return o.radius(); },
            [](CircularObstacle& o, double val) { o.radius() = val; });

    py::class_<LineObstacle, Obstacle, std::shared_ptr<LineObstacle>>(m, "LineObstacle")
        .def(py::init<>())
        .def(py::init<double, double, double, double>(),
             py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"))
        .def(py::init<const Eigen::Vector2d&, const Eigen::Vector2d&>(),
             py::arg("line_start"), py::arg("line_end"))
        .def("start", &LineObstacle::start)
        .def("end", &LineObstacle::end)
        .def("set_start", &LineObstacle::setStart)
        .def("set_end", &LineObstacle::setEnd);

    py::class_<PolygonObstacle, Obstacle, std::shared_ptr<PolygonObstacle>>(m, "PolygonObstacle")
        .def(py::init<>())
        .def(py::init<const Point2dContainer&>(), py::arg("vertices"))
        .def("push_back_vertex",
            static_cast<void (PolygonObstacle::*)(double, double)>(&PolygonObstacle::pushBackVertex))
        .def("finalize_polygon", &PolygonObstacle::finalizePolygon)
        .def("clear_vertices", &PolygonObstacle::clearVertices)
        .def("no_vertices", &PolygonObstacle::noVertices)
        .def("vertices", static_cast<const Point2dContainer& (PolygonObstacle::*)() const>(&PolygonObstacle::vertices));

    // ==================== ObstacleContainer ====================
    py::class_<ObstacleContainer>(m, "ObstacleContainer")
        .def(py::init<>())
        .def("add", &ObstacleContainer::add)
        .def("clear", &ObstacleContainer::clear)
        .def("size", &ObstacleContainer::size)
        .def("__len__", &ObstacleContainer::size)
        .def("__getitem__", [](ObstacleContainer& c, size_t idx) -> ObstaclePtr& {
            if (idx >= c.size()) throw py::index_error();
            return c[idx];
        }, py::return_value_policy::reference_internal);

    // ==================== ViaPointContainerWrapper ====================
    py::class_<ViaPointContainerWrapper>(m, "ViaPointContainer")
        .def(py::init<>())
        .def("add", &ViaPointContainerWrapper::add)
        .def("clear", &ViaPointContainerWrapper::clear)
        .def("size", &ViaPointContainerWrapper::size)
        .def("__len__", &ViaPointContainerWrapper::size);

    // ==================== CostmapConverter ====================
    py::class_<CostmapConverterConfig>(m, "CostmapConverterConfig")
        .def(py::init<>())
        .def_readwrite("max_distance", &CostmapConverterConfig::max_distance,
            "DBSCAN: maximum distance to neighbors [m]")
        .def_readwrite("min_pts", &CostmapConverterConfig::min_pts,
            "DBSCAN: minimum number of points that define a cluster")
        .def_readwrite("max_pts", &CostmapConverterConfig::max_pts,
            "DBSCAN: maximum number of points per cluster")
        .def_readwrite("min_keypoint_separation", &CostmapConverterConfig::min_keypoint_separation,
            "Douglas-Peucker: simplification threshold [m]");

    py::class_<CostmapConverter>(m, "CostmapConverter")
        .def(py::init<const CostmapConverterConfig&>(),
             py::arg("config") = CostmapConverterConfig())
        .def("set_config", &CostmapConverter::setConfig)
        .def("get_config", &CostmapConverter::getConfig, py::return_value_policy::reference)
        .def("compute", [](CostmapConverter& self,
                           py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& costmap,
                           double resolution,
                           double origin_x, double origin_y,
                           uint8_t threshold) {
            py::buffer_info buf = costmap.request();
            if (buf.ndim != 2)
                throw std::runtime_error("costmap must be 2D array");
            self.compute(
                static_cast<uint8_t*>(buf.ptr),
                static_cast<int>(buf.shape[1]),  // width (cols)
                static_cast<int>(buf.shape[0]),  // height (rows)
                resolution, origin_x, origin_y, threshold
            );
        }, py::arg("costmap"), py::arg("resolution"),
           py::arg("origin_x"), py::arg("origin_y"),
           py::arg("threshold") = 200,
           "Process costmap and extract polygon obstacles")
        .def("add_to_obstacles", [](CostmapConverter& self, ObstacleContainer& obstacles) {
            self.addToObstacles(*obstacles.ptr());
        }, py::arg("obstacles"),
           "Add extracted obstacles to an ObstacleContainer")
        .def("num_polygons", &CostmapConverter::numPolygons,
            "Get number of extracted polygons")
        .def("num_noise_points", &CostmapConverter::numNoisePoints,
            "Get number of noise points (isolated points)")
        .def("get_polygons", &CostmapConverter::getPolygons,
            py::return_value_policy::reference_internal,
            "Get extracted polygons")
        .def("get_noise_points", &CostmapConverter::getNoisePoints,
            py::return_value_policy::reference_internal,
            "Get noise points");

    // ==================== Robot Footprint Models ====================
    py::class_<BaseRobotFootprintModel, std::shared_ptr<BaseRobotFootprintModel>>(m, "BaseRobotFootprintModel")
        .def("get_inscribed_radius", &BaseRobotFootprintModel::getInscribedRadius);

    py::class_<PointRobotFootprint, BaseRobotFootprintModel, std::shared_ptr<PointRobotFootprint>>(m, "PointRobotFootprint")
        .def(py::init<>());

    py::class_<CircularRobotFootprint, BaseRobotFootprintModel, std::shared_ptr<CircularRobotFootprint>>(m, "CircularRobotFootprint")
        .def(py::init<double>(), py::arg("radius"))
        .def("set_radius", &CircularRobotFootprint::setRadius);

    py::class_<TwoCirclesRobotFootprint, BaseRobotFootprintModel, std::shared_ptr<TwoCirclesRobotFootprint>>(m, "TwoCirclesRobotFootprint")
        .def(py::init<double, double, double, double>(),
             py::arg("front_offset"), py::arg("front_radius"),
             py::arg("rear_offset"), py::arg("rear_radius"))
        .def("set_parameters", &TwoCirclesRobotFootprint::setParameters);

    py::class_<PolygonRobotFootprint, BaseRobotFootprintModel, std::shared_ptr<PolygonRobotFootprint>>(m, "PolygonRobotFootprint")
        .def(py::init<const Point2dContainer&>(), py::arg("vertices"))
        .def("set_vertices", &PolygonRobotFootprint::setVertices);

    // ==================== TebConfig ====================
    py::class_<TebConfig::Trajectory>(m, "TrajectoryConfig")
        .def_readwrite("teb_autosize", &TebConfig::Trajectory::teb_autosize)
        .def_readwrite("dt_ref", &TebConfig::Trajectory::dt_ref)
        .def_readwrite("dt_hysteresis", &TebConfig::Trajectory::dt_hysteresis)
        .def_readwrite("min_samples", &TebConfig::Trajectory::min_samples)
        .def_readwrite("max_samples", &TebConfig::Trajectory::max_samples)
        .def_readwrite("global_plan_overwrite_orientation", &TebConfig::Trajectory::global_plan_overwrite_orientation)
        .def_readwrite("allow_init_with_backwards_motion", &TebConfig::Trajectory::allow_init_with_backwards_motion)
        .def_readwrite("global_plan_viapoint_sep", &TebConfig::Trajectory::global_plan_viapoint_sep)
        .def_readwrite("via_points_ordered", &TebConfig::Trajectory::via_points_ordered)
        .def_readwrite("max_global_plan_lookahead_dist", &TebConfig::Trajectory::max_global_plan_lookahead_dist)
        .def_readwrite("exact_arc_length", &TebConfig::Trajectory::exact_arc_length)
        .def_readwrite("feasibility_check_no_poses", &TebConfig::Trajectory::feasibility_check_no_poses)
        .def_readwrite("control_look_ahead_poses", &TebConfig::Trajectory::control_look_ahead_poses);

    py::class_<TebConfig::Robot>(m, "RobotConfig")
        .def_readwrite("max_vel_x", &TebConfig::Robot::max_vel_x)
        .def_readwrite("max_vel_x_backwards", &TebConfig::Robot::max_vel_x_backwards)
        .def_readwrite("max_vel_y", &TebConfig::Robot::max_vel_y)
        .def_readwrite("max_vel_theta", &TebConfig::Robot::max_vel_theta)
        .def_readwrite("acc_lim_x", &TebConfig::Robot::acc_lim_x)
        .def_readwrite("acc_lim_y", &TebConfig::Robot::acc_lim_y)
        .def_readwrite("acc_lim_theta", &TebConfig::Robot::acc_lim_theta)
        .def_readwrite("min_turning_radius", &TebConfig::Robot::min_turning_radius)
        .def_readwrite("wheelbase", &TebConfig::Robot::wheelbase)
        .def_readwrite("cmd_angle_instead_rotvel", &TebConfig::Robot::cmd_angle_instead_rotvel)
        .def_readwrite("is_footprint_dynamic", &TebConfig::Robot::is_footprint_dynamic);

    py::class_<TebConfig::GoalTolerance>(m, "GoalToleranceConfig")
        .def_readwrite("xy_goal_tolerance", &TebConfig::GoalTolerance::xy_goal_tolerance)
        .def_readwrite("free_goal_vel", &TebConfig::GoalTolerance::free_goal_vel);

    py::class_<TebConfig::Obstacles>(m, "ObstaclesConfig")
        .def_readwrite("min_obstacle_dist", &TebConfig::Obstacles::min_obstacle_dist)
        .def_readwrite("inflation_dist", &TebConfig::Obstacles::inflation_dist)
        .def_readwrite("dynamic_obstacle_inflation_dist", &TebConfig::Obstacles::dynamic_obstacle_inflation_dist)
        .def_readwrite("include_dynamic_obstacles", &TebConfig::Obstacles::include_dynamic_obstacles)
        .def_readwrite("include_costmap_obstacles", &TebConfig::Obstacles::include_costmap_obstacles)
        .def_readwrite("obstacle_poses_affected", &TebConfig::Obstacles::obstacle_poses_affected)
        .def_readwrite("legacy_obstacle_association", &TebConfig::Obstacles::legacy_obstacle_association)
        .def_readwrite("obstacle_association_force_inclusion_factor", &TebConfig::Obstacles::obstacle_association_force_inclusion_factor)
        .def_readwrite("obstacle_association_cutoff_factor", &TebConfig::Obstacles::obstacle_association_cutoff_factor);

    py::class_<TebConfig::Optimization>(m, "OptimizationConfig")
        .def_readwrite("no_inner_iterations", &TebConfig::Optimization::no_inner_iterations)
        .def_readwrite("no_outer_iterations", &TebConfig::Optimization::no_outer_iterations)
        .def_readwrite("optimization_activate", &TebConfig::Optimization::optimization_activate)
        .def_readwrite("optimization_verbose", &TebConfig::Optimization::optimization_verbose)
        .def_readwrite("penalty_epsilon", &TebConfig::Optimization::penalty_epsilon)
        .def_readwrite("weight_max_vel_x", &TebConfig::Optimization::weight_max_vel_x)
        .def_readwrite("weight_max_vel_y", &TebConfig::Optimization::weight_max_vel_y)
        .def_readwrite("weight_max_vel_theta", &TebConfig::Optimization::weight_max_vel_theta)
        .def_readwrite("weight_acc_lim_x", &TebConfig::Optimization::weight_acc_lim_x)
        .def_readwrite("weight_acc_lim_y", &TebConfig::Optimization::weight_acc_lim_y)
        .def_readwrite("weight_acc_lim_theta", &TebConfig::Optimization::weight_acc_lim_theta)
        .def_readwrite("weight_kinematics_nh", &TebConfig::Optimization::weight_kinematics_nh)
        .def_readwrite("weight_kinematics_forward_drive", &TebConfig::Optimization::weight_kinematics_forward_drive)
        .def_readwrite("weight_kinematics_turning_radius", &TebConfig::Optimization::weight_kinematics_turning_radius)
        .def_readwrite("weight_optimaltime", &TebConfig::Optimization::weight_optimaltime)
        .def_readwrite("weight_shortest_path", &TebConfig::Optimization::weight_shortest_path)
        .def_readwrite("weight_obstacle", &TebConfig::Optimization::weight_obstacle)
        .def_readwrite("weight_inflation", &TebConfig::Optimization::weight_inflation)
        .def_readwrite("weight_dynamic_obstacle", &TebConfig::Optimization::weight_dynamic_obstacle)
        .def_readwrite("weight_viapoint", &TebConfig::Optimization::weight_viapoint)
        .def_readwrite("weight_adapt_factor", &TebConfig::Optimization::weight_adapt_factor);

    py::class_<TebConfig::HomotopyClasses>(m, "HomotopyClassesConfig")
        .def_readwrite("enable_homotopy_class_planning", &TebConfig::HomotopyClasses::enable_homotopy_class_planning)
        .def_readwrite("enable_multithreading", &TebConfig::HomotopyClasses::enable_multithreading)
        .def_readwrite("simple_exploration", &TebConfig::HomotopyClasses::simple_exploration)
        .def_readwrite("max_number_classes", &TebConfig::HomotopyClasses::max_number_classes)
        .def_readwrite("selection_cost_hysteresis", &TebConfig::HomotopyClasses::selection_cost_hysteresis)
        .def_readwrite("selection_obst_cost_scale", &TebConfig::HomotopyClasses::selection_obst_cost_scale)
        .def_readwrite("roadmap_graph_no_samples", &TebConfig::HomotopyClasses::roadmap_graph_no_samples)
        .def_readwrite("roadmap_graph_area_width", &TebConfig::HomotopyClasses::roadmap_graph_area_width)
        .def_readwrite("h_signature_prescaler", &TebConfig::HomotopyClasses::h_signature_prescaler)
        .def_readwrite("h_signature_threshold", &TebConfig::HomotopyClasses::h_signature_threshold)
        .def_readwrite("visualize_hc_graph", &TebConfig::HomotopyClasses::visualize_hc_graph);

    py::class_<TebConfig::Recovery>(m, "RecoveryConfig")
        .def_readwrite("shrink_horizon_backup", &TebConfig::Recovery::shrink_horizon_backup)
        .def_readwrite("shrink_horizon_min_duration", &TebConfig::Recovery::shrink_horizon_min_duration)
        .def_readwrite("oscillation_recovery", &TebConfig::Recovery::oscillation_recovery)
        .def_readwrite("oscillation_v_eps", &TebConfig::Recovery::oscillation_v_eps)
        .def_readwrite("oscillation_omega_eps", &TebConfig::Recovery::oscillation_omega_eps)
        .def_readwrite("divergence_detection_enable", &TebConfig::Recovery::divergence_detection_enable);

    py::class_<TebConfig>(m, "TebConfig")
        .def(py::init<>())
        .def_readwrite("trajectory", &TebConfig::trajectory)
        .def_readwrite("robot", &TebConfig::robot)
        .def_readwrite("goal_tolerance", &TebConfig::goal_tolerance)
        .def_readwrite("obstacles", &TebConfig::obstacles)
        .def_readwrite("optim", &TebConfig::optim)
        .def_readwrite("hcp", &TebConfig::hcp)
        .def_readwrite("recovery", &TebConfig::recovery)
        .def_readwrite("robot_model", &TebConfig::robot_model)
        .def("check_parameters", &TebConfig::checkParameters);

    // ==================== TrajectoryPoint ====================
    py::class_<TrajectoryPoint>(m, "TrajectoryPoint")
        .def(py::init<>())
        .def_readwrite("pose", &TrajectoryPoint::pose)
        .def_readwrite("velocity", &TrajectoryPoint::velocity)
        .def_readwrite("acceleration", &TrajectoryPoint::acceleration)
        .def_readwrite("time_from_start", &TrajectoryPoint::time_from_start);

    // ==================== TimedElasticBand ====================
    py::class_<TimedElasticBand>(m, "TimedElasticBand")
        .def(py::init<>())
        .def("size_poses", &TimedElasticBand::sizePoses)
        .def("size_time_diffs", &TimedElasticBand::sizeTimeDiffs)
        .def("is_init", &TimedElasticBand::isInit)
        .def("pose", [](const TimedElasticBand& teb, int index) -> PoseSE2 {
            return teb.Pose(index);
        }, py::arg("index"))
        .def("time_diff", [](const TimedElasticBand& teb, int index) -> double {
            return teb.TimeDiff(index);
        }, py::arg("index"))
        .def("get_sum_of_all_time_diffs", &TimedElasticBand::getSumOfAllTimeDiffs)
        .def("get_accumulated_distance", &TimedElasticBand::getAccumulatedDistance)
        .def("add_pose", static_cast<void (TimedElasticBand::*)(double, double, double, bool)>(
            &TimedElasticBand::addPose),
            py::arg("x"), py::arg("y"), py::arg("theta"), py::arg("fixed") = false)
        .def("add_time_diff", &TimedElasticBand::addTimeDiff,
            py::arg("dt"), py::arg("fixed") = false)
        .def("clear", &TimedElasticBand::clearTimedElasticBand)
        .def("init_trajectory_to_goal",
            static_cast<bool (TimedElasticBand::*)(const PoseSE2&, const PoseSE2&, double, double, int, bool)>(
                &TimedElasticBand::initTrajectoryToGoal),
            py::arg("start"), py::arg("goal"), py::arg("diststep") = 0,
            py::arg("max_vel_x") = 0.5, py::arg("min_samples") = 3,
            py::arg("guess_backwards_motion") = false)
        .def("get_all_poses", [](const TimedElasticBand& teb) {
            std::vector<PoseSE2> poses;
            for (int i = 0; i < teb.sizePoses(); ++i) {
                poses.push_back(teb.Pose(i));
            }
            return poses;
        })
        .def("get_all_time_diffs", [](const TimedElasticBand& teb) {
            std::vector<double> dts;
            for (int i = 0; i < teb.sizeTimeDiffs(); ++i) {
                dts.push_back(teb.TimeDiff(i));
            }
            return dts;
        });

    // ==================== TebOptimalPlanner ====================
    py::class_<TebOptimalPlanner, std::shared_ptr<TebOptimalPlanner>>(m, "TebOptimalPlanner")
        .def(py::init<>())
        .def(py::init([](const TebConfig& cfg, ObstacleContainer* obstacles, ViaPointContainerWrapper* via_points) {
            return std::make_shared<TebOptimalPlanner>(
                cfg,
                obstacles ? obstacles->ptr() : nullptr,
                via_points ? via_points->ptr() : nullptr
            );
        }), py::arg("cfg"), py::arg("obstacles") = nullptr, py::arg("via_points") = nullptr,
           py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
        .def("initialize", [](TebOptimalPlanner& self, const TebConfig& cfg,
                             ObstacleContainer* obstacles, ViaPointContainerWrapper* via_points) {
            self.initialize(cfg, obstacles ? obstacles->ptr() : nullptr,
                           via_points ? via_points->ptr() : nullptr);
        }, py::arg("cfg"), py::arg("obstacles") = nullptr, py::arg("via_points") = nullptr,
           py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
        .def("set_obstacles", [](TebOptimalPlanner& self, ObstacleContainer* obstacles) {
            self.setObstVector(obstacles ? obstacles->ptr() : nullptr);
        }, py::arg("obstacles"), py::keep_alive<1, 2>())
        .def("plan", static_cast<bool (TebOptimalPlanner::*)(const PoseSE2&, const PoseSE2&, const Velocity2D*, bool)>(
            &TebOptimalPlanner::plan),
            py::arg("start"), py::arg("goal"), py::arg("start_vel") = nullptr, py::arg("free_goal_vel") = false)
        .def("plan", static_cast<bool (TebOptimalPlanner::*)(const std::vector<PoseSE2>&, const Velocity2D*, bool)>(
            &TebOptimalPlanner::plan),
            py::arg("initial_plan"), py::arg("start_vel") = nullptr, py::arg("free_goal_vel") = false)
        .def("get_velocity_command", [](const TebOptimalPlanner& planner, int look_ahead_poses) {
            double vx, vy, omega;
            bool success = planner.getVelocityCommand(vx, vy, omega, look_ahead_poses);
            return py::make_tuple(success, vx, vy, omega);
        }, py::arg("look_ahead_poses") = 1)
        .def("optimize_teb", &TebOptimalPlanner::optimizeTEB,
             py::arg("iterations_innerloop"), py::arg("iterations_outerloop"),
             py::arg("compute_cost_afterwards") = false,
             py::arg("obst_cost_scale") = 1.0, py::arg("viapoint_cost_scale") = 1.0,
             py::arg("alternative_time_cost") = false)
        .def("set_velocity_start", &TebOptimalPlanner::setVelocityStart)
        .def("set_velocity_goal", &TebOptimalPlanner::setVelocityGoal)
        .def("set_velocity_goal_free", &TebOptimalPlanner::setVelocityGoalFree)
        .def("teb", static_cast<TimedElasticBand& (TebOptimalPlanner::*)()>(&TebOptimalPlanner::teb),
             py::return_value_policy::reference_internal)
        .def("is_optimized", &TebOptimalPlanner::isOptimized)
        .def("has_diverged", &TebOptimalPlanner::hasDiverged)
        .def("get_current_cost", &TebOptimalPlanner::getCurrentCost)
        .def("clear_planner", &TebOptimalPlanner::clearPlanner)
        .def("set_preferred_turning_dir", &TebOptimalPlanner::setPreferredTurningDir)
        .def("get_velocity_profile", [](const TebOptimalPlanner& planner) {
            std::vector<Velocity2D> profile;
            planner.getVelocityProfile(profile);
            return profile;
        })
        .def("get_full_trajectory", [](const TebOptimalPlanner& planner) {
            std::vector<TrajectoryPoint> trajectory;
            planner.getFullTrajectory(trajectory);
            return trajectory;
        })
        .def("is_trajectory_feasible", [](TebOptimalPlanner& planner,
                                          const RobotFootprintModelPtr& robot_model,
                                          const std::vector<ObstaclePtr>& obstacles,
                                          int look_ahead_idx,
                                          double feasibility_check_lookahead_distance) {
            return planner.isTrajectoryFeasible(robot_model, obstacles, look_ahead_idx,
                                                feasibility_check_lookahead_distance);
        }, py::arg("robot_model"), py::arg("obstacles"),
           py::arg("look_ahead_idx") = -1,
           py::arg("feasibility_check_lookahead_distance") = -1.0);

    // ==================== HomotopyClassPlanner ====================
    py::class_<HomotopyClassPlanner, std::shared_ptr<HomotopyClassPlanner>>(m, "HomotopyClassPlanner")
        .def(py::init<>())
        .def(py::init([](const TebConfig& cfg, ObstacleContainer* obstacles, ViaPointContainerWrapper* via_points) {
            return std::make_shared<HomotopyClassPlanner>(
                cfg,
                obstacles ? obstacles->ptr() : nullptr,
                via_points ? via_points->ptr() : nullptr
            );
        }), py::arg("cfg"), py::arg("obstacles") = nullptr, py::arg("via_points") = nullptr,
           py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
        .def("initialize", [](HomotopyClassPlanner& self, const TebConfig& cfg,
                             ObstacleContainer* obstacles, ViaPointContainerWrapper* via_points) {
            self.initialize(cfg, obstacles ? obstacles->ptr() : nullptr,
                           via_points ? via_points->ptr() : nullptr);
        }, py::arg("cfg"), py::arg("obstacles") = nullptr, py::arg("via_points") = nullptr,
           py::keep_alive<1, 3>(), py::keep_alive<1, 4>())
        .def("plan", static_cast<bool (HomotopyClassPlanner::*)(const PoseSE2&, const PoseSE2&, const Velocity2D*, bool)>(
            &HomotopyClassPlanner::plan),
            py::arg("start"), py::arg("goal"), py::arg("start_vel") = nullptr, py::arg("free_goal_vel") = false)
        .def("plan", static_cast<bool (HomotopyClassPlanner::*)(const std::vector<PoseSE2>&, const Velocity2D*, bool)>(
            &HomotopyClassPlanner::plan),
            py::arg("initial_plan"), py::arg("start_vel") = nullptr, py::arg("free_goal_vel") = false)
        .def("get_velocity_command", [](const HomotopyClassPlanner& planner, int look_ahead_poses) {
            double vx, vy, omega;
            bool success = planner.getVelocityCommand(vx, vy, omega, look_ahead_poses);
            return py::make_tuple(success, vx, vy, omega);
        }, py::arg("look_ahead_poses") = 1)
        .def("best_teb", &HomotopyClassPlanner::bestTeb)
        .def("find_best_teb", &HomotopyClassPlanner::findBestTeb)
        .def("clear_planner", &HomotopyClassPlanner::clearPlanner)
        .def("set_preferred_turning_dir", &HomotopyClassPlanner::setPreferredTurningDir)
        .def("has_diverged", &HomotopyClassPlanner::hasDiverged)
        .def("is_initialized", &HomotopyClassPlanner::isInitialized)
        .def("best_teb_idx", &HomotopyClassPlanner::bestTebIdx)
        .def("get_trajectory_container", &HomotopyClassPlanner::getTrajectoryContainer,
             py::return_value_policy::reference_internal)
        .def("is_trajectory_feasible", [](HomotopyClassPlanner& planner,
                                          const RobotFootprintModelPtr& robot_model,
                                          const std::vector<ObstaclePtr>& obstacles,
                                          int look_ahead_idx,
                                          double feasibility_check_lookahead_distance) {
            return planner.isTrajectoryFeasible(robot_model, obstacles, look_ahead_idx,
                                                feasibility_check_lookahead_distance);
        }, py::arg("robot_model"), py::arg("obstacles"),
           py::arg("look_ahead_idx") = -1,
           py::arg("feasibility_check_lookahead_distance") = -1.0);

    // ==================== Utility Functions ====================
    m.def("normalize_theta", &normalize_theta, "Normalize angle to [-pi, pi]");
    m.def("average_angle", &average_angle, "Average two angles");
}
