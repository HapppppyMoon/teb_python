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
 * Author: Christoph Rösmann
 * Modified: ROS dependencies removed for standalone use
 *********************************************************************/

#ifndef OPTIMAL_PLANNER_H_
#define OPTIMAL_PLANNER_H_

#include <math.h>
#include <limits.h>
#include <memory>

// teb stuff
#include "teb_local_planner/teb_config.h"
#include "teb_local_planner/misc.h"
#include "teb_local_planner/timed_elastic_band.h"
#include "teb_local_planner/planner_interface.h"
#include "teb_local_planner/robot_footprint_model.h"

// g2o lib stuff
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/csparse/linear_solver_csparse.h"
// #include "g2o/solvers/cholmod/linear_solver_cholmod.h"  // Optional - only if cholmod is available

// g2o custom edges and vertices for the TEB planner
#include "teb_local_planner/g2o_types/edge_velocity.h"
#include "teb_local_planner/g2o_types/edge_acceleration.h"
#include "teb_local_planner/g2o_types/edge_kinematics.h"
#include "teb_local_planner/g2o_types/edge_time_optimal.h"
#include "teb_local_planner/g2o_types/edge_shortest_path.h"
#include "teb_local_planner/g2o_types/edge_obstacle.h"
#include "teb_local_planner/g2o_types/edge_dynamic_obstacle.h"
#include "teb_local_planner/g2o_types/edge_via_point.h"
#include "teb_local_planner/g2o_types/edge_prefer_rotdir.h"

namespace teb_local_planner
{

//! Typedef for the block solver utilized for optimization
typedef g2o::BlockSolverX TEBBlockSolver;

//! Typedef for the linear solver utilized for optimization
typedef g2o::LinearSolverCSparse<TEBBlockSolver::PoseMatrixType> TEBLinearSolver;

//! Typedef for a container storing via-points
typedef std::vector< Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > ViaPointContainer;

//! Trajectory point for output
struct TrajectoryPoint
{
  PoseSE2 pose;
  Velocity2D velocity;
  Velocity2D acceleration;
  double time_from_start;
};

/**
 * @class TebOptimalPlanner
 * @brief This class optimizes an internal Timed Elastic Band trajectory using the g2o-framework.
 */
class TebOptimalPlanner : public PlannerInterface
{
public:

  /**
   * @brief Default constructor
   */
  TebOptimalPlanner();

  /**
   * @brief Construct and initialize the TEB optimal planner.
   * @param cfg Const reference to the TebConfig class for internal parameters
   * @param obstacles Container storing all relevant obstacles (see Obstacle)
   * @param via_points Container storing via-points (optional)
   */
  TebOptimalPlanner(const TebConfig& cfg, ObstContainer* obstacles = nullptr,
                    const ViaPointContainer* via_points = nullptr);

  /**
   * @brief Destruct the optimal planner.
   */
  virtual ~TebOptimalPlanner();

  /**
    * @brief Initializes the optimal planner
    * @param cfg Const reference to the TebConfig class for internal parameters
    * @param obstacles Container storing all relevant obstacles (see Obstacle)
    * @param via_points Container storing via-points (optional)
    */
  void initialize(const TebConfig& cfg, ObstContainer* obstacles = nullptr,
                  const ViaPointContainer* via_points = nullptr);

  /** @name Plan a trajectory  */
  //@{

  /**
   * @brief Plan a trajectory based on an initial reference plan.
   */
  virtual bool plan(const std::vector<PoseSE2>& initial_plan, const Velocity2D* start_vel = nullptr, bool free_goal_vel=false) override;

  /**
   * @brief Plan a trajectory between a given start and goal pose
   */
  virtual bool plan(const PoseSE2& start, const PoseSE2& goal, const Velocity2D* start_vel = nullptr, bool free_goal_vel=false) override;


  /**
   * @brief Get the velocity command from a previously optimized plan to control the robot at the current sampling interval.
   */
  virtual bool getVelocityCommand(double& vx, double& vy, double& omega, int look_ahead_poses) const override;


  /**
   * @brief Optimize a previously initialized trajectory (actual TEB optimization loop).
   */
  bool optimizeTEB(int iterations_innerloop, int iterations_outerloop, bool compute_cost_afterwards = false,
                   double obst_cost_scale=1.0, double viapoint_cost_scale=1.0, bool alternative_time_cost=false);

  //@}


  /** @name Desired initial and final velocity */
  //@{


  /**
   * @brief Set the initial velocity at the trajectory's start pose
   */
  void setVelocityStart(const Velocity2D& vel_start);

  /**
   * @brief Set the desired final velocity at the trajectory's goal pose.
   */
  void setVelocityGoal(const Velocity2D& vel_goal);

  /**
   * @brief Set the desired final velocity at the trajectory's goal pose to be the maximum velocity limit
   */
  void setVelocityGoalFree() {vel_goal_.first = false;}

  //@}


  /** @name Take obstacles into account */
  //@{


  /**
   * @brief Assign a new set of obstacles
   */
  void setObstVector(ObstContainer* obst_vector) {obstacles_ = obst_vector;}

  /**
   * @brief Access the internal obstacle container.
   */
  const ObstContainer& getObstVector() const {return *obstacles_;}

  //@}

  /** @name Take via-points into account */
  //@{


  /**
   * @brief Assign a new set of via-points
   */
  void setViaPoints(const ViaPointContainer* via_points) {via_points_ = via_points;}

  /**
   * @brief Access the internal via-point container.
   */
  const ViaPointContainer& getViaPoints() const {return *via_points_;}

  //@}


  /** @name Utility methods and more */
  //@{

  /**
   * @brief Reset the planner by clearing the internal graph and trajectory.
   */
  virtual void clearPlanner() override
  {
    clearGraph();
    teb_.clearTimedElasticBand();
  }

  /**
   * @brief Prefer a desired initial turning direction
   */
  virtual void setPreferredTurningDir(RotType dir) override {prefer_rotdir_=dir;}

  /**
   * @brief Register the vertices and edges defined for the TEB to the g2o::Factory.
   */
  static void registerG2OTypes();

  /**
   * @brief Access the internal TimedElasticBand trajectory.
   */
  TimedElasticBand& teb() {return teb_;};

  /**
   * @brief Access the internal TimedElasticBand trajectory (read-only).
   */
  const TimedElasticBand& teb() const {return teb_;};

  /**
   * @brief Access the internal g2o optimizer.
   */
  std::shared_ptr<g2o::SparseOptimizer> optimizer() {return optimizer_;};

  /**
   * @brief Access the internal g2o optimizer (read-only).
   */
  std::shared_ptr<const g2o::SparseOptimizer> optimizer() const {return optimizer_;};

  /**
   * @brief Check if last optimization was successful
   */
  bool isOptimized() const {return optimized_;};

  /**
   * @brief Returns true if the planner has diverged.
   */
  bool hasDiverged() const override;

  /**
   * @brief Compute the cost vector of a given optimization problem
   */
  void computeCurrentCost(double obst_cost_scale=1.0, double viapoint_cost_scale=1.0, bool alternative_time_cost=false);

  /**
   * Compute and return the cost of the current optimization graph
   */
  virtual void computeCurrentCost(std::vector<double>& cost, double obst_cost_scale=1.0, double viapoint_cost_scale=1.0, bool alternative_time_cost=false)
  {
    computeCurrentCost(obst_cost_scale, viapoint_cost_scale, alternative_time_cost);
    cost.push_back( getCurrentCost() );
  }

  /**
   * @brief Access the cost vector.
   */
  double getCurrentCost() const {return cost_;}


  /**
   * @brief Extract the velocity from consecutive poses and a time difference
   */
  inline void extractVelocity(const PoseSE2& pose1, const PoseSE2& pose2, double dt, double& vx, double& vy, double& omega) const;

  /**
   * @brief Compute the velocity profile of the trajectory
   */
  void getVelocityProfile(std::vector<Velocity2D>& velocity_profile) const;

  /**
   * @brief Return the complete trajectory including poses, velocity profiles and temporal information
   */
  void getFullTrajectory(std::vector<TrajectoryPoint>& trajectory) const;

  /**
   * @brief Check whether the planned trajectory is feasible or not.
   */
  virtual bool isTrajectoryFeasible(const RobotFootprintModelPtr& robot_model,
                                     const std::vector<ObstaclePtr>& obstacles,
                                     int look_ahead_idx=-1,
                                     double feasibility_check_lookahead_distance=-1.0) override;

  //@}

protected:

  /** @name Hyper-Graph creation and optimization */
  //@{

  /**
   * @brief Build the hyper-graph representing the TEB optimization problem.
   */
  bool buildGraph(double weight_multiplier=1.0);

  /**
   * @brief Optimize the previously constructed hyper-graph to deform / optimize the TEB.
   */
  bool optimizeGraph(int no_iterations, bool clear_after=true);

  /**
   * @brief Clear an existing internal hyper-graph.
   */
  void clearGraph();

  /**
   * @brief Add all relevant vertices to the hyper-graph as optimizable variables.
   */
  void AddTEBVertices();

  /**
   * @brief Add all edges (local cost functions) for limiting the translational and angular velocity.
   */
  void AddEdgesVelocity();

  /**
   * @brief Add all edges (local cost functions) for limiting the translational and angular acceleration.
   */
  void AddEdgesAcceleration();

  /**
   * @brief Add all edges (local cost functions) for minimizing the transition time
   */
  void AddEdgesTimeOptimal();

  /**
   * @brief Add all edges (local cost functions) for minimizing the path length
   */
  void AddEdgesShortestPath();

  /**
   * @brief Add all edges (local cost functions) related to keeping a distance from static obstacles
   */
  void AddEdgesObstacles(double weight_multiplier=1.0);

  /**
   * @brief Add all edges (local cost functions) related to keeping a distance from static obstacles (legacy)
   */
  void AddEdgesObstaclesLegacy(double weight_multiplier=1.0);

  /**
   * @brief Add all edges (local cost functions) related to minimizing the distance to via-points
   */
  void AddEdgesViaPoints();

  /**
   * @brief Add all edges (local cost functions) related to keeping a distance from dynamic obstacles.
   */
  void AddEdgesDynamicObstacles(double weight_multiplier=1.0);

  /**
   * @brief Add all edges (local cost functions) for satisfying kinematic constraints of a differential drive robot
   */
  void AddEdgesKinematicsDiffDrive();

  /**
   * @brief Add all edges (local cost functions) for satisfying kinematic constraints of a carlike robot
   */
  void AddEdgesKinematicsCarlike();

  /**
   * @brief Add all edges (local cost functions) for prefering a specifiy turning direction
   */
  void AddEdgesPreferRotDir();

  /**
   * @brief Add all edges (local cost function) for reducing the velocity due to obstacles
   */
  void AddEdgesVelocityObstacleRatio();

  //@}


  /**
   * @brief Initialize and configure the g2o sparse optimizer.
   */
  std::shared_ptr<g2o::SparseOptimizer> initOptimizer();


  // external objects (store weak pointers)
  const TebConfig* cfg_; //!< Config class that stores and manages all related parameters
  ObstContainer* obstacles_; //!< Store obstacles that are relevant for planning
  const ViaPointContainer* via_points_; //!< Store via points for planning
  std::vector<ObstContainer> obstacles_per_vertex_; //!< Store the obstacles associated with the n-1 initial vertices

  double cost_; //!< Store cost value of the current hyper-graph
  RotType prefer_rotdir_; //!< Store whether to prefer a specific initial rotation in optimization

  // internal objects (memory management owned)
  TimedElasticBand teb_; //!< Actual trajectory object
  RobotFootprintModelPtr robot_model_; //!< Robot model
  std::shared_ptr<g2o::SparseOptimizer> optimizer_; //!< g2o optimizer for trajectory optimization
  std::pair<bool, Velocity2D> vel_start_; //!< Store the initial velocity at the start pose
  std::pair<bool, Velocity2D> vel_goal_; //!< Store the final velocity at the goal pose

  bool initialized_; //!< Keeps track about the correct initialization of this class
  bool optimized_; //!< This variable is \c true as long as the last optimization has been completed successful

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

//! Abbrev. for shared instances of the TebOptimalPlanner
typedef std::shared_ptr<TebOptimalPlanner> TebOptimalPlannerPtr;
//! Abbrev. for shared const TebOptimalPlanner pointers
typedef std::shared_ptr<const TebOptimalPlanner> TebOptimalPlannerConstPtr;
//! Abbrev. for containers storing multiple teb optimal planners
typedef std::vector< TebOptimalPlannerPtr > TebOptPlannerContainer;

} // namespace teb_local_planner

#endif /* OPTIMAL_PLANNER_H_ */
