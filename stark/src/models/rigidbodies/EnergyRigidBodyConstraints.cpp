#include "EnergyRigidBodyConstraints.h"

#include <fmt/format.h>
#include <algorithm>
#include <cmath>
#include <limits>

#include "rigidbody_transformations.h"

namespace
{
	constexpr double kMinAlResidualSmoothing = 1e-12;

	symx::Scalar smooth_l2_norm(const symx::Vector& v, const symx::Scalar& eps)
	{
		return symx::sqrt(v.squared_norm() + eps.powN(2)) - eps;
	}
}

stark::EnergyRigidBodyConstraints::EnergyRigidBodyConstraints(stark::core::Stark& stark, spRigidBodyDynamics rb)
	: rb(rb)
{
	// Callbacks
	stark.callbacks.add_is_converged_state_valid([&]() { return this->_is_converged_state_valid(stark); });
	stark.callbacks.add_on_time_step_accepted([&]() { this->_on_time_step_accepted(stark); });

	// Constraint containers initialization
	this->global_points = std::make_shared<RigidBodyConstraints::GlobalPoints>();
	this->global_directions = std::make_shared<RigidBodyConstraints::GlobalDirections>();
	this->points = std::make_shared<RigidBodyConstraints::Points>();
	this->point_on_axes = std::make_shared<RigidBodyConstraints::PointOnAxes>();
	this->distances = std::make_shared<RigidBodyConstraints::Distance>();
	this->distance_limits = std::make_shared<RigidBodyConstraints::DistanceLimits>();
	this->directions = std::make_shared<RigidBodyConstraints::Directions>();
	this->angle_limits = std::make_shared<RigidBodyConstraints::AngleLimits>();
	this->damped_springs = std::make_shared<RigidBodyConstraints::DampedSprings>();
	this->linear_velocity = std::make_shared<RigidBodyConstraints::LinearVelocity>();
	this->angular_velocity = std::make_shared<RigidBodyConstraints::AngularVelocity>();
	this->al_residual_smoothing_flag = std::max(kMinAlResidualSmoothing, this->al_params.residual_smoothing);

	// Energy declarations
	stark.global_energy.add_energy("rb_constraint_global_points", this->global_points->conn,
		[&](symx::Energy& energy, symx::Element& conn)
		{
			auto& data = this->global_points;

			symx::Vector loc = energy.make_vector(data->loc, conn["idx"]);
			symx::Vector target_glob = energy.make_vector(data->target_glob, conn["idx"]);
			symx::Scalar stiffness = energy.make_scalar(data->stiffness, conn["idx"]);
			symx::Vector al_lambda = energy.make_vector(data->al_lambda_vec, conn["idx"]);
			symx::Scalar al_rho = energy.make_scalar(data->al_rho, conn["idx"]);
			symx::Scalar is_active = energy.make_scalar(data->is_active, conn["idx"]);
			symx::Scalar dt = energy.make_scalar(stark.dt);
			symx::Scalar use_al = energy.make_scalar(this->al_use_flag);

			symx::Vector glob = this->rb->get_x1(energy, conn["rb"], loc, dt);
			symx::Vector u = glob - target_glob;
			symx::Scalar E_penalty = RigidBodyConstraints::GlobalPoints::energy(stiffness, target_glob, glob);
			symx::Scalar E_al = al_lambda.dot(u) + 0.5 * al_rho * u.squared_norm();
			symx::Scalar E = symx::branch(use_al > 0.5, E_al, E_penalty);
			energy.set_with_condition(E, is_active > 0.0);
		}
	);

	stark.global_energy.add_energy("rb_constraint_global_directions", this->global_directions->conn,
		[&](symx::Energy& energy, symx::Element& conn)
		{
			auto& data = this->global_directions;

			symx::Vector d_loc = energy.make_vector(data->d_loc, conn["idx"]);
			symx::Vector target_d_glob = energy.make_vector(data->target_d_glob, conn["idx"]);
			symx::Scalar stiffness = energy.make_scalar(data->stiffness, conn["idx"]);
			symx::Vector al_lambda = energy.make_vector(data->al_lambda_vec, conn["idx"]);
			symx::Scalar al_rho = energy.make_scalar(data->al_rho, conn["idx"]);
			symx::Scalar is_active = energy.make_scalar(data->is_active, conn["idx"]);
			symx::Scalar dt = energy.make_scalar(stark.dt);
			symx::Scalar use_al = energy.make_scalar(this->al_use_flag);

			symx::Vector d_glob = this->rb->get_d1(energy, conn["rb"], d_loc, dt);
			symx::Vector u = d_glob - target_d_glob;
			symx::Scalar E_penalty = RigidBodyConstraints::GlobalDirections::energy(stiffness, target_d_glob, d_glob);
			symx::Scalar E_al = al_lambda.dot(u) + 0.5 * al_rho * u.squared_norm();
			symx::Scalar E = symx::branch(use_al > 0.5, E_al, E_penalty);
			energy.set_with_condition(E, is_active > 0.0);
		}
	);

	stark.global_energy.add_energy("rb_constraint_points", this->points->conn,
		[&](symx::Energy& energy, symx::Element& conn)
		{
			auto& data = this->points;

			symx::Vector a_loc = energy.make_vector(data->a_loc, conn["idx"]);
			symx::Vector b_loc = energy.make_vector(data->b_loc, conn["idx"]);
			symx::Scalar stiffness = energy.make_scalar(data->stiffness, conn["idx"]);
			symx::Vector al_lambda = energy.make_vector(data->al_lambda_vec, conn["idx"]);
			symx::Scalar al_rho = energy.make_scalar(data->al_rho, conn["idx"]);
			symx::Scalar is_active = energy.make_scalar(data->is_active, conn["idx"]);
			symx::Scalar dt = energy.make_scalar(stark.dt);
			symx::Scalar use_al = energy.make_scalar(this->al_use_flag);

			symx::Vector a1 = this->rb->get_x1(energy, conn["a"], a_loc, dt);
			symx::Vector b1 = this->rb->get_x1(energy, conn["b"], b_loc, dt);
			symx::Vector u = b1 - a1;
			symx::Scalar E_penalty = RigidBodyConstraints::Points::energy(stiffness, a1, b1);
			symx::Scalar E_al = al_lambda.dot(u) + 0.5 * al_rho * u.squared_norm();
			symx::Scalar E = symx::branch(use_al > 0.5, E_al, E_penalty);
			energy.set_with_condition(E, is_active > 0.0);
		}
	);

	stark.global_energy.add_energy("rb_constraint_point_on_axis", this->point_on_axes->conn,
		[&](symx::Energy& energy, symx::Element& conn)
		{
			auto& data = this->point_on_axes;

			symx::Vector a_loc = energy.make_vector(data->a_loc, conn["idx"]);
			symx::Vector da_loc = energy.make_vector(data->da_loc, conn["idx"]);
			symx::Vector b_loc = energy.make_vector(data->b_loc, conn["idx"]);
			symx::Scalar stiffness = energy.make_scalar(data->stiffness, conn["idx"]);
			symx::Scalar al_lambda = energy.make_scalar(data->al_lambda, conn["idx"]);
			symx::Scalar al_rho = energy.make_scalar(data->al_rho, conn["idx"]);
			symx::Scalar is_active = energy.make_scalar(data->is_active, conn["idx"]);
			symx::Scalar dt = energy.make_scalar(stark.dt);
			symx::Scalar use_al = energy.make_scalar(this->al_use_flag);
			symx::Scalar al_eps = energy.make_scalar(this->al_residual_smoothing_flag);

			auto [a1, da1] = this->rb->get_x1_d1(energy, conn["a"], a_loc, da_loc, dt);
			symx::Vector b1 = this->rb->get_x1(energy, conn["b"], b_loc, dt);
			symx::Vector u = b1 - a1;
			symx::Vector u_perp = u - u.dot(da1) * da1;
			symx::Scalar C = smooth_l2_norm(u_perp, al_eps);
			symx::Scalar E_penalty = RigidBodyConstraints::PointOnAxes::energy(stiffness, a1, da1, b1);
			symx::Scalar E_al = al_lambda * C + 0.5 * al_rho * C.powN(2);
			symx::Scalar E = symx::branch(use_al > 0.5, E_al, E_penalty);
			energy.set_with_condition(E, is_active > 0.0);
		}
	);

	stark.global_energy.add_energy("rb_constraint_distances", this->distances->conn,
		[&](symx::Energy& energy, symx::Element& conn)
		{
			auto& data = this->distances;

			symx::Vector a_loc = energy.make_vector(data->a_loc, conn["idx"]);
			symx::Vector b_loc = energy.make_vector(data->b_loc, conn["idx"]);
			symx::Scalar target_distance = energy.make_scalar(data->target_distance, conn["idx"]);
			symx::Scalar stiffness = energy.make_scalar(data->stiffness, conn["idx"]);
			symx::Scalar al_lambda = energy.make_scalar(data->al_lambda, conn["idx"]);
			symx::Scalar al_rho = energy.make_scalar(data->al_rho, conn["idx"]);
			symx::Scalar is_active = energy.make_scalar(data->is_active, conn["idx"]);
			symx::Scalar dt = energy.make_scalar(stark.dt);
			symx::Scalar use_al = energy.make_scalar(this->al_use_flag);
			symx::Scalar al_eps = energy.make_scalar(this->al_residual_smoothing_flag);

			symx::Vector a1 = this->rb->get_x1(energy, conn["a"], a_loc, dt);
			symx::Vector b1 = this->rb->get_x1(energy, conn["b"], b_loc, dt);
			symx::Scalar C = smooth_l2_norm(b1 - a1, al_eps) - target_distance;
			symx::Scalar E_penalty = RigidBodyConstraints::Distance::energy(stiffness, a1, b1, target_distance);
			symx::Scalar E_al = al_lambda * C + 0.5 * al_rho * C.powN(2);
			symx::Scalar E = symx::branch(use_al > 0.5, E_al, E_penalty);
			energy.set_with_condition(E, is_active > 0.0);
		}
	);

	stark.global_energy.add_energy("rb_constraint_distance_limits", this->distance_limits->conn,
		[&](symx::Energy& energy, symx::Element& conn)
		{
			auto& data = this->distance_limits;

			symx::Vector a_loc = energy.make_vector(data->a_loc, conn["idx"]);
			symx::Vector b_loc = energy.make_vector(data->b_loc, conn["idx"]);
			symx::Scalar min_distance = energy.make_scalar(data->min_distance, conn["idx"]);
			symx::Scalar max_distance = energy.make_scalar(data->max_distance, conn["idx"]);
			symx::Scalar stiffness = energy.make_scalar(data->stiffness, conn["idx"]);
			symx::Scalar al_lambda = energy.make_scalar(data->al_lambda, conn["idx"]);
			symx::Scalar al_rho = energy.make_scalar(data->al_rho, conn["idx"]);
			symx::Scalar is_active = energy.make_scalar(data->is_active, conn["idx"]);
			symx::Scalar dt = energy.make_scalar(stark.dt);
			symx::Scalar use_al = energy.make_scalar(this->al_use_flag);
			symx::Scalar al_eps = energy.make_scalar(this->al_residual_smoothing_flag);

			symx::Vector a1 = this->rb->get_x1(energy, conn["a"], a_loc, dt);
			symx::Vector b1 = this->rb->get_x1(energy, conn["b"], b_loc, dt);
			symx::Scalar d = smooth_l2_norm(b1 - a1, al_eps);
			symx::Scalar C = symx::branch(d < min_distance, min_distance - d, symx::branch(d > max_distance, d - max_distance, 0.0));
			symx::Scalar E_penalty = RigidBodyConstraints::DistanceLimits::energy(stiffness, a1, b1, min_distance, max_distance);
			symx::Scalar E_al = al_lambda * C + 0.5 * al_rho * C.powN(2);
			symx::Scalar E = symx::branch(use_al > 0.5, E_al, E_penalty);
			energy.set_with_condition(E, is_active > 0.0);
		}
	);

	stark.global_energy.add_energy("rb_constraint_directions", this->directions->conn,
		[&](symx::Energy& energy, symx::Element& conn)
		{
			auto& data = this->directions;

			symx::Vector da_loc = energy.make_vector(data->da_loc, conn["idx"]);
			symx::Vector db_loc = energy.make_vector(data->db_loc, conn["idx"]);
			symx::Scalar stiffness = energy.make_scalar(data->stiffness, conn["idx"]);
			symx::Vector al_lambda = energy.make_vector(data->al_lambda_vec, conn["idx"]);
			symx::Scalar al_rho = energy.make_scalar(data->al_rho, conn["idx"]);
			symx::Scalar is_active = energy.make_scalar(data->is_active, conn["idx"]);
			symx::Scalar dt = energy.make_scalar(stark.dt);
			symx::Scalar use_al = energy.make_scalar(this->al_use_flag);

			symx::Vector da = this->rb->get_d1(energy, conn["a"], da_loc, dt);
			symx::Vector db = this->rb->get_d1(energy, conn["b"], db_loc, dt);
			symx::Vector u = db - da;
			symx::Scalar E_penalty = RigidBodyConstraints::Directions::energy(stiffness, da, db);
			symx::Scalar E_al = al_lambda.dot(u) + 0.5 * al_rho * u.squared_norm();
			symx::Scalar E = symx::branch(use_al > 0.5, E_al, E_penalty);
			energy.set_with_condition(E, is_active > 0.0);
		}
	);

	stark.global_energy.add_energy("rb_constraint_angle_limits", this->angle_limits->conn,
		[&](symx::Energy& energy, symx::Element& conn)
		{
			auto& data = this->angle_limits;

			symx::Vector da_loc = energy.make_vector(data->da_loc, conn["idx"]);
			symx::Vector db_loc = energy.make_vector(data->db_loc, conn["idx"]);
			symx::Scalar max_distance = energy.make_scalar(data->max_distance, conn["idx"]);
			symx::Scalar stiffness = energy.make_scalar(data->stiffness, conn["idx"]);
			symx::Scalar al_lambda = energy.make_scalar(data->al_lambda, conn["idx"]);
			symx::Scalar al_rho = energy.make_scalar(data->al_rho, conn["idx"]);
			symx::Scalar is_active = energy.make_scalar(data->is_active, conn["idx"]);
			symx::Scalar dt = energy.make_scalar(stark.dt);
			symx::Scalar use_al = energy.make_scalar(this->al_use_flag);
			symx::Scalar al_eps = energy.make_scalar(this->al_residual_smoothing_flag);

			symx::Vector da1 = this->rb->get_d1(energy, conn["a"], da_loc, dt);
			symx::Vector db1 = this->rb->get_d1(energy, conn["b"], db_loc, dt);
			symx::Scalar d = smooth_l2_norm(db1 - da1, al_eps);
			symx::Scalar C = symx::branch(d > max_distance, d - max_distance, 0.0);
			symx::Scalar E_penalty = RigidBodyConstraints::AngleLimits::energy(stiffness, da1, db1, max_distance);
			symx::Scalar E_al = al_lambda * C + 0.5 * al_rho * C.powN(2);
			symx::Scalar E = symx::branch(use_al > 0.5, E_al, E_penalty);
			energy.set_with_condition(E, is_active > 0.0);
		}
	);

	stark.global_energy.add_energy("rb_constraint_damped_spring", this->damped_springs->conn,
		[&](symx::Energy& energy, symx::Element& conn)
		{
			auto& data = this->damped_springs;

			symx::Vector a_loc = energy.make_vector(data->a_loc, conn["idx"]);
			symx::Vector b_loc = energy.make_vector(data->b_loc, conn["idx"]);
			symx::Scalar rest_length = energy.make_scalar(data->rest_length, conn["idx"]);
			symx::Scalar stiffness = energy.make_scalar(data->stiffness, conn["idx"]);
			symx::Scalar damping = energy.make_scalar(data->damping, conn["idx"]);
			symx::Scalar is_active = energy.make_scalar(data->is_active, conn["idx"]);
			symx::Scalar dt = energy.make_scalar(stark.dt);

			auto [a0, a1] = this->rb->get_x0_x1(energy, conn["a"], a_loc, dt);
			auto [b0, b1] = this->rb->get_x0_x1(energy, conn["b"], b_loc, dt);

			symx::Scalar E = RigidBodyConstraints::DampedSprings::energy(stiffness, damping, a0, a1, b0, b1, rest_length, dt);
			energy.set_with_condition(E, is_active > 0.0);
		}
	);

	stark.global_energy.add_energy("rb_constraint_linear_velocity", this->linear_velocity->conn,
		[&](symx::Energy& energy, symx::Element& conn)
		{
			auto& data = this->linear_velocity;

			symx::Vector da_loc = energy.make_vector(data->da_loc, conn["idx"]);
			symx::Scalar target_v = energy.make_scalar(data->target_v, conn["idx"]);
			symx::Scalar max_force = energy.make_scalar(data->max_force, conn["idx"]);
			symx::Scalar delay = energy.make_scalar(data->delay, conn["idx"]);
			symx::Scalar is_active = energy.make_scalar(data->is_active, conn["idx"]);
			symx::Vector va1 = energy.make_dof_vector(this->rb->dof_v, this->rb->v1, conn["a"]);
			symx::Vector vb1 = energy.make_dof_vector(this->rb->dof_v, this->rb->v1, conn["b"]);
			symx::Vector wa1 = energy.make_dof_vector(this->rb->dof_w, this->rb->w1, conn["a"]);
			symx::Vector qa0 = energy.make_vector(this->rb->q0_, conn["a"]);
			symx::Scalar dt = energy.make_scalar(stark.dt);

			symx::Vector da1 = integrate_loc_direction(da_loc, qa0, wa1, dt);
			symx::Scalar E = RigidBodyConstraints::LinearVelocity::energy(da1, va1, vb1, target_v, max_force, delay, dt);
			energy.set_with_condition(E, is_active > 0.0);
		}
	);

	stark.global_energy.add_energy("rb_constraint_angular_velocity", this->angular_velocity->conn,
		[&](symx::Energy& energy, symx::Element& conn)
		{
			auto& data = this->angular_velocity;

			symx::Vector da_loc = energy.make_vector(data->da_loc, conn["idx"]);
			symx::Scalar target_w = energy.make_scalar(data->target_w, conn["idx"]);
			symx::Scalar max_torque = energy.make_scalar(data->max_torque, conn["idx"]);
			symx::Scalar delay = energy.make_scalar(data->delay, conn["idx"]);
			symx::Scalar is_active = energy.make_scalar(data->is_active, conn["idx"]);
			symx::Vector wa1 = energy.make_dof_vector(this->rb->dof_w, this->rb->w1, conn["a"]);
			symx::Vector wb1 = energy.make_dof_vector(this->rb->dof_w, this->rb->w1, conn["b"]);
			symx::Vector qa0 = energy.make_vector(this->rb->q0_, conn["a"]);
			symx::Scalar dt = energy.make_scalar(stark.dt);

			symx::Vector da1 = integrate_loc_direction(da_loc, qa0, wa1, dt);
			symx::Scalar E = RigidBodyConstraints::AngularVelocity::energy(da1, wa1, wb1, target_w, max_torque, delay, dt);
			energy.set_with_condition(E, is_active > 0.0);
		}
	);

	this->_initialize_al_state_if_needed();
}

void stark::EnergyRigidBodyConstraints::set_augmented_lagrangian_params(const AugmentedLagrangianParams& params)
{
	const bool was_enabled = this->al_params.enabled;
	this->al_params = params;
	this->al_residual_smoothing_flag = std::max(kMinAlResidualSmoothing, this->al_params.residual_smoothing);
	this->al_use_flag = this->al_params.enabled ? 1.0 : 0.0;
	this->_initialize_al_state_if_needed();

	if (this->al_params.enabled && !was_enabled) {
		auto reset = [&](auto& data)
		{
			for (std::size_t i = 0; i < data->al_lambda.size(); ++i) {
				data->al_lambda[i] = 0.0;
				data->al_prev_violation[i] = std::numeric_limits<double>::infinity();
				if (this->al_params.rho0 > 0.0) {
					data->al_rho[i] = this->al_params.rho0;
				}
			}
		};
		auto reset_vector = [&](auto& data)
		{
			for (std::size_t i = 0; i < data->al_lambda_vec.size(); ++i) {
				data->al_lambda_vec[i].setZero();
				data->al_prev_violation[i] = std::numeric_limits<double>::infinity();
				if (this->al_params.rho0 > 0.0) {
					data->al_rho[i] = this->al_params.rho0;
				}
			}
		};
		reset_vector(this->global_points);
		reset_vector(this->global_directions);
		reset_vector(this->points);
		reset(this->point_on_axes);
		reset(this->distances);
		reset(this->distance_limits);
		reset_vector(this->directions);
		reset(this->angle_limits);
	}

	if (!this->al_params.enabled) {
		this->al_outer_iteration = 0;
	}
}

stark::EnergyRigidBodyConstraints::AugmentedLagrangianParams stark::EnergyRigidBodyConstraints::get_augmented_lagrangian_params() const
{
	return this->al_params;
}

void stark::EnergyRigidBodyConstraints::_initialize_al_state_if_needed()
{
	auto ensure = [&](auto& data)
	{
		const std::size_t n = data->stiffness.size();
		if (data->al_lambda.size() != n) {
			data->al_lambda.resize(n, 0.0);
		}
		if (data->al_prev_violation.size() != n) {
			data->al_prev_violation.resize(n, std::numeric_limits<double>::infinity());
		}
		if (data->al_rho.size() < n) {
			const std::size_t old = data->al_rho.size();
			data->al_rho.resize(n, 0.0);
			for (std::size_t i = old; i < n; ++i) {
				data->al_rho[i] = (this->al_params.rho0 > 0.0) ? this->al_params.rho0 : std::max(1e-12, data->stiffness[i]);
			}
		}
		for (std::size_t i = 0; i < n; ++i) {
			if (data->al_rho[i] <= 0.0 || !std::isfinite(data->al_rho[i])) {
				data->al_rho[i] = (this->al_params.rho0 > 0.0) ? this->al_params.rho0 : std::max(1e-12, data->stiffness[i]);
			}
		}
	};
	auto ensure_vector = [&](auto& data)
	{
		const std::size_t n = data->stiffness.size();
		if (data->al_lambda.size() != n) {
			data->al_lambda.resize(n, 0.0);
		}
		if (data->al_lambda_vec.size() != n) {
			data->al_lambda_vec.resize(n, Eigen::Vector3d::Zero());
		}
		if (data->al_prev_violation.size() != n) {
			data->al_prev_violation.resize(n, std::numeric_limits<double>::infinity());
		}
		if (data->al_rho.size() < n) {
			const std::size_t old = data->al_rho.size();
			data->al_rho.resize(n, 0.0);
			for (std::size_t i = old; i < n; ++i) {
				data->al_rho[i] = (this->al_params.rho0 > 0.0) ? this->al_params.rho0 : std::max(1e-12, data->stiffness[i]);
			}
		}
		for (std::size_t i = 0; i < n; ++i) {
			if (data->al_rho[i] <= 0.0 || !std::isfinite(data->al_rho[i])) {
				data->al_rho[i] = (this->al_params.rho0 > 0.0) ? this->al_params.rho0 : std::max(1e-12, data->stiffness[i]);
			}
		}
	};

	ensure_vector(this->global_points);
	ensure_vector(this->global_directions);
	ensure_vector(this->points);
	ensure(this->point_on_axes);
	ensure(this->distances);
	ensure(this->distance_limits);
	ensure_vector(this->directions);
	ensure(this->angle_limits);
}

bool stark::EnergyRigidBodyConstraints::_is_converged_state_valid(core::Stark& stark)
{
	if (this->al_params.enabled) {
		return this->_run_augmented_lagrangian_outer_iteration(stark, /*are_positions_set=*/false);
	}

	/*
		Hardens every constraints that has gone beyond the input tolerance.
		If no constraint needs to be hardened, return true.
	*/
	const bool valid = this->_adjust_constraints_stiffness_and_log(stark, 1.0, this->stiffness_hard_multiplier, /* are_positions_set = */ false);
	if (!valid) {
		stark.console.add_error_msg("Rigid body constraints are not within tolerance. Hardening bending_stiffness.");
	}
	return valid;
}

bool stark::EnergyRigidBodyConstraints::_run_augmented_lagrangian_outer_iteration(core::Stark& stark, bool are_positions_set)
{
	this->_initialize_al_state_if_needed();

	const double dt = stark.dt;
	auto get_x1 = [&](int rb, Eigen::Vector3d& loc) { return (are_positions_set) ? this->rb->get_position_at(rb, loc) : this->rb->get_x1(rb, loc, dt); };
	auto get_d1 = [&](int rb, Eigen::Vector3d& loc) { return (are_positions_set) ? this->rb->get_direction(rb, loc) : this->rb->get_d1(rb, loc, dt); };

	bool is_converged = true;
	int active_constraints = 0;
	int violated_constraints = 0;
	double max_joint_error_l2 = 0.0;
	double max_joint_error_deg = 0.0;

	auto process = [&](double C, double tol, bool signed_constraint, bool contributes_to_l2, double angle_error_deg,
		double& lambda, double& rho, double& prev_violation)
	{
		const double violation = std::abs(C);
		const bool is_violated = violation > tol;
		if (contributes_to_l2) {
			max_joint_error_l2 = std::max(max_joint_error_l2, violation);
		}
		if (angle_error_deg >= 0.0 && std::isfinite(angle_error_deg)) {
			max_joint_error_deg = std::max(max_joint_error_deg, angle_error_deg);
		}

		if (is_violated) {
			is_converged = false;
			violated_constraints++;

			if (this->al_params.adaptive_rho
				&& std::isfinite(prev_violation)
				&& prev_violation > 0.0
				&& violation > this->al_params.sufficient_decrease_ratio * prev_violation)
			{
				rho *= this->al_params.rho_update_ratio;
				stark.logger.add_to_counter("joint_rho_update_count", 1);
			}
		}

		if (signed_constraint) {
			lambda += rho * C;
		}
		else if (is_violated) {
			lambda = std::max(0.0, lambda + rho * std::max(0.0, C));
		}

		prev_violation = violation;
	};
	auto process_vector_equality = [&](const Eigen::Vector3d& u, double tol, bool contributes_to_l2, double angle_error_deg, double& rho, double& prev_violation, Eigen::Vector3d& lambda)
	{
		const double violation = u.norm();
		const bool is_violated = violation > tol;
		if (contributes_to_l2) {
			max_joint_error_l2 = std::max(max_joint_error_l2, violation);
		}
		if (angle_error_deg >= 0.0 && std::isfinite(angle_error_deg)) {
			max_joint_error_deg = std::max(max_joint_error_deg, angle_error_deg);
		}

		if (is_violated) {
			is_converged = false;
			violated_constraints++;

			if (this->al_params.adaptive_rho
				&& std::isfinite(prev_violation)
				&& prev_violation > 0.0
				&& violation > this->al_params.sufficient_decrease_ratio * prev_violation)
			{
				rho *= this->al_params.rho_update_ratio;
				stark.logger.add_to_counter("joint_rho_update_count", 1);
			}
		}

		lambda += rho * u;
		prev_violation = violation;
	};

	// Global points
	for (int i = 0; i < (int)this->global_points->conn.size(); ++i) {
		auto& data = this->global_points;
		auto [idx, a] = data->conn[i];
		if (data->is_active[idx] <= 0.0) { continue; }
		active_constraints++;

		const Eigen::Vector3d p = get_x1(a, data->loc[idx]);
		const Eigen::Vector3d u = p - data->target_glob[idx];
		process_vector_equality(u, data->tolerance_in_m[idx], true, -1.0, data->al_rho[idx], data->al_prev_violation[idx], data->al_lambda_vec[idx]);
	}

	// Global directions
	for (int i = 0; i < (int)this->global_directions->conn.size(); ++i) {
		auto& data = this->global_directions;
		auto [idx, a] = data->conn[i];
		if (data->is_active[idx] <= 0.0) { continue; }
		active_constraints++;

		const Eigen::Vector3d d = get_d1(a, data->d_loc[idx]);
		const Eigen::Vector3d u = d - data->target_d_glob[idx];
		auto [C_deg, t] = RigidBodyConstraints::GlobalDirections::violation_in_deg_and_torque(data->stiffness[idx], data->target_d_glob[idx], d);
		const double tol_norm = RigidBodyConstraints::AngleLimits::opening_distance_of_angle(std::max(0.0, data->tolerance_in_deg[idx]));
		process_vector_equality(u, tol_norm, false, C_deg, data->al_rho[idx], data->al_prev_violation[idx], data->al_lambda_vec[idx]);
	}

	// Points
	for (int i = 0; i < (int)this->points->conn.size(); ++i) {
		auto& data = this->points;
		auto [idx, a, b] = data->conn[i];
		if (data->is_active[idx] <= 0.0) { continue; }
		active_constraints++;

		const Eigen::Vector3d a1 = get_x1(a, data->a_loc[idx]);
		const Eigen::Vector3d b1 = get_x1(b, data->b_loc[idx]);
		const Eigen::Vector3d u = b1 - a1;
		process_vector_equality(u, data->tolerance_in_m[idx], true, -1.0, data->al_rho[idx], data->al_prev_violation[idx], data->al_lambda_vec[idx]);
	}

	// Point on axis
	for (int i = 0; i < (int)this->point_on_axes->conn.size(); ++i) {
		auto& data = this->point_on_axes;
		auto [idx, a, b] = data->conn[i];
		if (data->is_active[idx] <= 0.0) { continue; }
		active_constraints++;

		const Eigen::Vector3d a1 = get_x1(a, data->a_loc[idx]);
		const Eigen::Vector3d da1 = get_d1(a, data->da_loc[idx]);
		const Eigen::Vector3d b1 = get_x1(b, data->b_loc[idx]);
		auto [C, f] = RigidBodyConstraints::PointOnAxes::violation_in_m_and_force(data->stiffness[idx], a1, da1, b1);
		process(C, data->tolerance_in_m[idx], false, true, -1.0, data->al_lambda[idx], data->al_rho[idx], data->al_prev_violation[idx]);
	}

	// Distance equality
	for (int i = 0; i < (int)this->distances->conn.size(); ++i) {
		auto& data = this->distances;
		auto [idx, a, b] = data->conn[i];
		if (data->is_active[idx] <= 0.0) { continue; }
		active_constraints++;

		const Eigen::Vector3d a1 = get_x1(a, data->a_loc[idx]);
		const Eigen::Vector3d b1 = get_x1(b, data->b_loc[idx]);
		auto [C, f] = RigidBodyConstraints::Distance::signed_violation_in_m_and_force(data->stiffness[idx], a1, b1, data->target_distance[idx]);
		process(C, data->tolerance_in_m[idx], true, true, -1.0, data->al_lambda[idx], data->al_rho[idx], data->al_prev_violation[idx]);
	}

	// Distance limits (two-sided inequality)
	for (int i = 0; i < (int)this->distance_limits->conn.size(); ++i) {
		auto& data = this->distance_limits;
		auto [idx, a, b] = data->conn[i];
		if (data->is_active[idx] <= 0.0) { continue; }
		active_constraints++;

		const Eigen::Vector3d a1 = get_x1(a, data->a_loc[idx]);
		const Eigen::Vector3d b1 = get_x1(b, data->b_loc[idx]);
		auto [C_signed, f] = RigidBodyConstraints::DistanceLimits::signed_violation_in_m_and_force(data->stiffness[idx], a1, b1, data->min_distance[idx], data->max_distance[idx]);
		const double C = std::abs(C_signed);
		process(C, data->tolerance_in_m[idx], false, true, -1.0, data->al_lambda[idx], data->al_rho[idx], data->al_prev_violation[idx]);
	}

	// Direction alignment
	for (int i = 0; i < (int)this->directions->conn.size(); ++i) {
		auto& data = this->directions;
		auto [idx, a, b] = data->conn[i];
		if (data->is_active[idx] <= 0.0) { continue; }
		active_constraints++;

		const Eigen::Vector3d da = get_d1(a, data->da_loc[idx]);
		const Eigen::Vector3d db = get_d1(b, data->db_loc[idx]);
		const Eigen::Vector3d u = db - da;
		auto [C_deg, t] = RigidBodyConstraints::Directions::violation_in_deg_and_torque(data->stiffness[idx], da, db);
		const double tol_norm = RigidBodyConstraints::AngleLimits::opening_distance_of_angle(std::max(0.0, data->tolerance_in_deg[idx]));
		process_vector_equality(u, tol_norm, false, C_deg, data->al_rho[idx], data->al_prev_violation[idx], data->al_lambda_vec[idx]);
	}

	// Angle limits
	for (int i = 0; i < (int)this->angle_limits->conn.size(); ++i) {
		auto& data = this->angle_limits;
		auto [idx, a, b] = data->conn[i];
		if (data->is_active[idx] <= 0.0) { continue; }
		active_constraints++;

		const Eigen::Vector3d da = get_d1(a, data->da_loc[idx]);
		const Eigen::Vector3d db = get_d1(b, data->db_loc[idx]);
		const double d = (db - da).norm();
		const double C = std::max(0.0, d - data->max_distance[idx]);
		const double tol = RigidBodyConstraints::AngleLimits::opening_distance_of_angle(std::max(0.0, data->tolerance_in_deg[idx]));
		const double angle_deg = RigidBodyConstraints::AngleLimits::angle_of_opening_distance(C);
		process(C, tol, false, false, angle_deg, data->al_lambda[idx], data->al_rho[idx], data->al_prev_violation[idx]);
	}

	this->last_joint_error_max_l2 = max_joint_error_l2;
	this->last_joint_error_max_deg = max_joint_error_deg;
	this->last_joint_active_constraints = active_constraints;
	this->last_joint_violated_constraints = violated_constraints;

	if (is_converged) {
		this->al_outer_iteration = 0;
		return true;
	}

	this->al_outer_iteration++;
	stark.logger.add_to_counter("joint_al_outer_iterations_total", 1);
	stark.console.add_error_msg(fmt::format(
		"Joint AL not converged. outer={:d}/{:d}, max_l2={:.2e} m, max_deg={:.2e} deg, violated={:d}/{:d}.",
		this->al_outer_iteration,
		std::max(1, this->al_params.max_outer_iterations),
		this->last_joint_error_max_l2,
		this->last_joint_error_max_deg,
		this->last_joint_violated_constraints,
		this->last_joint_active_constraints));

	if (this->al_outer_iteration >= std::max(1, this->al_params.max_outer_iterations)) {
		stark.console.add_error_msg("Joint AL reached max outer iterations. Accepting current step with residual.");
		stark.logger.add_to_counter("joint_al_outer_limit_hits", 1);
		this->al_outer_iteration = 0;
		return true;
	}

	return false;
}

void stark::EnergyRigidBodyConstraints::_log_joint_metrics(core::Stark& stark) const
{
	stark.logger.append_to_series("joint_error_max_l2", this->last_joint_error_max_l2);
	stark.logger.append_to_series("joint_error_max_deg", this->last_joint_error_max_deg);
	stark.logger.append_to_series("joint_active_constraints", this->last_joint_active_constraints);
	stark.logger.append_to_series("joint_violated_constraints", this->last_joint_violated_constraints);
	stark.logger.append_to_series("joint_al_outer_iterations", this->al_outer_iteration);

	stark.logger.set("joint_error_max_l2", this->last_joint_error_max_l2);
	stark.logger.set("joint_error_max_deg", this->last_joint_error_max_deg);
	stark.logger.set("joint_active_constraints", this->last_joint_active_constraints);
	stark.logger.set("joint_violated_constraints", this->last_joint_violated_constraints);
	stark.logger.set("joint_al_outer_iterations", this->al_outer_iteration);
}

void stark::EnergyRigidBodyConstraints::_on_time_step_accepted(core::Stark& stark)
{
	if (this->al_params.enabled) {
		this->_log_joint_metrics(stark);
		this->al_outer_iteration = 0;
		return;
	}

	/*
	*	Logs the state of the constraints.
	*	Also, increases constraint stiffnesses at the end of a successful time step to preemtively adapt to harder conditions if occur smoothly.
	*	This is an easy and cheap way to avoid restarting future successful time steps due to predictable load increases.
	*	Adaptive soft decrease is not done as it would require a base bending_stiffness value which is added responsibility to the user.
	*	It's too easy to have an overly soft constraint parametrization that runs into force time restarts too frequently.
	*/
	this->_adjust_constraints_stiffness_and_log(stark, this->soft_constraint_capacity_hardening_point, this->stiffness_soft_multiplier, /* are_positions_set = */ true);
	this->_log_joint_metrics(stark);
	this->al_outer_iteration = 0;
}

bool stark::EnergyRigidBodyConstraints::_adjust_constraints_stiffness_and_log(core::Stark& stark, double cap, double multiplier, bool are_positions_set)
{
	/*
		This function evaluates all the constraints and serves multiple purposes:
			- Hard increase of bending_stiffness within the newton step
			- Soft increase bending_stiffness at the end of a successful time step
			- Log the state of constraints
	*/

	const double dt = stark.dt;

	// Get correct points and directions according to `are_positions_set`
	auto get_x1 = [&](int rb, Eigen::Vector3d& loc) { return (are_positions_set) ? this->rb->get_position_at(rb, loc) : this->rb->get_x1(rb, loc, dt); };
	auto get_d1 = [&](int rb, Eigen::Vector3d& loc) { return (are_positions_set) ? this->rb->get_direction(rb, loc) : this->rb->get_d1(rb, loc, dt); };

	// Set true and make false if a constraint is violated beyond tolerance
	bool is_valid = true;
	int hardening_events = 0;
	int active_constraints = 0;
	int violated_constraints = 0;
	double max_joint_error_l2 = 0.0;
	double max_joint_error_deg = 0.0;

	auto maybe_harden = [&](bool violated, double& stiffness)
	{
		if (!violated) {
			return;
		}
		is_valid = false;
		violated_constraints++;
		if (multiplier > 1.0) {
			stiffness *= multiplier;
			hardening_events++;
		}
	};

	// Global Points
	for (int i = 0; i < (int)this->global_points->conn.size(); i++) {
		auto& data = this->global_points;
		auto [idx, a] = data->conn[i];
		if (data->is_active[idx] <= 0.0) { continue; }
		active_constraints++;

		const Eigen::Vector3d p = get_x1(a, data->loc[idx]);
		auto [C, f] = RigidBodyConstraints::GlobalPoints::violation_in_m_and_force(data->stiffness[idx], data->target_glob[idx], p);
		max_joint_error_l2 = std::max(max_joint_error_l2, C);
		maybe_harden(C > cap * data->tolerance_in_m[idx], data->stiffness[idx]);
	}

	// Global Directions
	for (int i = 0; i < (int)this->global_directions->conn.size(); i++) {
		auto& data = this->global_directions;
		auto [idx, a] = data->conn[i];
		if (data->is_active[idx] <= 0.0) { continue; }
		active_constraints++;

		const Eigen::Vector3d d = get_d1(a, data->d_loc[idx]);
		auto [C, t] = RigidBodyConstraints::GlobalDirections::violation_in_deg_and_torque(data->stiffness[idx], data->target_d_glob[idx], d);
		max_joint_error_deg = std::max(max_joint_error_deg, C);
		maybe_harden(C > cap * data->tolerance_in_deg[idx], data->stiffness[idx]);
	}

	// Points
	for (int i = 0; i < (int)this->points->conn.size(); i++) {
		auto& data = this->points;
		auto [idx, a, b] = data->conn[i];
		if (data->is_active[idx] <= 0.0) { continue; }
		active_constraints++;

		const Eigen::Vector3d a1 = get_x1(a, data->a_loc[idx]);
		const Eigen::Vector3d b1 = get_x1(b, data->b_loc[idx]);
		auto [C, f] = RigidBodyConstraints::Points::violation_in_m_and_force(data->stiffness[idx], a1, b1);
		max_joint_error_l2 = std::max(max_joint_error_l2, C);
		maybe_harden(C > cap * data->tolerance_in_m[idx], data->stiffness[idx]);
	}

	// PointOnAxis
	for (int i = 0; i < (int)this->point_on_axes->conn.size(); i++) {
		auto& data = this->point_on_axes;
		auto [idx, a, b] = data->conn[i];
		if (data->is_active[idx] <= 0.0) { continue; }
		active_constraints++;

		const Eigen::Vector3d a1 = get_x1(a, data->a_loc[idx]);
		const Eigen::Vector3d da1 = get_d1(a, data->da_loc[idx]);
		const Eigen::Vector3d b1 = get_x1(b, data->b_loc[idx]);
		auto [C, f] = RigidBodyConstraints::PointOnAxes::violation_in_m_and_force(data->stiffness[idx], a1, da1, b1);
		max_joint_error_l2 = std::max(max_joint_error_l2, C);
		maybe_harden(C > cap * data->tolerance_in_m[idx], data->stiffness[idx]);
	}

	// Distance
	for (int i = 0; i < (int)this->distances->conn.size(); i++) {
		auto& data = this->distances;
		auto [idx, a, b] = data->conn[i];
		if (data->is_active[idx] <= 0.0) { continue; }
		active_constraints++;

		const Eigen::Vector3d a1 = get_x1(a, data->a_loc[idx]);
		const Eigen::Vector3d b1 = get_x1(b, data->b_loc[idx]);
		auto [C, f] = RigidBodyConstraints::Distance::signed_violation_in_m_and_force(data->stiffness[idx], a1, b1, data->target_distance[idx]);
		max_joint_error_l2 = std::max(max_joint_error_l2, std::abs(C));
		maybe_harden(std::abs(C) > cap * data->tolerance_in_m[idx], data->stiffness[idx]);
	}

	// Distance Limits
	for (int i = 0; i < (int)this->distance_limits->conn.size(); i++) {
		auto& data = this->distance_limits;
		auto [idx, a, b] = data->conn[i];
		if (data->is_active[idx] <= 0.0) { continue; }
		active_constraints++;

		const Eigen::Vector3d a1 = get_x1(a, data->a_loc[idx]);
		const Eigen::Vector3d b1 = get_x1(b, data->b_loc[idx]);
		auto [C, f] = RigidBodyConstraints::DistanceLimits::signed_violation_in_m_and_force(data->stiffness[idx], a1, b1, data->min_distance[idx], data->max_distance[idx]);
		max_joint_error_l2 = std::max(max_joint_error_l2, std::abs(C));
		maybe_harden(std::abs(C) > cap * data->tolerance_in_m[idx], data->stiffness[idx]);
	}

	// Directions
	for (int i = 0; i < (int)this->directions->conn.size(); i++) {
		auto& data = this->directions;
		auto [idx, a, b] = data->conn[i];
		if (data->is_active[idx] <= 0.0) { continue; }
		active_constraints++;

		const Eigen::Vector3d da = get_d1(a, data->da_loc[idx]);
		const Eigen::Vector3d db = get_d1(b, data->db_loc[idx]);
		auto [C, t] = RigidBodyConstraints::Directions::violation_in_deg_and_torque(data->stiffness[idx], da, db);
		max_joint_error_deg = std::max(max_joint_error_deg, C);
		maybe_harden(C > cap * data->tolerance_in_deg[idx], data->stiffness[idx]);
	}

	// Angle Limits
	for (int i = 0; i < (int)this->angle_limits->conn.size(); i++) {
		auto& data = this->angle_limits;
		auto [idx, a, b] = data->conn[i];
		if (data->is_active[idx] <= 0.0) { continue; }
		active_constraints++;

		const Eigen::Vector3d da = get_d1(a, data->da_loc[idx]);
		const Eigen::Vector3d db = get_d1(b, data->db_loc[idx]);
		auto [C, t] = RigidBodyConstraints::AngleLimits::violation_in_deg_and_torque(data->stiffness[idx], da, db, data->max_distance[idx]);
		max_joint_error_deg = std::max(max_joint_error_deg, C);
		maybe_harden(C > cap * data->tolerance_in_deg[idx], data->stiffness[idx]);
	}

	this->last_joint_error_max_l2 = max_joint_error_l2;
	this->last_joint_error_max_deg = max_joint_error_deg;
	this->last_joint_active_constraints = active_constraints;
	this->last_joint_violated_constraints = violated_constraints;

	if (hardening_events > 0) {
		if (are_positions_set) {
			stark.logger.add_to_counter("joint_soft_hardening_count", hardening_events);
		}
		else {
			stark.logger.add_to_counter("hardening_count", hardening_events);
			stark.logger.add_to_counter("joint_hardening_count", hardening_events);
		}
	}

	return is_valid;
}
