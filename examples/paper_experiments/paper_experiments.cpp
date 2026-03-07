#include "paper_experiments.h"
#include <stark>
#include "../paths.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <memory>
#include <stdexcept>

namespace {
    const std::string kCodegenDir = "e:/workspace/stark/codegen";
    const std::string kOutputBase = "e:/workspace/stark/output/paper_experiments";
    const std::string kModelsDir = "e:/workspace/stark/models";
    const std::string kChronoDataDir = "E:/Anaconda/envs/chrono-baseline/Library/data";

    bool env_flag(const char* key, bool default_value = false)
    {
        const char* raw = std::getenv(key);
        if (raw == nullptr) {
            return default_value;
        }
        const std::string v(raw);
        return v == "1" || v == "true" || v == "TRUE" || v == "on" || v == "ON";
    }

    double env_double(const char* key, double default_value)
    {
        const char* raw = std::getenv(key);
        if (raw == nullptr) {
            return default_value;
        }
        const double parsed = std::atof(raw);
        return std::isfinite(parsed) ? parsed : default_value;
    }

    int env_int(const char* key, int default_value)
    {
        const char* raw = std::getenv(key);
        if (raw == nullptr) {
            return default_value;
        }
        const int parsed = std::atoi(raw);
        return (parsed > 0) ? parsed : default_value;
    }

    int env_int_any(const char* key, int default_value)
    {
        const char* raw = std::getenv(key);
        if (raw == nullptr) {
            return default_value;
        }
        return std::atoi(raw);
    }

    std::string env_string(const char* key, const std::string& default_value)
    {
        const char* raw = std::getenv(key);
        if (raw == nullptr || raw[0] == '\0') {
            return default_value;
        }
        return std::string(raw);
    }

    bool configure_joint_al_from_env(stark::Simulation& sim)
    {
        const bool enabled = env_flag("STARK_JOINT_AL_ENABLED", false);
        if (!enabled) {
            return false;
        }

        stark::EnergyRigidBodyConstraints::AugmentedLagrangianParams params;
        params.enabled = true;
        params.lagged_local_mode = env_flag("STARK_JOINT_AL_LAGGED_LOCAL_MODE", false);
        params.post_corrector_enabled = env_flag("STARK_JOINT_AL_POST_CORRECTOR_ENABLED", true);
        if (params.post_corrector_enabled) {
            params.lagged_local_mode = false;
        }
        params.adaptive_rho = env_flag("STARK_JOINT_AL_ADAPTIVE_RHO", true);
        params.rho0 = env_double("STARK_JOINT_AL_RHO0", 0.0);
        params.rho_update_ratio = env_double("STARK_JOINT_AL_RHO_UPDATE_RATIO", 1.5);
        params.sufficient_decrease_ratio = env_double("STARK_JOINT_AL_SUFFICIENT_DECREASE_RATIO", 0.9);
        params.max_outer_iterations = env_int("STARK_JOINT_AL_MAX_OUTER", 8);
        params.residual_smoothing = env_double("STARK_JOINT_AL_RESIDUAL_SMOOTHING", 1e-4);
        params.post_corrector_max_iterations = env_int("STARK_JOINT_AL_POST_CORRECTOR_MAX_ITERS", 3);
        params.post_corrector_relaxation = env_double("STARK_JOINT_AL_POST_CORRECTOR_RELAXATION", 0.8);
        params.post_corrector_target_tolerance_ratio = env_double("STARK_JOINT_AL_POST_CORRECTOR_TARGET_TOL_RATIO", 0.1);
        params.post_corrector_required_reduction_ratio = env_double("STARK_JOINT_AL_POST_CORRECTOR_REQUIRED_REDUCTION_RATIO", 1e-3);
        params.post_corrector_contact_pairs_threshold = env_int_any("STARK_JOINT_AL_POST_CORRECTOR_CONTACT_PAIRS_THRESHOLD", 0);
        params.post_corrector_min_gap_threshold = env_double("STARK_JOINT_AL_POST_CORRECTOR_MIN_GAP_THRESHOLD", 0.0);

        sim.rigidbodies->set_joint_augmented_lagrangian_params(params);
        return true;
    }

    void configure_solver_from_env(stark::Settings& settings)
    {
        settings.newton.residual.tolerance = env_double("STARK_NEWTON_TOL", settings.newton.residual.tolerance);
        settings.newton.cg_tolerance_multiplier = env_double("STARK_LINEAR_TOL", settings.newton.cg_tolerance_multiplier);
        settings.newton.cg_max_iterations_multiplier = env_double("STARK_CG_MAX_IT_MUL", settings.newton.cg_max_iterations_multiplier);
        const std::string preconditioner = env_string("STARK_NEWTON_PRECONDITIONER", "block_diagonal");
        if (preconditioner == "diag" || preconditioner == "diagonal" || preconditioner == "DIAGONAL") {
            settings.newton.preconditioner = stark::NewtonPreconditioner::Diagonal;
        }
        else {
            settings.newton.preconditioner = stark::NewtonPreconditioner::BlockDiagonal;
        }
    }

    std::tuple<std::vector<Eigen::Vector3d>, std::vector<std::array<int, 3>>> load_and_merge_obj_mesh(const std::string& path, double scale_factor)
    {
        const std::vector<stark::Mesh<3>> parts = stark::load_obj(path);
        std::vector<Eigen::Vector3d> vertices;
        std::vector<std::array<int, 3>> triangles;

        for (const auto& part : parts) {
            const int offset = (int)vertices.size();
            vertices.insert(vertices.end(), part.vertices.begin(), part.vertices.end());
            triangles.reserve(triangles.size() + part.conn.size());
            for (const auto& tri : part.conn) {
                triangles.push_back({ tri[0] + offset, tri[1] + offset, tri[2] + offset });
            }
        }

        std::vector<Eigen::Vector3d> clean_vertices;
        std::vector<std::array<int, 3>> clean_triangles;
        stark::clean_triangle_mesh(clean_vertices, clean_triangles, vertices, triangles, 0.0);

        std::vector<std::array<int, 3>> filtered_triangles;
        filtered_triangles.reserve(clean_triangles.size());
        for (const auto& tri : clean_triangles) {
            const double area = stark::triangle_area(clean_vertices[tri[0]], clean_vertices[tri[1]], clean_vertices[tri[2]]);
            if (area > 1e-14) {
                filtered_triangles.push_back(tri);
            }
        }

        stark::scale(clean_vertices, scale_factor);
        return { clean_vertices, filtered_triangles };
    }

    Eigen::Vector3d compute_aabb_size(const std::vector<Eigen::Vector3d>& vertices)
    {
        Eigen::Vector3d min_v = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
        Eigen::Vector3d max_v = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
        for (const auto& v : vertices) {
            min_v = min_v.cwiseMin(v);
            max_v = max_v.cwiseMax(v);
        }
        return (max_v - min_v).cwiseMax(Eigen::Vector3d::Constant(1e-6));
    }

    std::pair<Eigen::Vector3d, Eigen::Vector3d> compute_aabb_bounds(const std::vector<Eigen::Vector3d>& vertices)
    {
        Eigen::Vector3d min_v = Eigen::Vector3d::Constant(std::numeric_limits<double>::max());
        Eigen::Vector3d max_v = Eigen::Vector3d::Constant(std::numeric_limits<double>::lowest());
        for (const auto& v : vertices) {
            min_v = min_v.cwiseMin(v);
            max_v = max_v.cwiseMax(v);
        }
        return { min_v, max_v };
    }

    std::string resolve_model_path(const std::string& relative_path)
    {
        std::vector<std::string> candidates;
        const std::string override_models_dir = env_string("STARK_MODELS_DIR", "");
        const std::string override_chrono_data = env_string("STARK_CHRONO_DATA_DIR", "");

        if (!override_models_dir.empty()) {
            candidates.push_back(override_models_dir + "/" + relative_path);
        }
        candidates.push_back(kModelsDir + "/" + relative_path);

        if (!override_chrono_data.empty()) {
            candidates.push_back(override_chrono_data + "/models/" + relative_path);
        }
        candidates.push_back(kChronoDataDir + "/models/" + relative_path);

        for (const auto& candidate : candidates) {
            std::ifstream f(candidate);
            if (f.good()) {
                return candidate;
            }
        }

        throw std::runtime_error("Missing model file: " + relative_path);
    }

    struct LocalizedMeshData
    {
        std::vector<Eigen::Vector3d> vertices;
        std::vector<std::array<int, 3>> triangles;
        Eigen::Vector3d min_v = Eigen::Vector3d::Zero();
        Eigen::Vector3d max_v = Eigen::Vector3d::Zero();
        Eigen::Vector3d aabb_size = Eigen::Vector3d::Ones();
    };

    LocalizedMeshData load_localized_obj_mesh(const std::string& path, double scale_factor, const Eigen::Vector3d& reference_point)
    {
        auto [vertices, triangles] = load_and_merge_obj_mesh(path, scale_factor);
        const Eigen::Vector3d ref = reference_point * scale_factor;
        for (auto& v : vertices) {
            v -= ref;
        }
        const auto [min_v, max_v] = compute_aabb_bounds(vertices);

        LocalizedMeshData out;
        out.vertices = std::move(vertices);
        out.triangles = std::move(triangles);
        out.min_v = min_v;
        out.max_v = max_v;
        out.aabb_size = (max_v - min_v).cwiseMax(Eigen::Vector3d::Constant(1e-6));
        return out;
    }

    stark::RigidBody::Handler add_obj_rigidbody(
        stark::Simulation& sim,
        const std::string& output_label,
        const std::string& obj_path,
        double density,
        double scale_factor,
        const stark::ContactParams& contact_params)
    {
        auto [vertices, triangles] = load_and_merge_obj_mesh(obj_path, scale_factor);
        const Eigen::Vector3d aabb_size = compute_aabb_size(vertices);
        const double approx_volume = aabb_size.x() * aabb_size.y() * aabb_size.z();
        const double mass = std::max(1e-4, density * approx_volume);
        const Eigen::Matrix3d inertia = stark::inertia_tensor_box(mass, aabb_size);
        return sim.presets->rigidbodies->add(output_label, mass, inertia, vertices, triangles, contact_params);
    }

    stark::RigidBody::Handler add_localized_obj_rigidbody(
        stark::Simulation& sim,
        const std::string& output_label,
        const std::string& obj_path,
        double mass,
        const Eigen::Vector3d& reference_point,
        const stark::ContactParams& contact_params,
        double scale_factor = 1.0)
    {
        const auto mesh = load_localized_obj_mesh(obj_path, scale_factor, reference_point);
        const Eigen::Matrix3d inertia = stark::inertia_tensor_box(mass, mesh.aabb_size);
        auto handler = sim.presets->rigidbodies->add(output_label, mass, inertia, mesh.vertices, mesh.triangles, contact_params);
        handler.rigidbody.set_translation(reference_point * scale_factor);
        return handler;
    }

    void append_mesh(
        std::vector<Eigen::Vector3d>& vertices,
        std::vector<std::array<int, 3>>& triangles,
        const stark::Mesh<3>& mesh,
        const Eigen::Vector3d& translation)
    {
        const int offset = (int)vertices.size();
        vertices.reserve(vertices.size() + mesh.vertices.size());
        triangles.reserve(triangles.size() + mesh.conn.size());
        for (const auto& v : mesh.vertices) {
            vertices.push_back(v + translation);
        }
        for (const auto& tri : mesh.conn) {
            triangles.push_back({ tri[0] + offset, tri[1] + offset, tri[2] + offset });
        }
    }

    stark::RigidBody::Handler add_compound_box_rigidbody(
        stark::Simulation& sim,
        const std::string& output_label,
        double mass,
        const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& size_and_center_local,
        const Eigen::Vector3d& reference_point,
        const stark::ContactParams& contact_params)
    {
        std::vector<Eigen::Vector3d> vertices;
        std::vector<std::array<int, 3>> triangles;
        for (const auto& part : size_and_center_local) {
            const stark::Mesh<3> box_mesh = stark::make_box(part.first);
            append_mesh(vertices, triangles, box_mesh, part.second);
        }
        const Eigen::Matrix3d inertia = stark::inertia_tensor_box(mass, compute_aabb_size(vertices));
        auto handler = sim.presets->rigidbodies->add(output_label, mass, inertia, vertices, triangles, contact_params);
        handler.rigidbody.set_translation(reference_point);
        return handler;
    }

    Eigen::Vector3d make_planar_point(double x, double y)
    {
        return { x, y, 0.0 };
    }

    Eigen::Vector3d rotate_planar_vector(double angle_rad, double length)
    {
        return { length * std::cos(angle_rad), length * std::sin(angle_rad), 0.0 };
    }

    std::pair<Eigen::Vector3d, Eigen::Vector3d> circle_circle_intersection_xy(
        const Eigen::Vector3d& c0,
        double r0,
        const Eigen::Vector3d& c1,
        double r1)
    {
        const Eigen::Vector3d d = c1 - c0;
        const double dist = d.head<2>().norm();
        if (!(std::isfinite(dist) && dist > 1e-12)) {
            throw std::runtime_error("Degenerate four-bar geometry: coincident circle centers.");
        }
        if (dist > r0 + r1 || dist < std::abs(r0 - r1)) {
            throw std::runtime_error("Invalid four-bar geometry: circle intersection does not exist.");
        }

        const double a = (r0 * r0 - r1 * r1 + dist * dist) / (2.0 * dist);
        const double h_sq = std::max(0.0, r0 * r0 - a * a);
        const double h = std::sqrt(h_sq);

        const Eigen::Vector2d ex = d.head<2>() / dist;
        const Eigen::Vector2d ey(-ex.y(), ex.x());
        const Eigen::Vector2d p2 = c0.head<2>() + a * ex;

        const Eigen::Vector2d upper = p2 + h * ey;
        const Eigen::Vector2d lower = p2 - h * ey;
        return { { upper.x(), upper.y(), 0.0 }, { lower.x(), lower.y(), 0.0 } };
    }

    double planar_angle_deg(const Eigen::Vector3d& from, const Eigen::Vector3d& to)
    {
        const Eigen::Vector3d d = to - from;
        return std::atan2(d.y(), d.x()) * 180.0 / M_PI;
    }

    struct JointDriftSample
    {
        stark::RigidBodyHandler body_a;
        stark::RigidBodyHandler body_b;
        Eigen::Vector3d local_point_a = Eigen::Vector3d::Zero();
        Eigen::Vector3d local_point_b = Eigen::Vector3d::Zero();
    };
}

// --- Exp 1: Collision-Free & Stiffness Update ---
void run_exp1_variation(const std::string& name, bool stiffness_update, double min_stiffness, bool adaptive_scheduling = false, bool inertia_consistent = false, double mass_ratio = 100.0)
{
    stark::Settings settings;
    settings.output.simulation_name = "exp1_" + name;
    settings.output.output_directory = kOutputBase + "/exp1_" + name;
    settings.output.codegen_directory = kCodegenDir;
    settings.debug.symx_suppress_compiler_output = false;
    settings.debug.symx_force_load = false;
    settings.execution.end_simulation_time = 3.0; 
    settings.simulation.init_frictional_contact = true;
    
    stark::Simulation sim(settings);

    auto c_params = sim.interactions->contact->get_global_params();
    c_params.default_contact_thickness = 0.005;
    c_params.stiffness_update_enabled = stiffness_update;
    c_params.min_contact_stiffness = min_stiffness;
    c_params.adaptive_stiffness_scheduling = adaptive_scheduling;
    c_params.inertia_consistent_kappa = inertia_consistent;
    sim.interactions->contact->set_global_params(c_params);

    stark::EnergyFrictionalContact::Params contact_params;
    contact_params.contact_thickness = 0.005; 
    auto ground = sim.presets->rigidbodies->add_box("ground", 1.0, {10.0, 10.0, 0.1}, contact_params);
    ground.handler.rigidbody.set_translation({0, 0, -0.05});
    sim.rigidbodies->add_constraint_fix(ground.handler.rigidbody);

    std::vector<stark::RigidBodyHandler> rbs;
    for(int i = 0; i < 20; ++i) {
        double mass = (i % 5 == 0) ? mass_ratio : 1.0; 
        auto box = sim.presets->rigidbodies->add_box("box_" + std::to_string(i), mass, 0.2, contact_params);
        const double x = 0.35 * (i % 5) - 0.7;
        const double y = 0.35 * ((i / 5) % 4) - 0.525;
        const double z = 0.35 + 0.30 * i;
        box.handler.rigidbody.set_translation({x, y, z});
        box.handler.rigidbody.set_rotation(15.0 * i, {1.0, 1.0, 0.0});
        rbs.push_back(box.handler.rigidbody);
    }

    {
        std::string dir = sim.get_settings().output.output_directory;
        std::ofstream f_min(dir + "/min_z.csv", std::ios::trunc);
        f_min << "t,min_z\n";

        std::ofstream f_centers(dir + "/centers_z.csv", std::ios::trunc);
        f_centers << "t";
        for (int i = 0; i < 20; ++i) {
            f_centers << ",z_" << i;
        }
        f_centers << "\n";
    }

    sim.add_time_event(0.0, 3.0, [&sim, rbs](double t) {
        std::string dir = sim.get_settings().output.output_directory;
        std::ofstream f(dir + "/min_z.csv", std::ios::app);
        std::ofstream f_centers(dir + "/centers_z.csv", std::ios::app);

        double min_z = 1000.0;
        f_centers << t;
        for (auto& rb : rbs) {
            Eigen::Vector3d pos = rb.get_translation();
            f_centers << "," << pos.z();
            min_z = std::min(min_z, pos.z() - 0.1); // approx minus half-size
        }
        f_centers << "\n";
        f << t << "," << min_z << "\n";
    });

    sim.run();
}

void exp1_collision_free_and_stiffness() {
    std::cout << "Running Exp 1: Collision-Free & Stiffness Update..." << std::endl;
    run_exp1_variation("adaptive", true, 1e3);
    run_exp1_variation("gap_adaptive", true, 1e3, true);  // gap-ratio-driven adaptive scheduling
    run_exp1_variation("fixed_soft", false, 1e4);
    run_exp1_variation("fixed_stiff", false, 1e9);
}

void exp1_adaptive_only() {
    std::cout << "Running Exp 1: adaptive only..." << std::endl;
    run_exp1_variation("adaptive", true, 1e3);
}

void exp1_gap_adaptive_only() {
    std::cout << "Running Exp 1: gap-ratio adaptive only..." << std::endl;
    run_exp1_variation("gap_adaptive", true, 1e3, true);
}

void exp1_mass_adaptive_only() {
    std::cout << "Running Exp 1: mass+gap adaptive..." << std::endl;
    run_exp1_variation("mass_adaptive", true, 1e3, true, true);
}

void exp1_fixed_soft_only() {
    std::cout << "Running Exp 1: fixed soft..." << std::endl;
    run_exp1_variation("fixed_soft", false, 1e4);
}

void exp1_mass_ratio_sweep() {
    std::cout << "Running Exp 1: mass ratio sweep..." << std::endl;
    const std::vector<double> ratios = {1.0, 10.0, 100.0, 1000.0};
    for (double r : ratios) {
        std::string tag = "mr" + std::to_string((int)r);
        // Full method (inertia init + gap-ratio scheduling)
        std::cout << "  mass_ratio=" << r << " full..." << std::endl;
        run_exp1_variation(tag + "_full", true, 1e3, true, true, r);
        // Fixed-rate baseline (x2 / x0.99)
        std::cout << "  mass_ratio=" << r << " fixed_rate..." << std::endl;
        run_exp1_variation(tag + "_fixed_rate", true, 1e3, false, false, r);
    }
}

// --- Exp 2: High-Speed Impact ---
void run_exp2_variation(double v0) {
    stark::Settings settings;
    settings.output.simulation_name = "exp2_v" + std::to_string((int)v0);
    settings.output.output_directory = kOutputBase + "/exp2_v" + std::to_string((int)v0);
    settings.output.codegen_directory = kCodegenDir;
    settings.debug.symx_suppress_compiler_output = false;
    settings.debug.symx_force_load = false;
settings.execution.end_simulation_time = 0.05; // Short time is enough
    settings.simulation.init_frictional_contact = true;
    settings.simulation.max_time_step_size = 0.001; // Need fine steps for high speed
    
    stark::Simulation sim(settings);

    auto c_params = sim.interactions->contact->get_global_params();
    c_params.default_contact_thickness = 0.005;
    sim.interactions->contact->set_global_params(c_params);

    stark::EnergyFrictionalContact::Params contact_params;
    contact_params.contact_thickness = 0.005;

    auto wall = sim.presets->rigidbodies->add_box("wall", 1.0, {1.0, 5.0, 5.0}, contact_params);
    wall.handler.rigidbody.set_translation({2.0, 0.0, 2.5});
    sim.rigidbodies->add_constraint_fix(wall.handler.rigidbody);

    auto bullet = sim.presets->rigidbodies->add_box("bullet", 1.0, 0.2, contact_params);
    bullet.handler.rigidbody.set_translation({0.0, 0.0, 2.5});
    bullet.handler.rigidbody.set_velocity({v0, 0.0, 0.0});

    {
        std::string dir = sim.get_settings().output.output_directory;
        std::ofstream f(dir + "/impact_state.csv", std::ios::app);
        f << "t,v0,x,y,z,vx,vy,vz\n";
        const Eigen::Vector3d x = bullet.handler.rigidbody.get_translation();
        const Eigen::Vector3d v = bullet.handler.rigidbody.get_velocity();
        f << 0.0 << "," << v0 << "," << x[0] << "," << x[1] << "," << x[2] << "," << v[0] << "," << v[1] << "," << v[2] << "\n";
    }

    sim.add_time_event(0.0, 0.1, [&sim, rb = bullet.handler.rigidbody, v0](double t) {
        std::string dir = sim.get_settings().output.output_directory;
        std::ofstream f(dir + "/impact_state.csv", std::ios::app);
        const Eigen::Vector3d x = rb.get_translation();
        const Eigen::Vector3d v = rb.get_velocity();
        f << t << "," << v0 << "," << x[0] << "," << x[1] << "," << x[2] << "," << v[0] << "," << v[1] << "," << v[2] << "\n";
    });

    sim.run();	
}

void exp2_high_speed_impact() {
    std::cout << "Running Exp 2: High-Speed Impact..." << std::endl;
    run_exp2_variation(10.0);
    run_exp2_variation(100.0);
    run_exp2_variation(500.0);
}

void exp2_crank_slider_impact()
{
    const bool joint_al_enabled = env_flag("STARK_JOINT_AL_ENABLED", false);
    const std::string default_run_name = joint_al_enabled ? "exp2_crank_slider_al" : "exp2_crank_slider_soft";
    const std::string run_name = env_string("STARK_EXP2_RUN_NAME", default_run_name);

    const double joint_stiffness = env_double("STARK_EXP2_JOINT_STIFFNESS", 1e6);
    const double joint_tol_m = env_double("STARK_EXP2_JOINT_TOL_M", 1e-4);
    const double joint_tol_deg = env_double("STARK_EXP2_JOINT_TOL_DEG", 0.1);
    const bool guide_al_enabled = env_flag("STARK_EXP2_GUIDE_AL_ENABLED", false);
    const double dt = env_double("STARK_EXP2_DT", 0.002);
    const double end_time = env_double("STARK_EXP2_END_TIME", 1.2);
    const double motor_w = env_double("STARK_EXP2_MOTOR_W", -3.0);
    const double motor_max_torque = env_double("STARK_EXP2_MOTOR_MAX_TORQUE", 50.0);

    stark::Settings settings;
    settings.output.simulation_name = run_name;
    settings.output.output_directory = kOutputBase + "/" + run_name;
    settings.output.codegen_directory = kCodegenDir;
    settings.debug.symx_suppress_compiler_output = false;
    settings.debug.symx_force_load = false;
    settings.execution.end_simulation_time = end_time;
    settings.simulation.gravity = { 0.0, -9.81, 0.0 };
    settings.simulation.init_frictional_contact = true;
    settings.simulation.use_adaptive_time_step = false;
    settings.simulation.max_time_step_size = dt;
    configure_solver_from_env(settings);

    stark::Simulation sim(settings);
    if (joint_al_enabled) {
        configure_joint_al_from_env(sim);
    }
    sim.rigidbodies->set_default_constraint_stiffness(joint_stiffness);
    sim.rigidbodies->set_default_constraint_distance_tolerance(joint_tol_m);
    sim.rigidbodies->set_default_constraint_angle_tolerance(joint_tol_deg);

    auto c_params = sim.interactions->contact->get_global_params();
    c_params.default_contact_thickness = env_double("STARK_EXP2_CONTACT_THICKNESS", 1e-3);
    sim.interactions->contact->set_global_params(c_params);

    const double link_width = 0.05;
    const double link_thickness = 0.05;
    const double link_mass = 1.0;
    const double crank_length = env_double("STARK_EXP2_CRANK_LENGTH", 0.35);
    const double coupler_length = env_double("STARK_EXP2_COUPLER_LENGTH", 0.85);
    const double crank_geom_length = 0.90 * crank_length;
    const double coupler_geom_length = 0.90 * coupler_length;
    const double guide_y = env_double("STARK_EXP2_GUIDE_Y", 0.0);
    const double support_y = env_double("STARK_EXP2_SUPPORT_Y", 0.0);
    const double stop_face_x = env_double("STARK_EXP2_STOP_FACE_X", 1.28);
    const double slider_length = env_double("STARK_EXP2_SLIDER_LENGTH", 0.24);
    const double slider_height = env_double("STARK_EXP2_SLIDER_HEIGHT", 0.18);
    const double crank_angle_deg = env_double("STARK_EXP2_CRANK_ANGLE_DEG", 60.0);
    const double crank_angle_rad = crank_angle_deg * M_PI / 180.0;

    const Eigen::Vector3d A = make_planar_point(0.0, support_y);
    const Eigen::Vector3d B = A + rotate_planar_vector(crank_angle_rad, crank_length);
    const double dy = guide_y - B.y();
    const double dx_sq = coupler_length * coupler_length - dy * dy;
    if (dx_sq <= 0.0) {
        throw std::runtime_error("Invalid crank-slider geometry: coupler too short for guide line.");
    }
    const Eigen::Vector3d pin_C = make_planar_point(B.x() + std::sqrt(dx_sq), guide_y);
    const Eigen::Vector3d slider_center = pin_C + make_planar_point(0.5 * slider_length, 0.0);

    auto support = sim.presets->rigidbodies->add_box("support", 1.0, {0.12, 0.12, 0.12});
    support.handler.rigidbody.set_translation({ A.x(), A.y() - 0.8, 0.0 });
    sim.rigidbodies->add_constraint_fix(support.handler.rigidbody).set_label("support_fix");

    const double stop_thickness = 0.08;
    auto stop = sim.presets->rigidbodies->add_box("stop", 1.0, {stop_thickness, 0.6, 0.12});
    stop.handler.rigidbody.set_translation({ stop_face_x + 0.5 * stop_thickness, guide_y, 0.0 });
    sim.rigidbodies->add_constraint_fix(stop.handler.rigidbody).set_label("stop_fix");

    auto crank = sim.presets->rigidbodies->add_box("crank", link_mass, {crank_geom_length, link_width, link_thickness});
    crank.handler.rigidbody.set_translation(0.5 * (A + B));
    crank.handler.rigidbody.set_rotation(planar_angle_deg(A, B), Eigen::Vector3d::UnitZ());

    auto coupler = sim.presets->rigidbodies->add_box("coupler", link_mass, {coupler_geom_length, link_width, link_thickness});
    coupler.handler.rigidbody.set_translation(0.5 * (B + pin_C));
    coupler.handler.rigidbody.set_rotation(planar_angle_deg(B, pin_C), Eigen::Vector3d::UnitZ());

    auto slider = sim.presets->rigidbodies->add_box("slider", 1.2, {slider_length, slider_height, 0.12});
    slider.handler.rigidbody.set_translation(slider_center);

    auto support_motor = sim.rigidbodies->add_constraint_motor(
        support.handler.rigidbody,
        crank.handler.rigidbody,
        A,
        Eigen::Vector3d::UnitZ(),
        motor_w,
        motor_max_torque);
    support_motor.set_label("support_motor");
    support_motor.set_stiffness(joint_stiffness);
    support_motor.set_tolerance_in_m(joint_tol_m);
    support_motor.set_tolerance_in_deg(joint_tol_deg);

    auto crank_coupler_hinge = sim.rigidbodies->add_constraint_hinge(
        crank.handler.rigidbody,
        coupler.handler.rigidbody,
        B,
        Eigen::Vector3d::UnitZ());
    crank_coupler_hinge.set_label("crank_coupler_hinge");
    crank_coupler_hinge.set_stiffness(joint_stiffness);
    crank_coupler_hinge.set_tolerance_in_m(joint_tol_m);
    crank_coupler_hinge.set_tolerance_in_deg(joint_tol_deg);

    auto coupler_slider_hinge = sim.rigidbodies->add_constraint_hinge(
        coupler.handler.rigidbody,
        slider.handler.rigidbody,
        pin_C,
        Eigen::Vector3d::UnitZ());
    coupler_slider_hinge.set_label("coupler_slider_hinge");
    coupler_slider_hinge.set_stiffness(joint_stiffness);
    coupler_slider_hinge.set_tolerance_in_m(joint_tol_m);
    coupler_slider_hinge.set_tolerance_in_deg(joint_tol_deg);

    auto slider_guide = sim.rigidbodies->add_constraint_prismatic_slider(
        support.handler.rigidbody,
        slider.handler.rigidbody,
        slider_center,
        Eigen::Vector3d::UnitX());
    slider_guide.set_label("slider_guide");
    slider_guide.set_stiffness(joint_stiffness);
    slider_guide.set_tolerance_in_m(joint_tol_m);
    slider_guide.set_tolerance_in_deg(joint_tol_deg);
    slider_guide.set_augmented_lagrangian_enabled(guide_al_enabled);

    auto angle_deg = [](const stark::RigidBodyHandler& rb) {
        const Eigen::Vector3d x_axis = rb.transform_local_to_global_direction(Eigen::Vector3d::UnitX());
        return std::atan2(x_axis.y(), x_axis.x()) * 180.0 / M_PI;
    };

    {
        std::string dir = sim.get_settings().output.output_directory;
        std::ofstream f_state(dir + "/crank_slider_state.csv", std::ios::trunc);
        f_state << "t,slider_cx,slider_cy,slider_vx,slider_vy,crank_theta_deg,coupler_theta_deg,gap_stop\n";

        std::ofstream f_proxy(dir + "/crank_slider_constraint_proxy.csv", std::ios::trunc);
        f_proxy << "t,"
                << "support_point_force_proxy,support_direction_torque_proxy,motor_torque_proxy,"
                << "crank_coupler_point_force_proxy,crank_coupler_direction_torque_proxy,"
                << "coupler_slider_point_force_proxy,coupler_slider_direction_torque_proxy,"
                << "guide_axis_force_proxy,guide_long_direction_torque_proxy,guide_orth_direction_torque_proxy\n";
    }

    sim.add_time_event(
        0.0,
        settings.execution.end_simulation_time,
        [&sim,
         stop_face_x,
         slider_length,
         crank_rb = crank.handler.rigidbody,
         coupler_rb = coupler.handler.rigidbody,
         slider_rb = slider.handler.rigidbody,
         support_motor,
         crank_coupler_hinge,
         coupler_slider_hinge,
         slider_guide,
         angle_deg](double t) mutable {
            std::string dir = sim.get_settings().output.output_directory;
            const Eigen::Vector3d slider_c = slider_rb.get_translation();
            const Eigen::Vector3d slider_v = slider_rb.get_velocity();
            const double gap_stop = stop_face_x - (slider_c.x() + 0.5 * slider_length);

            std::ofstream f_state(dir + "/crank_slider_state.csv", std::ios::app);
            f_state << t << ","
                    << slider_c.x() << "," << slider_c.y() << ","
                    << slider_v.x() << "," << slider_v.y() << ","
                    << angle_deg(crank_rb) << "," << angle_deg(coupler_rb) << ","
                    << gap_stop << "\n";

            std::ofstream f_proxy(dir + "/crank_slider_constraint_proxy.csv", std::ios::app);
            f_proxy << t << ","
                    << support_motor.get_hinge_joint().get_point_constraint().get_violation_in_m_and_force()[1] << ","
                    << support_motor.get_hinge_joint().get_direction_constraint().get_violation_in_deg_and_torque()[1] << ","
                    << support_motor.get_angular_velocity_constraint().get_signed_angular_velocity_violation_in_deg_per_s_and_torque()[1] << ","
                    << crank_coupler_hinge.get_point_constraint().get_violation_in_m_and_force()[1] << ","
                    << crank_coupler_hinge.get_direction_constraint().get_violation_in_deg_and_torque()[1] << ","
                    << coupler_slider_hinge.get_point_constraint().get_violation_in_m_and_force()[1] << ","
                    << coupler_slider_hinge.get_direction_constraint().get_violation_in_deg_and_torque()[1] << ","
                    << slider_guide.get_slider().get_point_on_axis().get_violation_in_m_and_force()[1] << ","
                    << slider_guide.get_slider().get_direction_lock().get_violation_in_deg_and_torque()[1] << ","
                    << slider_guide.get_direction_lock().get_violation_in_deg_and_torque()[1] << "\n";
        });

    sim.run();
}

void exp3_limit_stop_hinge()
{
    const bool joint_al_enabled = env_flag("STARK_JOINT_AL_ENABLED", false);
    const std::string default_run_name = joint_al_enabled ? "exp3_limit_stop_al" : "exp3_limit_stop_soft";
    const std::string run_name = env_string("STARK_EXP3_RUN_NAME", default_run_name);

    const double joint_stiffness = env_double("STARK_EXP3_JOINT_STIFFNESS", 1e6);
    const double joint_tol_m = env_double("STARK_EXP3_JOINT_TOL_M", 1e-4);
    const double joint_tol_deg = env_double("STARK_EXP3_JOINT_TOL_DEG", 0.1);
    const double dt = env_double("STARK_EXP3_DT", 1e-3);
    const double end_time = env_double("STARK_EXP3_END_TIME", 2.0);
    const double limit_angle_deg = env_double("STARK_EXP3_LIMIT_DEG", 35.0);
    const bool stop_projection_enabled = env_flag("STARK_EXP3_STOP_PROJECTION_ENABLED", true);
    const double stop_restitution = env_double("STARK_EXP3_STOP_RESTITUTION", 0.0);
    const double stop_projection_eps_deg = env_double("STARK_EXP3_STOP_PROJECTION_EPS_DEG", 1e-8);

    stark::Settings settings;
    settings.output.simulation_name = run_name;
    settings.output.output_directory = kOutputBase + "/" + run_name;
    settings.output.codegen_directory = kCodegenDir;
    settings.debug.symx_suppress_compiler_output = false;
    settings.debug.symx_force_load = false;
    settings.execution.end_simulation_time = end_time;
    settings.simulation.gravity = { 0.0, -9.81, 0.0 };
    settings.simulation.init_frictional_contact = false;
    settings.simulation.use_adaptive_time_step = false;
    settings.simulation.max_time_step_size = dt;
    configure_solver_from_env(settings);

    stark::Simulation sim(settings);
    if (joint_al_enabled) {
        configure_joint_al_from_env(sim);
    }
    sim.rigidbodies->set_default_constraint_stiffness(joint_stiffness);
    sim.rigidbodies->set_default_constraint_distance_tolerance(joint_tol_m);
    sim.rigidbodies->set_default_constraint_angle_tolerance(joint_tol_deg);

    const double rod_length = env_double("STARK_EXP3_ROD_LENGTH", 1.0);
    const double rod_width = env_double("STARK_EXP3_ROD_WIDTH", 0.06);
    const double rod_thickness = env_double("STARK_EXP3_ROD_THICKNESS", 0.06);
    const double rod_mass = env_double("STARK_EXP3_ROD_MASS", 1.0);
    const Eigen::Vector3d pivot = make_planar_point(0.0, 0.0);

    auto support = sim.presets->rigidbodies->add_box("support", 1.0, { 0.12, 0.12, 0.12 });
    support.handler.rigidbody.set_translation({ -0.06, 0.0, 0.0 });
    sim.rigidbodies->add_constraint_fix(support.handler.rigidbody).set_label("support_fix");

    auto rod = sim.presets->rigidbodies->add_box("rod", rod_mass, { rod_length, rod_width, rod_thickness });
    rod.handler.rigidbody.set_translation(make_planar_point(0.5 * rod_length, 0.0));
    rod.handler.rigidbody.set_rotation(0.0, Eigen::Vector3d::UnitZ());
    rod.handler.rigidbody.set_angular_velocity(Eigen::Vector3d::Zero());

    auto hinge_stop = sim.rigidbodies->add_constraint_hinge_with_angle_limit(
        support.handler.rigidbody,
        rod.handler.rigidbody,
        pivot,
        Eigen::Vector3d::UnitZ(),
        limit_angle_deg);
    hinge_stop.set_label("support_limit_hinge");
    hinge_stop.set_stiffness(joint_stiffness);
    hinge_stop.set_tolerance_in_m(joint_tol_m);
    hinge_stop.set_tolerance_in_deg(joint_tol_deg);

    auto angle_deg = [](const stark::RigidBodyHandler& rb) {
        const Eigen::Vector3d x_axis = rb.transform_local_to_global_direction(Eigen::Vector3d::UnitX());
        return std::atan2(x_axis.y(), x_axis.x()) * 180.0 / M_PI;
    };
    auto center_at_angle = [pivot, rod_length](double theta_deg) {
        const double theta_rad = theta_deg * M_PI / 180.0;
        return pivot + Eigen::Vector3d(
            0.5 * rod_length * std::cos(theta_rad),
            0.5 * rod_length * std::sin(theta_rad),
            0.0);
    };
    auto projected_flag = std::make_shared<int>(0);

    {
        std::string dir = sim.get_settings().output.output_directory;
        std::ofstream f_state(dir + "/limit_stop_state.csv", std::ios::trunc);
        f_state << "t,theta_deg,omega_deg_s,tip_x,tip_y,"
                << "limit_violation_deg,limit_torque_proxy,"
                << "support_point_force_proxy,support_direction_torque_proxy,"
                << "projection_applied\n";
    }

    sim.add_time_event(
        0.0,
        settings.execution.end_simulation_time,
        [rod_rb = rod.handler.rigidbody,
         angle_deg,
          center_at_angle,
          limit_angle_deg,
          pivot,
          stop_projection_enabled,
          stop_restitution,
          stop_projection_eps_deg,
         projected_flag](double t) mutable {
            *projected_flag = 0;
            if (!stop_projection_enabled || t <= 0.0) {
                return;
            }

            const double theta = angle_deg(rod_rb);
            const double omega_deg_s = rod_rb.get_angular_velocity().z() * 180.0 / M_PI;
            const bool hit_negative = theta < -(limit_angle_deg + stop_projection_eps_deg) && omega_deg_s < 0.0;
            const bool hit_positive = theta > +(limit_angle_deg + stop_projection_eps_deg) && omega_deg_s > 0.0;
            if (!hit_negative && !hit_positive) {
                return;
            }

            const double clamped_theta_deg = hit_negative ? -limit_angle_deg : +limit_angle_deg;
            const double projected_omega_deg_s = -stop_restitution * omega_deg_s;
            const double projected_omega_rad_s = projected_omega_deg_s * M_PI / 180.0;
            const Eigen::Vector3d center = center_at_angle(clamped_theta_deg);
            const Eigen::Vector3d r = center - pivot;
            const Eigen::Vector3d projected_velocity(
                -projected_omega_rad_s * r.y(),
                +projected_omega_rad_s * r.x(),
                0.0);

            rod_rb.set_rotation(clamped_theta_deg, Eigen::Vector3d::UnitZ());
            rod_rb.set_translation(center);
            rod_rb.set_velocity(projected_velocity);
            rod_rb.set_angular_velocity({ 0.0, 0.0, projected_omega_rad_s });
            *projected_flag = 1;
        });

    sim.add_time_event(
        0.0,
        settings.execution.end_simulation_time,
        [&sim,
         rod_rb = rod.handler.rigidbody,
         hinge_stop,
         rod_length,
         angle_deg,
         projected_flag](double t) mutable {
            const double theta_deg = angle_deg(rod_rb);
            const double omega_deg_s = rod_rb.get_angular_velocity().z() * 180.0 / M_PI;
            const Eigen::Vector3d tip = rod_rb.transform_local_to_global_point({ 0.5 * rod_length, 0.0, 0.0 });
            const auto limit_metrics = hinge_stop.get_angle_limit().get_violation_in_deg_and_torque();
            const double support_point_force_proxy = hinge_stop.get_hinge_joint().get_point_constraint().get_violation_in_m_and_force()[1];
            const double support_direction_torque_proxy = hinge_stop.get_hinge_joint().get_direction_constraint().get_violation_in_deg_and_torque()[1];

            std::string dir = sim.get_settings().output.output_directory;
            std::ofstream f_state(dir + "/limit_stop_state.csv", std::ios::app);
            f_state << t << ","
                    << theta_deg << ","
                    << omega_deg_s << ","
                    << tip.x() << ","
                    << tip.y() << ","
                    << limit_metrics[0] << ","
                    << limit_metrics[1] << ","
                    << support_point_force_proxy << ","
                    << support_direction_torque_proxy << ","
                    << *projected_flag << "\n";
        });

    sim.run();
}

// --- Exp 4 / Exp 4 Four-Bar: Coupled Joints & Closed Loops ---
namespace {
    void run_exp4_scene(bool use_fourbar_scene)
    {
    const bool joint_al_enabled = env_flag("STARK_JOINT_AL_ENABLED", false);
    const std::string default_run_name = use_fourbar_scene
        ? (joint_al_enabled ? "exp4_fourbar_al" : "exp4_fourbar")
        : (joint_al_enabled ? "exp4_coupled_joints_al" : "exp4_coupled_joints");
    const std::string run_name = env_string("STARK_EXP4_RUN_NAME", default_run_name);
    std::cout << "Running "
              << (use_fourbar_scene ? "Exp 4 Four-Bar: Closed Loop" : "Exp 4: Coupled Joints & Impacts (10-link chain)")
              << std::endl;
    const double exp4_joint_stiffness = env_double("STARK_EXP4_JOINT_STIFFNESS", 1e6);
    const double exp4_joint_tol_m = env_double("STARK_EXP4_JOINT_TOL_M", 1e-4);
    const double exp4_joint_tol_deg = env_double("STARK_EXP4_JOINT_TOL_DEG", 0.1);
    const double exp4_dt = env_double("STARK_EXP4_DT", 0.01);
    const double exp4_end_time = env_double("STARK_EXP4_END_TIME", 3.0);
    stark::Settings settings;
    settings.output.simulation_name = run_name;
    settings.output.output_directory = kOutputBase + "/" + run_name;
    settings.output.codegen_directory = kCodegenDir;
    settings.debug.symx_suppress_compiler_output = false;
    settings.debug.symx_force_load = false;
    settings.execution.end_simulation_time = exp4_end_time;
    settings.simulation.gravity = { 0.0, -9.81, 0.0 };
    settings.simulation.init_frictional_contact = false;
    settings.simulation.use_adaptive_time_step = false;
    settings.simulation.max_time_step_size = exp4_dt;
    configure_solver_from_env(settings);

    stark::Simulation sim(settings);
    if (joint_al_enabled) {
        configure_joint_al_from_env(sim);
    }
    sim.rigidbodies->set_default_constraint_stiffness(exp4_joint_stiffness);
    sim.rigidbodies->set_default_constraint_distance_tolerance(exp4_joint_tol_m);
    sim.rigidbodies->set_default_constraint_angle_tolerance(exp4_joint_tol_deg);

    const std::string joint_drift_run_file =
        "joint_drift_" + settings.output.simulation_name + "__" + settings.output.time_stamp + ".csv";

    if (use_fourbar_scene) {
        const double link_width = 0.06;
        const double link_thickness = 0.06;
        const double link_mass = 1.0;
        const double crank_length = 0.55;
        const double coupler_length = 1.10;
        const double rocker_length = 0.95;
        const double ground_span = 1.40;
        const double crank_angle_deg = env_double("STARK_EXP4_CRANK_ANGLE_DEG", 55.0);
        const double crank_angle_rad = crank_angle_deg * M_PI / 180.0;

        const Eigen::Vector3d A = make_planar_point(0.0, 0.0);
        const Eigen::Vector3d D = make_planar_point(ground_span, 0.0);
        const Eigen::Vector3d B = A + rotate_planar_vector(crank_angle_rad, crank_length);
        auto [C_upper, C_lower] = circle_circle_intersection_xy(B, coupler_length, D, rocker_length);
        const Eigen::Vector3d C = (C_upper.y() >= C_lower.y()) ? C_upper : C_lower;

        auto ground = sim.presets->rigidbodies->add_box("ground", 1.0, {ground_span + 0.2, 0.12, 0.12});
        ground.handler.rigidbody.set_translation({0.5 * ground_span, 0.0, 0.0});
        sim.rigidbodies->add_constraint_fix(ground.handler.rigidbody).set_label("ground_fix");
        const auto ground_rb = ground.handler.rigidbody;

        auto crank = sim.presets->rigidbodies->add_box("crank", link_mass, {crank_length, link_width, link_thickness});
        crank.handler.rigidbody.set_translation(0.5 * (A + B));
        crank.handler.rigidbody.set_rotation(planar_angle_deg(A, B), Eigen::Vector3d::UnitZ());

        auto coupler = sim.presets->rigidbodies->add_box("coupler", link_mass, {coupler_length, link_width, link_thickness});
        coupler.handler.rigidbody.set_translation(0.5 * (B + C));
        coupler.handler.rigidbody.set_rotation(planar_angle_deg(B, C), Eigen::Vector3d::UnitZ());

        auto rocker = sim.presets->rigidbodies->add_box("rocker", link_mass, {rocker_length, link_width, link_thickness});
        rocker.handler.rigidbody.set_translation(0.5 * (C + D));
        rocker.handler.rigidbody.set_rotation(planar_angle_deg(D, C), Eigen::Vector3d::UnitZ());

        auto support_hinge = sim.rigidbodies->add_constraint_hinge(
            ground_rb,
            crank.handler.rigidbody,
            A,
            Eigen::Vector3d::UnitZ());
        support_hinge.set_label("support_hinge");
        support_hinge.set_stiffness(exp4_joint_stiffness);
        support_hinge.set_tolerance_in_m(exp4_joint_tol_m);
        support_hinge.set_tolerance_in_deg(exp4_joint_tol_deg);

        auto crank_coupler_hinge = sim.rigidbodies->add_constraint_hinge(
            crank.handler.rigidbody,
            coupler.handler.rigidbody,
            B,
            Eigen::Vector3d::UnitZ());
        crank_coupler_hinge.set_label("crank_coupler_hinge");
        crank_coupler_hinge.set_stiffness(exp4_joint_stiffness);
        crank_coupler_hinge.set_tolerance_in_m(exp4_joint_tol_m);
        crank_coupler_hinge.set_tolerance_in_deg(exp4_joint_tol_deg);

        auto coupler_rocker_hinge = sim.rigidbodies->add_constraint_hinge(
            coupler.handler.rigidbody,
            rocker.handler.rigidbody,
            C,
            Eigen::Vector3d::UnitZ());
        coupler_rocker_hinge.set_label("coupler_rocker_hinge");
        coupler_rocker_hinge.set_stiffness(exp4_joint_stiffness);
        coupler_rocker_hinge.set_tolerance_in_m(exp4_joint_tol_m);
        coupler_rocker_hinge.set_tolerance_in_deg(exp4_joint_tol_deg);

        auto rocker_support_hinge = sim.rigidbodies->add_constraint_hinge(
            rocker.handler.rigidbody,
            ground_rb,
            D,
            Eigen::Vector3d::UnitZ());
        rocker_support_hinge.set_label("rocker_support_hinge");
        rocker_support_hinge.set_stiffness(exp4_joint_stiffness);
        rocker_support_hinge.set_tolerance_in_m(exp4_joint_tol_m);
        rocker_support_hinge.set_tolerance_in_deg(exp4_joint_tol_deg);

        const std::vector<JointDriftSample> joints = {
            { support_hinge.get_point_constraint().get_body_a(),
              support_hinge.get_point_constraint().get_body_b(),
              support_hinge.get_point_constraint().get_local_point_body_a(),
              support_hinge.get_point_constraint().get_local_point_body_b() },
            { crank_coupler_hinge.get_point_constraint().get_body_a(),
              crank_coupler_hinge.get_point_constraint().get_body_b(),
              crank_coupler_hinge.get_point_constraint().get_local_point_body_a(),
              crank_coupler_hinge.get_point_constraint().get_local_point_body_b() },
            { coupler_rocker_hinge.get_point_constraint().get_body_a(),
              coupler_rocker_hinge.get_point_constraint().get_body_b(),
              coupler_rocker_hinge.get_point_constraint().get_local_point_body_a(),
              coupler_rocker_hinge.get_point_constraint().get_local_point_body_b() },
            { rocker_support_hinge.get_point_constraint().get_body_a(),
              rocker_support_hinge.get_point_constraint().get_body_b(),
              rocker_support_hinge.get_point_constraint().get_local_point_body_a(),
              rocker_support_hinge.get_point_constraint().get_local_point_body_b() },
        };

        {
            std::string dir = sim.get_settings().output.output_directory;
            std::ofstream f_latest(dir + "/joint_drift.csv", std::ios::trunc);
            f_latest << "t,max_drift\n";
            std::ofstream f_run(dir + "/" + joint_drift_run_file, std::ios::trunc);
            f_run << "t,max_drift\n";

            std::ofstream f_state(dir + "/fourbar_state.csv", std::ios::trunc);
            f_state << "t,"
                    << "crank_cx,crank_cy,crank_theta_deg,"
                    << "coupler_cx,coupler_cy,coupler_theta_deg,"
                    << "rocker_cx,rocker_cy,rocker_theta_deg\n";

            std::ofstream f_reaction(dir + "/fourbar_reaction.csv", std::ios::trunc);
            f_reaction << "t,"
                       << "support_fx,support_fy,support_fz,support_f_norm,"
                       << "support_tx,support_ty,support_tz,support_t_norm,"
                       << "crank_coupler_f_norm,crank_coupler_t_norm,"
                       << "coupler_rocker_f_norm,coupler_rocker_t_norm,"
                       << "rocker_support_fx,rocker_support_fy,rocker_support_fz,rocker_support_f_norm,"
                       << "rocker_support_tx,rocker_support_ty,rocker_support_tz,rocker_support_t_norm\n";
        }

        sim.add_time_event(
            0.0,
            settings.execution.end_simulation_time,
            [&sim,
             joints,
             joint_drift_run_file,
             crank_rb = crank.handler.rigidbody,
             coupler_rb = coupler.handler.rigidbody,
             rocker_rb = rocker.handler.rigidbody,
             support_hinge,
             crank_coupler_hinge,
             coupler_rocker_hinge,
             rocker_support_hinge](double t) mutable {
            std::string dir = sim.get_settings().output.output_directory;
            std::ofstream f_latest(dir + "/joint_drift.csv", std::ios::app);
            std::ofstream f_run(dir + "/" + joint_drift_run_file, std::ios::app);
            
            double max_drift = 0.0;
            for (const auto& joint : joints) {
                const Eigen::Vector3d pa = joint.body_a.transform_local_to_global_point(joint.local_point_a);
                const Eigen::Vector3d pb = joint.body_b.transform_local_to_global_point(joint.local_point_b);
                max_drift = std::max(max_drift, (pa - pb).norm());
            }
            f_latest << t << "," << max_drift << "\n";
            f_run << t << "," << max_drift << "\n";

            auto angle_deg = [](const stark::RigidBodyHandler& rb) {
                const Eigen::Vector3d x_axis = rb.transform_local_to_global_direction(Eigen::Vector3d::UnitX());
                return std::atan2(x_axis.y(), x_axis.x()) * 180.0 / M_PI;
            };

            std::ofstream f_state(dir + "/fourbar_state.csv", std::ios::app);
            const Eigen::Vector3d crank_c = crank_rb.get_translation();
            const Eigen::Vector3d coupler_c = coupler_rb.get_translation();
            const Eigen::Vector3d rocker_c = rocker_rb.get_translation();
            f_state << t << ","
                    << crank_c.x() << "," << crank_c.y() << "," << angle_deg(crank_rb) << ","
                    << coupler_c.x() << "," << coupler_c.y() << "," << angle_deg(coupler_rb) << ","
                    << rocker_c.x() << "," << rocker_c.y() << "," << angle_deg(rocker_rb) << "\n";

            const Eigen::Vector3d support_force = support_hinge.get_reaction_force_on_body_b();
            const Eigen::Vector3d support_torque = support_hinge.get_reaction_torque_on_body_b();
            const Eigen::Vector3d crank_coupler_force = crank_coupler_hinge.get_reaction_force_on_body_b();
            const Eigen::Vector3d crank_coupler_torque = crank_coupler_hinge.get_reaction_torque_on_body_b();
            const Eigen::Vector3d coupler_rocker_force = coupler_rocker_hinge.get_reaction_force_on_body_b();
            const Eigen::Vector3d coupler_rocker_torque = coupler_rocker_hinge.get_reaction_torque_on_body_b();
            const Eigen::Vector3d rocker_support_force = rocker_support_hinge.get_reaction_force_on_body_b();
            const Eigen::Vector3d rocker_support_torque = rocker_support_hinge.get_reaction_torque_on_body_b();

            std::ofstream f_reaction(dir + "/fourbar_reaction.csv", std::ios::app);
            f_reaction << t << ","
                       << support_force.x() << "," << support_force.y() << "," << support_force.z() << "," << support_force.norm() << ","
                       << support_torque.x() << "," << support_torque.y() << "," << support_torque.z() << "," << support_torque.norm() << ","
                       << crank_coupler_force.norm() << "," << crank_coupler_torque.norm() << ","
                       << coupler_rocker_force.norm() << "," << coupler_rocker_torque.norm() << ","
                       << rocker_support_force.x() << "," << rocker_support_force.y() << "," << rocker_support_force.z() << "," << rocker_support_force.norm() << ","
                       << rocker_support_torque.x() << "," << rocker_support_torque.y() << "," << rocker_support_torque.z() << "," << rocker_support_torque.norm() << "\n";
        });
    } else {
        auto c_params = sim.interactions->contact->get_global_params();
        c_params.default_contact_thickness = 0.005;
        sim.interactions->contact->set_global_params(c_params);

        auto ground = sim.presets->rigidbodies->add_box("ground", 1.0, {10.0, 10.0, 0.1});
        ground.handler.rigidbody.set_translation({0.0, 0.0, -0.05});
        sim.rigidbodies->add_constraint_fix(ground.handler.rigidbody);
        const auto ground_rb = ground.handler.rigidbody;

        const int N_chain = 10;
        std::vector<stark::RigidBodyHandler> links;
        std::vector<JointDriftSample> joints;
        stark::RigidBodyHandler prev;

        for (int i = 0; i < N_chain; i++) {
            auto link = sim.presets->rigidbodies->add_box("link_" + std::to_string(i), 1.0, {0.3, 0.1, 0.1});
            link.handler.rigidbody.set_translation({ i * 0.35, 0.0, 1.0 });
            links.push_back(link.handler.rigidbody);

            if (i == 0) {
                auto hinge = sim.rigidbodies->add_constraint_hinge(
                    ground_rb,
                    link.handler.rigidbody,
                    { 0.0, 0.0, 1.0 },
                    Eigen::Vector3d::UnitY());
                hinge.set_stiffness(exp4_joint_stiffness);
                hinge.set_tolerance_in_m(exp4_joint_tol_m);
                hinge.set_tolerance_in_deg(exp4_joint_tol_deg);
                joints.push_back({
                    hinge.get_point_constraint().get_body_a(),
                    hinge.get_point_constraint().get_body_b(),
                    hinge.get_point_constraint().get_local_point_body_a(),
                    hinge.get_point_constraint().get_local_point_body_b() });
            } else {
                auto point = sim.rigidbodies->add_constraint_point(prev, link.handler.rigidbody, { i * 0.35 - 0.175, 0.0, 1.0 });
                point.set_stiffness(exp4_joint_stiffness);
                point.set_tolerance_in_m(exp4_joint_tol_m);
                joints.push_back({
                    point.get_body_a(),
                    point.get_body_b(),
                    point.get_local_point_body_a(),
                    point.get_local_point_body_b() });
            }
            prev = link.handler.rigidbody;
        }

        {
            std::string dir = sim.get_settings().output.output_directory;
            std::ofstream f_latest(dir + "/joint_drift.csv", std::ios::trunc);
            f_latest << "t,max_drift\n";
            std::ofstream f_run(dir + "/" + joint_drift_run_file, std::ios::trunc);
            f_run << "t,max_drift\n";
        }

        sim.add_time_event(
            0.0,
            settings.execution.end_simulation_time,
            [&sim, joints, joint_drift_run_file](double t) {
            std::string dir = sim.get_settings().output.output_directory;
            std::ofstream f_latest(dir + "/joint_drift.csv", std::ios::app);
            std::ofstream f_run(dir + "/" + joint_drift_run_file, std::ios::app);

            double max_drift = 0.0;
            for (const auto& joint : joints) {
                const Eigen::Vector3d pa = joint.body_a.transform_local_to_global_point(joint.local_point_a);
                const Eigen::Vector3d pb = joint.body_b.transform_local_to_global_point(joint.local_point_b);
                max_drift = std::max(max_drift, (pa - pb).norm());
            }
            f_latest << t << "," << max_drift << "\n";
            f_run << t << "," << max_drift << "\n";
        });
    }

    sim.run();
    }
}

void exp4_coupled_joints_and_impacts()
{
    run_exp4_scene(false);
}

void exp4_fourbar_closed_loop()
{
    run_exp4_scene(true);
}

void exp5_bolt_from_models()
{
    std::cout << "Running Exp 5: Screw-Nut from models OBJ..." << std::endl;

    const std::string run_name = env_string("STARK_EXP5_RUN_NAME", "exp5_bolt");
    const double end_time = env_double("STARK_EXP5_END_TIME", 5.0);
    const double dt = env_double("STARK_EXP5_DT", 0.01);

    stark::Settings settings;
    settings.output.simulation_name = run_name;
    settings.output.output_directory = kOutputBase + "/" + run_name;
    settings.output.codegen_directory = kCodegenDir;
    settings.debug.symx_suppress_compiler_output = false;
    settings.debug.symx_force_load = false;
    settings.execution.end_simulation_time = end_time;
    settings.simulation.max_time_step_size = dt;
    settings.simulation.init_frictional_contact = true;
    configure_solver_from_env(settings);

    stark::Simulation sim(settings);

    auto c_params = sim.interactions->contact->get_global_params();
    c_params.default_contact_thickness = 1e-4;
    sim.interactions->contact->set_global_params(c_params);

    const double density = 8050.0;
    const double scale = 0.01;
    stark::ContactParams contact_params;
    contact_params.contact_thickness = 1e-4;

    auto pick_model_path = [](const std::string& file_name) {
        const std::string abs_path = kModelsDir + "/" + file_name;
        std::ifstream f(abs_path);
        if (f.good()) {
            return abs_path;
        }
        return MODELS_PATH + "/" + file_name;
    };

    auto nut = add_obj_rigidbody(sim, "nut", pick_model_path("nut-big.obj"), density, scale, contact_params);
    nut.rigidbody.set_translation({0.0, 0.0, 0.0});
    sim.rigidbodies->add_constraint_fix(nut.rigidbody);

    auto screw = add_obj_rigidbody(sim, "screw", pick_model_path("screw-big.obj"), density, scale, contact_params);
    screw.rigidbody.set_translation({0.0, 0.03, 0.0});

    sim.set_gravity({0.0, -9.81, 0.0});

    {
        std::string dir = sim.get_settings().output.output_directory;
        std::ofstream f(dir + "/screw_state.csv", std::ios::trunc);
        f << "t,x,y,z,vx,vy,vz\n";
    }

    sim.add_time_event(0.0, settings.execution.end_simulation_time, [&sim, rb = screw.rigidbody](double t) {
        std::string dir = sim.get_settings().output.output_directory;
        std::ofstream f(dir + "/screw_state.csv", std::ios::app);
        const Eigen::Vector3d x = rb.get_translation();
        const Eigen::Vector3d v = rb.get_velocity();
        f << t << "," << x.x() << "," << x.y() << "," << x.z() << "," << v.x() << "," << v.y() << "," << v.z() << "\n";
    });

    sim.run();
}

void exp6_double_pendulum()
{
    std::cout << "Running Exp 6: Double Pendulum (STARK)..." << std::endl;

    const double exp6_end_time = env_double("STARK_EXP6_END_TIME", 2.0);
    const double exp6_dt = env_double("STARK_EXP6_DT", 1e-3);
    const double exp6_joint_stiffness = env_double("STARK_EXP6_JOINT_STIFFNESS", 1e6);
    const double exp6_joint_tol_m = env_double("STARK_EXP6_JOINT_TOL_M", 1e-4);
    const double exp6_joint_tol_deg = env_double("STARK_EXP6_JOINT_TOL_DEG", 0.5);

    stark::Settings settings;
    settings.output.simulation_name = "exp6_double_pendulum_stark";
    settings.output.output_directory = kOutputBase + "/exp6_double_pendulum_stark";
    settings.output.codegen_directory = kCodegenDir;
    settings.debug.symx_suppress_compiler_output = false;
    settings.debug.symx_force_load = false;

    settings.execution.end_simulation_time = exp6_end_time;
    settings.simulation.gravity = { 0.0, -9.81, 0.0 };
    settings.simulation.init_frictional_contact = false;
    settings.simulation.use_adaptive_time_step = false;
    settings.simulation.max_time_step_size = exp6_dt;
    settings.newton.residual.tolerance = 1e-6;
    settings.newton.max_newton_iterations = 80;

    stark::Simulation sim(settings);
    sim.rigidbodies->set_default_constraint_stiffness(exp6_joint_stiffness);
    sim.rigidbodies->set_default_constraint_distance_tolerance(exp6_joint_tol_m);
    sim.rigidbodies->set_default_constraint_angle_tolerance(exp6_joint_tol_deg);

    std::cout << "Exp6 joint stiffness=" << exp6_joint_stiffness
              << ", tol_m=" << exp6_joint_tol_m
              << ", tol_deg=" << exp6_joint_tol_deg
              << ", dt=" << exp6_dt
              << ", end_time=" << exp6_end_time << std::endl;

    const double L = 1.0;
    const double W = 0.05;
    const double link_mass = 1.0;

    stark::ContactParams contact_params;
    contact_params.contact_thickness = 1e-4;

    auto ground = sim.presets->rigidbodies->add_box("ground", 1.0, { 0.1, 0.1, 0.1 }, contact_params);
    ground.handler.rigidbody.set_translation({ -0.05, 0.0, 0.0 });
    sim.rigidbodies->add_constraint_fix(ground.handler.rigidbody).set_label("support_fix");

    auto rod1 = sim.presets->rigidbodies->add_box("rod1", link_mass, { L, W, W }, contact_params);
    rod1.handler.rigidbody.set_translation({ 0.5 * L, 0.0, 0.0 });

    auto rod2 = sim.presets->rigidbodies->add_box("rod2", link_mass, { L, W, W }, contact_params);
    rod2.handler.rigidbody.set_translation({ 1.5 * L, 0.0, 0.0 });

    auto support_hinge = sim.rigidbodies->add_constraint_hinge(
        ground.handler.rigidbody,
        rod1.handler.rigidbody,
        { 0.0, 0.0, 0.0 },
        Eigen::Vector3d::UnitZ());
    support_hinge.set_label("support_hinge");

    auto middle_hinge = sim.rigidbodies->add_constraint_hinge(
        rod1.handler.rigidbody,
        rod2.handler.rigidbody,
        { L, 0.0, 0.0 },
        Eigen::Vector3d::UnitZ());
    middle_hinge.set_label("middle_hinge");

    const double total_mass = rod1.handler.rigidbody.get_mass() + rod2.handler.rigidbody.get_mass();
    const Eigen::Vector3d gravity = settings.simulation.gravity;

    {
        std::string dir = sim.get_settings().output.output_directory;
        std::ofstream f_state(dir + "/double_pendulum_state.csv", std::ios::trunc);
        f_state << "t,"
                << "rod1_x,rod1_y,rod1_z,rod1_vx,rod1_vy,rod1_vz,"
                << "rod2_x,rod2_y,rod2_z,rod2_vx,rod2_vy,rod2_vz,"
                << "vcm_x,vcm_y,vcm_z\n";

        std::ofstream f_react(dir + "/support_reaction_est.csv", std::ios::trunc);
        f_react << "t,fx_est,fy_est,fz_est,f_norm,"
                << "support_point_force_proxy,support_direction_torque_proxy,"
                << "middle_point_force_proxy,middle_direction_torque_proxy\n";
    }

    sim.add_time_event(0.0, settings.execution.end_simulation_time,
        [&sim, rod1 = rod1.handler.rigidbody, rod2 = rod2.handler.rigidbody, support_hinge, middle_hinge, total_mass, gravity,
         prev_vcm = Eigen::Vector3d(0.0, 0.0, 0.0), prev_t = -1.0](double t) mutable {
            const Eigen::Vector3d x1 = rod1.get_translation();
            const Eigen::Vector3d v1 = rod1.get_velocity();
            const Eigen::Vector3d x2 = rod2.get_translation();
            const Eigen::Vector3d v2 = rod2.get_velocity();
            const Eigen::Vector3d vcm = (rod1.get_mass() * v1 + rod2.get_mass() * v2) / total_mass;

            Eigen::Vector3d support_reaction_est = Eigen::Vector3d::Zero();
            if (prev_t >= 0.0) {
                const double dt = t - prev_t;
                if (dt > 1e-12) {
                    const Eigen::Vector3d acm = (vcm - prev_vcm) / dt;
                    support_reaction_est = total_mass * (acm - gravity);
                }
            }

            const double support_point_force_proxy = support_hinge.get_point_constraint().get_violation_in_m_and_force()[1];
            const double support_direction_torque_proxy = support_hinge.get_direction_constraint().get_violation_in_deg_and_torque()[1];
            const double middle_point_force_proxy = middle_hinge.get_point_constraint().get_violation_in_m_and_force()[1];
            const double middle_direction_torque_proxy = middle_hinge.get_direction_constraint().get_violation_in_deg_and_torque()[1];

            std::string dir = sim.get_settings().output.output_directory;
            std::ofstream f_state(dir + "/double_pendulum_state.csv", std::ios::app);
            f_state << t << ","
                    << x1.x() << "," << x1.y() << "," << x1.z() << ","
                    << v1.x() << "," << v1.y() << "," << v1.z() << ","
                    << x2.x() << "," << x2.y() << "," << x2.z() << ","
                    << v2.x() << "," << v2.y() << "," << v2.z() << ","
                    << vcm.x() << "," << vcm.y() << "," << vcm.z() << "\n";

            std::ofstream f_react(dir + "/support_reaction_est.csv", std::ios::app);
            f_react << t << ","
                    << support_reaction_est.x() << ","
                    << support_reaction_est.y() << ","
                    << support_reaction_est.z() << ","
                    << support_reaction_est.norm() << ","
                    << support_point_force_proxy << ","
                    << support_direction_torque_proxy << ","
                    << middle_point_force_proxy << ","
                    << middle_direction_torque_proxy << "\n";

            prev_vcm = vcm;
            prev_t = t;
        });

    sim.run();
}

void exp7_forklift_lift()
{
    std::cout << "Running Exp 7: Forklift lift benchmark (STARK)..." << std::endl;

    const std::string run_name = env_string("STARK_EXP7_RUN_NAME", "exp7_forklift_lift");
    const double end_time = env_double("STARK_EXP7_END_TIME", 3.0);
    const double dt = env_double("STARK_EXP7_DT", 5e-3);
    const double lift_start = env_double("STARK_EXP7_LIFT_START", 0.25);
    const double lift_end = env_double("STARK_EXP7_LIFT_END", 2.25);
    const double lift_speed = env_double("STARK_EXP7_LIFT_SPEED", -0.08);
    const double lift_max_force = env_double("STARK_EXP7_LIFT_MAX_FORCE", 2e4);
    const double pallet_y = env_double("STARK_EXP7_PALLET_Y", 0.4);
    const double pallet_z = env_double("STARK_EXP7_PALLET_Z", 3.0);
    const double joint_stiffness = env_double("STARK_EXP7_JOINT_STIFFNESS", 1e6);
    const double joint_tol_m = env_double("STARK_EXP7_JOINT_TOL_M", 1e-4);
    const double joint_tol_deg = env_double("STARK_EXP7_JOINT_TOL_DEG", 0.1);
    const bool guide_al_enabled = env_flag("STARK_EXP7_GUIDE_AL_ENABLED", false);
    const bool joint_al_enabled = env_flag("STARK_JOINT_AL_ENABLED", false);
    const double contact_thickness = env_double("STARK_EXP7_CONTACT_THICKNESS", 1e-3);
    const double min_contact_stiffness = env_double("STARK_EXP7_MIN_CONTACT_STIFFNESS", 1e3);
    const double ground_pallet_friction = env_double("STARK_EXP7_GROUND_PALLET_FRICTION", 0.5);
    const double fork_pallet_friction = env_double("STARK_EXP7_FORK_PALLET_FRICTION", 0.4);
    const bool contact_stiffness_update = env_flag("STARK_EXP7_CONTACT_STIFFNESS_UPDATE", true);
    const bool contact_adaptive_scheduling = env_flag("STARK_EXP7_CONTACT_ADAPTIVE_SCHEDULING", joint_al_enabled);
    const bool contact_inertia_consistent = env_flag("STARK_EXP7_CONTACT_INERTIA_CONSISTENT", false);
    const bool adaptive_dt = env_flag("STARK_EXP7_ADAPTIVE_DT", true);

    stark::Settings settings;
    settings.output.simulation_name = run_name;
    settings.output.output_directory = kOutputBase + "/" + run_name;
    settings.output.codegen_directory = kCodegenDir;
    settings.debug.symx_suppress_compiler_output = false;
    settings.debug.symx_force_load = false;
    settings.execution.end_simulation_time = end_time;
    settings.simulation.gravity = { 0.0, -9.81, 0.0 };
    settings.simulation.init_frictional_contact = true;
    settings.simulation.use_adaptive_time_step = adaptive_dt;
    settings.simulation.max_time_step_size = dt;
    settings.newton.max_newton_iterations = 100;
    configure_solver_from_env(settings);

    stark::Simulation sim(settings);
    sim.rigidbodies->set_default_constraint_stiffness(joint_stiffness);
    sim.rigidbodies->set_default_constraint_distance_tolerance(joint_tol_m);
    sim.rigidbodies->set_default_constraint_angle_tolerance(joint_tol_deg);
    if (joint_al_enabled) {
        configure_joint_al_from_env(sim);
    }

    auto global_contact = sim.interactions->contact->get_global_params();
    global_contact.default_contact_thickness = contact_thickness;
    global_contact.min_contact_stiffness = min_contact_stiffness;
    global_contact.stiffness_update_enabled = contact_stiffness_update;
    global_contact.adaptive_stiffness_scheduling = contact_adaptive_scheduling;
    global_contact.inertia_consistent_kappa = contact_inertia_consistent;
    sim.interactions->contact->set_global_params(global_contact);

    stark::ContactParams contact_params;
    contact_params.contact_thickness = contact_thickness;

    const Eigen::Vector3d COG_truss(0.0, 0.4, 0.5);
    const Eigen::Vector3d COG_arm(0.0, 1.300, 1.855);
    const Eigen::Vector3d COG_fork(0.0, 0.362, 2.100);
    const Eigen::Vector3d POS_pivotarm(0.0, 0.150, 1.855);
    const Eigen::Vector3d POS_prismatic(0.0, 0.150, 1.855);
    const Eigen::Vector3d pallet_pos(0.0, pallet_y, pallet_z);

    auto ground = sim.presets->rigidbodies->add_box("forklift_ground", 1000.0, { 40.0, 2.0, 40.0 }, contact_params);
    ground.handler.rigidbody.set_translation({ 0.0, -1.0, 0.0 });
    sim.rigidbodies->add_constraint_fix(ground.handler.rigidbody).set_label("ground_fix");

    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> chassis_boxes = {
        { { 1.227, 1.621, 1.864 }, { -0.003, 1.019, 0.192 } },
        { { 0.187, 0.773, 1.201 }, { 0.486, 0.153, -0.047 } },
        { { 0.187, 0.773, 1.201 }, { -0.486, 0.153, -0.047 } },
    };
    auto chassis = add_compound_box_rigidbody(
        sim, "forklift_chassis", 200.0, chassis_boxes, COG_truss, contact_params);
    sim.rigidbodies->add_constraint_fix(chassis.rigidbody).set_label("chassis_fix");

    auto arm = sim.presets->rigidbodies->add_box("forklift_arm", 100.0, { 0.9, 2.2, 0.31 }, contact_params);
    arm.handler.rigidbody.set_translation(COG_arm);

    const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> fork_boxes = {
        { { 0.100, 0.032, 1.033 }, { -0.352, -0.312, 0.613 } },
        { { 0.100, 0.032, 1.033 }, { 0.352, -0.312, 0.613 } },
        { { 0.344, 1.134, 0.101 }, { 0.0, 0.321, -0.009 } },
    };
    auto fork = add_compound_box_rigidbody(
        sim, "forklift_fork", 60.0, fork_boxes, COG_fork, contact_params);
    auto pallet = add_obj_rigidbody(
        sim, "forklift_pallet", resolve_model_path("pallet.obj"), 300.0, 1.0, contact_params);
    pallet.rigidbody.set_translation(pallet_pos);

    sim.interactions->contact->disable_collision(ground.handler.contact, chassis.contact);
    sim.interactions->contact->disable_collision(ground.handler.contact, arm.handler.contact);
    sim.interactions->contact->disable_collision(chassis.contact, arm.handler.contact);
    sim.interactions->contact->disable_collision(chassis.contact, fork.contact);
    sim.interactions->contact->disable_collision(arm.handler.contact, fork.contact);
    sim.interactions->contact->disable_collision(chassis.contact, pallet.contact);
    sim.interactions->contact->disable_collision(arm.handler.contact, pallet.contact);
    sim.interactions->contact->set_friction(ground.handler.contact, pallet.contact, ground_pallet_friction);
    sim.interactions->contact->set_friction(fork.contact, pallet.contact, fork_pallet_friction);

    sim.rigidbodies->add_constraint_fix(arm.handler.rigidbody).set_label("arm_fix");

    auto fork_lift = sim.rigidbodies->add_constraint_prismatic_press(
        fork.rigidbody,
        arm.handler.rigidbody,
        POS_prismatic,
        Eigen::Vector3d::UnitY(),
        0.0,
        lift_max_force);
    fork_lift.set_label("fork_lift");
    fork_lift.set_stiffness(joint_stiffness);
    fork_lift.set_tolerance_in_m(joint_tol_m);
    fork_lift.set_tolerance_in_deg(joint_tol_deg);
    fork_lift.set_augmented_lagrangian_enabled(guide_al_enabled);

    const auto pallet_mesh = load_localized_obj_mesh(resolve_model_path("pallet.obj"), 1.0, Eigen::Vector3d::Zero());
    const double fork_tine_top_local = -0.296;

    {
        std::string dir = sim.get_settings().output.output_directory;
        std::ofstream f(dir + "/forklift_state.csv", std::ios::trunc);
        f << "t,target_lift_v,fork_cx,fork_cy,fork_cz,fork_vx,fork_vy,fork_vz,"
             "pallet_cx,pallet_cy,pallet_cz,pallet_vx,pallet_vy,pallet_vz,"
             "fork_top_y,pallet_bottom_y,vertical_gap,actuator_force_proxy\n";
    }

    sim.add_time_event(
        0.0,
        settings.execution.end_simulation_time,
         [&sim,
         fork_rb = fork.rigidbody,
         pallet_rb = pallet.rigidbody,
         fork_lift,
         fork_top_y_local = fork_tine_top_local,
         pallet_bottom_y_local = pallet_mesh.min_v.y(),
         lift_start,
         lift_end,
         lift_speed](double t) mutable {
            const double target_v = (t >= lift_start && t <= lift_end) ? lift_speed : 0.0;
            fork_lift.get_linear_velocity_constraint().set_target_velocity(target_v);

            const Eigen::Vector3d fork_x = fork_rb.get_translation();
            const Eigen::Vector3d fork_v = fork_rb.get_velocity();
            const Eigen::Vector3d pallet_x = pallet_rb.get_translation();
            const Eigen::Vector3d pallet_v = pallet_rb.get_velocity();
            const double fork_top_y = fork_x.y() + fork_top_y_local;
            const double pallet_bottom_y = pallet_x.y() + pallet_bottom_y_local;
            const double vertical_gap = pallet_bottom_y - fork_top_y;
            const double actuator_force_proxy =
                fork_lift.get_linear_velocity_constraint().get_signed_velocity_violation_and_force()[1];

            std::string dir = sim.get_settings().output.output_directory;
            std::ofstream f(dir + "/forklift_state.csv", std::ios::app);
            f << t << ","
              << target_v << ","
              << fork_x.x() << "," << fork_x.y() << "," << fork_x.z() << ","
              << fork_v.x() << "," << fork_v.y() << "," << fork_v.z() << ","
              << pallet_x.x() << "," << pallet_x.y() << "," << pallet_x.z() << ","
              << pallet_v.x() << "," << pallet_v.y() << "," << pallet_v.z() << ","
              << fork_top_y << ","
              << pallet_bottom_y << ","
              << vertical_gap << ","
              << actuator_force_proxy << "\n";
        });

    sim.run();
}
