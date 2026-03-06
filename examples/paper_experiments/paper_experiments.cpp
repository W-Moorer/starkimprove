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

namespace {
    const std::string kCodegenDir = "e:/workspace/stark/codegen";
    const std::string kOutputBase = "e:/workspace/stark/output/paper_experiments";
    const std::string kModelsDir = "e:/workspace/stark/models";

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
        params.adaptive_rho = env_flag("STARK_JOINT_AL_ADAPTIVE_RHO", true);
        params.rho0 = env_double("STARK_JOINT_AL_RHO0", 1e5);
        params.rho_update_ratio = env_double("STARK_JOINT_AL_RHO_UPDATE_RATIO", 1.5);
        params.sufficient_decrease_ratio = env_double("STARK_JOINT_AL_SUFFICIENT_DECREASE_RATIO", 0.9);
        params.max_outer_iterations = env_int("STARK_JOINT_AL_MAX_OUTER", 8);
        params.residual_smoothing = env_double("STARK_JOINT_AL_RESIDUAL_SMOOTHING", 1e-4);

        sim.rigidbodies->set_joint_augmented_lagrangian_params(params);
        return true;
    }

    void configure_solver_from_env(stark::Settings& settings)
    {
        settings.newton.residual.tolerance = env_double("STARK_NEWTON_TOL", settings.newton.residual.tolerance);
        settings.newton.cg_tolerance_multiplier = env_double("STARK_LINEAR_TOL", settings.newton.cg_tolerance_multiplier);
        settings.newton.cg_max_iterations_multiplier = env_double("STARK_CG_MAX_IT_MUL", settings.newton.cg_max_iterations_multiplier);
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

// --- Exp 4: Coupled Joints & Impacts ---
void exp4_coupled_joints_and_impacts() {
    std::cout << "Running Exp 4: Coupled Joints & Impacts..." << std::endl;
    const bool joint_al_enabled = env_flag("STARK_JOINT_AL_ENABLED", false);
    const std::string default_run_name = joint_al_enabled ? "exp4_coupled_joints_al" : "exp4_coupled_joints";
    const std::string run_name = env_string("STARK_EXP4_RUN_NAME", default_run_name);
    stark::Settings settings;
    settings.output.simulation_name = run_name;
    settings.output.output_directory = kOutputBase + "/" + run_name;
    settings.output.codegen_directory = kCodegenDir;
    settings.debug.symx_suppress_compiler_output = false;
    settings.debug.symx_force_load = false;
    settings.execution.end_simulation_time = 3.0;
    settings.simulation.init_frictional_contact = true;
    configure_solver_from_env(settings);

    stark::Simulation sim(settings);
    if (joint_al_enabled) {
        configure_joint_al_from_env(sim);
    }

    auto c_params = sim.interactions->contact->get_global_params();
    c_params.default_contact_thickness = 0.005;
    sim.interactions->contact->set_global_params(c_params);

    auto ground = sim.presets->rigidbodies->add_box("ground", 1.0, {10.0, 10.0, 0.1});
    ground.handler.rigidbody.set_translation({0.0, 0.0, -0.05});
    sim.rigidbodies->add_constraint_fix(ground.handler.rigidbody);

    stark::RigidBodyHandler prev;
    const int N_chain = 10;
    std::vector<stark::RigidBodyHandler> links;
    const std::string joint_drift_run_file =
        "joint_drift_" + settings.output.simulation_name + "__" + settings.output.time_stamp + ".csv";
    
    for (int i = 0; i < N_chain; i++) {
        auto link = sim.presets->rigidbodies->add_box("link_" + std::to_string(i), 1.0, {0.3, 0.1, 0.1});      
        // Position them along X initially
        link.handler.rigidbody.set_translation({i * 0.35, 0.0, 1.0}); // Gap 0.05
        links.push_back(link.handler.rigidbody);
        
        if (i == 0) {
            sim.rigidbodies->add_constraint_hinge(ground.handler.rigidbody, link.handler.rigidbody, {0.0, 0.0, 1.0}, Eigen::Vector3d::UnitY());
        } else {
            sim.rigidbodies->add_constraint_point(prev, link.handler.rigidbody, {i*0.35 - 0.175, 0.0, 1.0});   
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

    sim.add_time_event(0.0, 3.0, [&sim, links, joint_drift_run_file](double t) {
        std::string dir = sim.get_settings().output.output_directory;
        std::ofstream f_latest(dir + "/joint_drift.csv", std::ios::app);
        std::ofstream f_run(dir + "/" + joint_drift_run_file, std::ios::app);
        
        double max_drift = 0.0;
        for (int i = 1; i < (int)links.size(); i++) {
            // Anchor point in local coords
            Eigen::Vector3d p_prev = links[i-1].transform_local_to_global_point({0.175, 0.0, 0.0});
            Eigen::Vector3d p_curr = links[i].transform_local_to_global_point({-0.175, 0.0, 0.0});
            double drift = (p_prev - p_curr).norm();
            max_drift = std::max(max_drift, drift); 
        }
        f_latest << t << "," << max_drift << "\n";
        f_run << t << "," << max_drift << "\n";
    });

    sim.run();
}

void exp5_bolt_from_models()
{
    std::cout << "Running Exp 5: Screw-Nut from models OBJ..." << std::endl;

    stark::Settings settings;
    settings.output.simulation_name = "exp5_bolt";
    settings.output.output_directory = kOutputBase + "/exp5_bolt";
    settings.output.codegen_directory = kCodegenDir;
    settings.debug.symx_suppress_compiler_output = false;
    settings.debug.symx_force_load = false;
    settings.execution.end_simulation_time = 5.0;
    settings.simulation.max_time_step_size = 0.01;
    settings.simulation.init_frictional_contact = true;

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
