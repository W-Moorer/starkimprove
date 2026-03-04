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

namespace {
    const std::string kCodegenDir = "e:/workspace/stark/codegen";
    const std::string kOutputBase = "e:/workspace/stark/output/paper_experiments";
    const std::string kModelsDir = "e:/workspace/stark/models";

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

// --- Exp 3: Stick-Slip Transition ---
void exp3_stick_slip_transition() {
    std::cout << "Running Exp 3: Fixed-Angle Friction Validation..." << std::endl;

    auto run_exp3_case = [](const std::string& case_name, const double theta_deg) {
        stark::Settings settings;
        settings.output.simulation_name = "exp3_" + case_name;
        settings.output.output_directory = kOutputBase + "/exp3_" + case_name;
        settings.output.codegen_directory = kCodegenDir;
        settings.debug.symx_suppress_compiler_output = false;
        settings.debug.symx_force_load = false;
        settings.execution.end_simulation_time = 4.0;
        settings.simulation.init_frictional_contact = true;
        settings.simulation.max_time_step_size = 2e-4;

        stark::Simulation sim(settings);

        auto c_params = sim.interactions->contact->get_global_params();
        c_params.default_contact_thickness = 5e-4;
        c_params.min_contact_stiffness = 2e4;
        c_params.friction_stick_slide_threshold = 0.01;
        sim.interactions->contact->set_global_params(c_params);

        stark::EnergyFrictionalContact::Params contact_params;
        contact_params.contact_thickness = 5e-4;
        auto slope = sim.presets->rigidbodies->add_box("slope", 1.0, {10.0, 2.0, 0.1}, contact_params);
        slope.handler.rigidbody.set_translation({0.0, 0.0, -0.05});
        sim.rigidbodies->add_constraint_fix(slope.handler.rigidbody);

        auto block = sim.presets->rigidbodies->add_box("block", 1.0, {0.2, 0.2, 0.2}, contact_params);
        block.handler.rigidbody.set_translation({0.0, 0.0, 0.1005});

        const double mu = 0.3;
        sim.interactions->contact->set_friction(slope.handler.contact, block.handler.contact, mu);

        const double settle_duration = 1.0;
        const double total_duration = settings.execution.end_simulation_time;
        const double theta_rad = theta_deg * 3.14159265 / 180.0;
        const double g = 9.81;

        {
            std::string dir = sim.get_settings().output.output_directory;
            std::ofstream f(dir + "/velocity.csv", std::ios::trunc);
            f << "t,phase,theta,v_x,v_y,v_z\n";
        }

        sim.add_time_event(0.0, total_duration, [&sim,
                                                 block_rb = block.handler.rigidbody,
                                                 settle_duration,
                                                 theta_deg,
                                                 theta_rad,
                                                 g](double t) {
            int phase = 0;
            double gx = 0.0;
            if (t >= settle_duration) {
                phase = 1;
                gx = g * std::sin(theta_rad);
            }
            const double gz = -g * std::cos(theta_rad);
            sim.set_gravity({gx, 0.0, gz});

            std::string dir = sim.get_settings().output.output_directory;
            std::ofstream f(dir + "/velocity.csv", std::ios::app);
            Eigen::Vector3d v = block_rb.get_velocity();
            f << t << "," << phase << "," << theta_deg << "," << v[0] << "," << v[1] << "," << v[2] << "\n";
        });

        sim.run();
    };

    run_exp3_case("theta12", 12.0);
    run_exp3_case("theta20", 20.0);
}

// --- Exp 4: Coupled Joints & Impacts ---
void exp4_coupled_joints_and_impacts() {
    std::cout << "Running Exp 4: Coupled Joints & Impacts..." << std::endl;
    stark::Settings settings;
    settings.output.simulation_name = "exp4_coupled_joints";
    settings.output.output_directory = kOutputBase + "/exp4_coupled_joints";
    settings.output.codegen_directory = kCodegenDir;
    settings.debug.symx_suppress_compiler_output = false;
    settings.debug.symx_force_load = false;
    settings.execution.end_simulation_time = 3.0;
    settings.simulation.init_frictional_contact = true;

    stark::Simulation sim(settings);

    auto c_params = sim.interactions->contact->get_global_params();
    c_params.default_contact_thickness = 0.005;
    sim.interactions->contact->set_global_params(c_params);

    auto ground = sim.presets->rigidbodies->add_box("ground", 1.0, {10.0, 10.0, 0.1});
    ground.handler.rigidbody.set_translation({0.0, 0.0, -0.05});
    sim.rigidbodies->add_constraint_fix(ground.handler.rigidbody);

    stark::RigidBodyHandler prev;
    const int N_chain = 10;
    std::vector<stark::RigidBodyHandler> links;
    
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
    
    sim.add_time_event(0.0, 3.0, [&sim, links](double t) {
        std::string dir = sim.get_settings().output.output_directory;
        std::ofstream f(dir + "/joint_drift.csv", std::ios::app);
        if (t == 0.0) f << "t,max_drift\n";
        
        double max_drift = 0.0;
        for (int i = 1; i < (int)links.size(); i++) {
            // Anchor point in local coords
            Eigen::Vector3d p_prev = links[i-1].transform_local_to_global_point({0.175, 0.0, 0.0});
            Eigen::Vector3d p_curr = links[i].transform_local_to_global_point({-0.175, 0.0, 0.0});
            double drift = (p_prev - p_curr).norm();
            max_drift = std::max(max_drift, drift); 
        }
        f << t << "," << max_drift << "\n";
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

void exp6_friction_effect()
{
    std::cout << "Running Exp 6: Friction Effect on Screw-Nut (mu=0 vs 0.3)..." << std::endl;

    auto run_exp6_case = [](const std::string& case_name, const double mu) {
        stark::Settings settings;
        settings.output.simulation_name = "exp6_" + case_name;
        settings.output.output_directory = kOutputBase + "/exp6_" + case_name;
        settings.output.codegen_directory = kCodegenDir;
        settings.debug.symx_suppress_compiler_output = false;
        settings.debug.symx_force_load = false;
        settings.execution.end_simulation_time = 5.0;
        settings.simulation.init_frictional_contact = true;
        settings.simulation.max_time_step_size = 0.01;

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

        sim.interactions->contact->set_friction(nut.contact, screw.contact, mu);
        sim.set_gravity({0.0, -9.81, 0.0});

        {
            std::string dir = sim.get_settings().output.output_directory;
            std::ofstream f(dir + "/state.csv", std::ios::trunc);
            f << "t,mu,x,y,z,vx,vy,vz\n";
        }

        sim.add_time_event(0.0, settings.execution.end_simulation_time,
            [&sim, screw_rb = screw.rigidbody, mu](double t) {
                std::string dir = sim.get_settings().output.output_directory;
                std::ofstream f(dir + "/state.csv", std::ios::app);
                const Eigen::Vector3d x = screw_rb.get_translation();
                const Eigen::Vector3d v = screw_rb.get_velocity();
                f << t << "," << mu << ","
                  << x.x() << "," << x.y() << "," << x.z() << ","
                  << v.x() << "," << v.y() << "," << v.z() << "\n";
            }
        );

        sim.run();
    };

    run_exp6_case("mu0", 0.0);
    run_exp6_case("mu03", 0.3);
}

void exp7_anchor_cube_kappa_sweep()
{
    std::cout << "Running Exp 7: Anchor-Cube adaptive stiffness sweep..." << std::endl;

    auto run_exp7_case = [](const std::string& case_name, const double kappa) {
        stark::Settings settings;
        settings.output.simulation_name = "exp7_" + case_name;
        settings.output.output_directory = kOutputBase + "/exp7_" + case_name;
        settings.output.codegen_directory = kCodegenDir;
        settings.debug.symx_suppress_compiler_output = false;
        settings.debug.symx_force_load = false;
        settings.execution.end_simulation_time = 0.3;
        settings.simulation.init_frictional_contact = true;
        settings.simulation.max_time_step_size = 0.001;

        stark::Simulation sim(settings);

        auto c_params = sim.interactions->contact->get_global_params();
        c_params.default_contact_thickness = 1e-4;
        c_params.stiffness_update_enabled = true;
        c_params.min_contact_stiffness = kappa;
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

        auto cube = add_obj_rigidbody(sim, "cube", pick_model_path("cube.obj"), density, scale, contact_params);
        cube.rigidbody.set_translation({0.0, 0.0, 0.0});
        sim.rigidbodies->add_constraint_fix(cube.rigidbody);

        auto anchor = add_obj_rigidbody(sim, "anchor", pick_model_path("anchor.obj"), density, scale, contact_params);
        anchor.rigidbody.set_translation({0.0, 0.0, 0.0});

        sim.interactions->contact->set_friction(cube.contact, anchor.contact, 0.3);
        sim.set_gravity({0.0, -9.81, 0.0});

        {
            std::string dir = sim.get_settings().output.output_directory;
            std::ofstream f(dir + "/state.csv", std::ios::trunc);
            f << "t,kappa,x,y,z,vx,vy,vz\n";
        }

        sim.add_time_event(0.0, settings.execution.end_simulation_time,
            [&sim, anchor_rb = anchor.rigidbody, kappa](double t) {
                std::string dir = sim.get_settings().output.output_directory;
                std::ofstream f(dir + "/state.csv", std::ios::app);
                const Eigen::Vector3d x = anchor_rb.get_translation();
                const Eigen::Vector3d v = anchor_rb.get_velocity();
                f << t << "," << kappa << ","
                  << x.x() << "," << x.y() << "," << x.z() << ","
                  << v.x() << "," << v.y() << "," << v.z() << "\n";
            }
        );

        sim.run();
    };

    run_exp7_case("k1e6", 1e6);
    run_exp7_case("k1e7", 1e7);
    run_exp7_case("k1e8", 1e8);
    run_exp7_case("k1e9", 1e9);
}
