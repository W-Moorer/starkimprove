import pandas as pd
import matplotlib.pyplot as plt
import os

# Settings for the plots
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['lines.linewidth'] = 2.0

data_dir = r"E:\workspace\stark\output\paper_experiments"
out_dir = r"E:\workspace\stark\documents\local\paper1\figs"

os.makedirs(out_dir, exist_ok=True)

# Helper function to configure axes
def setup_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.6)

def load_ref_curve(csv_path, scale=1.0):
    df = pd.read_csv(csv_path)
    x = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    y = pd.to_numeric(df.iloc[:, 1], errors='coerce') * scale
    mask = x.notna() & y.notna()
    out = pd.DataFrame({'t': x[mask], 'v': y[mask]}).sort_values('t')
    return out

# 1. Experiment 1: all box center z-curves
try:
    centers_path = os.path.join(data_dir, "exp1_adaptive", "centers_z.csv")
    minz_path = os.path.join(data_dir, "exp1_adaptive", "min_z.csv")
    fig, ax = plt.subplots(figsize=(6, 4))
    setup_axes(ax)
    if os.path.exists(centers_path):
        df1 = pd.read_csv(centers_path)
        z_cols = [c for c in df1.columns if c.startswith('z_')]
        for c in z_cols:
            ax.plot(df1['t'], df1[c], alpha=0.7, linewidth=1.2)
        ax.set_ylabel('Center Height z (m)')
        ax.set_title('All Box Center-z Trajectories (Exp1)')
    else:
        df1 = pd.read_csv(minz_path)
        ax.plot(df1['t'], df1['min_z'], color='#1f77b4', label='Minimum Object Z-Coordinate')
        ax.set_ylabel('Vertical Position (m)')
        ax.set_title('Settling Dynamics and Stability (Fallback)')
        ax.legend()
    ax.set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp1_settling.svg"), format='svg')
    plt.savefig(os.path.join(out_dir, "exp1_settling.pdf"), format='pdf')
    plt.close()
    print("Saved exp1_settling.svg")
except Exception as e:
    print(f"Error plotting Exp1: {e}")

# 2. Experiment 2: High-speed impact
try:
    fig, ax = plt.subplots(figsize=(6, 4))
    setup_axes(ax)
    colors = ['#2ca02c', '#ff7f0e', '#d62728']
    
    for c, v in zip(colors, [10, 100, 500]):
        filepath = os.path.join(data_dir, f"exp2_v{v}", "impact_state.csv")
        if os.path.exists(filepath):
            df2 = pd.read_csv(filepath)
            df2 = df2.drop_duplicates(subset=['t'], keep='last')
            ax.plot(df2['t'], df2['x'], color=c, label=f'v0 = {v} m/s')
            
    # Add a shaded region representing the Wall
    ax.axhspan(1.5, 2.5, color='gray', alpha=0.3, label='Wall Extent')
            
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Horizontal Position X (m)')
    ax.set_title('High-Speed Impact and Bouncing')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp2_impact.svg"), format='svg')
    plt.savefig(os.path.join(out_dir, "exp2_impact.pdf"), format='pdf')
    plt.close()
    print("Saved exp2_impact.svg")
except Exception as e:
    print(f"Error plotting Exp2: {e}")

# 3. Experiment 3: Stick-slip
try:
    df12 = pd.read_csv(os.path.join(data_dir, "exp3_theta12", "velocity.csv"))
    df20 = pd.read_csv(os.path.join(data_dir, "exp3_theta20", "velocity.csv"))

    df12 = df12.drop_duplicates(subset=['t'], keep='last').sort_values('t')
    df20 = df20.drop_duplicates(subset=['t'], keep='last').sort_values('t')

    # Reprocess by parameters: detect release time from phase flag for each case
    release12 = float(df12.loc[df12['phase'] == 1, 't'].iloc[0]) if (df12['phase'] == 1).any() else 0.0
    release20 = float(df20.loc[df20['phase'] == 1, 't'].iloc[0]) if (df20['phase'] == 1).any() else 0.0
    
    fig, ax = plt.subplots(figsize=(6, 4))
    setup_axes(ax)
    
    ax.plot(df12['t'], df12['v_x'], color='#1f77b4', label=r'$\theta=12^\circ$')
    ax.plot(df20['t'], df20['v_x'], color='#d62728', label=r'$\theta=20^\circ$')

    if abs(release12 - release20) < 1e-6:
        ax.axvline(release12, color='gray', linestyle='--', linewidth=1.2, label=f'Release at {release12:.2f} s')
    else:
        ax.axvline(release12, color='#1f77b4', linestyle='--', linewidth=1.2, alpha=0.8, label=f'Release 12° ({release12:.2f} s)')
        ax.axvline(release20, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.8, label=f'Release 20° ({release20:.2f} s)')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Downslope Velocity $v_x$ (m/s)')
    ax.set_title(r'Fixed-Angle Friction Test ($\mu=0.3$)')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp3_stick_slip.svg"), format='svg')
    plt.savefig(os.path.join(out_dir, "exp3_stick_slip.pdf"), format='pdf')
    plt.close()
    print("Saved exp3_stick_slip.svg")
except Exception as e:
    print(f"Error plotting Exp3: {e}")

# 4. Experiment 4: Joint Drift
try:
    df4 = pd.read_csv(os.path.join(data_dir, "exp4_coupled_joints", "joint_drift.csv"))
    fig, ax = plt.subplots(figsize=(6, 4))
    setup_axes(ax)
    # The drift can be very small, plot in log scale if it's strictly > 0
    ax.plot(df4['t'], df4['max_drift'] * 1000, color='#e377c2', label='Max Constraint Violation')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Joint Drift (mm)')
    ax.set_title('Coupled Joints Constraint Accuracy')
    
    # Use scientific notation for Y-axis if needed, but we scaled to mm
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp4_drift.svg"), format='svg')
    plt.savefig(os.path.join(out_dir, "exp4_drift.pdf"), format='pdf')
    plt.close()
    print("Saved exp4_drift.svg")
except Exception as e:
    print(f"Error plotting Exp4: {e}")

# 5. Experiment 5: Screw-nut y/vy vs reference (mu=0)
try:
    df5 = pd.read_csv(os.path.join(data_dir, "exp5_bolt", "screw_state.csv"))
    df5 = df5.drop_duplicates(subset=['t'], keep='last').sort_values('t')

    ref_pos5 = load_ref_curve(os.path.join(data_dir, "exp5_ref", "Pos_Ty_mu0.csv"), scale=0.01)
    ref_vel5 = load_ref_curve(os.path.join(data_dir, "exp5_ref", "Vel_Ty_mu0.csv"), scale=0.01)

    # Reprocess: align position zero-level between simulation and reference
    y0_sim = float(df5['y'].iloc[0])
    y0_ref = float(ref_pos5['v'].iloc[0])
    df5_y_aligned = df5['y'] - y0_sim
    ref_pos5_aligned = ref_pos5['v'] - y0_ref

    fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    setup_axes(axs[0])
    setup_axes(axs[1])

    axs[0].plot(df5['t'], df5_y_aligned, color='#1f77b4', label='IPC (mu=0, aligned)')
    axs[0].plot(ref_pos5['t'], ref_pos5_aligned, color='black', linestyle='--', linewidth=1.6, label='Reference (mu=0, aligned)')
    axs[0].set_ylabel('Position y (m)')
    axs[0].set_title('Exp5 Screw Y / Vy vs Reference')
    axs[0].legend(loc='upper right')

    axs[1].plot(df5['t'], df5['vy'], color='#1f77b4', label='IPC (mu=0)')
    axs[1].plot(ref_vel5['t'], ref_vel5['v'], color='black', linestyle='--', linewidth=1.6, label='Reference (mu=0)')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel(r'Velocity $v_y$ (m/s)')
    axs[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp5_bolt_vs_ref.svg"), format='svg')
    plt.savefig(os.path.join(out_dir, "exp5_bolt_vs_ref.pdf"), format='pdf')
    plt.close()
    print("Saved exp5_bolt_vs_ref.svg")
except Exception as e:
    print(f"Error plotting Exp5: {e}")

# 6. Experiment 6: Screw-nut friction effect + reference overlay
try:
    df_mu03 = pd.read_csv(os.path.join(data_dir, "exp6_mu03", "state.csv"))

    df_mu03 = df_mu03.drop_duplicates(subset=['t'], keep='last').sort_values('t')

    ref_pos_mu03_path = os.path.join(data_dir, "exp6_ref", "Pos_Ty_mu0.3.csv")
    ref_vel_mu03_path = os.path.join(data_dir, "exp6_ref", "Vel_Ty_mu0.3.csv")

    ref_pos_mu03 = load_ref_curve(ref_pos_mu03_path, scale=0.01) if os.path.exists(ref_pos_mu03_path) else None
    ref_vel_mu03 = load_ref_curve(ref_vel_mu03_path, scale=0.01) if os.path.exists(ref_vel_mu03_path) else None

    fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    setup_axes(axs[0])
    setup_axes(axs[1])

    axs[0].plot(df_mu03['t'], df_mu03['y'], color='#d62728', label=r'$\mu=0.3$')
    if ref_pos_mu03 is not None:
        axs[0].plot(ref_pos_mu03['t'], ref_pos_mu03['v'], color='black', linestyle='-', linewidth=1.2, label=r'Reference $\mu=0.3$')
    axs[0].set_ylabel('Position y (m)')
    axs[0].set_title('Screw Position/Velocity under Friction (Exp6)')
    axs[0].legend(loc='lower right')

    axs[1].plot(df_mu03['t'], df_mu03['vy'], color='#d62728', label=r'$\mu=0.3$')
    if ref_vel_mu03 is not None:
        axs[1].plot(ref_vel_mu03['t'], ref_vel_mu03['v'], color='black', linestyle='-', linewidth=1.2, label=r'Reference $\mu=0.3$')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel(r'Velocity $v_y$ (m/s)')
    axs[1].legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp6_friction_effect.svg"), format='svg')
    plt.savefig(os.path.join(out_dir, "exp6_friction_effect.pdf"), format='pdf')
    plt.close()
    print("Saved exp6_friction_effect.svg")
except Exception as e:
    print(f"Error plotting Exp6: {e}")

# 7. Experiment 7: Anchor-cube adaptive stiffness sensitivity (overlay kappa curves)
try:
    kappa_cases = [
        ("k1e6",  r'$\kappa=10^6$'),
        ("k1e7",  r'$\kappa=10^7$'),
        ("k1e8",  r'$\kappa=10^8$'),
        ("k1e9",  r'$\kappa=10^9$'),
    ]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

    axis_specs = [
        ('x', 'vx', 'X'),
        ('y', 'vy', 'Y'),
        ('z', 'vz', 'Z'),
    ]

    for idx_axis, (pos_col, vel_col, axis_tag) in enumerate(axis_specs):
        fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        setup_axes(axs[0])
        setup_axes(axs[1])

        plotted_any = False
        for (case_name, label), c in zip(kappa_cases, colors):
            state_path = os.path.join(data_dir, f"exp7_{case_name}", "state.csv")
            if not os.path.exists(state_path):
                print(f"Skip Exp7 {case_name}: missing state.csv")
                continue

            dfi = pd.read_csv(state_path)
            dfi = dfi.drop_duplicates(subset=['t'], keep='last').sort_values('t')
            if dfi.empty:
                print(f"Skip Exp7 {case_name}: empty state.csv")
                continue

            p0 = float(dfi[pos_col].iloc[0])
            axs[0].plot(dfi['t'], dfi[pos_col] - p0, color=c, label=label)
            axs[1].plot(dfi['t'], dfi[vel_col], color=c, label=label)
            plotted_any = True

        if plotted_any:
            axs[0].set_ylabel(f'Position {pos_col} (m, aligned)')
            axs[0].set_title(fr'Exp7 Adaptive Stiffness Sensitivity ({axis_tag}-axis)')

            if idx_axis == 1:  # Y-axis: upper plot legend at upper right
                axs[0].legend(loc='upper right')
            else:  # X-axis and Z-axis: legend at lower left
                axs[0].legend(loc='lower left')

            axs[1].set_xlabel('Time (s)')
            axs[1].set_ylabel(fr'Velocity ${vel_col}$ (m/s)')

            if idx_axis == 0:  # X-axis: lower plot legend at upper right
                axs[1].legend(loc='upper right')
            elif idx_axis == 1:  # Y-axis: lower plot legend at lower right
                axs[1].legend(loc='lower right')
            else:  # Z-axis: legend at lower left
                axs[1].legend(loc='lower left')

            plt.tight_layout()
            stem = f"exp7_kappa_sensitivity_{pos_col}"
            plt.savefig(os.path.join(out_dir, stem + ".svg"), format='svg')
            plt.savefig(os.path.join(out_dir, stem + ".pdf"), format='pdf')
            print(f"Saved {stem}.svg")
        else:
            print(f"Skip Exp7 {axis_tag}-axis plot: no valid exp7 state.csv found")
        plt.close()
except Exception as e:
    print(f"Error plotting Exp7: {e}")

print("All done.")
