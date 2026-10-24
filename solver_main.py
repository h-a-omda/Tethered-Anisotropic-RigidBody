import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid


m = 1.0
I = 0.5
L_tether = 3.0
anchor = np.array([0.0, 0.0])


b_parallel = 0.01
b_perp = 50.0


r_f = np.array([0.5, -0.5])


k_spring = 1000000.0
c_damper = 500.0


def sleigh_physics(t, state):
    X, Y, theta, Vx, Vy, Omega = state

    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    e_para = np.array([c, s])
    e_perp = np.array([-s, c])

    r_com = np.array([X, Y])
    v_com = np.array([Vx, Vy])


    rf_global = R @ r_f
    v_f = v_com + Omega * np.array([-rf_global[1], rf_global[0]])


    F_d_para = -b_parallel * np.dot(v_f, e_para) * e_para
    F_d_perp = -b_perp * np.dot(v_f, e_perp) * e_perp
    F_drag = F_d_para + F_d_perp


    tau_drag = rf_global[0] * F_drag[1] - rf_global[1] * F_drag[0]


    r_vec = r_com - anchor
    r_dist = np.linalg.norm(r_vec)
    F_tension = np.array([0.0, 0.0])

    if r_dist > L_tether:
        r_unit = r_vec / r_dist
        r_dot = np.dot(v_com, r_unit)

        magnitude = k_spring * (r_dist - L_tether) + c_damper * max(0, r_dot)
        F_tension = -magnitude * r_unit


    F_total = F_drag + F_tension
    ax = F_total[0] / m
    ay = F_total[1] / m
    alpha = tau_drag / I

    return [Vx, Vy, Omega, ax, ay, alpha]


state0 = [0.0, 2.5, 0.0, -3.0, 0.0, 1.0]
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)

sol = solve_ivp(
    sleigh_physics, t_span, state0, t_eval=t_eval, method='Radau', rtol=1e-8, atol=1e-10
)

t = sol.t
X, Y, theta, Vx, Vy, Omega = sol.y


r = np.sqrt(X**2 + Y**2)
force_magnitude = np.zeros_like(r)
T_kinetic = 0.5 * m * (Vx**2 + Vy**2) + 0.5 * I * Omega**2
V_spring = np.zeros_like(r)
p_drag_array = np.zeros_like(r)
p_tether_damper_array = np.zeros_like(r)

for i in range(len(t)):
    pos = np.array([X[i], Y[i]])
    vel = np.array([Vx[i], Vy[i]])
    om = Omega[i]
    th = theta[i]

    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, -s], [s, c]])
    e_para = np.array([c, s])
    e_perp = np.array([-s, c])

    rf_global = R @ r_f
    v_f = vel + om * np.array([-rf_global[1], rf_global[0]])

    F_d_para = -b_parallel * np.dot(v_f, e_para) * e_para
    F_d_perp = -b_perp * np.dot(v_f, e_perp) * e_perp
    F_drag = F_d_para + F_d_perp
    tau_drag = rf_global[0] * F_drag[1] - rf_global[1] * F_drag[0]


    p_drag_array[i] = np.dot(F_drag, vel) + tau_drag * om

    r_vec = pos - anchor
    r_dist = np.linalg.norm(r_vec)
    if r_dist > L_tether:
        r_unit = r_vec / r_dist
        r_dot = np.dot(vel, r_unit)
        mag = k_spring * (r_dist - L_tether) + c_damper * max(0, r_dot)
        force_magnitude[i] = mag
        V_spring[i] = 0.5 * k_spring * (r_dist - L_tether)**2

        if r_dot > 0:
            p_tether_damper_array[i] = -c_damper * (r_dot**2)


W_drag = -cumulative_trapezoid(p_drag_array, t, initial=0)
W_tether_damper = -cumulative_trapezoid(p_tether_damper_array, t, initial=0)
W_dissipated_total = W_drag + W_tether_damper

E_total = T_kinetic + V_spring


plt.rc('text', usetex=False)
plt.rc('font', family='serif')


fig1, ax = plt.subplots(figsize=(6, 6))
ax.plot(X, Y, label='Sleigh Path', color='b', lw=1.5)
ax.scatter([anchor[0]], [anchor[1]], color='r', s=100, label='Tether Anchor (0,0)', zorder=5)
ax.set_xlabel('X Position [m]')
ax.set_ylabel('Y Position [m]')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right')
ax.set_aspect('equal')
plt.savefig('figure1_trajectory.pdf', format='pdf', bbox_inches='tight')
plt.show()


fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
ax1.plot(t, r, color='b', lw=1.5, label=r'Radial Distance $r(t)$')
ax1.axhline(y=L_tether, color='r', linestyle='--', lw=1.2, label=r'Tether Limit $L$')
ax1.set_ylabel('Radius r [m]')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right', frameon=True)

ax2.plot(t, force_magnitude, color='k', lw=1.0, label=r'Penalty Force Magnitude $|\mathbf{F}_{\mathrm{tension}}|$')
ax2.set_ylabel('Penalty Force [N]')
ax2.set_xlabel('Time t [s]')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig('figure2_tethersnap.pdf', format='pdf', bbox_inches='tight')
plt.show()


fig3, ax = plt.subplots(figsize=(6, 6))
ax.plot(Vx, Vy, color='purple', lw=1.2, label='Phase Trajectory')
ax.scatter([Vx[0]], [Vy[0]], color='green', s=50, zorder=5, label='Initial State')
ax.scatter([Vx[-1]], [Vy[-1]], color='red', s=50, zorder=5, label='Asymptotic Attractor')
ax.set_xlabel(r'Velocity Component $v_x$ [m/s]')
ax.set_ylabel(r'Velocity Component $v_y$ [m/s]')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig('figure3_phasespace.pdf', format='pdf', bbox_inches='tight')
plt.show()


fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
ax1.plot(t, E_total, color='black', lw=1.5, label=r'$E_{\mathrm{total}}$')
ax1.plot(t, T_kinetic, color='blue', lw=1.2, linestyle='--', label=r'$T_{\mathrm{kinetic}}$')
ax1.plot(t, V_spring, color='red', lw=1.2, linestyle=':', label=r'$V_{\mathrm{spring}}$')
ax1.set_ylabel('Mechanical Energy [J]')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper right', frameon=True, fontsize=9.0)

ax2.plot(t, E_total, color='black', lw=1.5, label=r'$E_{\mathrm{total}}$')
ax2.plot(t, T_kinetic, color='blue', lw=1.2, linestyle='--', label=r'$T_{\mathrm{kinetic}}$')
ax2.plot(t, V_spring, color='red', lw=1.2, linestyle=':', label=r'$V_{\mathrm{spring}}$')
ax2.plot(t, W_dissipated_total, color='green', lw=1.2, linestyle='-.', label=r'$W_{\mathrm{dissipated}}$')
ax2.set_xlabel('Time t [s]')
ax2.set_ylabel('Energy / Work Ledger [J]')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='lower left', frameon=True, fontsize=9.0)
plt.tight_layout()
plt.savefig('figure4_energyledger.pdf', format='pdf', bbox_inches='tight')
plt.show()


max_penetration = np.max(r[r > L_tether] - L_tether) if np.any(r > L_tether) else 0.0
print(f"Maximum radial penetration: {max_penetration:.6f} m")
