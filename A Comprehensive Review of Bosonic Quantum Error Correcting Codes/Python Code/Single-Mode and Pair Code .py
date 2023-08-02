import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, coherent, tensor, destroy, mesolve, Options

def single_mode_cat_code(alpha, beta, N):
    """
    Create a single-mode cat code state.

    Parameters:
        alpha (float): Amplitude of the cat state.
        beta (float): Superposition parameter of the cat state.
        N (int): Number of Fock states for the harmonic oscillator.

    Returns:
        cat0 (qutip.Qobj): Single-mode cat code state.
    """
    # Create the coherent state |α⟩ and the superposition state β|α⟩
    cat0 = (alpha * basis(N, 0) + beta * coherent(N, alpha)).unit()
    return cat0

def pair_cat_code(alpha, beta, N):
    """
    Create a pair-cat code state.

    Parameters:
        alpha (float): Amplitude of the cat state.
        beta (float): Superposition parameter of the cat state.
        N (int): Number of Fock states for the harmonic oscillator.

    Returns:
        cat_state (qutip.Qobj): Pair-cat code state.
    """
    # Create the four components of the pair-cat state |αα⟩, |α(-α)⟩, |(-α)α⟩, |(-α)(-α)⟩
    cat00 = tensor(coherent(N, alpha), coherent(N, alpha))
    cat01 = tensor(coherent(N, alpha), coherent(N, -alpha))
    cat10 = tensor(coherent(N, -alpha), coherent(N, alpha))
    cat11 = tensor(coherent(N, -alpha), coherent(N, -alpha))

    # Create the pair-cat code state (β(|00⟩ + |11⟩) + α(|01⟩ + |10⟩)) / √2
    cat_state = (beta * (cat00 + cat11) + alpha * (cat01 + cat10)).unit()
    return cat_state

def simulate_quantum_dynamics(H, rho0, tlist):
    """
    Simulate quantum dynamics using QuTiP's mesolve function.

    Parameters:
        H (qutip.Qobj): Hamiltonian operator for the quantum system.
        rho0 (qutip.Qobj): Initial density matrix of the quantum state.
        tlist (numpy.array): Time points for simulation.

    Returns:
        result (qutip.solver.Result): Result of the quantum simulation.
    """
    # Configure simulation options
    options = Options(nsteps=10000)

    # Perform the quantum simulation using mesolve
    result = mesolve(H, rho0, tlist, [], [], options=options)
    return result

def plot_population(result, tlist):
    """
    Plot the populations of quantum states over time.

    Parameters:
        result (qutip.solver.Result): Result of the quantum simulation.
        tlist (numpy.array): Time points for simulation.
    """
    plt.figure()
    # Plot the populations of |0⟩⟨0| and |α⟩⟨α| over time
    plt.plot(tlist, result.expect[0], label="|0⟩⟨0|")
    plt.plot(tlist, result.expect[1], label="|α⟩⟨α|")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Parameters for the cat state and system size
    alpha = 2.0  # Cat state amplitude parameter
    beta = 1.0   # Cat state superposition parameter
    N = 10       # Number of Fock states for the harmonic oscillator

    # Time parameters for simulation
    tlist = np.linspace(0, 5, 100)

    # Define quantum operators for single-mode cat code and pair-cat code
    a = destroy(N)
    H_single_mode = alpha * a.dag() + alpha * a
    H_pair_cat = alpha * a.dag() * a.dag() + alpha * a * a

    # Encoding and simulation for single-mode cat code
    cat_single_mode = single_mode_cat_code(alpha, beta, N)
    rho0_single_mode = cat_single_mode * cat_single_mode.dag()
    result_single_mode = simulate_quantum_dynamics(H_single_mode, rho0_single_mode, tlist)

    # Encoding and simulation for pair-cat code
    cat_pair_cat = pair_cat_code(alpha, beta, N)
    rho0_pair_cat = cat_pair_cat * cat_pair_cat.dag()
    result_pair_cat = simulate_quantum_dynamics(H_pair_cat, rho0_pair_cat, tlist)

    # Plot the populations of the quantum states over time for both codes
    plot_population(result_single_mode, tlist)
    plot_population(result_pair_cat, tlist)
