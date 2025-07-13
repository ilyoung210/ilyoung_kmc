import mainrev1

# ----- User configurable parameters -----
DOPANT_CHOICE = '1'  # '1': Zr, '2': Si, '3': Al, others: use HZO defaults
DOPANT_PERCENT = 50.0  # doping percentage (0-100)

# Enable or disable random variations
APPLY_THICKNESS_VARIATION = False
APPLY_DOPANT_VARIATION = False

# Variation ranges
# Thickness variation range in nm (±). Ignored if APPLY_THICKNESS_VARIATION is False.
THICKNESS_VARIATION_NM = 1.0
# Dopant variation range in percent (±). Ignored if APPLY_DOPANT_VARIATION is False.
DOPANT_VARIATION_PERCENT = 2.0

LATTICE_NX = 50  # lattice size in x direction
LATTICE_NY = 50  # lattice size in y direction
REPEAT_COUNT = 10  # number of simulation repetitions

# Temperature profile segments: list of tuples (start_T, end_T, steps)
# Modify as needed
SEGMENTS = [
    (300, 1000, 30),
    (1000, 1000, 30),
]


def apply_dopant():
    """Apply selected dopant parameters."""
    if DOPANT_CHOICE == '1':
        mainrev1.set_material_params_zr_doped(DOPANT_PERCENT)
    elif DOPANT_CHOICE == '2':
        mainrev1.set_material_params_si_doped(DOPANT_PERCENT)
    elif DOPANT_CHOICE == '3':
        mainrev1.set_material_params_al_doped(DOPANT_PERCENT)
    else:
        mainrev1.apply_global_params(mainrev1.hzo_par)


def main():
    mainrev1.initialize_barrier_file()
    apply_dopant()
    h_var = THICKNESS_VARIATION_NM * 10.0 if APPLY_THICKNESS_VARIATION else 0.0
    doping_var = DOPANT_VARIATION_PERCENT if APPLY_DOPANT_VARIATION else 0.0
    mainrev1.run_scatter_mode(
        LATTICE_NX,
        LATTICE_NY,
        SEGMENTS,
        REPEAT_COUNT,
        DOPANT_CHOICE,
        DOPANT_PERCENT,
        doping_var,
        h_var,
    )
    print("Scatter simulation completed.")


if __name__ == "__main__":
    main()
