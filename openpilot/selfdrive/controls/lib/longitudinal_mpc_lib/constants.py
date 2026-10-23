# The long MPC's stopping geometry, apart from the solver so a reader of it does not load casadi and the generated
# solver. The solver is generated from these values: the SConscript lists this file among its sources.
COMFORT_BRAKE = 2.5
STOP_DISTANCE = 7.0


def get_stopped_equivalence_factor(v_lead):
  return (v_lead**2) / (2 * COMFORT_BRAKE)
