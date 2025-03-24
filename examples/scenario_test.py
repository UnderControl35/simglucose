from simglucose.patient.base import Patient
import numpy as np
from scipy.integrate import ode
import pandas as pd
from collections import namedtuple
import logging
import pkg_resources
import matplotlib.pyplot as plt

# Configure logger
logger = logging.getLogger(__name__)

# Define named tuples for actions and observations
Action = namedtuple("patient_action", ["CHO", "insulin"])
Observation = namedtuple("observation", ["Gsub"])

# Path to patient parameters file
PATIENT_PARA_FILE = pkg_resources.resource_filename("simglucose", "params/vpatient_params.csv")


class T1DPatient(Patient):
    SAMPLE_TIME = 1  # min
    EAT_RATE = 5  # g/min CHO

    def __init__(self, params, init_state=None, random_init_bg=False, seed=None, t0=0):
        """
        Initialize a T1DPatient instance.
        Inputs:
            - params: pandas Series with patient parameters
            - init_state: custom initial state (defaults to params.iloc[2:15])
            - random_init_bg: if True, randomize initial blood glucose
            - seed: random seed for reproducibility
            - t0: simulation start time (default: 0)
        """
        self._params = params
        self._init_state = init_state
        self.random_init_bg = random_init_bg
        self._seed = seed
        self.t0 = t0
        logger.debug(f"Initializing patient with name: {params.Name}")
        self.reset()

    @classmethod
    def withID(cls, patient_id, **kwargs):
        """Construct patient by ID (1-30: adolescent#001 to child#010)."""
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params.iloc[patient_id - 1, :]
        logger.info(f"Creating patient with ID {patient_id}: {params.Name}")
        return cls(params, **kwargs)

    @classmethod
    def withName(cls, name, **kwargs):
        """Construct patient by name (e.g., adolescent#001)."""
        patient_params = pd.read_csv(PATIENT_PARA_FILE)
        params = patient_params.loc[patient_params.Name == name].squeeze()
        logger.info(f"Creating patient with name: {name}")
        return cls(params, **kwargs)

    @property
    def state(self):
        return self._odesolver.y

    @property
    def t(self):
        return self._odesolver.t

    @property
    def sample_time(self):
        return self.SAMPLE_TIME

    def step(self, action):
        """Advance the simulation by one step based on the given action."""
        # Convert announced meal to amount to eat now
        to_eat = self._announce_meal(action.CHO)
        action = action._replace(CHO=to_eat)

        # Log eating start
        if action.CHO > 0 and self._last_action.CHO <= 0:
            logger.info(f"t={self.t:.1f}: Patient starts eating...")
            self._last_Qsto = self.state[0] + self.state[1]  # mg
            self._last_foodtaken = 0  # g
            self.is_eating = True

        # Log amount eaten
        if to_eat > 0:
            logger.debug(f"t={self.t:.1f}: Patient eats {to_eat:.2f} g of CHO")

        if self.is_eating:
            self._last_foodtaken += action.CHO

        # Log eating end
        if action.CHO <= 0 and self._last_action.CHO > 0:
            logger.info(f"t={self.t:.1f}: Patient finishes eating!")
            self.is_eating = False

        # Update last action
        self._last_action = action

        # Run ODE solver
        self._odesolver.set_f_params(action, self._params, self._last_Qsto, self._last_foodtaken)
        if self._odesolver.successful():
            self._odesolver.integrate(self._odesolver.t + self.sample_time)
            logger.debug(f"t={self.t:.1f}: State updated - BG={self.observation.Gsub:.2f} mg/dL")
        else:
            logger.error("ODE solver failed!")
            raise RuntimeError("ODE solver failed")

    @staticmethod
    def model(t, x, action, params, last_Qsto, last_foodtaken):
        """Differential equation model for T1D simulation."""
        dxdt = np.zeros(13)
        d = action.CHO * 1000  # g -> mg
        insulin = action.insulin * 6000 / params.BW  # U/min -> pmol/kg/min
        basal = params.u2ss * params.BW / 6000  # U/min

        # Log insulin injection if above basal
        if action.insulin > basal:
            logger.debug(f"t={t:.1f}: Injecting insulin: {action.insulin:.4f} U/min (basal: {basal:.4f})")

        # Stomach glucose dynamics
        qsto = x[0] + x[1]
        Dbar = last_Qsto + last_foodtaken * 1000  # mg
        dxdt[0] = -params.kmax * x[0] + d

        if Dbar > 0:
            aa = 5 / (2 * Dbar * (1 - params.b))
            cc = 5 / (2 * Dbar * params.d)
            kgut = params.kmin + (params.kmax - params.kmin) / 2 * (
                np.tanh(aa * (qsto - params.b * Dbar)) - np.tanh(cc * (qsto - params.d * Dbar)) + 2
            )
        else:
            kgut = params.kmax

        dxdt[1] = params.kmax * x[0] - x[1] * kgut
        dxdt[2] = kgut * x[1] - params.kabs * x[2]

        # Glucose dynamics
        Rat = params.f * params.kabs * x[2] / params.BW
        EGPt = params.kp1 - params.kp2 * x[3] - params.kp3 * x[8]
        Uiit = params.Fsnc
        Et = params.ke1 * (x[3] - params.ke2) if x[3] > params.ke2 else 0

        dxdt[3] = max(EGPt, 0) + Rat - Uiit - Et - params.k1 * x[3] + params.k2 * x[4]
        dxdt[3] = (x[3] >= 0) * dxdt[3]

        Vmt = params.Vm0 + params.Vmx * x[6]
        Kmt = params.Km0
        Uidt = Vmt * x[4] / (Kmt + x[4])
        dxdt[4] = -Uidt + params.k1 * x[3] - params.k2 * x[4]
        dxdt[4] = (x[4] >= 0) * dxdt[4]

        # Insulin dynamics
        dxdt[5] = -(params.m2 + params.m4) * x[5] + params.m1 * x[9] + params.ka1 * x[10] + params.ka2 * x[11]
        It = x[5] / params.Vi
        dxdt[5] = (x[5] >= 0) * dxdt[5]

        dxdt[6] = -params.p2u * x[6] + params.p2u * (It - params.Ib)
        dxdt[7] = -params.ki * (x[7] - It)
        dxdt[8] = -params.ki * (x[8] - x[7])
        dxdt[9] = -(params.m1 + params.m30) * x[9] + params.m2 * x[5]
        dxdt[9] = (x[9] >= 0) * dxdt[9]

        dxdt[10] = insulin - (params.ka1 + params.kd) * x[10]
        dxdt[10] = (x[10] >= 0) * dxdt[10]
        dxdt[11] = params.kd * x[10] - params.ka2 * x[11]
        dxdt[11] = (x[11] >= 0) * dxdt[11]

        # Subcutaneous glucose
        dxdt[12] = -params.ksc * x[12] + params.ksc * x[3]
        dxdt[12] = (x[12] >= 0) * dxdt[12]

        return dxdt

    @property
    def observation(self):
        """Return current observation (subcutaneous glucose level)."""
        GM = self.state[12]  # mg/kg
        Gsub = GM / self._params.Vg  # mg/dL
        return Observation(Gsub=Gsub)

    def _announce_meal(self, meal):
        """Process meal announcement and return amount to eat now."""
        self.planned_meal += meal
        if self.planned_meal > 0:
            to_eat = min(self.EAT_RATE, self.planned_meal)
            self.planned_meal -= to_eat
            self.planned_meal = max(0, self.planned_meal)
            logger.debug(f"Meal announced: {meal:.2f} g, eating now: {to_eat:.2f} g")
        else:
            to_eat = 0
        return to_eat

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        self._seed = seed
        self.reset()

    def reset(self):
        """Reset patient state to initial conditions."""
        if self._init_state is None:
            self.init_state = np.copy(self._params.iloc[2:15].values)
        else:
            self.init_state = self._init_state

        self.random_state = np.random.RandomState(self.seed)
        if self.random_init_bg:
            mean = [self.init_state[3], self.init_state[4], self.init_state[12]]
            cov = np.diag([0.1 * val for val in mean])
            bg_init = self.random_state.multivariate_normal(mean, cov)
            self.init_state[3], self.init_state[4], self.init_state[12] = bg_init
            logger.info("Randomized initial blood glucose levels")

        self._last_Qsto = self.init_state[0] + self.init_state[1]
        self._last_foodtaken = 0
        self.name = self._params.Name
        self._odesolver = ode(self.model).set_integrator("dopri5")
        self._odesolver.set_initial_value(self.init_state, self.t0)
        self._last_action = Action(CHO=0, insulin=0)
        self.is_eating = False
        self.planned_meal = 0
        logger.info(f"Patient {self.name} reset at t={self.t0}")


if __name__ == "__main__":
    # Configure logging
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Simulation setup
    logger.info("Starting T1D simulation...")
    p = T1DPatient.withName("adolescent#001")
    basal = p._params.u2ss * p._params.BW / 6000  # U/min
    logger.info(f"Basal insulin rate: {basal:.4f} U/min")

    t, CHO, insulin, BG = [], [], [], []
    simulation_time = 1000  # minutes

    # Run simulation
    while p.t < simulation_time:
        ins = basal
        carb = 0
        if p.t == 100:
            carb = 80  # g
            ins = 80.0 / 6.0 + basal  # Bolus + basal
            logger.info(f"t={p.t:.1f}: Administering 80g CHO and bolus insulin {ins:.4f} U/min")

        act = Action(insulin=ins, CHO=carb)
        t.append(p.t)
        CHO.append(act.CHO)
        insulin.append(act.insulin)
        BG.append(p.observation.Gsub)
        p.step(act)

    logger.info(f"Simulation completed at t={p.t:.1f} minutes")

    # Plot results
    logger.info("Generating simulation plots...")
    fig, ax = plt.subplots(3, sharex=True, figsize=(10, 8))
    ax[0].plot(t, BG, label="Subcutaneous Glucose")
    ax[0].set_ylabel("Glucose (mg/dL)")
    ax[0].legend()
    ax[1].plot(t, CHO, label="Carbohydrate Intake", color="orange")
    ax[1].set_ylabel("CHO (g)")
    ax[1].legend()
    ax[2].plot(t, insulin, label="Insulin", color="green")
    ax[2].set_ylabel("Insulin (U/min)")
    ax[2].set_xlabel("Time (minutes)")
    ax[2].legend()
    plt.tight_layout()
    plt.show()
    logger.info("Plots displayed successfully")