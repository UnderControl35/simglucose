from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
from datetime import timedelta
from datetime import datetime

from simglucose.controller.pid_ctrller import PIDController

# specify start_time as the beginning of today
now = datetime.now()
start_time = datetime.combine(now.date(), datetime.min.time())

# --------- Create Random Scenario --------------
# Specify results saving path
path = './results'

# Create a simulation environment
patient = T1DPatient.withName('adolescent#001')
sensor = CGMSensor.withName('Dexcom', seed=1)
pump = InsulinPump.withName('Insulet')
scenario = RandomScenario(start_time=start_time, seed=1)
env = T1DSimEnv(patient, sensor, pump, scenario)

# Create a controller
# controller = BBController()
controller = PIDController(P=-1.74e-04, I=-1e-07, D=-1e-02, target=120)
# Put them together to create a simulation object
s1 = SimObj(env, controller, timedelta(days=1), animate=True, path=path)
results1 = sim(s1)
print(results1)


#from simglucose.controller.pid_ctrller import PIDController
#from simglucose.simulation.user_interface import simulate


#pid_controller = PIDController(P=0.001, I=0.00001, D=0.001, target=140)
#s = simulate(controller=pid_controller)