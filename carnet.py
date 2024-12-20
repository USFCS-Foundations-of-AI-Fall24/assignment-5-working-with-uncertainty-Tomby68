from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

car_model = BayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("Starts","Moves"),
        ("keyPresent", "Starts")
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_keyPresent = TabularCPD(
    variable="keyPresent", variable_card=2,
    values=[[0.7], [0.3]], state_names={"keyPresent": ["yes", "no"]}
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    #values=[[0.95, 0.05, 0.05, 0.001], [0.05, 0.95, 0.95, 0.9999]],
    values=[[0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],[0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99]],
    evidence=["Ignition", "Gas", "keyPresent"],
    evidence_card=[2, 2, 2],
    state_names={"Starts":['yes','no'], "Ignition":["Works", "Doesn't work"], "Gas":['Full',"Empty"], "keyPresent": ["yes", "no"]},
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)



# Associating the parameters with the model structure
car_model.add_cpds( cpd_starts, cpd_ignition, cpd_gas, cpd_radio, cpd_battery, cpd_moves, cpd_keyPresent)

car_infer = VariableElimination(car_model)

#print(car_infer.query(variables=["Moves"],evidence={"Radio":"turns on", "Starts":"yes"}))

def main():
    #q_battery_not_working = car_infer.query(variables=["Battery"],evidence={"Moves": "no"})
    #print(q_battery_not_working)
    #q_car_not_start = car_infer.query(variables=["Starts"],evidence={"Radio": "Doesn't turn on"})
    #print(q_car_not_start)
    #q_radio_working = car_infer.query(variables=["Radio","Gas"],evidence={"Battery": "Works"})
    #print(q_radio_working)
    #q_ignition_failing = car_infer.query(variables=["Ignition", "Gas"],evidence={"Moves": "no"})
    #print(q_ignition_failing)
    #q_starts = car_infer.query(variables=["Starts"],evidence={"Radio": "turns on", "Gas": "Full"})
    #print(q_starts)

    q_key_not_present = car_infer.query(variables=["keyPresent"],evidence={"Moves": "no"})
    print(q_key_not_present)

if __name__ == "__main__":
    main()

