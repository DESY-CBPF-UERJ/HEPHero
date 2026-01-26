#--------------------------------------------------------------------------------------------------
# General Setup
#--------------------------------------------------------------------------------------------------
input_path = ""
output_path = ""
periods = ['0_22']


#--------------------------------------------------------------------------------------------------
# ML setup
#--------------------------------------------------------------------------------------------------
device = 'cuda' # 'cpu'
library = 'torch'
optimizer = ['ranger', 'adam', 'sgd']
loss_func = ['mse']
learning_rate = [[0.1, 0.01, 0.001], [0.01]]


#--------------------------------------------------------------------------------------------------
# Models setup
#--------------------------------------------------------------------------------------------------
model_type = 'APSNN'
model_parameters = {
    'affine_setups': [[100, 50], [300, 150, 100, 50]],
    'activation_func': ['elu', 'relu', 'tanh', 'selu', 'gelu'],
    'batch_norm': [True, False],
    'dropout': [None, 0.2, 0.5],
    # EFT parameters
    "parameter_max_power": [2, 2],
    "max_overall_power": 2,
    "basis": [[0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 2.0],
            [2.0, 0.0]]
    }


#-------------------------------------------------------------------------------------
# Training setup
#-------------------------------------------------------------------------------------
batch_size = [1000, 500]
load_size_stat = 120000
load_size_training = 100000
num_load_for_check = 4 # It must be smaller or equal to the maximum nSlices
train_frac = 0.5
eval_step_size = 250
eval_interval = 20
num_max_iterations = 10000
early_stopping = 20
initial_model_path = None


#--------------------------------------------------------------------------------------------------
# Inputs setup
#--------------------------------------------------------------------------------------------------
feature_info = False

scalar_variables = [
    ['jet_pt',      'Jet_pt',       'F'],
    ['jet_mass',    'Jet_mass',     'F'],
    ['theta',       'theta',        [0]],
    ]

vector_variables = []


#--------------------------------------------------------------------------------------------------
# Preprocessing setup
#--------------------------------------------------------------------------------------------------
reweight_variables = [
    #["jet_pt",      [200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 350, 400, 450, 500, 1000, 2500]],
    ]
normalization_method = "area" # "evtsum"

pca_transformation = None #None # "standard", "custom"
pca_custom_classes = {}

#--------------------------------------------------------------------------------------------------
# Classes setup
#--------------------------------------------------------------------------------------------------
classes = {
#<class_name>: [[<list_of_processes>], <mode>, <combination>, <label>, <color>]
"Signal_sample": [[
    "Zto2Q_PTQQ-100to200",
    "Zto2Q_PTQQ-200to400"
    ], "scalars", 'equal', "Signal", "green"],
"Background": [[
    "QCD_PT-15to30",
    "QCD_PT-30to50"
    ], "scalars", 'equal', "Background", "red"],
}



# Signal class names must start with "Signal"

# If a class has only one process, the class name must be equal to the process name

# Parameterized signal must have a class name starting with "Signal_parameterized"

# More than one signal class is allowed

# If a class name starts with "Signal_samples", the models will be trained to each signal point separately. In addition, combination and label are ignored.

# If two or more class names start with "Signal_samples", the signal points from these classes are paired together during the loop

# The code support a maximum of 2 reweight variables
