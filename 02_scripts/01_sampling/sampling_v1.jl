using JSON
using LatinHypercubeSampling

# Load JSON file 
json_dir = "01_data/parameter_ranges"
json_file = "geometrical_sampling_v1.json"

# Sample output
sample_amount = 100
output_dir = "01_data/parameter_files/geometrical_samples/v1/"

json_file = joinpath(json_dir, json_file)

@info("File $json_file will be processed for sample generation...")
@info("$sample_amount samples will be saved in $output_dir...")

# Function to load and parse the JSON file
function load_parameters(file_path::String)
    return JSON.parsefile(file_path)
end

# Function to identify fixed and variable parameters
function identify_parameters(params::Dict)
    fixed_params = Dict()
    variable_params = Dict()

    for (category, parameters) in params
        fixed_params[category] = Dict()
        variable_params[category] = Dict()
        for (param, value) in parameters
            if isa(value, Array)
                variable_params[category][param] = value
            else
                fixed_params[category][param] = value
            end
        end
    end

    return fixed_params, variable_params
end


# Load parameter set
params = load_parameters(json_file)

# Identify fixed and variable parameters
fixed_params, variable_params = identify_parameters(params)

# Count parameters
function count_parameters(fixed::Dict, variable::Dict)
    for category in keys(fixed)
        fixed_count = length(keys(fixed[category]))
        variable_count = length(keys(variable[category]))
        total_count = fixed_count + variable_count
        println("$fixed_count/$total_count fixed $category detected.")
        println("$variable_count/$total_count variable $category detected.")
    end
end

# Print results
count_parameters(fixed_params, variable_params)

# Calculate the number of variable parameters
total_variable_params = sum(length(v) for v in values(variable_params))

@info("\nSamples with $total_variable_params variable parameters will be generated through Latin Hypercube Sampling...")


# Create now directories for each sample
for i in 1:sample_amount
    dir_name = joinpath(output_dir, lpad(i, 3, '0'))
    mkpath(dir_name)
end