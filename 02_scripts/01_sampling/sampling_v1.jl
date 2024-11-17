using JSON
using LatinHypercubeSampling
using DelimitedFiles
using Random

random_seed = 1234
Random.seed!(random_seed) #set random seed to reproduce

# Load JSON file 
json_dir = "01_data/parameter_ranges"
json_file = "geometrical_sampling_v8-4.json"

# Sample output
num_samples = 100
output_dir = "01_data/parameter_files/geometrical_samples/v8-4/"
mkpath(output_dir) 

# Log the seed number in a text file
seed_file_path = joinpath(output_dir, "random_seed.txt")
open(seed_file_path, "w") do io
    write(io, "$random_seed")
end

json_file = joinpath(json_dir, json_file)

@info("File $json_file will be processed for sample generation...")
@info("$num_samples samples will be saved in $output_dir...")

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
num_variables = sum(length(v) for v in values(variable_params))

@info("Samples with $num_variables variable parameters will be generated through Latin Hypercube Sampling...")


# Create now directories for each sample
for i in 1:num_samples
    dir_name = joinpath(output_dir, lpad(i, 3, '0'))
    mkpath(dir_name)
end

# Generate LHS samples in normalized space [0, 1] using total_variable_params and num_samples
lhs_samples = randomLHC(num_samples, num_variables)
lhs_samples_normalized = lhs_samples ./ num_samples

# Save the normalized samples to a text file
normalized_file_path = joinpath(output_dir, "_lhs_samples_normalized.txt")
writedlm(normalized_file_path, lhs_samples_normalized)

# Extract variable ranges from the variable_params dictionary
variable_ranges = Vector{Tuple{Float64, Float64}}()
for category in keys(variable_params)
    for (param, range) in variable_params[category]
        if isa(range, Array) && length(range) == 2
            push!(variable_ranges, (range[1], range[2]))
        else
            error("Invalid range format for parameter $param. Expected a 2-element array.")
        end
    end
end

# Scale the normalized samples to the actual variable ranges
scaled_samples = scaleLHC(lhs_samples_normalized, variable_ranges)


#### THIS PART WAS MOVED TO THE PYTHON SCRIPT ####
# --- If blade_distance is variable, adapt ranges according to other variable geometrical parameters --- #
#if haskey(variable_params["geometrical_parameters"], "blade_distance")
#    blade_index = findfirst(==( "blade_distance"), collect(keys(variable_params["geometrical_parameters"])))
#
#    for i in 1:num_samples
#        fiber_diameter = scaled_samples[i, findfirst(==("fiber_diameter"), collect(keys(variable_params["geometrical_parameters"])))]
#        elliptical_fiber_ratio = scaled_samples[i, findfirst(==("elliptical_fiber_ratio"), collect(keys(variable_params["geometrical_parameters"])))]
#        fiber_rotation = scaled_samples[i, findfirst(==("fiber_rotation"), collect(keys(variable_params["geometrical_parameters"])))]
#        droplet_diameter = scaled_samples[i, findfirst(==("droplet_diameter"), collect(keys(variable_params["geometrical_parameters"])))]
#
#        # Calculate min and max blade distance
#        min_blade_distance = fiber_diameter * sqrt(elliptical_fiber_ratio^2 * cosd(fiber_rotation)^2 + sind(fiber_rotation)^2)
#        max_blade_distance = droplet_diameter
#
#        # Rescale the normalized LHS value to the new range
#        normalized_value = lhs_samples_normalized[i, blade_index]
#        scaled_samples[i, blade_index] = min_blade_distance + normalized_value * (max_blade_distance - min_blade_distance)
#    end
#end
# ------------------------------------------------------------------------------------------------------ #


# Save the scaled samples to a text file and define path
scaled_file_path = joinpath(output_dir, "_lhs_samples_scaled.txt")
writedlm(scaled_file_path, scaled_samples)

@info("Normalized and scaled LHS samples saved to $normalized_file_path and $scaled_file_path.")


# Function to map the scaled samples back to the JSON structure and save
function save_sampled_parameters(fixed_params, variable_params, scaled_samples, num_samples, output_dir)
    sample_index = 1
    for i in 1:num_samples
        # Create a copy of the fixed parameters
        sampled_params = deepcopy(fixed_params)
        
        # Map the scaled values to the corresponding parameters
        variable_index = 1
        for category in keys(variable_params)
            for param in keys(variable_params[category])
                sampled_params[category][param] = scaled_samples[sample_index, variable_index]
                variable_index += 1
            end
        end
        
        # Define the directory and file name
        sample_dir = joinpath(output_dir, lpad(i, 3, '0'))
        sample_file = joinpath(sample_dir, "parameters.json")
        
        # Save the JSON structure with pretty printing
        open(sample_file, "w") do io
            JSON.print(io, sampled_params, 4)  # Pass the indent value directly
        end
        
        sample_index += 1
    end
end

# Save each sample's parameters to its corresponding directory
save_sampled_parameters(fixed_params, variable_params, scaled_samples, num_samples, output_dir)
@info("Sampled parameters saved to individual directories with proper formatting.")
