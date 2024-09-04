### --- This part loads the parameters for training --- ###
using JSON
using DelimitedFiles

# Define the root directory of samples
root_dir = "C:\\Users\\Senk\\Desktop\\Droplet_Tests_FEA\\01_neural_network_project\\01_data\\parameter_files"
#samples = "mechanical_samples\\v2"
samples = "geometrical_samples\\v3"
root_dir = joinpath(root_dir, samples)

# Define the root directory for storing results
root_results_dir = "C:\\Users\\Senk\\Desktop\\Droplet_Tests_FEA\\01_neural_network_project\\03_results"
results_dir = joinpath(root_results_dir, samples, "01_filtered_data")

### Define the parameters which need to be extracted:
function extract_parameters(param_data)
    # Manually extract parameters and order as desired for the neural network
    param1 = param_data["mechanical_parameters"]["GI"]
    param2 = param_data["mechanical_parameters"]["GII,GIII"]
    param3 = param_data["mechanical_parameters"]["tI=tII=tIII"]
    param4 = param_data["mechanical_parameters"]["interface_friction"]
    param5 = param_data["mechanical_parameters"]["blade_friction"]
    #param1 = param_data["geometrical_parameters"]["fiber_diameter"]
    #param2 = param_data["geometrical_parameters"]["droplet_diameter"]
    #param3 = param_data["geometrical_parameters"]["ratio_droplet_embedded_length"]
    #param4 = param_data["geometrical_parameters"]["contact_angle"]
    #param5 = param_data["geometrical_parameters"]["elliptical_fiber_ratio"]
    #param6 = param_data["geometrical_parameters"]["fiber_rotation"]
    #param7 = param_data["geometrical_parameters"]["blade_distance"]

    # Return the extracted parameters as a vector
    return [param1, param2, param3, param4, param5]
    #return [param1, param2, param3, param4, param5, param6, param7]
end

# Create the results directory if it does not exist
if !isdir(results_dir)
    mkpath(results_dir)
    println("Created results directory at: $results_dir")
end

# Find all subdirectories of the samples (e.g., 001, 002, etc.)
subdirs = filter(d -> isdir(joinpath(root_dir, d)) && occursin(r"^\d{3}$", d), readdir(root_dir))

# Initialize containers for the selected parameters and force (or stress) - displacement data
all_params = []
all_XY_data = [] #right now the mIFSS data vs displacement is stored

for subdir in subdirs
    # Construct the path to the parameters.json file
    param_file = joinpath(root_dir, subdir, "parameters.json")

    # Read and parse the JSON file
    if isfile(param_file)
        param_data = JSON.parsefile(param_file)

        # Extract parameters using the function
        extracted_params = extract_parameters(param_data)

        # Store the extracted parameters as a vector of parameters
        push!(all_params, extracted_params)

    else
        println("Warning: parameters.json not found in $subdir")
        continue
    end
    
    # Construct paths to the load-displacement data files
    output_data_dir = joinpath(root_dir, subdir, "lhs_$(subdir)_v1_results", "outputdata")
    mifss_file = joinpath(output_data_dir, "mIFSS_from_RF2_BC3_zero_stresses_removed.txt")
    u2_file = joinpath(output_data_dir, "U2_BC3_displacement_corrected.txt")

    # Read the data files
    if isfile(mifss_file) && isfile(u2_file)
        mifss_data = readdlm(mifss_file)[:]
        u2_data = readdlm(u2_file)[:]
        
        # Store the data as a named tuple (X, Y) as Sebi did
        push!(all_XY_data, (X = u2_data, Y = mifss_data))
    else
        println("Warning: Data files not found in $output_data_dir")
    end
end



### --- Visualize existing data set --- ###
using Plots

# Initialize a plot object
p = plot(
    title="Original mIFSS vs. Displacement",
    xlabel="Displacement (mm)",
    ylabel="mIFSS (N/mm^2)",
    size=(1200, 900)
)


# Loop through each dataset and plot it
for i in 1:length(all_XY_data)
    X = all_XY_data[i].X
    Y = all_XY_data[i].Y

    # Check if X and Y have the same length
    if length(X) == length(Y)
        # Plot the original data
        plot!(p, X, Y, label="Sample $i", legend=:outerright)
    else
        # If they don't match, trim both arrays to the length of the shorter one
        min_length = min(length(X), length(Y))
        X = X[1:min_length]
        Y = Y[1:min_length]
        println("Warning: Sample $i had mismatched lengths. Adjusted X and Y to length $min_length.")
        
        # Plot the corrected data
        plot!(p, X, Y, label="Sample $i (corrected)", legend=:outerright)
    end
end

# Save the plot in the new results directory
plot_file = joinpath(results_dir, "00_Original_mIFSS_vs_Displacement.png")
savefig(p, plot_file)

println("Original input data plot saved to: $plot_file")

# Display the plot
display(p)

### --- --- ###

### --- Plot the samles, which didn't run until the end --- ###

# Calculate the overall maximum x value across all samples
#x_max = maximum([maximum(d.X) for d in all_XY_data])
#threshold = 0.90 * x_max  # Define threshold which samples to take in
x_max = 0.10

# Initialize counters
count_reach = 0
count_no_reach = 0
# Loop through the data to count how many reach the threshold
for i in 1:length(all_XY_data)
    X = all_XY_data[i].X
    if maximum(X) >= x_max
        count_reach += 1  # Increment counter for reaching samples
    else
        count_no_reach += 1  # Increment counter for non-reaching samples
    end
end

# Initialize two plot objects: one for samples reaching the threshold, one for those that don't
p_reach = plot(
    title="Samples Reaching x_max = 0.10: $count_reach / $(length(all_XY_data)) samples ",
    xlabel="Displacement (mm)",
    ylabel="mIFSS (N/mm^2)",
    size=(1200, 900)
)

p_no_reach = plot(
    title="Samples NOT Reaching x_max = 0.10: $count_no_reach / $(length(all_XY_data)) samples",
    xlabel="Displacement (mm)",
    ylabel="mIFSS (N/mm^2)",
    size=(1200, 900)
)

# Loop through each dataset and check if it reaches 95% of x_max
for i in 1:length(all_XY_data)
    X = all_XY_data[i].X
    Y = all_XY_data[i].Y

    # Check for mismatched lengths and adjust if necessary
    if length(X) != length(Y)
        min_length = min(length(X), length(Y))
        X = X[1:min_length]
        Y = Y[1:min_length]
        println("Warning: Sample $i had mismatched lengths. Adjusted X and Y to length $min_length.")
    end
    
    if maximum(X) >= x_max
        # Plot the data that reaches the threshold
        plot!(p_reach, X, Y, label="Sample $i", legend=:outerright)
    else
        # Plot the data that doesn't reach the threshold
        plot!(p_no_reach, X, Y, label="Sample $i", legend=:outerright)
    end
end

# Save the plot in the new results directory
plot_file = joinpath(results_dir,
    "00_NICE_SAMPLES.png")
savefig(p_reach, plot_file)

plot_file = joinpath(results_dir,
    "00_NOT_NICE_SAMPLES.png")
savefig(p_no_reach, plot_file)

# Display both plots
display(p_reach)
display(p_no_reach)

# Print out the counts
println("Number of samples that reach the x_max threshold: $count_reach")
println("Number of samples that do NOT reach the x_max threshold: $count_no_reach")





#### SAVE THE FILTERED AND EXTRAPOLATED DATA FOR LATER USE ####
using JLD2

function save_data(filename::String, clean_XY_data, clean_params)
    @save filename clean_XY_data clean_params
    println("Data saved to $filename")
end

save_data(joinpath(results_dir, "clean_data.jld2"), clean_XY_data, clean_params)

#### AND THIS IS FOR LOADING IN THE ANN SCRIPT