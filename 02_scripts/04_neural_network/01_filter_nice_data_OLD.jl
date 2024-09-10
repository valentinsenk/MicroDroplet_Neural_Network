### --- This part loads the parameters for training --- ###
using JSON
using DelimitedFiles

# Define the root directory of samples
root_dir = "C:\\Users\\Senk\\Desktop\\Droplet_Tests_FEA\\01_neural_network_project\\01_data\\parameter_files"
#samples = "mechanical_samples\\v2"
samples = "mechanical_samples\\v2"
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

# Define a list of line styles to iterate through
line_styles = [:solid, :dash, :dot, :dashdot]

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
        line_style = line_styles[mod(i-1, length(line_styles)) + 1]
        plot!(p, X, Y, label="Sample $i", legend=:outerright)
    else
        # If they don't match, trim both arrays to the length of the shorter one
        min_length = min(length(X), length(Y))
        X = X[1:min_length]
        Y = Y[1:min_length]
        println("Warning: Sample $i had mismatched lengths. Adjusted X and Y to length $min_length.")
        
        # Plot the corrected data
        line_style = line_styles[mod(i-1, length(line_styles)) + 1]
        plot!(p, X, Y, label="Sample $i (corrected)", legend=:outerright, linestyle=line_style)
    end
end

# Save the plot in the new results directory
plot_file = joinpath(results_dir, "00_Original_mIFSS_vs_Displacement.png")
savefig(p, plot_file)
display(p)

println("Original input data plot saved to: $plot_file")

### --- Visulize filtered data --- ###

#define x_max threshold; filter out other
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
        line_style = line_styles[mod(i-1, length(line_styles)) + 1]
        plot!(p_reach, X, Y, label="Sample $i", legend=:outerright, linestyle=line_style)
    else
        # Plot the data that doesn't reach the threshold
        line_style = line_styles[mod(i-1, length(line_styles)) + 1]
        plot!(p_no_reach, X, Y, label="Sample $i", legend=:outerright, linestyle=line_style)
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
println("Filtered plots with x_max threshold saved")

### --- Extrapolate data to better filter out stuff --- ###

#Function to extrapolate data if the last values are close to zero (no friction)

function extrapolate_at_zero(X, Y, x_max, zero_threshold=0.1, N_last_points=3, extrap_length=10)
    # Ensure we only check the available number of points in Y
    N_to_check = min(N_last_points, length(Y))
    
    # Check if the last N Y values are close to zero and less than x_max
    if all(abs.(Y[end-N_to_check+1:end]) .<= zero_threshold) && X[end] < x_max
        # If yes, add a few new X points up to x_max
        new_xs = collect(range(X[end], x_max, length=extrap_length))
        new_ys = fill(0.0, extrap_length)  # Add corresponding Y values as zeros
        
        # Return data with added extrapolation
        return vcat(X, new_xs), vcat(Y, new_ys), true
    end
    
    # If no extrapolation needed, return empty arrays
    return [], [], false
end


# initialize plot
p_extrapolated = plot(
    title="Extrapolated data at y=0 to x_max = 0.10",#: $count_extra / $(length(all_XY_data)) samples ",
    xlabel="Displacement (mm)",
    ylabel="mIFSS (N/mm^2)",
    size=(1200, 900)
)

# x_max already defined...
count_extra = 0
# Loop through each dataset and check for extrapolation
for i in 1:length(all_XY_data)
    X = all_XY_data[i].X
    Y = all_XY_data[i].Y
    
    # Adjust X and Y if lengths are mismatched
    if length(X) != length(Y)
        min_length = min(length(X), length(Y))
        X = X[1:min_length]
        Y = Y[1:min_length]
    end
    
    # Apply extrapolation
    X_extrapolated, Y_extrapolated, was_extrapolated = extrapolate_at_zero(X, Y, x_max)
    
    # Plot the extrapolated data
    if was_extrapolated
        # Select a line style from the list (cycle through styles)
        line_style = line_styles[mod(i-1, length(line_styles)) + 1]
        count_extra += 1
        plot!(p_extrapolated, X_extrapolated, Y_extrapolated, label="Sample $i", legend=:outerright, linestyle=line_style)
    end
end

# Update the title with the count of extrapolated samples
plot_title = "Extrapolated data at y=0 to x_max = 0.10: $count_extra / $(length(all_XY_data)) samples"
title!(p_extrapolated, plot_title)

plot_file = joinpath(results_dir,
    "00_EXTRAPOLATED_SAMPLES.png")
savefig(p_extrapolated, plot_file)

display(p_extrapolated)


### look also at data where I got non-zero plateaus ###

function extrapolate_at_plateau(X, Y, x_max, slope_threshold=0.01, N_last_points=20, extrap_length=50)
    slopes = diff(Y[end-N_last_points+1:end]) ./ diff(X[end-N_last_points+1:end])
    
    if all(abs.(slopes) .< slope_threshold) && X[end] < x_max
        plateau_y = mean(Y[end-N_last_points+1:end])
        new_xs = collect(range(X[end], x_max, length=extrap_length))
        new_ys = fill(plateau_y, extrap_length)
        
        println("Extrapolating plateau for sample. X[end]: $(X[end]), plateau_y: $plateau_y")
        return vcat(X, new_xs), vcat(Y, new_ys), true
    end
    
    return [], [], false
end

p_extrapolated_plateau = plot(
    title="Extrapolated data at y=const. to x_max = 0.10",
    xlabel="Displacement (mm)",
    ylabel="mIFSS (N/mm^2)",
    size=(1200, 900)
)

count_extra_plateau = 0

for i in 1:length(all_XY_data)
    X = all_XY_data[i].X
    Y = all_XY_data[i].Y
    
    if length(X) != length(Y)
        min_length = min(length(X), length(Y))
        X = X[1:min_length]
        Y = Y[1:min_length]
    end
    
    X_extrapolated, Y_extrapolated, was_extrapolated = extrapolate_at_plateau(X, Y, x_max, 0.01, 20, 50)
    
    if was_extrapolated
        line_style = line_styles[mod(i-1, length(line_styles)) + 1]
        count_extra_plateau += 1
        plot!(p_extrapolated_plateau, X_extrapolated, Y_extrapolated, label="Sample $i", legend=:outerright, linestyle=line_style)
    end
end

plot_title = "Extrapolated data at y=const. to x_max = 0.10: $count_extra_plateau / $(length(all_XY_data)) samples"
title!(p_extrapolated_plateau, plot_title)

if count_extra_plateau > 0
    plot_file = joinpath(results_dir, "00_EXTRAPOLATED_SAMPLES_PLATEAU.png")
    savefig(p_extrapolated_plateau, plot_file)
    display(p_extrapolated_plateau)
    println("Saved extrapolated plot to: $plot_file")
else
    println("No samples were extrapolated.")
end



### --- Extrapolate data for plateau cases with a drop greater than 50% --- ###
using Statistics
using Plots

# Function for extrapolating data with the custom criteria
function extrapolate_with_custom_criteria(X, Y, x_max; drop_threshold=0.5, relative_tolerance=0.02, extrap_length=10)
    # Maximum Y value
    Y_max = maximum(Y)

    # Identify the last X and its corresponding Y
    X_last = X[end]
    Y_last = Y[end]
    
    # Ensure the curve has dropped at least 50% from the peak value (Y_max)
    if Y_last > drop_threshold * Y_max
        return [], [], false  # No extrapolation if Y_last hasn't dropped enough
    end
    
    # Find the point where X is approximately 90% of the last X value
    X_target = 0.95 * X_last
    idx_near_target = findmin(abs.(X .- X_target))[2]  # Index of closest point to 90% of X_last
    
    # Get corresponding Y value
    Y_target = Y[idx_near_target]
    
    # Check if the Y values at these two points are within the relative tolerance
    if abs(Y_last - Y_target) / Y_max < relative_tolerance && X_last < x_max
        # Plateau detected, so extrapolate Y_last to x_max
        new_xs = collect(range(X_last, x_max, length=extrap_length))
        new_ys = fill(Y_last, extrap_length)  # Keep Y constant for plateau
        
        return vcat(X, new_xs), vcat(Y, new_ys), true
    end
    
    return [], [], false  # No extrapolation needed if criteria not met
end

### --- Plotting extrapolated data --- ###

# Initialize a plot object
p_extrapolated_custom = plot(
    title="Extrapolated Data for Plateau Detection with Custom Criteria",
    xlabel="Displacement (mm)",
    ylabel="mIFSS (N/mm²)",
    size=(1200, 900)
)

# Define the line styles to cycle through
line_styles = [:solid, :dash, :dot, :dashdot]

# Counter for the number of extrapolated samples
count_custom_extrapolated = 0

# Loop through each dataset (already loaded)
for i in 1:length(all_XY_data)
    X = all_XY_data[i].X
    Y = all_XY_data[i].Y
    
    # Ensure X and Y have the same length
    if length(X) != length(Y)
        min_length = min(length(X), length(Y))
        X = X[1:min_length]
        Y = Y[1:min_length]
    end
    
    # Apply the custom extrapolation function
    X_extrapolated, Y_extrapolated, was_extrapolated = extrapolate_with_custom_criteria(X, Y, x_max)
    
    # Plot the extrapolated data
    if was_extrapolated
        # Select a line style from the list (cycle through styles)
        line_style = line_styles[mod(i-1, length(line_styles)) + 1]
        count_custom_extrapolated += 1
        plot!(p_extrapolated_custom, X_extrapolated, Y_extrapolated, label="Sample $i", legend=:outerright, linestyle=line_style)
    end
end

# Update the plot title to reflect the number of extrapolated samples
plot_title_custom = "Extrapolated Data for Plateau Detection: $count_custom_extrapolated / $(length(all_XY_data)) samples"
title!(p_extrapolated_custom, plot_title_custom)

# Save the plot
plot_file_custom = joinpath(results_dir, "00_Custom_Extrapolated_Samples.png")
savefig(p_extrapolated_custom, plot_file_custom)

# Display the plot
display(p_extrapolated_custom)

# Print out the saved file location
println("Custom extrapolated data plot saved to: $plot_file_custom")
















#### SAVE THE FILTERED AND EXTRAPOLATED DATA FOR LATER USE ####
using JLD2

function save_data(filename::String, clean_XY_data, clean_params)
    @save filename clean_XY_data clean_params
    println("Data saved to $filename")
end

save_data(joinpath(results_dir, "clean_data.jld2"), clean_XY_data, clean_params)

#### AND THIS IS FOR LOADING IN THE ANN SCRIPT