using Dates
using Printf
using Logging
using LoggingExtras

#get the current working directory
sample_dir = pwd()

#extract the basename of the directory
sample_basename = basename(sample_dir)

inpfilename = "lhs_$sample_basename"

new_version = "_v1"

# What are the Sets called?
fiber_set = "Set-2" # Fiber-Set     #watch out for case-sensitivity
matrix_set = "Set-1" # Matrix-Set

fiber_material = "Material-2"
matrix_material = "Material-1"

# What should the new parts be called?
part_name_f = "Fiber"
part_name_m = "Matrix" 


### CREATE NEW DIR AND FILES ###

# new clean inp file name
clean_inpfilename = chop(inpfilename, tail=4) * new_version * ".inp"

logfile = chop(clean_inpfilename, tail=4) * "_clean_inp_generation.log"

# Open the log file
logio = open(logfile, "w")


# Create a logger that writes to both console and file
console_logger = ConsoleLogger()
file_logger = SimpleLogger(logio)

# Combine loggers to log to both destinations
multi_logger = TeeLogger(console_logger, file_logger)

# Set the global logger to the combined logger
global_logger(multi_logger)


# Create new directory
new_dir = chop(clean_inpfilename, tail=4) * "_inp-files"
if !isdir(new_dir)
    mkdir(new_dir)
end

# List of new inp filenames
new_inp_files = ["00_Part-1", "00_Part-2", "00_Part-3", "Assembly", "Material", "Step"]

# Create new inp files in the new directory
for file in new_inp_files
    open("$new_dir/$file.inp", "w") do f
        # This will create empty files, add content to these files as needed
        write(f, "")
    end
end

# Create the main clean inp file
open(clean_inpfilename, "w") do f
    write(f, "**$(clean_inpfilename)\n**\n")
    for file in new_inp_files
        write(f, "*Include, Input=$new_dir/$file.inp\n")
    end
    write(f, "**\n")
end

@info "New directory and files created successfully. Main inp file: $clean_inpfilename"

### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

# Assembly instance-name...
instance_f = "$part_name_f-1"
instance_m = "$part_name_m-1"

######################
### PART-3 (Blade) ###
######################

# Function to read the input file, save Part-3, and exclude it from processing
function read_and_process_file(filename::String, new_dir::String)
    # Read the input file into an array of strings
    inpfile = open(filename) do file
        readlines(file)
    end
    
    # Initialize variables
    new_inpfile = String[]
    part3_lines = String[]
    skip_lines = false
    part3 = false
    
    # Iterate over each line to extract and filter out Part-3
    for line in inpfile
        if occursin("*Part, name=Part-3", line)
            skip_lines = true
            part3 = true
        elseif occursin("*End Part", line) && skip_lines
            skip_lines = false
            part3 = false
            push!(part3_lines, line)
            continue
        end
        
        if skip_lines
            if part3
                push!(part3_lines, line)
            end
        else
            push!(new_inpfile, line)
        end
    end
    
    # Save Part-3 to 00_Part-3.inp
    open("$new_dir/00_Part-3.inp", "w") do f
        for line in part3_lines
            write(f, line * "\n")
        end
    end
    
    return new_inpfile
end

# Read the input file, save Part-3, and filter out Part-3 for further processing
inpfile = read_and_process_file(inpfilename, new_dir)

@info "[PART-3] Part-3 written to file"


@info "[PART-1/-2] Start splitting PART-ALL to PART-1 and PART-2..."

###########################################
###########################################
### SPLIT PART-ALL to PART-1 and PART-2 ###
###########################################
###########################################

### get the index, where certain keywords are
node_entry = findfirst(contains("*Node"), inpfile)
element_entry = findfirst(contains("*Element, type="), inpfile)
nset_fiber_entry = findfirst(contains("*Nset, nset=$fiber_set"), inpfile)
elset_fiber_entry = findfirst(contains("*Elset, elset=$fiber_set"), inpfile)
nset_matrix_entry = findfirst(contains("*Nset, nset=$matrix_set"), inpfile)
elset_matrix_entry = findfirst(contains("*Elset, elset=$matrix_set"), inpfile)

### function for saving all data after keyword to Array{String,1}
function get_strings_after_keyword(keyword_index::Int64)
    local var = String[] #this is the Array of Strings, where stuff after keyword gets pushed in
    for i in keyword_index+1:length(inpfile)
        bool = occursin("*", inpfile[i]) #gives back true, if "*" occurs in string
        if bool == false
            push!(var, inpfile[i])
        else
            break
        end
    end
    return var
end

### apply function for all variables needed
nodes = get_strings_after_keyword(node_entry)
elements = get_strings_after_keyword(element_entry)
nset_fiber = get_strings_after_keyword(nset_fiber_entry)
elset_fiber = get_strings_after_keyword(elset_fiber_entry)
nset_matrix = get_strings_after_keyword(nset_matrix_entry)
elset_matrix = get_strings_after_keyword(elset_matrix_entry)

###################################
####    PROCESSING NODES....   ####
###################################

#get one node-number per line from nset...
function nset_to_array(nset::Array{String,1})
    local var = String[]
    for i in 1:length(nset)
        splitstrings = split(nset[i]) #split strings into array of substrings (= every node is in new line)
        for i in 1:length(splitstrings)
            push!(var, splitstrings[i])
        end
    end
    var = replace.(var, r"[,]" => "") #get rid of the commas after the node-numbers with regex
    return var
end

nset_f_array = nset_to_array(nset_fiber)
nset_m_array = nset_to_array(nset_matrix)

#identify shared nodes, which have to be doubled
shared_nodes = String[]
# loop over nsets and compare
for i in 1:length(nset_m_array)
    for j in 1:length(nset_f_array)      
        if nset_m_array[i] == nset_f_array[j]
            push!(shared_nodes, nset_f_array[j])
        end
    end
end

# function, whichs links shared node numbers to full list:
# this only works, if nodes and elements are continousliy written from e.g. 1:end
function link_to_full_string(link::Array{String,1}, nodes_or_elements::Array{String,1})
    #this functions links e.g. shared_nodes-numbers to the corresponding full nodes-strings (nodes+coordinates)
    local INT = Int64[]
    local list = String[]
    for i in 1:length(link)
        push!(INT, parse(Int64, link[i]))
        push!(list, nodes_or_elements[INT[i]])
    end
    return list
end

shared_nodes_coord_list = link_to_full_string(shared_nodes, nodes)

#add new nodes to the list (last node number +1)
shared_nodes_new = String[]
for i in 1:length(shared_nodes)
    push!(shared_nodes_new, string(i+length(nodes)))
end

#link coordinates from shared_nodes_coord_list to new nodes
new_nodes_list = String[]
for i in 1:length(shared_nodes)
    push!(new_nodes_list ,replace(shared_nodes_coord_list[i], r"^[ 0-9]+," => "")) #delete node numbers from string 
    new_nodes_list[i] = shared_nodes_new[i]*","*new_nodes_list[i] #generate new node numbers with coordinates from the shared ones
end

# all nodes (old+new ones):
nodes_new = copy(nodes)
for i in 1:length(new_nodes_list)
    push!(nodes_new, new_nodes_list[i])
end

#----------------------------#
#--- nodes for fiber-part ---# assign old nodes
#----------------------------#

nset_f_array_list = link_to_full_string(nset_f_array, nodes)

#-----------------------------#
#--- nodes for matrix-part ---# assign new nodes
#-----------------------------#

nset_m_array_new = copy(nset_m_array) #copy to be able to compare later

for i in 1:length(nset_m_array)
    for j in 1:length(shared_nodes)
        if nset_m_array[i] == shared_nodes[j] #only overwrite the lines, where it matches the shared_nodes
            nset_m_array_new[i] = string(j+length(nodes))
        end
    end
end

nset_m_array_new_list = link_to_full_string(nset_m_array_new, nodes_new)


@info "[PART-1/-2] NODES: Shared nodes on matrix/fiber interface identified, doubled and assigned to matrix or fiber"

#-----------------------------#
### - SAVE NODES TO FILES - ###
#-----------------------------#
# Function to save nodes to a specified file
function save_nodes_to_file(nodes::Array{String,1}, filename::String, part_name::String, part_type::String)
    open(filename, "w") do f
        write(f, "** $part_type PART\n")
        write(f, "*Part, name=$part_name\n")
        write(f, "*Node\n")
        for node in nodes
            write(f, node * "\n")
        end
    end
end

# Set-1 to Part-1...
#fiber_set = lowercase(fiber_set) #ensure case-insensitive matching
#matrix_set = lowercase(matrix_set)
part1_type = (fiber_set in ["Set-1", "SET-1"]) ? "FIBER" : "MATRIX"
part2_type = (matrix_set in ["Set-2", "SET-2"]) ? "MATRIX" : "FIBER"

# Save fiber nodes to the correct part file
if part1_type == "FIBER"
    save_nodes_to_file(nset_f_array_list, "$new_dir/00_Part-1.inp", "Part-1", "FIBER")
    save_nodes_to_file(nset_m_array_new_list, "$new_dir/00_Part-2.inp", "Part-2", "MATRIX")
else
    save_nodes_to_file(nset_f_array_list, "$new_dir/00_Part-2.inp", "Part-2", "FIBER")
    save_nodes_to_file(nset_m_array_new_list, "$new_dir/00_Part-1.inp", "Part-1", "MATRIX")
end

@info "[PART-1/-2] NODES: Nodes saved to corresponding Part-1 ($part1_type) and Part-2 ($part2_type) files."


###################################
###    PROCESSING ELEMENTS...  ####
###################################

function elset_to_array(elset::Array{String,1})
    local var = String[]

    ### IF ELEMENTS ARE DEFINED VIA RANGE (=generate in ABAQUS), this has to be done:
    if size(elset)[1] == 1 #indicates, that "generate"; from 1 to n is used
        local splitstrings = split(elset[1])
        splitstrings = replace.(splitstrings, r"[,]" => "")
        pop!(splitstrings) #delete last element in array

        range1 = Int64[]
        push!(range1, parse(Int64, splitstrings[1]))
        push!(range1, parse(Int64, splitstrings[2]))

        range1 = collect(range1[1]:range1[2])
        
        for i in 1:length(range1)
            push!(var, string(range1[i]))
        end
    ### OTHERWISE THE SAME STUFF AS in function "nset_to_array"
    else
        for i in 1:length(elset)
            local splitstrings = split(elset[i]) #split strings into array of substrings
            for i in 1:length(splitstrings)
                push!(var, splitstrings[i])
            end
        end          
    end
    var = replace.(var, r"[,]" => "")
    return var
end

elset_f_array = elset_to_array(elset_fiber)
elset_m_array = elset_to_array(elset_matrix)


#-------------------------------#
#--- elements for fiber-part ---# assign elements
#-------------------------------#

elset_f_array_list = link_to_full_string(elset_f_array, elements)

#--------------------------------#
#--- elements for matrix-part ---# assign elements and change old node-no. (node-connectivity) to new ones
#--------------------------------#

elset_m_array_list = link_to_full_string(elset_m_array, elements)

# nodes in elset_m_array_list have to be changed to new nodes

elset_m_array_list_new = String[]

for i in 1:length(elset_m_array_list)

    local splitstrings = split(elset_m_array_list[i])
    local splitstrings = replace.(splitstrings, r"[,]" => "") #splitstrings magically converts here from Array{SubString{String},1} to Array{String,1}
    #skip first entry in the search, because it's the element; the others are the nodes
    for j in 1:length(shared_nodes)
        for k in 2:length(splitstrings) #skip first one (=element not node)
            if shared_nodes[j] == splitstrings[k]
                splitstrings[k] = shared_nodes_new[j]
            end
        end
    end

    local new_string = join(splitstrings, ", ")
    push!(elset_m_array_list_new, new_string)

end

@info "[PART-1/-2] ELEMENTS: Node-Connectivity changed for MATRIX (new nodes for matrix-elements assigned)"

# Extract the element type from the input file
element_type = ""
for line in inpfile
    if occursin("*Element, type=", line)
        global element_type = String(match(r"\*Element, type=(\w+)", line).captures[1])
        break
    end
end

# Function to append elements to a specified file
function append_elements_to_file(elements::Array{String,1}, filename::String, element_type::String)
    open(filename, "a") do f  # "a" for append mode
        write(f, "*Element, type=$element_type\n")
        for element in elements
            write(f, element * "\n")
        end
    end
end

# Append fiber elements to the correct part file
if part1_type == "FIBER"
    append_elements_to_file(elset_f_array_list, "$new_dir/00_Part-1.inp", element_type)
    append_elements_to_file(elset_m_array_list_new, "$new_dir/00_Part-2.inp", element_type)
else
    append_elements_to_file(elset_f_array_list, "$new_dir/00_Part-2.inp", element_type)
    append_elements_to_file(elset_m_array_list_new, "$new_dir/00_Part-1.inp", element_type)
end

@info "[PART-1/-2] ELEMENTS: Elements appended to corresponding Part-1 ($part1_type) and Part-2 ($part2_type) files."



### --- Element Sets --- ###

## Function to append element sets to a specified file (using the original elsets)
#function append_element_sets_to_file(elset::Array{String,1}, filename::String, set_name::String)
#    open(filename, "a") do f  # "a" for append mode
#        write(f, "*Elset, elset=$set_name, generate\n")
#        for element_set in elset
#            write(f, element_set * "\n")
#        end
#    end
#end

# NOT TESTED: Function to append element sets to a specified file (using the original elsets)
function append_element_sets_to_file(elset::Array{String,1}, filename::String, set_name::String)
    open(filename, "a") do f  # "a" for append mode
        if length(elset) == 1
            write(f, "*Elset, elset=$set_name, generate\n")
        else
            write(f, "*Elset, elset=$set_name\n")
        end
        for element_set in elset
            write(f, element_set * "\n")
        end
    end
end


# Append fiber and matrix element sets to the correct part file
if part1_type == "FIBER"
    append_element_sets_to_file(elset_fiber, "$new_dir/00_Part-1.inp", "Set-1")
    append_element_sets_to_file(elset_matrix, "$new_dir/00_Part-2.inp", "Set-2")
else
    append_element_sets_to_file(elset_fiber, "$new_dir/00_Part-2.inp", "Set-2")
    append_element_sets_to_file(elset_matrix, "$new_dir/00_Part-1.inp", "Set-1")
end

@info "[PART-1/-2] ELEMENT SETS: Element sets appended to corresponding Part-1 ($part1_type) and Part-2 ($part2_type) files."


### --- Section entries --- ###

nset_BC_fiber_entry = findfirst(contains("*Nset, nset=Set-BC-3"), inpfile)
elset_BC_fiber_entry = findfirst(contains("*Elset, elset=Set-BC-3"), inpfile)

ori_fiber_entry = findfirst(contains("*Orientation"), inpfile)
section_f_entry = findfirst(contains("*Solid Section, elset=$fiber_set"), inpfile)
section_m_entry = findfirst(contains("*Solid Section, elset=$matrix_set"), inpfile)

nset_BC_fiber = get_strings_after_keyword(nset_BC_fiber_entry)
elset_BC_fiber = get_strings_after_keyword(elset_BC_fiber_entry)

ori_fiber = get_strings_after_keyword(ori_fiber_entry)
section_f = get_strings_after_keyword(section_f_entry)
section_m = get_strings_after_keyword(section_m_entry) 

if part1_type == "FIBER"
    open("$new_dir/00_Part-1.inp", "a") do f  # "a" for append mode
        write(f, "*Nset, nset=Set-BC-3\n")
        for lines in nset_BC_fiber
            write(f, lines * "\n")
        end
        write(f, "*Elset, elset=Set-BC-3\n")
        for lines in elset_BC_fiber
            write(f, lines * "\n")
        end
        write(f, "*Orientation, name=Ori-1\n")
        for lines in ori_fiber
            write(f, lines * "\n")
        end
        write(f, "*Solid Section, elset=$fiber_set, orientation=Ori-1, material=$fiber_material\n")
        write(f, section_f[1] * "\n")
        write(f, "*End Part\n")
    end
    open("$new_dir/00_Part-2.inp", "a") do f  # "a" for append mode
        write(f, "*Solid Section, elset=$matrix_set, material=$matrix_material\n")
        write(f, section_m[1] * "\n")
        write(f, "*End Part\n")
    end
else
    open("$new_dir/00_Part-2.inp", "a") do f  # "a" for append mode
        write(f, "*Nset, nset=Set-BC-3\n")
        for lines in nset_BC_fiber
            write(f, lines * "\n")
        end
        write(f, "*Elset, elset=Set-BC-3\n")
        for lines in elset_BC_fiber
            write(f, lines * "\n")
        end
        write(f, "*Orientation, name=Ori-1\n")
        for lines in ori_fiber
            write(f, lines * "\n")
        end
        write(f, "*Solid Section, elset=$fiber_set, orientation=Ori-1, material=$fiber_material\n")
        write(f, section_f[1] * "\n")
        write(f, "*End Part\n")
    end
    open("$new_dir/00_Part-1.inp", "a") do f  # "a" for append mode
        write(f, "*Solid Section, elset=$matrix_set, material=$matrix_material\n")
        write(f, section_m[1] * "\n")
        write(f, "*End Part\n")
    end
end

@info "[PART-1/-2] SECTIONS: Sections written and part ended"

### --- --- ###


###########################################
###########################################
### ASSEMBLY ### Watch out! This part is heavily written with Chatgpt
###########################################
###########################################

# Function to extract the assembly section from the input file
function extract_assembly_section(inpfile::Array{String,1})
    assembly_section = String[]
    inside_assembly = false

    for line in inpfile
        if occursin("*Assembly", line)
            inside_assembly = true
        end
        if inside_assembly
            push!(assembly_section, line)
            if occursin("*End Assembly", line)
                break
            end
        end
    end
    return assembly_section
end

# Function to filter out general contact related lines
function filter_general_contact_lines(assembly_section::Array{String,1})
    filtered_section = String[]
    skip_lines = false

    for line in assembly_section
        if occursin("*Elset, elset=__GENERAL_CONTACT", line) || occursin("*Surface, type=ELEMENT, name=_GENERAL_CONTACT", line)
            skip_lines = true
        end
        if !skip_lines
            push!(filtered_section, line)
        end
        if skip_lines && occursin("*", line) && !occursin("*Elset, elset=__GENERAL_CONTACT", line) && !occursin("*Surface, type=ELEMENT, name=_GENERAL_CONTACT", line)
            skip_lines = false
        end
    end
    return filtered_section
end

# Function to duplicate and modify Part-ALL instances
function modify_instances(assembly_section::Array{String,1})
    modified_section = String[]
    inside_instance = false
    part_all_instance_lines = String[]
    part_all_instance_detected = false

    for line in assembly_section
        if occursin("*Instance, name=Part-ALL-1, part=Part-ALL", line)
            inside_instance = true
            part_all_instance_detected = true
            part_all_instance_lines = [replace(line, "Part-ALL-1" => "Part-1-1", "Part-ALL" => "Part-1")]
        elseif occursin("*End Instance", line) && inside_instance
            push!(part_all_instance_lines, line)
            # Add the lines for Part-1 instance
            append!(modified_section, part_all_instance_lines)
            # Modify and add lines for Part-2 instance
            part_2_instance_lines = replace.(part_all_instance_lines, "Part-1-1" => "Part-2-1", "Part-1" => "Part-2")
            append!(modified_section, part_2_instance_lines)
            inside_instance = false
        elseif inside_instance
            push!(part_all_instance_lines, line)
        else
            push!(modified_section, line)
        end
    end

    # Ensure the assembly section ends with "*End Assembly"
    if !occursin("*End Assembly", modified_section[end])
        push!(modified_section, "*End Assembly")
    end

    return modified_section
end

# Function to replace Part-ALL-1 with Part-2-1 for Nset and Elset
function replace_part_all_with_fiber_instance(assembly_section::Array{String,1})
    modified_section = String[]
    for line in assembly_section
        if occursin("*Nset, nset=Set-BC-3, instance=Part-ALL-1", line) || occursin("*Elset, elset=Set-BC-3, instance=Part-ALL-1", line)
            push!(modified_section, replace(line, "Part-ALL-1" => "Part-2-1"))
        else
            push!(modified_section, line)
        end
    end
    return modified_section
end

# Read the input file and extract the assembly section
assembly_section = extract_assembly_section(inpfile)

# Filter out general contact related lines
filtered_assembly_section = filter_general_contact_lines(assembly_section)

# Modify instances for Part-ALL to Part-1 and Part-2
modified_assembly_section = modify_instances(filtered_assembly_section)

# Replace Part-ALL-1 with Part-2-1 for Nset and Elset definitions
final_assembly_section = replace_part_all_with_fiber_instance(modified_assembly_section)

# Save the cleaned and modified assembly section to the assembly file
open("$new_dir/Assembly.inp", "w") do f
    for line in final_assembly_section
        write(f, line * "\n")
    end
end

@info "[ASSEMBLY] Assembly section cleaned, modified, and written to Assembly.inp"



###########################################
###########################################
### MATERIALS ### Watch out! This part is heavily written with Chatgpt
###########################################
###########################################

# Function to extract the materials section from the input file
function extract_materials_section(inpfile::Array{String,1})
    materials_section = String[]
    inside_materials = false

    for line in inpfile
        if occursin("** MATERIALS", line)
            inside_materials = true
        end
        if inside_materials
            push!(materials_section, line)
            if occursin("** BOUNDARY CONDITIONS", line)
                break
            end
        end
    end
    return materials_section
end

# Read the input file and extract the materials section
materials_section = extract_materials_section(inpfile)

# Save the materials section to the Material.inp file
open("$new_dir/Material.inp", "w") do f
    for line in materials_section
        write(f, line * "\n")
    end
end

@info "[MATERIALS] Materials section extracted and written to Material.inp"


###########################################
###########################################
### BOUNDARY, STEP and OUTPUT #############
###########################################
###########################################

# Function to extract the ending section from the input file
function extract_ending_section(inpfile::Array{String,1})
    ending_section = String[]
    inside_ending = false

    for line in inpfile
        if occursin("** BOUNDARY CONDITIONS", line)
            inside_ending = true
        end
        if inside_ending
            push!(ending_section, line)
        end
    end
    return ending_section
end

# Function to filter out surface property assignment lines and everything after it
function filter_surface_property_assignment(lines::Array{String,1})
    filtered_lines = String[]
    skip_lines = false

    for line in lines
        if occursin("*Surface Property Assignment, property=GEOMETRIC CORRECTION", line)
            skip_lines = true
        end
        if !skip_lines
            push!(filtered_lines, line)
        end
        if skip_lines && occursin("*End Step", line)
            skip_lines = false
        end
    end
    return filtered_lines
end

# Predefined step and boundary conditions to append
predefined_step = [
    "**",
    "** STEP: Step-1",
    "**",
    "*Step, name=Step-1, nlgeom=YES, inc=500",
    "*Static",
    "0.01, 1., 1e-11, 0.01",
    "**",
    "** BOUNDARY CONDITIONS",
    "**",
    "** Name: BC-2 Type: Displacement/Rotation",
    "*Boundary",
    "Part-2-1.Set-BC-3, 1, 1",
    "Part-2-1.Set-BC-3, 2, 2, 0.10",
    "Part-2-1.Set-BC-3, 3, 3",
    "**",
    "** CONTROLS",
    "*Controls, reset",
    "*Controls, parameters=time incrementation",
    "8, 10, , , , , , 60, , ,",
    "*Controls, parameters=line search",
    "5, 1, 0.0001, 0.25, 0.1",
    "**",
    "** OUTPUT REQUESTS",
    "**Restart, write, number interval=200, time marks=yes",
    "*Print, plasticity=YES",
    "*Output, field, variable=PRESELECT",
    "*Output, field",
    "*Contact Output",
    "CDISP, CSTRESS, CSDMG, CSMAXSCRT, CSTRESSERI, CNAREA, CSTATUS",
    "*Node Output",
    "RF, U, CF, VF",
    "**, PHILSM, PSILSM",
    "*Element Output, directions=YES",
    "S, E, LE, PE, PEEQ, PEMAG,",
    "*Output, history, variable=PRESELECT",
    "*End Step"
]

# Read the input file and extract the ending section
ending_section = extract_ending_section(inpfile)

# Filter out surface property assignment lines and everything after it
filtered_ending_section = filter_surface_property_assignment(ending_section)

# Save the cleaned and modified ending section to the Step.inp file
open("$new_dir/Step.inp", "w") do f
    for line in filtered_ending_section
        write(f, line * "\n")
    end
    for line in predefined_step
        write(f, line * "\n")
    end
end

@info "[STEP] Step section cleaned, predefined step appended, and written to Step.inp"



#### Save embedded area to inp-files ####

log_filename = chop(inpfilename, tail=4) * "_model_generation.log" # logfile name

# extract embedded area
open(log_filename, "r") do file
    for line in eachline(file)
        if occursin("The EXCACT embedded area of this fiber-droplet configuration using", line)
            parts = split(line)
            global embedded_area = parse(Float64, parts[end-1])
        end
    end
end

emb_area_filename = joinpath(new_dir, "xx__embedded_area__xx.txt")

open(emb_area_filename, "w") do file
    write(file, "$embedded_area\n")
end

@info "Embedded area extracted from model_generation log-file and saved to the inp-files directory."

########################################

println("Script finished.")

# Close the log file
close(logio)


