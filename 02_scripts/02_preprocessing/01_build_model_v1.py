#################################################
import logging
import sys
from datetime import datetime
import os
import json
from abaqus import session
#################################################

# Directory for the current sample (assuming the correct path is set via the slurm bash-script)
sample_dir = os.getcwd()

# define inpfilename 
inpfile = 'lhs_' + os.path.basename(sample_dir)

# Load parameters from JSON file in the current directory
json_file_path = os.path.join(sample_dir, "parameters.json")
with open(json_file_path, 'r') as json_file:
    params = json.load(json_file)

### --- Setting up log-file --- ###

log_filename = os.path.join(sample_dir, inpfile + '_model_generation.log')

logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

### this scripts name
script_name = '01_build_model_v1.py'

# Redirect stdout to log file
class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass

sys.stdout = Logger(log_filename)

# Add header to log file
now = datetime.now()
header = f"Model generation {inpfile} with {script_name} on {now.strftime('%d-%m-%y %H:%M:%S')}\n\n"
with open(log_filename, 'a') as log_file:
    log_file.write(header)

print(f"Generated Input File name: {inpfile}")

#########################

### Standard Abaqus import ###
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=100.907814025879, 
    height=123.342597961426)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
##############################

### some standard session stuff; don't know if needed
session.viewports['Viewport: 1'].setValues(displayedObject=None)
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
###

### These imports are for DROPLET calculations ###
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipkinc, ellipeinc
import math
##################################################

### INPUT PARAMETERS ###

### Geometrical parameters for fiber-droplet system and blades ###
h = round(params['geometrical_parameters']['droplet_diameter']/2, 4)  # bead height in mm (=Droplet diameter / 2)
L = round(params['geometrical_parameters']['ratio_droplet_embedded_length']*2*h, 4)  # embedded length in mm
r = round(params['geometrical_parameters']['fiber_diameter']/2, 4) #0.0105  # fiber radius in mm (=Fiber diameter / 2)
o = round(params['geometrical_parameters']['contact_angle'], 2) #25.7  # contact angle in degrees (=in Pantalonis paper this is probaly 90deg minus contact angle)
ell = round(params['geometrical_parameters']['elliptical_fiber_ratio'], 2) #1.5 #crosssection ratio of elliptical fiber (define 1.0001 for "circle") 
rot = round(params['geometrical_parameters']['fiber_rotation'], 1) #0.0, 30.0, 45.0, 60.0 rotation of elliptical fiber 

### --- Moved dynamical blade distance stuff from sampling script here ... --- ###

normalized_blade_distance = params['geometrical_parameters']['blade_distance']

fiber_diameter = params['geometrical_parameters']['fiber_diameter']
elliptical_fiber_ratio = params['geometrical_parameters']['elliptical_fiber_ratio']
fiber_rotation = params['geometrical_parameters']['fiber_rotation']
droplet_diameter = params['geometrical_parameters']['droplet_diameter']

# Calculate min and max blade distance
min_blade_distance = fiber_diameter * np.sqrt(elliptical_fiber_ratio**2 * np.cos(np.radians(fiber_rotation))**2 + np.sin(np.radians(fiber_rotation))**2)
max_blade_distance = droplet_diameter

# Rescale the normalized blade distance value
actual_blade_distance = min_blade_distance + normalized_blade_distance * (max_blade_distance - min_blade_distance)

### --- ------------------------------------------------------------------ --- ###

b = round(actual_blade_distance/2, 4) #CHANGED AFTER MOVING PARAMETER FROM JULIA SCRIPT HERE
blade = 0 #blade 0: 20 degrees angle, 1:flat, 2: rounded with fillet radius=0.005
l_free = round(2*r, 4) #3000 # free fiber length in mm
l_end = round(L, 4) # length of fiber from loose end to the end of the droplet

### Some mechanical parameters for fiber-droplet system ###
G_modeI = round(params['mechanical_parameters']['GI'], 4) #interface normal energy
G_modeII = round(params['mechanical_parameters']['GII,GIII'], 4) #interface shear energy
tI = round(params['mechanical_parameters']['tI=tII=tIII']*9.74, 4) #interface strength
interface_fric = round(params['mechanical_parameters']['interface_friction'], 2)#friction const. between fiber and droplet
blade_fric = round(params['mechanical_parameters']['blade_friction'], 2)#friction between blade and droplet

### Mesh Parameters ###
b_seed = 0.0040 # Finest Seed for Blades in mm #Choose e.g. 0.002 for finest, 0.008 for most coarse
fd_seed = 2 # Choose between 1 - 4 #1 = finest seed for fiber-droplet system

### --- log file --- ###

logging.info("Sript Parameters ")
logging.info(f"-----------------------------------------")
# Log input parameters
logging.info(f"h = {h} mm #bead height")
logging.info(f"L = {L} mm #embedded length")
logging.info(f"r = {r} mm #fiber radius")
logging.info(f"o = {o} degrees #contact angle")
logging.info(f"l_free = {l_free} mm #free fiber length")
logging.info(f"l_end = {l_end} mm # fiber length from loose end to droplet")
logging.info(f"ell = {ell} #crosssection ratio of elliptical fiber")
logging.info(f"rot = {rot} degrees #0.0, 30.0, 45.0, 60.0 rotation of elliptical fiber")
logging.info(f"blade = no. {blade} #blade type (blade 0: 20 degrees angle, 1:flat, 2: rounded with fillet radius=0.005)")
logging.info(f"normalized_blade_distance = {normalized_blade_distance}")
logging.info(f"b = {b} mm #actual blade distance (from middle axis)")
logging.info(f"-----------------------------------------")
logging.info(f"GI = {G_modeI} N/mm #")
logging.info(f"GII = {G_modeII} N/mm #")
logging.info(f"tI = tII = tIII = {tI} N/mm² #")
logging.info(f"interface_fric = {interface_fric} #coulomb friction parameter")
logging.info(f"blade_fric = {blade_fric} #coulomb friction parameter")
logging.info(f"-----------------------------------------")
logging.info(f"b_seed = {b_seed} mm #finest mesh seed for blades")
logging.info(f"fd_seed = no. {fd_seed} #finest mesh seed for fiber-droplet system")

print(f"")

# Abaqus session options to log messages
session.journalOptions.setValues(replayGeometry=COORDINATE, recoverGeometry=COORDINATE)

# Function to capture messages generated during mesh generation
def generate_mesh_and_capture_messages(part):
    # Temporarily redirect stdout and stderr to capture messages
    import io
    import contextlib

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        part.generateMesh()
    
    # Print captured messages to ensure they are logged
    captured_output = buffer.getvalue()
    print(captured_output)



########################################################

### Create a step
mdb.models['Model-1'].StaticStep(name='Step-1', previous='Initial')

#### This function calculates the x,y-data for the droplet geometry ###
#### This is based on the Carroll's method (Carroll1976, Ash2003) ###
def bead_profile(h, r, L, o):
    # Calculate parameters
    a = (h * np.cos(np.radians(o)) - r) / (h - r * np.cos(np.radians(o)))
    k = np.sqrt((h**2 - (a * r)**2) / h**2)
    theta_max = np.arcsin(1 / k * np.sqrt(1 - (r**2 / h**2)))
    
    # Profile calculation
    phi = np.linspace(0, theta_max, 50)
    x = h * np.sqrt(1 - k**2 * np.sin(phi)**2)
    y = a * r * ellipkinc(phi, k**2) + h * ellipeinc(phi, k**2)

    # Scale y-values to range from 0 to L/2 to ensure the embedded length stays correct
    y_scaled = (y - np.min(y)) * (L / 2) / (np.max(y) - np.min(y))
    
    return x, y_scaled
#######################################################################

### Execute function
x, y = bead_profile(h, r, L, o)
###

######## FOR LATER IN THE SCRIPT: move blade and droplet closer together in the y-direction
# to avoid too many increments with no force before touching
x_filtered = x[x < b]
closest_x = np.max(x_filtered)
closest_index = np.where(x == closest_x)[0][0]
# Get the corresponding y-value
blade_distance_y = y[closest_index]
########

### DROPLET GEOMETRY (Part-1) ###
### open geometry / sketch editor in ABAQUS 
#del mdb.models['Model-1'].sketches['__profile__'] ##eventual
s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
    sheetSize=2.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
s.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, 100.0)) # this is probably the revolving axis
s.FixedConstraint(entity=g[2]) #don't know what this is

# Create splines with x,y-data
s.Spline(points=[(x[i], y[i]) for i in range(len(x))])
s.Spline(points=[(x[i], -y[i]) for i in range(len(x))])

s.Line(point1=(x[-1], y[-1]), point2=(0.0, y[-1])) 
s.Line(point1=(0.0, y[-1]), point2=(0.0, -y[-1]))
s.Line(point1=(0.0, -y[-1]), point2=(x[-1], -y[-1])) #connect ends

p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=THREE_D, 
    type=DEFORMABLE_BODY)
p = mdb.models['Model-1'].parts['Part-1']
p.BaseSolidRevolve(sketch=s, angle=360.0, flipRevolveDirection=OFF)
s.unsetPrimaryObject()
p = mdb.models['Model-1'].parts['Part-1']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
del mdb.models['Model-1'].sketches['__profile__']


### FIBER GEOMETRY (Part-2) ###
s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=20.0)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)

# draw circle or ellipse for fiber cross section
if ell == 1.0:
    s.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(r, 0.0))
else:
    s.EllipseByCenterPerimeter(center=(0.0, 0.0), axisPoint1=(ell * r, 0.0), axisPoint2=(0.0, r))

p = mdb.models['Model-1'].Part(name='Part-2', dimensionality=THREE_D, 
    type=DEFORMABLE_BODY)
p = mdb.models['Model-1'].parts['Part-2']
p.BaseSolidExtrude(sketch=s, depth=l_free+L+l_end) #depth is sum of l_free, L and l_end
s.unsetPrimaryObject()
p = mdb.models['Model-1'].parts['Part-2']


# make sets for geometry...
p = mdb.models['Model-1'].parts['Part-1']
c = p.cells
#cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
cells = p.cells[:] #select all cells instead of mask
p.Set(cells=cells, name='Set-1')

p = mdb.models['Model-1'].parts['Part-2']
c = p.cells
#cells = c.getSequenceFromMask(mask=('[#1 ]', ), )
cells = p.cells[:] #select all cells instead of mask
p.Set(cells=cells, name='Set-2')


### Assembly Drop and fiber, move in position then CUT AND MERGE...
a1 = mdb.models['Model-1'].rootAssembly
p = mdb.models['Model-1'].parts['Part-1']
a1.Instance(name='Part-1-1', part=p, dependent=ON)
p = mdb.models['Model-1'].parts['Part-2']
a1.Instance(name='Part-2-1', part=p, dependent=ON)

a1.rotate(instanceList=('Part-2-1', ), axisPoint=(0.0, 0.0, 0.0), 
    axisDirection=(10.0, 0.0, 0.0), angle=270.0)

a1.translate(instanceList=('Part-2-1', ), vector=(0.0, -l_end-L/2, 0.0))

a1.InstanceFromBooleanCut(name='Part-1_CUT', 
    instanceToBeCut=mdb.models['Model-1'].rootAssembly.instances['Part-1-1'], 
    cuttingInstances=(a1.instances['Part-2-1'], ), originalInstances=SUPPRESS)

a1.features['Part-2-1'].resume()

a1.InstanceFromBooleanMerge(name='Part-ALL', instances=(
    a1.instances['Part-2-1'], a1.instances['Part-1_CUT-1'], ), 
    keepIntersections=ON, originalInstances=SUPPRESS, domain=GEOMETRY)
p1 = mdb.models['Model-1'].parts['Part-ALL']

del mdb.models['Model-1'].sketches['__profile__']

#############
### BLADE ###
#############

# function partition blade in middle, make set, and set mesh type
def partition_meshtype_set():
    ### Partition the BLADE for MESHING...
    p = mdb.models['Model-1'].parts['Part-3']
    pickedCells = p.cells[:] #THIS APPROACH SELECTS ALL CELLS
    pickedRegions =(pickedCells, ) #regions here are just cells in a tuple??
    # And this is a different approach for selecting all cells:
    #pickedCells = p.cells.getByBoundingBox(xMin=-1e30, xMax=1e30, yMin=-1e30, yMax=1e30, zMin=-1e30, zMax=1e30)
    #
    ### Partition Cell in the 'middle' -define cutting plane by points:
    point1 = (0.0, 0.0, 2.0)  # These are the coordinates of the point of first contact 
    point2 = (-1.0, 1.0, 2.0)  # random points in the desired plane
    point3 = (-1.0, -1.0, 2.0)  # random points in the desired plane
    p.PartitionCellByPlaneThreePoints(cells=pickedCells, point1=point1, point2=point2, point3=point3)
    # Refresh after partition:
    pickedCells = p.cells[:] 
    pickedRegions =(pickedCells, )
    # Assign MESH Type
    p.setMeshControls(regions=pickedCells, elemShape=TET, technique=FREE)
    elemType1 = mesh.ElemType(elemCode=C3D20R)
    elemType2 = mesh.ElemType(elemCode=C3D15)
    elemType3 = mesh.ElemType(elemCode=C3D10)
    p = mdb.models['Model-1'].parts['Part-3']
    #
    p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, 
        elemType3))
    ###
    p.Set(cells=pickedCells, name='Set-3') #also refer to as Set-3
    ##

def blade_assembly_translation():
    ### Blade distance - this should be independant of blade type ###
    a = mdb.models['Model-1'].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    p = mdb.models['Model-1'].parts['Part-3']
    a.Instance(name='Part-3-1', part=p, dependent=ON)
    a.Instance(name='Part-3-2', part=p, dependent=ON)
    ##
    # translate to origin + embedded length / 2
    a.translate(instanceList=('Part-3-1', ), vector=(0.0, blade_distance_y, -2.0))
    a.translate(instanceList=('Part-3-2', ), vector=(0.0, blade_distance_y, -2.0))
    ##
    # mirror 1 blade instance
    a.rotate(instanceList=('Part-3-2', ), axisPoint=(0.0, 0.0, 0.0), 
        axisDirection=(0.0, 10.0, 0.0), angle=180.0)
    ##
    # now translate both according to the b
    a.translate(instanceList=('Part-3-1', ), vector=(-b, 0.0, 0.0))
    a.translate(instanceList=('Part-3-2', ), vector=(b, 0.0, 0.0))


if blade == 0: 
    #
    ########################
    ### BLADE 20 degrees ###
    ########################
    #
    ### Go on with BLADE GEOMETRY ###
    #
    ### This is the geometry with the 20deg angle
    s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=4.0)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=STANDALONE)
    ## see paper notes on blade geometry...
    s1.Line(point1=(0.0, 0.0), point2=(-0.1, 0.0342))
    s1.Line(point1=(-0.1, 0.0342), point2=(-0.8, 0.11))
    s1.Line(point1=(-0.8, 0.11), point2=(-3.0, 0.11))
    s1.HorizontalConstraint(entity=g[4], addUndoState=False)
    s1.Line(point1=(-3.0, 0.11), point2=(-3.0, -0.11))
    s1.VerticalConstraint(entity=g[5], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[4], entity2=g[5], addUndoState=False)
    s1.Line(point1=(-3.0, -0.11), point2=(-0.8, -0.11))
    s1.HorizontalConstraint(entity=g[6], addUndoState=False)
    s1.PerpendicularConstraint(entity1=g[5], entity2=g[6], addUndoState=False)
    s1.Line(point1=(-0.8, -0.11), point2=(-0.1, -0.0342))
    s1.Line(point1=(-0.1, -0.0342), point2=(0.0, 0.0))
    ##
    p = mdb.models['Model-1'].Part(name='Part-3', dimensionality=THREE_D, 
        type=DEFORMABLE_BODY)
    p = mdb.models['Model-1'].parts['Part-3']
    p.BaseSolidExtrude(sketch=s1, depth=4.0)
    s1.unsetPrimaryObject()
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models['Model-1'].sketches['__profile__']
    ##
    ##
    partition_meshtype_set()
    ##
    ## TODO ACTUAL SEEDS FOR MESHES ARE SELECTED MANUALLY AND DEPEND ON INTERNAL MASKS RIGHT NOW:
    ## TODO This could potentially lead to issuses:
    # outer edges are very roughly meshed
    e = p.edges
    pickedEdges = e.getSequenceFromMask(mask=('[#7f55cd20 #5 ]', ), )
    p.seedEdgeBySize(edges=pickedEdges, size=0.22, deviationFactor=0.1, 
        constraint=FINER)
    # inner edges slightly finer
    pickedEdges1 = e.getSequenceFromMask(mask=('[#281210 ]', ), )
    pickedEdges2 = e.getSequenceFromMask(mask=('[#40 ]', ), )
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=pickedEdges1, 
        end2Edges=pickedEdges2, minSize=0.11, maxSize=0.22, constraint=FINER)
    pickedEdges1 = e.getSequenceFromMask(mask=('[#81010 ]', ), )
    pickedEdges2 = e.getSequenceFromMask(mask=('[#200240 ]', ), )
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=pickedEdges1, 
        end2Edges=pickedEdges2, minSize=0.11, maxSize=0.22, constraint=FINER)
    # edges at sharp contact
    pickedEdges1 = e.getSequenceFromMask(mask=('[#0 #2 ]', ), )
    pickedEdges2 = e.getSequenceFromMask(mask=('[#2000 ]', ), )
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=pickedEdges1, 
        end2Edges=pickedEdges2, minSize=b_seed, maxSize=0.22, constraint=FINER)
    # edges smallest transistion
    pickedEdges1 = e.getSequenceFromMask(mask=('[#4 ]', ), )
    pickedEdges2 = e.getSequenceFromMask(mask=('[#2 ]', ), )
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=pickedEdges1, 
        end2Edges=pickedEdges2, minSize=b_seed, maxSize=2*b_seed, constraint=FINER)
    # some inner edges close to contact point
    pickedEdges1 = e.getSequenceFromMask(mask=('[#80800008 ]', ), )
    pickedEdges2 = e.getSequenceFromMask(mask=('[#20081 ]', ), )
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=pickedEdges1, 
        end2Edges=pickedEdges2, minSize=2*b_seed, maxSize=0.22, constraint=FINER)
    pickedEdges1 = e.getSequenceFromMask(mask=('[#8 ]', ), )
    pickedEdges2 = e.getSequenceFromMask(mask=('[#1 ]', ), )
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=pickedEdges1, 
        end2Edges=pickedEdges2, minSize=2*b_seed, maxSize=0.11, constraint=FINER)
    ##
    #p.generateMesh()
    generate_mesh_and_capture_messages(p)
    ##
    blade_assembly_translation()
    ##
    # generate BCs
    # Access the part
    p = mdb.models['Model-1'].parts['Part-3']
    # Access faces
    f = p.faces
    # Define coordinates to identify faces (replace with your actual coordinates)
    coord1 = (-3.0, 0.0, 3.0)  # Coordinate on face 13
    coord2 = (-3.0, 0.0, 1.0)  # Coordinate on face 2

    # Select faces using findAt method
    face1 = f.findAt((coord1,))
    face2 = f.findAt((coord2,))

    # Print the faces to verify
    print("Selected faces using findAt:","Face 1:", face1, "Face 2:", face2)
    faces = (face1, face2)

    # Create a new set with the identified faces
    try:
        p.Set(faces=faces, name='Set-BC')
        print("Faces have been selected and added to the set 'Set-BC'.")
    except Exception as e:
        print("Error creating set with faces using findAt:", e)
    ##
    ######################################################

elif blade == 1:
    #
    ##################
    ### BLADE flat ###
    ##################
    ##
    s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=4.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    ##
    s.Line(point1=(0.0, 0.0), point2=(-0.604, 0.22))
    s.Line(point1=(-0.604, 0.22), point2=(-3.0, 0.22))
    s.HorizontalConstraint(entity=g[3], addUndoState=False)
    s.Line(point1=(-3.0, 0.22), point2=(-3.0, 0.0))
    s.VerticalConstraint(entity=g[4], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[3], entity2=g[4], addUndoState=False)
    s.Line(point1=(-3.0, 0.0), point2=(-0.604, 0.0))
    s.HorizontalConstraint(entity=g[5], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[4], entity2=g[5], addUndoState=False)
    s.Line(point1=(-0.604, 0.0), point2=(0.0, 0.0))
    s.HorizontalConstraint(entity=g[6], addUndoState=False)
    s.ParallelConstraint(entity1=g[5], entity2=g[6], addUndoState=False)
    ##
    p = mdb.models['Model-1'].Part(name='Part-3', dimensionality=THREE_D, 
        type=DEFORMABLE_BODY)
    p = mdb.models['Model-1'].parts['Part-3']
    p.BaseSolidExtrude(sketch=s, depth=4.0)
    s.unsetPrimaryObject()
    p = mdb.models['Model-1'].parts['Part-3']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models['Model-1'].sketches['__profile__']
    ##
    ### Partition the BLADE MESHING...

    p = mdb.models['Model-1'].parts['Part-3']
    pickedCells = p.cells[:] #THIS APPROACH SELECTS ALL CELLS
    pickedRegions =(pickedCells, ) #regions here are just cells in a tuple??
    ### Additional Partition Cell:
    point1 = (-0.604, 0.0, 0.0)  # These are the coordinates of the point of first contact 
    point2 = (-0.604, 0.0, 4.0)  # random points in the desired plane
    point3 = (-0.604, -1.0, 4.0)  # random points in the desired plane
    p.PartitionCellByPlaneThreePoints(cells=pickedCells, point1=point1, point2=point2, point3=point3)

    partition_meshtype_set()

    ##
    # MESH Seed:
    p = mdb.models['Model-1'].parts['Part-3']
    e = p.edges
    # coordinates of all edges (point on the middle of the edge/line):
    #finest to most coarse
    coord01 = (0,0,3)
    coord02 = (0,0,1)

    pickedEdge = e.findAt(coord01)
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=(pickedEdge,), minSize=b_seed,
    maxSize=0.22, constraint=FINER)
    pickedEdge = e.findAt(coord02)
    p.seedEdgeByBias(biasMethod=SINGLE, end2Edges=(pickedEdge,), minSize=b_seed, 
    maxSize=0.22, constraint=FINER)

    #finest to most coarse/2
    coord03 = (-0.302,0.11,2)
    coord04 = (-0.302,0,2)

    pickedEdge = e.findAt(coord03)
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=(pickedEdge,), minSize=b_seed,
    maxSize=0.11, constraint=FINER)
    pickedEdge = e.findAt(coord04)
    p.seedEdgeByBias(biasMethod=SINGLE, end2Edges=(pickedEdge,), minSize=b_seed, 
    maxSize=0.11, constraint=FINER)

    #most coarse to most coarse/2
    coord05 = (-0.604,0.22,3)
    coord06 = (-0.604,0,3)
    coord07 = (-0.604,0.22,1)
    coord08 = (-0.604,0,1)
    coord09 = (-1.802,0.22,2)
    coord10 = (-1.802,0,2)

    pickedEdge = e.findAt(coord05)
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=(pickedEdge,), minSize=0.11,
    maxSize=0.22, constraint=FINER)
    pickedEdge = e.findAt(coord06)
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=(pickedEdge,), minSize=0.11, 
    maxSize=0.22, constraint=FINER)
    pickedEdge = e.findAt(coord07)
    p.seedEdgeByBias(biasMethod=SINGLE, end2Edges=(pickedEdge,), minSize=0.11,
    maxSize=0.22, constraint=FINER)
    pickedEdge = e.findAt(coord08)
    p.seedEdgeByBias(biasMethod=SINGLE, end2Edges=(pickedEdge,), minSize=0.11, 
    maxSize=0.22, constraint=FINER)
    pickedEdge = e.findAt(coord09)
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=(pickedEdge,), minSize=0.11,
    maxSize=0.22, constraint=FINER)
    pickedEdge = e.findAt(coord10)
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=(pickedEdge,), minSize=0.11, 
    maxSize=0.22, constraint=FINER)

    #most coarse
    coord11 = (-0.302,0.11,0) 
    coord12 = (-0.302,0,0)
    coord13 = (-0.302,0.11,4)
    coord14 = (-0.302,0,4)
    coord15 = (-1.802,0.22,4)
    coord16 = (-1.802,0,4)
    coord17 = (-1.802,0.22,0)
    coord18 = (-1.802,0,0)
    coord19 = (-3,0.22,3)
    coord20 = (-3,0,3)
    coord21 = (-3,0.22,1)
    coord22 = (-3,0,1)

    pickedEdges = (e.findAt(coord11),e.findAt(coord12),e.findAt(coord13),e.findAt(coord14),
                   e.findAt(coord15),e.findAt(coord16),e.findAt(coord17),e.findAt(coord18),
                   e.findAt(coord19),e.findAt(coord20),e.findAt(coord21),e.findAt(coord22),)
    p.seedEdgeBySize(edges=pickedEdges, size=0.22, deviationFactor=0.1, 
    constraint=FINER)
    ##
    #p.generateMesh()
    generate_mesh_and_capture_messages(p)
    ##
    blade_assembly_translation()
    ##

    # generate BCs
    # Access the part
    p = mdb.models['Model-1'].parts['Part-3']
    # Access faces
    f = p.faces
    # Define coordinates to identify faces (replace with your actual coordinates)
    coord1 = (-3.0, 0.11, 3.0)  # Coordinate on face 13
    coord2 = (-3.0, 0.11, 1.0)  # Coordinate on face 2

    # Select faces using findAt method
    face1 = f.findAt((coord1,))
    face2 = f.findAt((coord2,))

    # Print the faces to verify
    print("Selected faces using findAt:","Face 1:", face1, "Face 2:", face2)
    faces = (face1, face2)

    # Create a new set with the identified faces
    try:
        p.Set(faces=faces, name='Set-BC')
        print("Faces have been selected and added to the set 'Set-BC'.")
    except Exception as e:
        print("Error creating set with faces using findAt:", e)
    ##

elif blade == 2:
    #
    #####################
    ### BLADE rounded ###
    #####################
    s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
    sheetSize=200.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.Line(point1=(0.0, 0.0), point2=(-0.604, 0.22))
    s.Line(point1=(-0.604, 0.22), point2=(-3.0, 0.22))
    s.HorizontalConstraint(entity=g[3], addUndoState=False)
    s.Line(point1=(-3.0, 0.22), point2=(-3.0, 0.0))
    s.VerticalConstraint(entity=g[4], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[3], entity2=g[4], addUndoState=False)
    s.Line(point1=(-3.0, 0.0), point2=(-0.604, 0.0))
    s.HorizontalConstraint(entity=g[5], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[4], entity2=g[5], addUndoState=False)
    s.Line(point1=(-0.604, 0.0), point2=(0.0, 0.0))
    s.HorizontalConstraint(entity=g[6], addUndoState=False)
    s.ParallelConstraint(entity1=g[5], entity2=g[6], addUndoState=False)
    s.FilletByRadius(radius=0.005, curve1=g[6], nearPoint1=(-0.257631868124008, 
    -0.00103402137756348), curve2=g[2], nearPoint2=(-0.172356456518173, 
    0.0663262605667114))
    ##
    p = mdb.models['Model-1'].Part(name='Part-3', dimensionality=THREE_D, 
    type=DEFORMABLE_BODY)
    p = mdb.models['Model-1'].parts['Part-3']
    p.BaseSolidExtrude(sketch=s, depth=4.0)
    s.unsetPrimaryObject()
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models['Model-1'].sketches['__profile__']
    # Partition Model
    p = mdb.models['Model-1'].parts['Part-3']
    pickedCells = p.cells[:] #THIS APPROACH SELECTS ALL CELLS
    pickedRegions =(pickedCells, ) #regions here are just cells in a tuple??
    ### Additional Partition Cell:
    point1 = (-0.604, 0.0, 0.0)  # These are the coordinates of the point of first contact 
    point2 = (-0.604, 0.0, 4.0)  # random points in the desired plane
    point3 = (-0.604, -1.0, 4.0)  # random points in the desired plane
    p.PartitionCellByPlaneThreePoints(cells=pickedCells, point1=point1, point2=point2, point3=point3)
    #
    partition_meshtype_set()
    ##
    # MESH Seed:
    p = mdb.models['Model-1'].parts['Part-3']
    e = p.edges
    # coordinates of all edges (point on the middle of the edge/line):
    #finest to most coarse
    #coord01 = (0,0,3)
    coord01_1 = (-26.626E-03,9.698E-03,3.)
    coord01_2 = (-28.337E-03,0.,3.)
    #coord02 = (0,0,1)
    coord02_1 = (-26.626E-03,9.698E-03,1.)
    coord02_2 = (-28.337E-03,0.,1.)

    pickedEdge = e.findAt(coord01_1)
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=(pickedEdge,), minSize=b_seed,
    maxSize=0.22, constraint=FINER)
    pickedEdge = e.findAt(coord01_2)
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=(pickedEdge,), minSize=b_seed,
    maxSize=0.22, constraint=FINER)
    pickedEdge = e.findAt(coord02_1)
    p.seedEdgeByBias(biasMethod=SINGLE, end2Edges=(pickedEdge,), minSize=b_seed, 
    maxSize=0.22, constraint=FINER)
    pickedEdge = e.findAt(coord02_2)
    p.seedEdgeByBias(biasMethod=SINGLE, end2Edges=(pickedEdge,), minSize=b_seed, 
    maxSize=0.22, constraint=FINER)

    #finest to most coarse/2
    coord03 = (-0.302,0.11,2)
    coord04 = (-0.302,0,2)

    pickedEdge = e.findAt(coord03)
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=(pickedEdge,), minSize=b_seed,
    maxSize=0.11, constraint=FINER)
    pickedEdge = e.findAt(coord04)
    p.seedEdgeByBias(biasMethod=SINGLE, end2Edges=(pickedEdge,), minSize=b_seed, 
    maxSize=0.11, constraint=FINER)

    #most coarse to most coarse/2
    coord05 = (-0.604,0.22,3)
    coord06 = (-0.604,0,3)
    coord07 = (-0.604,0.22,1)
    coord08 = (-0.604,0,1)
    coord09 = (-1.802,0.22,2)
    coord10 = (-1.802,0,2)

    pickedEdge = e.findAt(coord05)
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=(pickedEdge,), minSize=0.11,
    maxSize=0.22, constraint=FINER)
    pickedEdge = e.findAt(coord06)
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=(pickedEdge,), minSize=0.11, 
    maxSize=0.22, constraint=FINER)
    pickedEdge = e.findAt(coord07)
    p.seedEdgeByBias(biasMethod=SINGLE, end2Edges=(pickedEdge,), minSize=0.11,
    maxSize=0.22, constraint=FINER)
    pickedEdge = e.findAt(coord08)
    p.seedEdgeByBias(biasMethod=SINGLE, end2Edges=(pickedEdge,), minSize=0.11, 
    maxSize=0.22, constraint=FINER)
    pickedEdge = e.findAt(coord09)
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=(pickedEdge,), minSize=0.11,
    maxSize=0.22, constraint=FINER)
    pickedEdge = e.findAt(coord10)
    p.seedEdgeByBias(biasMethod=SINGLE, end1Edges=(pickedEdge,), minSize=0.11, 
    maxSize=0.22, constraint=FINER)

    #most coarse
    coord11 = (-0.302,0.11,0) 
    coord12 = (-0.302,0,0)
    coord13 = (-0.302,0.11,4)
    coord14 = (-0.302,0,4)
    coord15 = (-1.802,0.22,4)
    coord16 = (-1.802,0,4)
    coord17 = (-1.802,0.22,0)
    coord18 = (-1.802,0,0)
    coord19 = (-3,0.22,3)
    coord20 = (-3,0,3)
    coord21 = (-3,0.22,1)
    coord22 = (-3,0,1)

    pickedEdges = (e.findAt(coord11),e.findAt(coord12),e.findAt(coord13),e.findAt(coord14),
                   e.findAt(coord15),e.findAt(coord16),e.findAt(coord17),e.findAt(coord18),
                   e.findAt(coord19),e.findAt(coord20),e.findAt(coord21),e.findAt(coord22),)
    p.seedEdgeBySize(edges=pickedEdges, size=0.22, deviationFactor=0.1, 
    constraint=FINER)
    ##
    #p.generateMesh()
    generate_mesh_and_capture_messages(p)
    ##
    ### Blade distance - slightly different here: ###
    a = mdb.models['Model-1'].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    p = mdb.models['Model-1'].parts['Part-3']
    a.Instance(name='Part-3-1', part=p, dependent=ON)
    a.Instance(name='Part-3-2', part=p, dependent=ON)
    ##
    # translate to origin + embedded length / 2
    a.translate(instanceList=('Part-3-1', ), vector=(0.0, L/2, -2.0))
    a.translate(instanceList=('Part-3-2', ), vector=(0.0, L/2, -2.0))
    ##
    # mirror 1 blade instance
    a.rotate(instanceList=('Part-3-2', ), axisPoint=(0.0, 0.0, 0.0), 
        axisDirection=(0.0, 10.0, 0.0), angle=180.0)
    ##
    # now translate both according to b (correct for fillet here)
    a.translate(instanceList=('Part-3-1', ), vector=(-b+0.023, -0.0047, 0.0))
    a.translate(instanceList=('Part-3-2', ), vector=(b-0.023, -0.0047, 0.0))
    ##
    ##
    # generate BCs
    # Access the part
    p = mdb.models['Model-1'].parts['Part-3']
    # Access faces
    f = p.faces
    # Define coordinates to identify faces (replace with your actual coordinates)
    coord1 = (-3.0, 0.11, 3.0)  # Coordinate on face 13
    coord2 = (-3.0, 0.11, 1.0)  # Coordinate on face 2

    # Select faces using findAt method
    face1 = f.findAt((coord1,))
    face2 = f.findAt((coord2,))

    # Print the faces to verify
    print("Selected faces using findAt:","Face 1:", face1, "Face 2:", face2)
    faces = (face1, face2)

    # Create a new set with the identified faces
    try:
        p.Set(faces=faces, name='Set-BC')
        print("Faces have been selected and added to the set 'Set-BC'.")
    except Exception as e:
        print("Error creating set with faces using findAt:", e)
    ##


########################################
### Rotation of Fiber-Droplet System ### (in Assembly)
########################################
# Translations and rotations of individual parts (instances) can also easily be changed in the input file...
a = mdb.models['Model-1'].rootAssembly
a.rotate(instanceList=('Part-ALL-1', ), axisPoint=(0.0, 0.0, 0.0), 
    axisDirection=(0.0, 10.0, 0.0), angle=rot)

### Mesh the Fiber-Droplet System

#############################
### Todo, quite difficult ###
#############################

#Start with partitioning:
p = mdb.models['Model-1'].parts['Part-ALL']
#get all cells
pickedCells = p.cells[:] #THIS APPROACH SELECTS ALL CELLS
# Create datum points for partitioning and save them as variables
datum_pt1 = p.DatumPointByCoordinate(coords=(0.0, 1.0, 0.0))
datum_pt2 = p.DatumPointByCoordinate(coords=(0.0, -1.0, 0.0))
datum_pt3 = p.DatumPointByCoordinate(coords=(1.0, 0.0, 0.0))
datum_pt4 = p.DatumPointByCoordinate(coords=(0.0, 0.0, 1.0))
datum_pt5 = p.DatumPointByCoordinate(coords=(0.0, 0.0, -1.0))
# Create a datum planes using the datum points
datum_plane1 = p.DatumPlaneByThreePoints(point1=p.datums[datum_pt1.id],
    point2=p.datums[datum_pt2.id], point3=p.datums[datum_pt3.id])
datum_plane2 = p.DatumPlaneByThreePoints(point1=p.datums[datum_pt1.id],
    point2=p.datums[datum_pt2.id], point3=p.datums[datum_pt4.id])
datum_plane3 = p.DatumPlaneByThreePoints(point1=p.datums[datum_pt3.id],
    point2=p.datums[datum_pt4.id], point3=p.datums[datum_pt5.id])
# Partition the cell using the datum plane
p.PartitionCellByDatumPlane(datumPlane=p.datums[datum_plane1.id], cells=pickedCells)
pickedCells = p.cells[:]
p.PartitionCellByDatumPlane(datumPlane=p.datums[datum_plane2.id], cells=pickedCells)
pickedCells = p.cells[:]
p.PartitionCellByDatumPlane(datumPlane=p.datums[datum_plane3.id], cells=pickedCells)


### SEED VIA COORDINATES ###

# Define seeding parameters for each configuration
if fd_seed == 1:
    seed_fiber_ell_axis = 6 ##only this finer here
    seed_fiber_ell_circum = 8 ##only this finer here
    seed_free_fiber = 5
    seed_vert_lines_droplet_upper_half = 13
    ratio_vert_lines_upper = 3
    seed_outer_lines_droplet_upper_half = 18
    ratio_outer_lines_upper = 3
    seed_vertical_upper_middle_line = seed_vert_lines_droplet_upper_half + seed_free_fiber
    ratio_vertical_upper_middle = 4
    seed_outer_lines_droplet_lower_half = 7
    seed_vert_lines_droplet_lower_half = 6
    seed_end_fiber = 12
    seed_vertical_lower_middle_line = seed_vert_lines_droplet_lower_half + seed_end_fiber
    seed_droplet_bead = 7
elif fd_seed == 2:
    seed_fiber_ell_axis = 4
    seed_fiber_ell_circum = 6
    seed_free_fiber = 5
    seed_vert_lines_droplet_upper_half = 13
    ratio_vert_lines_upper = 3
    seed_outer_lines_droplet_upper_half = 18
    ratio_outer_lines_upper = 3
    seed_vertical_upper_middle_line = seed_vert_lines_droplet_upper_half + seed_free_fiber
    ratio_vertical_upper_middle = 4
    seed_outer_lines_droplet_lower_half = 7
    seed_vert_lines_droplet_lower_half = 6
    seed_end_fiber = 12
    seed_vertical_lower_middle_line = seed_vert_lines_droplet_lower_half + seed_end_fiber
    seed_droplet_bead = 7
elif fd_seed == 3:
    seed_fiber_ell_axis = 3
    seed_fiber_ell_circum = 4
    seed_free_fiber = 4
    seed_vert_lines_droplet_upper_half = 9
    ratio_vert_lines_upper = 4
    seed_outer_lines_droplet_upper_half = 12
    ratio_outer_lines_upper = 4
    seed_vertical_upper_middle_line = seed_vert_lines_droplet_upper_half + seed_free_fiber
    ratio_vertical_upper_middle = 6
    seed_outer_lines_droplet_lower_half = 7
    seed_vert_lines_droplet_lower_half = 4
    seed_end_fiber = 8
    seed_vertical_lower_middle_line = seed_vert_lines_droplet_lower_half + seed_end_fiber
    seed_droplet_bead = 5
elif fd_seed == 4:
    seed_fiber_ell_axis = 3
    seed_fiber_ell_circum = 4
    seed_free_fiber = 3
    seed_vert_lines_droplet_upper_half = 7
    ratio_vert_lines_upper = 4.0
    seed_outer_lines_droplet_upper_half = 8
    ratio_outer_lines_upper = 4.0
    seed_vertical_upper_middle_line = seed_vert_lines_droplet_upper_half + seed_free_fiber
    ratio_vertical_upper_middle = 6.0 
    seed_outer_lines_droplet_lower_half = 3
    seed_vert_lines_droplet_lower_half = 4
    seed_end_fiber = 6
    seed_vertical_lower_middle_line = seed_vert_lines_droplet_lower_half + seed_end_fiber
    seed_droplet_bead = 3

### TRY TO SEED VIA COORDINATES ###
print("")
print("mesh seed selection of Fiber-Droplet system")
# Define seeding functions
def seed_edges_uniformly(part, coordinates_list, seed_number, constraint=FINER):
    selected_edges = []
    for coords in coordinates_list:
        edge = part.edges.getClosest(coordinates=(coords,))
        selected_edges.append(edge[0][0])
    part.seedEdgeByNumber(edges=tuple(selected_edges), number=seed_number, constraint=constraint)
    print(f"Uniformly seeded edges: {[edge.index for edge in selected_edges]}")

def seed_edges_bias_direction1(part, coordinates_list, seed_number, ratio, constraint=FINER):
    selected_edges = []
    for coords in coordinates_list:
        edge = part.edges.getClosest(coordinates=(coords,))
        selected_edges.append(edge[0][0])
    part.seedEdgeByBias(end1Edges=tuple(selected_edges), biasMethod=SINGLE, ratio=ratio, number=seed_number, constraint=constraint)
    print(f"Bias direction 1 seeded edges: {[edge.index for edge in selected_edges]}")

def seed_edges_bias_direction2(part, coordinates_list, seed_number, ratio, constraint=FINER):
    selected_edges = []
    for coords in coordinates_list:
        edge = part.edges.getClosest(coordinates=(coords,))
        selected_edges.append(edge[0][0])
    part.seedEdgeByBias(end2Edges=tuple(selected_edges), biasMethod=SINGLE, ratio=ratio, number=seed_number, constraint=constraint)
    print(f"Bias direction 2 seeded edges: {[edge.index for edge in selected_edges]}")

# Uniform seeding for fiber ellipsis axis
coordinates_list = [
    (0.0, 0.0, 0.001),
    (0.0, 0.0, -0.001),
    (0.001, 0.0, 0.0),
    (-0.001, 0.0, 0.0)
]
seed_edges_uniformly(p, coordinates_list, seed_fiber_ell_axis)

# Uniform seeding for fiber ellipsis circumference
coordinates_list = [
    (0.001, L/2, r),
    (0.001, L/2, -r),
    (-0.001, L/2, r),
    (-0.001, L/2, -r),
    (0.001, -L/2, r),
    (0.001, -L/2, -r),
    (-0.001, -L/2, r),
    (-0.001, -L/2, -r),
    (0.001, L/2+l_free, r),
    (0.001, L/2+l_free, -r),
    (-0.001, L/2+l_free, r),
    (-0.001, L/2+l_free, -r),
    (0.001, -L/2-l_end, r),
    (0.001, -L/2-l_end, -r),
    (-0.001, -L/2-l_end, r),
    (-0.001, -L/2-l_end, -r),
    (0.001, 0.0, r),
    (0.001, 0.0, -r),
    (-0.001, 0.0, r),
    (-0.001, 0.0, -r),
    (h, 0.0, 0.001),
    (-h, 0.0, 0.001),
    (h, 0.0, -0.001),
    (-h, 0.0, -0.001)
]
seed_edges_uniformly(p, coordinates_list, seed_fiber_ell_circum)

# Uniform seeding for free fiber
coordinates_list = [
    (r*ell, L/2 + l_free - 0.002, 0.0),
    (-r*ell, L/2 + l_free - 0.002, 0.0),
    (0.0, L/2 + l_free - 0.001, r),
    (0.0, L/2 + l_free - 0.001, -r)
]
seed_edges_uniformly(p, coordinates_list, seed_free_fiber)

# Bias seeding for vertical lines in the upper half of the droplet
coordinates_list = [
    (r*ell, 0.002, 0.0),
    (0.0, 0.002, r)
]
seed_edges_bias_direction2(p, coordinates_list, seed_vert_lines_droplet_upper_half, ratio_vert_lines_upper)

coordinates_list = [
    (-r*ell, 0.002, 0.0),
    (0.0, 0.002, -r)
]
seed_edges_bias_direction1(p, coordinates_list, seed_vert_lines_droplet_upper_half, ratio_vert_lines_upper)

# Bias seeding for outer lines in the upper half of the droplet
coordinates_list = [
    (h, 0.002, 0.0),
    (-h, 0.002, 0.0),
    (0.0, 0.002, -h)
]
seed_edges_bias_direction1(p, coordinates_list, seed_outer_lines_droplet_upper_half, ratio_outer_lines_upper)

coordinates_list = [
    (0.0, 0.002, h)
]
seed_edges_bias_direction2(p, coordinates_list, seed_outer_lines_droplet_upper_half, ratio_outer_lines_upper)

# Bias seeding for vertical upper middle line
coordinates_list = [
    (0.0, 0.002, 0.0)
]
seed_edges_bias_direction2(p, coordinates_list, seed_vertical_upper_middle_line, ratio_vertical_upper_middle)

######
# Lower half

# Uniform seeding for outer lines in the lower half of the droplet
coordinates_list = [
    (h, -0.002, 0.0),
    (-h, -0.002, 0.0),
    (0.0, -0.002, h),
    (0.0, -0.002, -h)
]
seed_edges_uniformly(p, coordinates_list, seed_outer_lines_droplet_lower_half)

# Uniform seeding for vertical lines in the lower half of the droplet
coordinates_list = [
    (r*ell, -0.002, 0.0),
    (0.0, -0.002, r),
    (-r*ell, -0.002, 0.0),
    (0.0, -0.002, -r)
]
seed_edges_uniformly(p, coordinates_list, seed_vert_lines_droplet_lower_half)

# Uniform seeding for end fiber
coordinates_list = [
    (r*ell, -L/2 - l_end + 0.002, 0.0),
    (0.0, -L/2 - l_end + 0.002, r),
    (-r*ell, -L/2 - l_end + 0.002, 0.0),
    (0.0, -L/2 - l_end + 0.002, -r)
]
seed_edges_uniformly(p, coordinates_list, seed_end_fiber)

# Uniform seeding for vertical lower middle line
coordinates_list = [
    (0.0, -L/2 - l_end + 0.002, 0.0)
]
seed_edges_uniformly(p, coordinates_list, seed_vertical_lower_middle_line)

# Uniform seeding for droplet bead
coordinates_list = [
    (0.0, 0.0, h-0.002),
    (0.0, 0.0, -h+0.002),
    (h-0.002, 0.0, 0.0),
    (-h+0.002, 0.0, 0.0)
]
seed_edges_uniformly(p, coordinates_list, seed_droplet_bead)



### Mesh Fiber-Droplet-Part ###
pickedCells = p.cells[:]
pickedRegions = (pickedCells, )

p.setMeshControls(regions=pickedCells, technique=SWEEP)
elemType1 = mesh.ElemType(elemCode=C3D8, elemLibrary=STANDARD, 
    secondOrderAccuracy=OFF, distortionControl=DEFAULT)
elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD)

p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2, 
    elemType3))
p.generateMesh()



### FOR TESTING ONLY HALF MODEL ###
#p = mdb.models['Model-1'].parts['Part-ALL']
#c = p.cells
#pickedRegions = c.findAt(((-0.002009, 0.131, 0.006734), ), ((-0.04806, 
#    0.075938, 0.003881), ), ((-0.002009, 0.131, -0.006734), ), ((-0.042003, 
#    0.063169, -0.042002), ), ((-0.000735, -0.183149, 0.010489), ), ((-0.042003, 
#    -0.063169, 0.042002), ), ((-0.000704, -0.183156, -0.010489), ), ((
#    -0.042002, -0.063169, -0.042002), ))
#p.generateMesh(regions=pickedRegions)
#p = mdb.models['Model-1'].parts['Part-ALL']
#f = p.faces
#faces = f.findAt(((0.015249, 0.0, 0.002022), ), ((-0.015249, 0.0, 0.002022), ), 
#    ((0.002009, 0.0, -0.006734), ), ((0.058731, 0.0, 0.005267), ), ((-0.058731, 
#    0.0, 0.005267), ), ((-0.002009, 0.0, -0.006734), ), ((0.005267, 0.0, 
#    -0.056981), ), ((-0.005267, 0.0, -0.056981), ), ((-0.058837, 0.00528, 0.0), 
#    ), ((0.058837, -0.00528, 0.0), ), ((0.0, 0.005636, -0.057028), ), ((
#    -0.0105, 0.03433, 0.0), ), ((-0.01573, -0.068692, -0.000534), ), ((
#    -0.015716, 0.03433, 0.000689), ), ((0.0, -0.005636, 0.057028), ), ((
#    0.000978, -0.073091, -0.01048), ), ((0.0, 0.036614, -0.007), ), ((0.015716, 
#    -0.03433, 0.000689), ), ((0.0105, -0.03433, 0.0), ), ((0.0, 0.036614, 
#    0.007), ), ((0.0, 0.005636, 0.057028), ), ((0.0, -0.005636, -0.057028), ), 
#    ((0.0, -0.036614, -0.007), ), ((0.0, -0.036614, 0.007), ), ((0.04806, 
#    -0.075938, 0.003881), ), ((-0.04806, 0.075938, 0.003881), ), ((0.042002, 
#    0.063169, -0.042002), ), ((-0.042002, -0.063169, -0.042002), ), ((0.000692, 
#    -0.183158, 0.01049), ), ((-0.000735, -0.073149, 0.010489), ), ((-0.000978, 
#    0.073091, -0.01048), ), ((0.000704, 0.116822, -0.010489), ), ((-0.000692, 
#    0.116825, 0.01049), ), ((-0.000704, -0.183156, -0.010489), ), ((0.002009, 
#    -0.33, 0.006734), ), ((0.002009, 0.131, -0.006734), ), ((-0.002009, 0.131, 
#    0.006734), ), ((-0.002009, -0.33, -0.006734), ), ((0.0105, 0.03433, 0.0), 
#    ), ((0.058778, 0.005636, 0.0), ), ((-0.058778, -0.005636, 0.0), ), ((
#    -0.0105, -0.03433, 0.0), ), ((0.002009, -0.33, -0.006734), ), ((0.002009, 
#    0.131, 0.006734), ), ((0.000978, -0.183091, -0.01048), ), ((0.000735, 
#    0.116816, 0.010489), ), ((0.000735, 0.073149, 0.010489), ), ((0.042003, 
#    -0.063169, -0.042002), ), ((-0.042003, 0.063169, -0.042002), ), ((0.042003, 
#    0.063169, 0.042002), ), ((-0.042003, -0.063169, 0.042002), ), ((-0.000735, 
#    -0.183149, 0.010489), ), ((0.015733, 0.068687, -0.000494), ), ((-0.000978, 
#    0.116757, -0.01048), ), ((-0.002009, 0.131, -0.006734), ), ((-0.002009, 
#    -0.33, 0.006734), ))
#leaf = dgm.LeafFromGeometry(faceSeq=faces)


#######################################
## Calculate embedded area and print ##

# Function to calculate the circumference of an ellipse using Ramanujan's approximation
def ellipse_circumference(ell, r):
    h = ((r*ell - r) ** 2) / ((r*ell + r) ** 2)
    circumference = math.pi * (r*ell + r) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))
    return circumference

def circle_circumference(r):
    return 2 * math.pi * r

# Inputs: semi-major axis (a), semi-minor axis (b), and embedded length (L)
#r = semi-major axis in mm
#r*ell =  semi-minor axis in mm
#L = embedded length in mm

if ell == 1.0:
    U = circle_circumference(r)
else:
    U = ellipse_circumference(ell, r)

# Calculate the embedded area
embedded_area = U * L

## Print the embedded area
print("")
print(f"The embedded area of this fiber-droplet configuration using L * U is {embedded_area:.8f} mm²")
##
######################################

# Calculate the embedded area using getSize() of the faces
p = mdb.models['Model-1'].parts['Part-ALL']
f = p.faces
faces = f.findAt(((r*ell*sqrt(2)/2, 0.01, r*sqrt(2)/2), ))
# Verify the selected faces
print(f"Selected face for getSize of face (times 8): {[face.index for face in faces]}")
embedded_area_selected = faces[0].getSize() * 8
print(f"The EXCACT embedded area of this fiber-droplet configuration using \"getSize()\" is {embedded_area_selected:.8f} mm²")

### SOME MORE GENERAL MODEL DEFINITIONS ###

## regenerate assembly
a = mdb.models['Model-1'].rootAssembly
a.regenerate()

## select upper end of fiber for BC-Set
## BUT DO THIS STILL IN PART!!!

## select upper end of fiber for BC-Set
## Surface selection for BC-3 via coordinates:

# Define the model and part
model = mdb.models['Model-1']
p = model.parts['Part-ALL']

# Define the set name
set_name = 'Set-BC-3'

# Define coordinates for face selection
coordinates_list = [
    ((0.002, L/2 + l_free, 0.002),),
    ((-0.002, L/2 + l_free, 0.002),),
    ((0.002, L/2 + l_free, -0.002),),
    ((-0.002, L/2 + l_free, -0.002),)
]

# Find faces at the specified coordinates using findAt
faces = p.faces.findAt(*coordinates_list)

# Verify the selected faces
print(f"Selected faces for Set_BC-3 creation: {[face.index for face in faces]}")

# Create a set with the selected faces
try:
    p.Set(faces=faces, name=set_name)
    print(f"Created set '{set_name}' with faces: {[face.index for face in faces]}")
except Exception as e:
    print(f"Failed to create set '{set_name}': {e}")


### Define displacement
region = a.instances['Part-ALL-1'].sets['Set-BC-3']
mdb.models['Model-1'].DisplacementBC(name='BC-3', createStepName='Step-1', 
    region=region, u1=0.0, u2=0.1, u3=0.0, ur1=UNSET, ur2=UNSET, ur3=UNSET, 
    amplitude=UNSET, fixed=OFF, distributionType=UNIFORM, fieldName='', 
    localCsys=None)

#### ENCASTRE BLADES
a = mdb.models['Model-1'].rootAssembly
region = a.instances['Part-3-1'].sets['Set-BC']
mdb.models['Model-1'].EncastreBC(name='BC-1', createStepName='Initial', 
    region=region, localCsys=None)
#
region = a.instances['Part-3-2'].sets['Set-BC']
mdb.models['Model-1'].EncastreBC(name='BC-2', createStepName='Initial', 
    region=region, localCsys=None)

####
####
#### DEFINE MATERIALS
#### Fiber material main direction is y-axis
mdb.models['Model-1'].Material(name='Material-2')
mdb.models['Model-1'].materials['Material-2'].Elastic(type=ORTHOTROPIC, table=(
    (7137.0, 2630.0, 51600.0, 2576.0, 2630.0, 7137.0, 3023.5, 2281.0, 3023.5), 
    ))
mdb.models['Model-1'].Material(name='Material-1')
mdb.models['Model-1'].materials['Material-1'].Elastic(table=((1580.0, 0.43), ))
mdb.models['Model-1'].materials['Material-1'].Plastic(scaleStress=None, table=(
    (6.461776, 0.0), (8.167821, 0.000244), (11.996452, 0.001171), (15.859888, 
    0.002663), (18.752369, 0.004446), (22.010481, 0.008066), (24.317039, 
    0.013031), (25.799861, 0.018859), (26.713848, 0.025963), (27.062125, 
    0.032978)))
mdb.models['Model-1'].Material(name='Material-3')
mdb.models['Model-1'].materials['Material-3'].Elastic(table=((210000.0, 0.3), 
    ))
####
mdb.models['Model-1'].HomogeneousSolidSection(name='Section-1', 
    material='Material-1', thickness=None)
mdb.models['Model-1'].HomogeneousSolidSection(name='Section-2', 
    material='Material-2', thickness=None)
mdb.models['Model-1'].HomogeneousSolidSection(name='Section-3', 
    material='Material-3', thickness=None)
###############################################
#### ASSIGN MATERIALS
p = mdb.models['Model-1'].parts['Part-ALL']
region = p.sets['Set-1']
p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, 
    offsetType=MIDDLE_SURFACE, offsetField='', 
    thicknessAssignment=FROM_SECTION)
region = p.sets['Set-2']
p.SectionAssignment(region=region, sectionName='Section-2', offset=0.0, 
    offsetType=MIDDLE_SURFACE, offsetField='', 
    thicknessAssignment=FROM_SECTION)
p1 = mdb.models['Model-1'].parts['Part-3']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
region = p1.sets['Set-3']
p1.SectionAssignment(region=region, sectionName='Section-3', offset=0.0, 
    offsetType=MIDDLE_SURFACE, offsetField='', 
    thicknessAssignment=FROM_SECTION)
#### Orientation for Fiber
p = mdb.models['Model-1'].parts['Part-ALL']
region = p.sets['Set-2']
orientation=None
mdb.models['Model-1'].parts['Part-ALL'].MaterialOrientation(region=region, 
    orientationType=GLOBAL, axis=AXIS_1, additionalRotationType=ROTATION_NONE, 
    localCsys=None, fieldName='', stackDirection=STACK_3)
###############################################
mdb.models['Model-1'].ContactProperty('CONTACT')
mdb.models['Model-1'].interactionProperties['CONTACT'].NormalBehavior(
    pressureOverclosure=HARD, allowSeparation=ON, 
    constraintEnforcementMethod=DEFAULT)
mdb.models['Model-1'].interactionProperties['CONTACT'].TangentialBehavior(
    formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF, 
    pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
    blade_fric, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION, 
    fraction=0.005, elasticSlipStiffness=None)
#: The interaction property "CONTACT" has been created.
mdb.models['Model-1'].ContactProperty('COHESIVE')
mdb.models['Model-1'].interactionProperties['COHESIVE'].TangentialBehavior(
    formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF, 
    pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
    interface_fric, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION, 
    fraction=0.005, elasticSlipStiffness=None)
mdb.models['Model-1'].interactionProperties['COHESIVE'].CohesiveBehavior(
    defaultPenalties=OFF, table=((500000.0, 50000.0, 50000.0), ))
mdb.models['Model-1'].interactionProperties['COHESIVE'].NormalBehavior(
    pressureOverclosure=HARD, allowSeparation=ON, 
    constraintEnforcementMethod=DEFAULT)
mdb.models['Model-1'].interactionProperties['COHESIVE'].Damage(initTable=((tI, tI, tI), ),
    useEvolution=ON, evolutionType=ENERGY, useMixedMode=ON, mixedModeType=POWER_LAW, 
    exponent=1.0, evolTable=((G_modeI, G_modeII, G_modeII), ), useStabilization=ON, viscosityCoef=1e-05)
#: The interaction property "COHESIVE" has been created.
mdb.models['Model-1'].ContactStd(name='GENERAL_CONTACT', 
    createStepName='Initial')
mdb.models['Model-1'].interactions['GENERAL_CONTACT'].includedPairs.setValuesInStep(
    stepName='Initial', useAllstar=ON)
m21=mdb.models['Model-1'].materials['Material-2']
m22=mdb.models['Model-1'].materials['Material-1']
mdb.models['Model-1'].interactions['GENERAL_CONTACT'].contactPropertyAssignments.appendInStep(
    stepName='Initial', assignments=((GLOBAL, SELF, 'CONTACT'), (m21, m22, 
    'CONTACT')))
mdb.models['Model-1'].interactions['GENERAL_CONTACT'].contactPropertyAssignments.changeValuesInStep(
    stepName='Initial', index=1, value='COHESIVE')
#: The interaction "GENERAL_CONTACT" has been created.
############################################################

##regenerate assembly
a = mdb.models['Model-1'].rootAssembly
a.regenerate()

########################
### write input file ###
########################

mdb.Job(name=inpfile, model='Model-1', description='', type=ANALYSIS, 
    atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
    memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
    explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
    modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
    scratch='', resultsFormat=ODB, numThreadsPerMpiProcess=1, 
    multiprocessingMode=DEFAULT, numCpus=1, numGPUs=0)
mdb.jobs[inpfile].writeInput(consistencyChecking=OFF)
#: The job input file has been written to "XXXXX.inp".


### Some output for the log-file: ###
print("")
mesh = a.getMeshStats(regions=[a.instances['Part-ALL-1'],])
print("Number of nodes for instance PART-ALL-1: ")
print(mesh.numNodes)
print("Number of elements for instance PART-ALL-1:")
print(mesh.numHexElems)

mesh1 = a.getMeshStats(regions=[a.instances['Part-3-1'],])
print("Number of nodes for instance PART-3-1:")
print(mesh1.numNodes)
print("Number of elements for instance PART-3-1:")
print(mesh1.numTetElems)
print("")
sum_number_nodes = mesh.numNodes + 2 * mesh1.numNodes
sum_number_elems = mesh.numTetElems + 2 * mesh1.numTetElems
print("Sum of all nodes:")
print(sum_number_nodes)
print("Sum of all elements:")
print(sum_number_elems)


#save cae
current_path = os.getcwd()
full_path = os.path.join(current_path, f'{inpfile}.cae')

mdb.saveAs(pathName=full_path)
