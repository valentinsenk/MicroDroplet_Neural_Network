import os
import sys

#import jobname
jobname = sys.argv[-1]
dir = os.getcwd()
jobdir = (dir + '\\' + jobname)

#some abaqus import stuff...
from abaqus import *
# abaqusConstants imports names like NODAL, etc.
from abaqusConstants import *
####
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=112.84375, 
    height=65.2055511474609)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
session.viewports['Viewport: 1'].setValues(displayedObject=None)

#open odb-file
o1 = session.openOdb(
    name=jobdir)
session.viewports['Viewport: 1'].setValues(displayedObject=o1)


### create directory + .txt-files for output data

resultsfolder = jobname + '_results'
datafolder = 'outputdata'

if not os.path.exists(os.path.join(resultsfolder, datafolder)):
    os.makedirs(os.path.join(resultsfolder, datafolder))
else:
    print("results-directory already exists")


####################################
#### Get Output from ABAQUS CAE ####
####################################

odb = session.odbs[jobdir]
#steps:
#stepnames = odb.steps.keys()

### REACTION FORCES and DISPLACEMENTS ###
#############################################################################
#session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', 
#    NODAL, ((COMPONENT, 'RF2'), )), ), nodeSets=("SET-RP-1", ))
#session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('U', 
#    NODAL, ((COMPONENT, 'U2'), )), ), nodeSets=("SET-RP-1", ))
##############################################################################

### Get REACTION FORCES SUMMED UP OVER ALL NODES OF BC-3: ###
session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('RF', 
    NODAL, ((COMPONENT, 'RF1'), (COMPONENT, 'RF2'), (COMPONENT, 'RF3'), )), ), 
    operator=ADD, nodeSets=("PART-2-1.SET-BC-3", ))
#############################################################

data_list = session.xyDataObjects.keys()

data_RF1_BC_3 = session.xyDataObjects[data_list[0]].data
data_RF2_BC_3 = session.xyDataObjects[data_list[1]].data
data_RF3_BC_3 = session.xyDataObjects[data_list[2]].data

# get Lists of Output-data

steptime_RF = []
for a_tuple in data_RF2_BC_3:
    steptime_RF.append(a_tuple[0])

path_time_RF = os.path.join(os.path.join(resultsfolder, datafolder), 'time_RF.txt')
with open(path_time_RF, 'w') as f:
    for item in steptime_RF:
        f.write("%s\n" % item)


#function for output
def get_output(dataname, filename):
    data = []
    for a_tuple in dataname:
        data.append(a_tuple[1])
    path1 = os.path.join(os.path.join(resultsfolder, datafolder), filename + '.txt')
    with open(path1, 'w') as f:
        for item in data:
            f.write("%s\n" % item)

#get output

get_output(data_RF1_BC_3, 'RF1_BC3')
get_output(data_RF2_BC_3, 'RF2_BC3')
get_output(data_RF3_BC_3, 'RF3_BC3')


# delete previous output:
for key in data_list:
    del session.xyDataObjects[key]

## get U2 - should be the same for all:
session.xyDataListFromField(odb=odb, outputPosition=NODAL, variable=(('U', 
    NODAL, ((COMPONENT, 'U2'), )), ), nodeSets=("PART-2-1.SET-BC-3", ))

data_list = session.xyDataObjects.keys()
data_U2_BC_3 = session.xyDataObjects[data_list[0]].data

get_output(data_U2_BC_3, 'U2_BC3')


#########################
### get energy output ###
#########################

# delete previous output:
for key in data_list:
    del session.xyDataObjects[key]


# List of history output variable names
history_variables = [
    'Artificial strain energy: ALLAE for Whole Model',
    'Contact constraint discontinuity work: ALLCCDW for Whole Model',
    'Contact constraint elastic energy: ALLCCE for Whole Model',
    'Contact constraint elastic normal energy: ALLCCEN for Whole Model',
    'Contact constraint elastic tangential energy: ALLCCET for Whole Model',
    'Contact constraint stabilization dissipation: ALLCCSD for Whole Model',
    'Contact constraint stabilization normal dissipation: ALLCCSDN for Whole Model',
    'Contact constraint stabilization tangential dissipation: ALLCCSDT for Whole Model',
    'Creep dissipation energy: ALLCD for Whole Model',
    'Damage dissipation energy: ALLDMD for Whole Model',
    'Dynamic time integration energy: ALLDTI for Whole Model',
    'Electrostatic energy: ALLEE for Whole Model',
    'Energy lost to quiet boundaries: ALLQB for Whole Model',
    'External work: ALLWK for Whole Model',
    'Frictional dissipation: ALLFD for Whole Model',
    'Internal energy: ALLIE for Whole Model',
    'Joule heat dissipation: ALLJD for Whole Model',
    'Kinetic energy: ALLKE for Whole Model',
    'Loss of kinetic energy at impact: ALLKL for Whole Model',
    'Plastic dissipation: ALLPD for Whole Model',
    'Static dissipation (stabilization): ALLSD for Whole Model',
    'Strain energy: ALLSE for Whole Model',
    'Total energy of the output set: ETOTAL for Whole Model',
    'Viscous dissipation: ALLVD for Whole Model'
]

# Function to save output data to a file
def save_output_to_file(short_name, data):
    path = os.path.join(os.path.join(resultsfolder, datafolder), short_name + '.txt')
    with open(path, 'w') as f:
        for item in data:
            f.write("%s\n" % item)


# Initialize the list for time data (for the first variable)
time_data = []

# Loop through each history variable, create XYDataFromHistory objects, and save the data to files
for i, var in enumerate(history_variables):
    short_name = var.split(':')[1].split()[0]  # Extract the short name
    xy_data = session.XYDataFromHistory(
        name=short_name, 
        odb=odb, 
        outputVariableName=var, 
        steps=('Step-1', ), 
        __linkedVpName__='Viewport: 1'
    )
    
    # Extract data and save to file
    data = [datum[1] for datum in xy_data.data]
    save_output_to_file(short_name, data)
    
    # Save time data for the first variable
    if i == 0:
        time_data = [datum[0] for datum in xy_data.data]
        save_output_to_file('time', time_data)


##### TEST AND SAVE ANIMATIONS AND STUFF #####

# Define the path for the new folder
pics_and_videos_folder = os.path.join(resultsfolder, "pics_and_videos")

# Create the new folder if it doesn't already exist
if not os.path.exists(pics_and_videos_folder):
    os.makedirs(pics_and_videos_folder)

### Make first pic of configuration
# New image size (4 times bigger)
new_image_size = (4096, 1976)
# Set new image size and save the file to the new directory
session.pngOptions.setValues(imageSize=new_image_size)
session.aviOptions.setValues(sizeDefinition=USER_DEFINED, imageSize=new_image_size)
# Save '01_View_ALL.png'
output_file_path = os.path.join(pics_and_videos_folder, '01_View_ALL.png')
session.printToFile(fileName=output_file_path, format=PNG, canvasObjects=(
    session.viewports['Viewport: 1'], ))

### Make 2nd viewport
session.Viewport(name='Viewport: 2', origin=(3.86250019073486, 
    3.04351854324341), width=216.782821655273, height=114.372222900391)
session.viewports['Viewport: 2'].makeCurrent()
session.viewports['Viewport: 2'].maximize()
session.viewports['Viewport: 1'].restore()
session.viewports['Viewport: 2'].restore()
session.viewports['Viewport: 1'].setValues(origin=(0.0, 3.04351806640625), 
    width=123.1171875, height=118.216667175293)
session.viewports['Viewport: 2'].setValues(origin=(123.1171875, 
    3.04351806640625), width=123.1171875, height=118.216667175293)
session.viewports['Viewport: 1'].makeCurrent()

### Zoom in Viewport 1
session.viewports['Viewport: 1'].view.setValues(nearPlane=12.4496, 
    farPlane=18.6602, width=0.156711, height=0.146062, viewOffsetX=-0.0931698, 
    viewOffsetY=0.387197)

### Zoom, 2D, perspective off and Cut in Viewport 2
session.viewports['Viewport: 2'].makeCurrent()
session.viewports['Viewport: 2'].view.setProjection(projection=PARALLEL)
session.viewports['Viewport: 2'].view.setValues(session.views['Front'])
session.viewports['Viewport: 2'].odbDisplay.setValues(viewCutNames=('Z-Plane', 
    ), viewCut=ON)
session.viewports['Viewport: 2'].view.setValues(nearPlane=12.353, 
    farPlane=16.7322, width=0.568504, height=0.529872, cameraPosition=(
    -0.000552788, 0.0638656, 14.5426), cameraTarget=(-0.000552788, 0.0638656, 
    0))

# Save '02_View_ZOOM.png'
output_file_path_zoom = os.path.join(pics_and_videos_folder, '02_View_ZOOM.png')
session.printToFile(fileName=output_file_path_zoom, format=PNG, canvasObjects=(
    session.viewports['Viewport: 2'], session.viewports['Viewport: 1']))

### Link viewport, switch to deformed config (at last inc)
session. linkedViewportCommands.setValues(linkViewports=True)
session.viewports['Viewport: 2'].odbDisplay.display.setValues(plotState=(
    DEFORMED, ))

# Save '03_View_ZOOM_Deformed.png'
output_file_path_zoom_def = os.path.join(pics_and_videos_folder, '03_View_ZOOM_Deformed.png')
session.printToFile(fileName=output_file_path_zoom_def, format=PNG, canvasObjects=(
    session.viewports['Viewport: 2'], session.viewports['Viewport: 1']))

### Show contours on def (von Mises is shown first)
session.viewports['Viewport: 2'].odbDisplay.display.setValues(plotState=(
    CONTOURS_ON_DEF, ))
session.viewports['Viewport: 2'].viewportAnnotationOptions.setValues(compass=OFF, 
    legendDecimalPlaces=2, legendNumberFormat=FIXED, legendBox=OFF)
session.viewports['Viewport: 2'].odbDisplay.contourOptions.setValues(
    numIntervals=5, maxAutoCompute=OFF, maxValue=27, minAutoCompute=OFF, minValue=0)

# Turn off averaging, for some reason for both viewports needed even though they are linked
session.viewports['Viewport: 2'].odbDisplay.basicOptions.setValues(
    averageElementOutput=False)
session.viewports['Viewport: 1'].odbDisplay.basicOptions.setValues(
    averageElementOutput=False)

# Animation Stuff
session.viewports['Viewport: 2'].animationController.setValues(
    animationType=TIME_HISTORY)
session.viewports['Viewport: 2'].animationController.play(duration=UNLIMITED)
session.aviOptions.setValues(sizeDefinition=USER_DEFINED)
session.imageAnimationOptions.setValues(vpDecorations=ON, vpBackground=OFF, 
    compass=OFF, timeScale=1, frameRate=10)

# Save '04_Anim_vonMises.avi'
output_file_path_ani_mises = os.path.join(pics_and_videos_folder, '04_Anim_vonMises.avi')
session.writeImageAnimation(fileName=output_file_path_ani_mises, format=AVI, 
    canvasObjects=(session.viewports['Viewport: 2'], 
    session.viewports['Viewport: 1']))

# remove fiber from view
leaf = dgo.LeafFromElementSets(elementSets=("PART-2-1.SET-2", ))
session.viewports['Viewport: 2'].odbDisplay.displayGroup.remove(leaf=leaf)
session.imageAnimationOptions.setValues(vpDecorations=ON, vpBackground=OFF, 
    compass=OFF)

# Save '05_Anim_vonMises_noFib.avi'
output_file_path_ani_mises = os.path.join(pics_and_videos_folder, '05_Anim_vonMises_noFib.avi')
session.writeImageAnimation(fileName=output_file_path_ani_mises, format=AVI, 
    canvasObjects=(session.viewports['Viewport: 2'], 
    session.viewports['Viewport: 1']))

# Show PEEQ

session.viewports['Viewport: 2'].odbDisplay.setPrimaryVariable(
    variableLabel='PEEQ', outputPosition=INTEGRATION_POINT, )
session.viewports['Viewport: 2'].odbDisplay.contourOptions.setValues(
    maxValue=0.10)

# Save '06_Anim_peeq.avi'
output_file_path_peeq = os.path.join(pics_and_videos_folder, '06_Anim_peeq.avi')
session.writeImageAnimation(fileName=output_file_path_peeq, format=AVI, 
    canvasObjects=(session.viewports['Viewport: 2'], 
    session.viewports['Viewport: 1']))

# Go to CSHEAR
session.viewports['Viewport: 2'].odbDisplay.setPrimaryVariable(
    variableLabel='CSHEAR2', outputPosition=ELEMENT_NODAL, )
session.viewports['Viewport: 2'].odbDisplay.contourOptions.setValues(
    maxValue=12.74,minValue=-12.74, numIntervals=9)
session.Spectrum(name="POS-NEG_SAME", colors =('#FF0000', '#FFB200', '#E7FF00', 
    '#00FF2E', '#0000FF', '#2EFF00', '#E7FF00', '#FFB200', '#FF0000', ))
session.viewports['Viewport: 2'].odbDisplay.contourOptions.setValues(
    spectrum='POS-NEG_SAME')

# Save '07_Anim_cshear2.avi'
output_file_path_peeq = os.path.join(pics_and_videos_folder, '07_Anim_cshear2.avi')
session.writeImageAnimation(fileName=output_file_path_peeq, format=AVI, 
    canvasObjects=(session.viewports['Viewport: 2'], 
    session.viewports['Viewport: 1']))