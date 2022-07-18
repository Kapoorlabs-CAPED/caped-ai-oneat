#@ File(label='Tracking image hyperstack', style='file') imagepath
#@ Integer(label="Integer Detection Channel", required=true, value=2, stepSize=1) integer_channel
#@ Boolean(label='Use Mari Principle', value=True) mari_principle
#@ Float(label="Mari Angle", required=true, value=30, stepSize=1) mari_angle
#@ Float(label="Prob Threshold", required=true, value=0.90, stepSize=0.1) oneat_prob_threshold

#@ Float(label="Linking Max Distance", required=true, value=14, stepSize=1) linking_maxdist
#@ Float(label="Gap Closing Max Distance", required=true, value=16, stepSize=1) gap_maxdist
#@ Integer(label="Max Frame Gap", required=true, value=3, stepSize=1) gap_maxframe

#@ File(label='Oneat Mitosis File', style='file') oneat_mitosis_file
#@ File(label='Save XML directory', style='directory') savedir
from pickle import FALSE
import sys

import java.io.File as File
from ij import IJ
from ij import WindowManager

import net.imagej.ImageJ as ImageJ
from fiji.plugin.trackmate.io import TmXmlWriter
from  fiji.plugin.trackmate.gui.wizard import TrackMateWizardSequence
from net.imagej.axis import Axes
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.detection import LabeImageDetectorFactory
from fiji.plugin.trackmate.tracking import LAPUtils
from fiji.plugin.trackmate.util import TMUtils

from fiji.plugin.trackmate.tracking.sparselap import SimpleSparseLAPTrackerFactory
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettingsIO
from fiji.plugin.trackmate.action.oneat import OneatCorrectorFactory, OneatExporterPanel
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
import fiji.plugin.trackmate.features.FeatureFilter as FeatureFilter

from  net.imglib2.img.display.imagej import ImgPlusViews
# We have to do the following to avoid errors with UTF8 chars generated in 
# TrackMate that will mess with our Fiji Jython.
reload(sys)
sys.setdefaultencoding('utf-8')

# Get the image path and open the image


imp = IJ.openImage(str(imagepath))


#Or use the currently open image 
#imp = WindowManager.getCurrentImage()


#----------------------------
# Create the model object now
#----------------------------

# Some of the parameters we configure below need to have
# a reference to the model at creation. So we create an
# empty model now.

model = Model()

# Send all messages to ImageJ log window.
model.setLogger(Logger.IJ_LOGGER)



#------------------------
# Prepare settings object
#------------------------

settings = Settings(imp)

# Configure detector - We use the Strings for the keys
settings.detectorFactory = LabeImageDetectorFactory()
settings.detectorSettings = {
    'TARGET_CHANNEL' : integer_channel,
    'SIMPLIFY_CONTOURS' : False,
}  


# Configure tracker - We want to allow merges and fusions
settings.trackerFactory = SimpleSparseLAPTrackerFactory()
settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap() # almost good enough
settings.trackerSettings['LINKING_MAX_DISTANCE'] = linking_maxdist
settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = gap_maxdist

settings.trackerSettings['MAX_FRAME_GAP'] = gap_maxframe

# Add ALL the feature analyzers known to TrackMate. They will 
# yield numerical features for the results, such as speed, mean intensity etc.
settings.addAllAnalyzers()

#-------------------
# Instantiate plugin
#-------------------

trackmate = TrackMate(model, settings)

#--------
# Process
#--------

ok = trackmate.checkInput()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))

ok = trackmate.process()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))

# A selection.
selectionModel = SelectionModel( model )

# Read the default display settings.
ds = DisplaySettingsIO.readUserDefault()

# Echo results with the logger we set at start:
model.getLogger().log( str( model ) )
savename = imp.getShortTitle()
img = TMUtils.rawWraps( settings.imp )

 

#Lets start the oneat part of track linking
settings = trackmate.getSettings()
trackmapsettings = settings.trackerSettings
detectorsettings = settings.detectorSettings

if img.dimensionIndex(Axes.CHANNEL) > 0:
		detectionimg = ImgPlusViews.hyperSlice( img, img.dimensionIndex( Axes.CHANNEL ),  integer_channel - 1 )
		
elif (img.dimensionIndex(Axes.CHANNEL) < 0 and img.numDimensions() < 5):
	    detectionimg = img
	    
elif (img.numDimensions() == 5):
		detectionimg = ImgPlusViews.hyperSlice( img, 2, integer_channel )
			
			
intimg =  detectionimg;

corrector = OneatCorrectorFactory()
oneatmap = { 'MITOSIS_FILE': oneat_mitosis_file,
          'DETECTION_THRESHOLD': oneat_prob_threshold,
          'USE_MARI_PRINCIPLE':mari_principle,
          'MARI_ANGLE':mari_angle,
          'MAX_FRAME_GAP':gap_maxframe,
          'CREATE_LINKS': True,
          'BREAK_LINKS': True,
          'ALLOW_GAP_CLOSING': True,
          'SPLITTING_MAX_DISTANCE' : linking_maxdist,
          'GAP_CLOSING_MAX_DISTANCE':gap_maxdist}          
calibration = [settings.dx,settings.dy,settings.dz]

oneatcorrector = corrector.create(intimg,model, trackmate, settings, ds,oneatmap,model.getLogger(), calibration, False)
oneatcorrector.checkInput()
oneatcorrector.process()
model = oneatcorrector.returnModel()
savefile = File(str(savedir) + '/' +   savename + ".xml") 

#Write the autocorrected tracks to xml file
writer = TmXmlWriter( savefile, model.getLogger() )
selectionModel = SelectionModel( model )
seq = TrackMateWizardSequence(trackmate, selectionModel, ds)
state = seq.configDescriptor()
writer.appendLog( model.getLogger().toString() )
writer.appendModel( trackmate.getModel() )
writer.appendSettings( trackmate.getSettings() )
writer.appendGUIState( state.getPanelDescriptorIdentifier() )
writer.appendDisplaySettings( ds )
writer.writeToFile()