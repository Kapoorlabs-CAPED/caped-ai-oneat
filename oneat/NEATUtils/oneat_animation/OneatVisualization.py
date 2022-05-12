class OneatVisualization:

    def __init__(self, viewer, image_to_read, imagename, csv_event_name, plot_event_name, savedir, savename):

        self.viewer = viewer
        self.image_to_read = image_to_read
        self.imagename = imagename
        self.csv_event_name = csv_event_name
        self.plot_event_name = plot_event_name
        self.savedir = savedir
        self.savename = savename