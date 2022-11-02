import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from napari import Viewer, layers
from scipy import spatial
from skimage import measure, morphology
from tifffile import imread

from ..utils import location_map


class OneatVolumeVisualization:
    def __init__(
        self,
        viewer: Viewer,
        key_categories: dict,
        csvdir: str,
        savedir: str,
        savename: str,
        ax,
        figure: plt.figure,
    ):

        self.viewer = viewer
        self.csvdir = csvdir
        self.savedir = savedir
        self.savename = savename
        self.key_categories = key_categories
        self.ax = ax
        self.figure = figure
        self.dataset = None
        self.event_name = None
        self.cell_count = None
        self.image = None
        self.seg_image = None
        self.event_locations = []
        self.event_locations_dict = {}
        self.event_locations_size_dict = {}
        self.size_locations = []
        self.score_locations = []
        self.confidence_locations = []
        self.event_locations_clean = []
        self.cleantimelist = []
        self.cleaneventlist = []
        self.cleannormeventlist = []
        self.cleancelllist = []
        self.labelsize = {}
        self.segimagedir = None
        self.plot_event_name = None
        self.event_count_plot = None
        self.event_norm_count_plot = None
        self.cell_count_plot = None
        self.imagename = None
        self.originalimage = None

    # To prevent early detectin of events
    def cluster_points(self, nms_space):

        print("before", len(self.event_locations_size_dict))

        for (k, v) in self.event_locations_dict.items():
            currenttime = k
            event_locations = v

            if len(event_locations) > 0:
                tree = spatial.cKDTree(event_locations)
                forwardtime = currenttime + 1
                if int(forwardtime) in self.event_locations_dict.keys():
                    forward_event_locations = self.event_locations_dict[
                        int(forwardtime)
                    ]
                    for location in forward_event_locations:
                        if (
                            int(forwardtime),
                            int(location[0]),
                            int(location[1]),
                            int(location[2]),
                        ) in self.event_locations_size_dict:
                            (
                                forwardsize,
                                forwardscore,
                                forwardconfidence,
                            ) = self.event_locations_size_dict[
                                int(forwardtime),
                                int(location[0]),
                                int(location[1]),
                                int(location[2]),
                            ]
                            distance, nearest_location = tree.query(location)
                            nearest_location = (
                                int(event_locations[nearest_location][0]),
                                int(event_locations[nearest_location][1]),
                                int(event_locations[nearest_location][2]),
                            )

                            if distance <= nms_space:
                                if (
                                    int(currenttime),
                                    int(nearest_location[0]),
                                    int(nearest_location[1]),
                                    int(nearest_location[2]),
                                ) in self.event_locations_size_dict:
                                    (
                                        currentsize,
                                        currentscore,
                                        currentconfidence,
                                    ) = self.event_locations_size_dict[
                                        int(currenttime),
                                        int(nearest_location[0]),
                                        int(nearest_location[1]),
                                        int(nearest_location[2]),
                                    ]
                                    if currentscore >= forwardscore:
                                        self.event_locations_size_dict.pop(
                                            (
                                                int(forwardtime),
                                                int(location[0]),
                                                int(location[1]),
                                                int(location[2]),
                                            )
                                        )

                                    if currentscore < forwardscore:
                                        self.event_locations_size_dict.pop(
                                            (
                                                int(currenttime),
                                                int(nearest_location[0]),
                                                int(nearest_location[1]),
                                                int(nearest_location[2]),
                                            )
                                        )

        print("after", len(self.event_locations_size_dict))
        self.show_clean_csv()

    def show_clean_csv(self):
        self.cleaneventlist = []
        self.cleantimelist = []
        self.event_locations_clean.clear()
        dict_locations = self.event_locations_size_dict.keys()
        tlocations = []
        zlocations = []
        ylocations = []
        xlocations = []
        scores = []
        radiuses = []
        confidences = []
        for location, sizescore in self.event_locations_size_dict.items():
            tlocations.append(float(location[0]))
            zlocations.append(float(location[1]))
            ylocations.append(float(location[2]))
            xlocations.append(float(location[3]))

            radiuses.append(float(sizescore[0]))
            scores.append(float(sizescore[1]))
            confidences.append(float(sizescore[2]))
        for location in dict_locations:
            self.event_locations_clean.append(location)

        event_count = np.column_stack(
            [
                tlocations,
                zlocations,
                ylocations,
                xlocations,
                scores,
                radiuses,
                confidences,
            ]
        )
        event_count = sorted(event_count, key=lambda x: x[0], reverse=False)

        event_data = []
        csvname = self.savedir + "/" + "Clean" + self.event_name + "Location"
        if os.path.exists(csvname + ".csv"):
            os.remove(csvname + ".csv")
        writer = csv.writer(open(csvname + ".csv", "a", newline=""))
        filesize = os.stat(csvname + ".csv").st_size

        if filesize < 1:
            writer.writerow(
                ["T", "Z", "Y", "X", "Score", "Size", "Confidence"]
            )
        for line in event_count:
            if line not in event_data:
                print(f"line{line}")
                event_data.append(line)
            writer.writerows(event_data)
            event_data = []
        name_remove = ("Clean Detections", "Clean Location Map")

        point_properties = {
            "score": scores,
            "confidence": confidences,
            "size": radiuses,
        }

        for layer in list(self.viewer.layers):

            if any(name in layer.name for name in name_remove):
                self.viewer.layers.remove(layer)
        self.viewer.add_points(
            self.event_locations_clean,
            properties=point_properties,
            symbol="square",
            blending="translucent_no_depth",
            name="Clean Detections",
            face_color=[0] * 4,
            edge_color="green",
        )

        df = pd.DataFrame(
            self.event_locations_clean, columns=["T", "Z", "Y", "X"]
        )
        T_pred = df[df.keys()[0]][0:]
        listtime_pred = T_pred.tolist()

        for j in range(self.image.shape[0]):
            cleanlist = []
            for i in range(len(listtime_pred)):

                if j == listtime_pred[i]:
                    cleanlist.append(listtime_pred[i])

            countT = len(cleanlist)
            self.cleantimelist.append(j)
            self.cleaneventlist.append(countT)

    def show_plot(
        self,
        plot_event_name,
        event_count_plot,
        cell_count_plot,
        event_norm_count_plot,
        segimagedir=None,
        event_threshold=0,
    ):

        timelist = []
        eventlist = []
        normeventlist = []
        celllist = []
        self.ax.cla()

        self.segimagedir = segimagedir
        self.plot_event_name = plot_event_name
        self.event_count_plot = event_count_plot
        self.event_norm_count_plot = event_norm_count_plot
        self.cell_count_plot = cell_count_plot

        if self.dataset is not None:

            for layer in list(self.viewer.layers):
                if isinstance(layer, layers.Image):
                    self.image = layer.data
                if isinstance(layer, layers.Labels):
                    self.seg_image = layer.data

            if self.image is not None:
                currentT = np.round(self.dataset["T"]).astype("int")
                try:
                    currentscore = self.dataset["Score"]
                except ValueError:
                    currentscore = currentT * 0 + 1.0

                for i in range(0, self.image.shape[0]):

                    condition = currentT == i
                    condition_indices = self.dataset_index[condition]
                    conditionScore = currentscore[condition_indices]
                    score_condition = conditionScore > event_threshold
                    countT = len(conditionScore[score_condition])
                    timelist.append(i)
                    eventlist.append(countT)
                    if (
                        self.segimagedir is not None
                        and self.seg_image is not None
                    ):

                        all_cells = self.cell_count[i]
                        celllist.append(all_cells + 1)
                        normeventlist.append(countT / (all_cells + 1))
                self.cleannormeventlist = []
                if len(self.cleaneventlist) > 0:
                    for k in range(len(self.cleaneventlist)):
                        self.cleannormeventlist.append(
                            self.cleaneventlist[k] / celllist[k]
                        )

                if self.plot_event_name == self.event_count_plot:
                    self.ax.plot(timelist, eventlist, "-r")
                    self.ax.plot(self.cleantimelist, self.cleaneventlist, "-g")
                    self.ax.set_title(self.event_name + "Events")
                    self.ax.set_xlabel("Time")
                    self.ax.set_ylabel("Counts")
                    self.figure.canvas.draw()
                    self.figure.canvas.flush_events()

                    self.figure.savefig(
                        self.savedir
                        + self.event_name
                        + self.event_count_plot
                        + ".png",
                        dpi=300,
                    )

                if (
                    self.plot_event_name == self.event_norm_count_plot
                    and len(normeventlist) > 0
                ):
                    self.ax.plot(timelist, normeventlist, "-r")
                    self.ax.plot(
                        self.cleantimelist, self.cleannormeventlist, "-g"
                    )
                    self.ax.set_title(self.event_name + "Normalized Events")
                    self.ax.set_xlabel("Time")
                    self.ax.set_ylabel("Normalized Counts")
                    self.figure.canvas.draw()
                    self.figure.canvas.flush_events()

                    self.figure.savefig(
                        self.savedir
                        + self.event_name
                        + self.event_norm_count_plot
                        + ".png",
                        dpi=300,
                    )

                if (
                    self.plot_event_name == self.cell_count_plot
                    and len(celllist) > 0
                ):
                    self.ax.plot(timelist, celllist, "-r")
                    self.ax.set_title("Total Cell counts")
                    self.ax.set_xlabel("Time")
                    self.ax.set_ylabel("Total Cell Counts")
                    self.figure.canvas.draw()
                    self.figure.canvas.flush_events()
                    self.figure.savefig(
                        self.savedir + self.cell_count_plot + ".png", dpi=300
                    )

    def show_image(
        self,
        image_toread,
        imagename,
        segimagedir=None,
        heatmapimagedir=None,
        heatname="_Heat",
    ):
        self.imagename = imagename
        name_remove = ("Image", "SegImage")
        for layer in list(self.viewer.layers):
            if any(name in layer.name for name in name_remove):
                self.viewer.layers.remove(layer)
        try:
            self.image = imread(image_toread)

            if heatmapimagedir is not None:
                try:
                    heat_image = imread(
                        heatmapimagedir + imagename + heatname + ".tif"
                    )
                except FileNotFoundError:
                    heat_image = None

            if segimagedir is not None:
                self.seg_image = imread(segimagedir + imagename + ".tif")

                self.viewer.add_labels(
                    self.seg_image.astype("uint16"),
                    name="SegImage" + imagename,
                )

            self.originalimage = self.image
            self.viewer.add_image(self.image, name="Image" + imagename)
            if heatmapimagedir is not None:
                try:
                    self.viewer.add_image(
                        heat_image,
                        name="Image" + imagename + heatname,
                        blending="additive",
                        colormap="inferno",
                    )
                except FileNotFoundError:
                    pass

        except FileNotFoundError:
            pass

    def show_csv(
        self,
        imagename,
        csv_event_name,
        segimagedir=None,
        event_threshold=0,
        heatmapsteps=0,
        nms_space=0,
    ):

        csvname = None
        self.event_locations_size_dict.clear()
        self.size_locations = []
        self.score_locations = []
        self.event_locations = []
        self.confidence_locations = []

        for layer in list(self.viewer.layers):
            if "Detections" in layer.name or layer.name in "Detections":
                self.viewer.layers.remove(layer)
        for (event_name, event_label) in self.key_categories.items():
            if event_label > 0 and csv_event_name == event_name:
                self.event_label = event_label
                csvname = list(Path(self.csvdir).glob("*.csv"))[0]
        if csvname is not None:

            self.event_name = csv_event_name
            self.dataset = pd.read_csv(csvname, delimiter=",")
            nrows = len(self.dataset.columns)
            for index, row in self.dataset.iterrows():
                tcenter = int(row[0])
                zcenter = row[1]
                ycenter = row[2]
                xcenter = row[3]
                if nrows > 4:
                    score = row[4]
                    size = row[5]
                    confidence = row[6]
                else:
                    score = 1.0
                    size = 10
                    confidence = 1.0
                self.dataset_index = self.dataset.index
                if score > event_threshold:
                    self.event_locations.append(
                        [
                            int(tcenter),
                            int(zcenter),
                            int(ycenter),
                            int(xcenter),
                        ]
                    )

                    if int(tcenter) in self.event_locations_dict.keys():
                        current_list = self.event_locations_dict[int(tcenter)]
                        current_list.append(
                            [int(zcenter), int(ycenter), int(xcenter)]
                        )
                        self.event_locations_dict[int(tcenter)] = current_list
                        self.event_locations_size_dict[
                            (
                                int(tcenter),
                                int(zcenter),
                                int(ycenter),
                                int(xcenter),
                            )
                        ] = [size, score, confidence]
                    else:
                        current_list = []
                        current_list.append(
                            [int(zcenter), int(ycenter), int(xcenter)]
                        )
                        self.event_locations_dict[int(tcenter)] = current_list
                        self.event_locations_size_dict[
                            int(tcenter),
                            int(zcenter),
                            int(ycenter),
                            int(xcenter),
                        ] = [size, score, confidence]

                    self.size_locations.append(size)
                    self.score_locations.append(score)
                    self.confidence_locations.append(confidence)
            point_properties = {
                "score": np.array(self.score_locations),
                "confidence": np.array(self.confidence_locations),
                "size": np.array(self.size_locations),
            }

            name_remove = ("Detections", "Location Map")
            for layer in list(self.viewer.layers):

                if any(name in layer.name for name in name_remove):
                    self.viewer.layers.remove(layer)
            if len(self.score_locations) > 0:
                self.viewer.add_points(
                    self.event_locations,
                    properties=point_properties,
                    symbol="square",
                    blending="translucent_no_depth",
                    name="Detections" + event_name,
                    face_color=[0] * 4,
                    edge_color="red",
                )

            if segimagedir is not None:
                for layer in list(self.viewer.layers):
                    if isinstance(layer, layers.Labels):
                        self.seg_image = layer.data

                        location_image, self.cell_count = location_map(
                            self.event_locations_dict,
                            self.seg_image,
                            heatmapsteps,
                            display_3d=True,
                        )
                        self.viewer.add_labels(
                            location_image.astype("uint16"),
                            name="Location Map" + imagename,
                        )

            self.cluster_points(nms_space)


def TimedDistance(pointA, pointB):

    spacedistance = float(
        np.sqrt(
            (pointA[1] - pointB[1]) * (pointA[1] - pointB[1])
            + (pointA[2] - pointB[2]) * (pointA[2] - pointB[2])
        )
    )

    timedistance = float(np.abs(pointA[0] - pointB[0]))

    return spacedistance, timedistance


def GetMarkers(image):

    MarkerImage = np.zeros(image.shape)
    waterproperties = measure.regionprops(image)
    Coordinates = [prop.centroid for prop in waterproperties]
    Coordinates = sorted(Coordinates, key=lambda k: [k[0], k[1]])
    coordinates_int = np.round(Coordinates).astype(int)
    MarkerImage[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))

    markers = morphology.dilation(MarkerImage, morphology.disk(2))

    return markers
