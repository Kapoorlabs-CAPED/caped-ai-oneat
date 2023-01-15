#!/usr/bin/env python3
"""
Created on Wed Aug  4 14:50:47 2021

@author: vkapoor
"""
import csv
import glob
import json
import os
from pathlib import Path


import numpy as np
import pandas as pd
import napari
from qtpy.QtCore import Qt
from oneat.NEATUtils.oneat_animation._qt import OneatVolumeWidget, OneatWidget
from scipy import spatial
from tifffile import imread

from oneat.NEATUtils.oneat_animation.OneatVisualization import MidSlices


class NEATViz:
    def __init__(
        self,
        imagedir: str,
        csvdir: str,
        savedir: str,
        categories_json: dict,
        heatmapimagedir: str = None,
        segimagedir: str = None,
        heatname: str = "_Heat",
        eventname: str = "_Event",
        fileextension: str = "*tif",
        blur_radius: int = 5,
        start_project_mid: int = None,
        end_project_mid: int = None,
        headless: bool = False,
        volume: bool = True,
        batch: bool = False,
        event_threshold: float = 0.999,
        nms_space: int = 10,
        nms_time: int = 3,
    ):

        self.imagedir = imagedir
        self.heatmapimagedir = heatmapimagedir
        self.segimagedir = segimagedir
        self.savedir = savedir
        self.volume = volume
        self.csvdir = csvdir
        self.heatname = heatname
        self.eventname = eventname
        self.headless = headless
        self.batch = batch
        self.event_threshold = event_threshold
        self.nms_space = nms_space
        self.nms_time = nms_time
        self.categories_json = categories_json
        self.start_project_mid = start_project_mid
        self.end_project_mid = end_project_mid
        self.fileextension = fileextension
        self.blur_radius = blur_radius

        Path(self.savedir).mkdir(exist_ok=True)
        Path(self.csvdir).mkdir(exist_ok=True)
        

        self.time = 0
        self.key_categories = self.load_json()
        if not self.headless:
            
            self.viewer = napari.Viewer() 
        if not self.headless and not self.volume:
            self.showNapari()
        if self.headless and not self.volume:
            self.donotshowNapari()
        if self.volume and not self.headless:
            self.showVolumeNapari()
        if self.volume and self.headless:
            self.donotshowVolumeNapari()    

    def load_json(self):
        with open(self.categories_json) as f:
            return json.load(f)

    def donotshowVolumeNapari(self):
        
        headlessvolumecall(
                self.key_categories,
                self.event_threshold,
                self.nms_space,
                self.nms_time,
                self.csvdir,
                self.savedir,
            )
        
    def donotshowNapari(self):

        headlesscall(
                
                self.key_categories,
                self.event_threshold,
                self.nms_space,
                self.nms_time,
                self.csvdir,
                self.savedir,
            )

    def showNapari(self):

        self.oneat_widget = OneatWidget(
            self.viewer,
            self.csvdir,
            self.savedir,
            "Name",
            self.key_categories,
            segimagedir=self.segimagedir,
            heatimagedir=self.heatmapimagedir,
            heatname=self.heatname,
            start_project_mid=self.start_project_mid,
            end_project_mid=self.end_project_mid,
        )
        Raw_path = os.path.join(self.imagedir, self.fileextension)
        X = glob.glob(Raw_path)
        Imageids = []
        self.oneat_widget.frameWidget.imageidbox.addItem("Select Image")
        self.oneat_widget.frameWidget.eventidbox.addItem("Select Event")
        for imagename in X:
            Imageids.append(imagename)

        for i in range(0, len(Imageids)):
            self.oneat_widget.frameWidget.imageidbox.addItem(str(Imageids[i]))

        for (event_name, event_label) in self.key_categories.items():
            if event_label > 0:
                self.oneat_widget.frameWidget.eventidbox.addItem(event_name)

        dock_widget = self.viewer.window.add_dock_widget(
            self.oneat_widget, area="right"
        )
        self.viewer.window._qt_window.resizeDocks(
            [dock_widget], [200], Qt.Horizontal
        )

        napari.run()

    def showVolumeNapari(self):

        self.oneat_widget = OneatVolumeWidget(
            self.viewer,
            self.csvdir,
            self.savedir,
            "Name",
            self.key_categories,
            segimagedir=self.segimagedir,
            heatimagedir=self.heatmapimagedir,
            heatname=self.heatname,
        )
        Raw_path = os.path.join(self.imagedir, self.fileextension)
        X = glob.glob(Raw_path)
        Imageids = []
        self.oneat_widget.frameWidget.imageidbox.addItem("Select Image")
        self.oneat_widget.frameWidget.eventidbox.addItem("Select Event")
        for imagename in X:
            Imageids.append(imagename)

        for i in range(0, len(Imageids)):
            self.oneat_widget.frameWidget.imageidbox.addItem(str(Imageids[i]))

        for (event_name, event_label) in self.key_categories.items():
            if event_label > 0:
                self.oneat_widget.frameWidget.eventidbox.addItem(event_name)

        dock_widget = self.viewer.window.add_dock_widget(
            self.oneat_widget, area="right"
        )
        self.viewer.window._qt_window.resizeDocks(
            [dock_widget], [200], Qt.Horizontal
        )

        napari.run()


def cluster_points(
    event_locations_dict, event_locations_size_dict, nms_space, nms_time
):

    for (k, v) in event_locations_dict.items():
        currenttime = k
        event_locations = v

        tree = spatial.cKDTree(event_locations)
        for i in range(1, nms_time):

            forwardtime = currenttime + i
            if int(forwardtime) in event_locations_dict.keys():
                forward_event_locations = event_locations_dict[
                    int(forwardtime)
                ]
                for location in forward_event_locations:
                    if (
                        int(forwardtime),
                        int(location[0]),
                        int(location[1]),
                    ) in event_locations_size_dict:
                        forwardsize, forwardscore = event_locations_size_dict[
                            int(forwardtime),
                            int(location[0]),
                            int(location[1]),
                        ]
                        distance, nearest_location = tree.query(location)
                        nearest_location = int(
                            event_locations[nearest_location][0]
                        ), int(event_locations[nearest_location][1])

                        if distance <= nms_space:
                            if (
                                int(currenttime),
                                int(nearest_location[0]),
                                int(nearest_location[1]),
                            ) in event_locations_size_dict:
                                (
                                    currentsize,
                                    currentscore,
                                ) = event_locations_size_dict[
                                    int(currenttime),
                                    int(nearest_location[0]),
                                    int(nearest_location[1]),
                                ]
                                if currentsize >= forwardsize:
                                    event_locations_size_dict.pop(
                                        (
                                            int(forwardtime),
                                            int(location[0]),
                                            int(location[1]),
                                        )
                                    )

                                if currentsize < forwardsize:
                                    event_locations_size_dict.pop(
                                        (
                                            int(currenttime),
                                            int(nearest_location[0]),
                                            int(nearest_location[1]),
                                        )
                                    )
    return event_locations_size_dict

def cluster_spheres(event_locations_dict, event_locations_size_dict, nms_space, nms_time):

        print("before", len(event_locations_size_dict))

        for (k, v) in event_locations_dict.items():
            currenttime = k
            event_locations = v

            if len(event_locations) > 0:
                tree = spatial.cKDTree(event_locations)
                forwardtime = currenttime + 1
                if int(forwardtime) in event_locations_dict.keys():
                    forward_event_locations = event_locations_dict[
                        int(forwardtime)
                    ]
                    for location in forward_event_locations:
                        if (
                            int(forwardtime),
                            int(location[0]),
                            int(location[1]),
                            int(location[2]),
                        ) in event_locations_size_dict:
                            (
                                forwardsize,
                                forwardscore,
                                forwardconfidence,
                            ) = event_locations_size_dict[
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
                                ) in event_locations_size_dict:
                                    (
                                        currentsize,
                                        currentscore,
                                        currentconfidence,
                                    ) = event_locations_size_dict[
                                        int(currenttime),
                                        int(nearest_location[0]),
                                        int(nearest_location[1]),
                                        int(nearest_location[2]),
                                    ]
                                    if currentscore >= forwardscore:
                                        event_locations_size_dict.pop(
                                            (
                                                int(forwardtime),
                                                int(location[0]),
                                                int(location[1]),
                                                int(location[2]),
                                            )
                                        )

                                    if currentscore < forwardscore:
                                        event_locations_size_dict.pop(
                                            (
                                                int(currenttime),
                                                int(nearest_location[0]),
                                                int(nearest_location[1]),
                                                int(nearest_location[2]),
                                            )
                                        )
                                        
        return event_locations_size_dict                                

def headlesscall(
    key_categories: dict,
    event_threshold: float,
    nms_space: int,
    nms_time: int,
    csvdir: str,
    savedir: str,
):
    for (event_name, event_label) in key_categories.items():
        if event_label > 0:
            event_locations = []
            size_locations = []
            score_locations = []
            event_locations = []
            confidence_locations = []
            event_locations_dict = {}
            event_locations_size_dict = {}
            csvnames = list(Path(csvdir).glob("*.csv"))
            for csvname in csvnames:
                event_locations = []
                size_locations = []
                score_locations = []
                event_locations = []
                confidence_locations = []
                event_locations_dict = {}
                event_locations_size_dict = {}
                savename = Path(csvname).stem
                print(savename)
                dataset = pd.read_csv(csvname, delimiter=",")
                # Data is written as T, Y, X, Score, Size, Confidence
                T = dataset[dataset.keys()[0]][0:]
                Y = dataset[dataset.keys()[2]][0:]
                X = dataset[dataset.keys()[3]][0:]
                Score = dataset[dataset.keys()[4]][0:]
                Size = dataset[dataset.keys()[5]][0:]
                Confidence = dataset[dataset.keys()[6]][0:]
                listtime = T.tolist()
                listy = Y.tolist()
                listx = X.tolist()
                listsize = Size.tolist()

                listscore = Score.tolist()
                listconfidence = Confidence.tolist()

                for i in range(len(listtime)):

                    tcenter = int(listtime[i])
                    ycenter = float(listy[i])
                    xcenter = float(listx[i])
                    size = float(listsize[i])
                    score = float(listscore[i])
                    confidence = listconfidence[i]
                    if score > event_threshold:
                        event_locations.append(
                            [int(tcenter), int(ycenter), int(xcenter)]
                        )

                        if int(tcenter) in event_locations_dict.keys():
                            current_list = event_locations_dict[int(tcenter)]
                            current_list.append([int(ycenter), int(xcenter)])
                            event_locations_dict[int(tcenter)] = current_list
                            event_locations_size_dict[
                                (int(tcenter), int(ycenter), int(xcenter))
                            ] = [size, score]
                        else:
                            current_list = []
                            current_list.append([int(ycenter), int(xcenter)])
                            event_locations_dict[int(tcenter)] = current_list
                            event_locations_size_dict[
                                int(tcenter), int(ycenter), int(xcenter)
                            ] = [size, score]

                        size_locations.append(size)
                        score_locations.append(score)
                        confidence_locations.append(confidence)

                event_locations_size_dict = cluster_points(
                    event_locations_dict,
                    event_locations_size_dict,
                    nms_space,
                    nms_time,
                )
                event_locations_clean = []
                dict_locations = event_locations_size_dict.keys()
                tlocations = []
                zlocations = []
                ylocations = []
                xlocations = []
                scores = []
                radiuses = []
                confidences = []
                for location, sizescore in event_locations_size_dict.items():
                    tlocations.append(float(location[0]))
                    zlocations.append(0)
                    ylocations.append(float(location[1]))
                    xlocations.append(float(location[2]))
                    scores.append(float(sizescore[1]))
                    radiuses.append(float(sizescore[0]))
                    confidences.append(1)
                for location in dict_locations:
                    event_locations_clean.append(location)

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
                event_count = sorted(
                    event_count, key=lambda x: x[0], reverse=False
                )

                event_data = []
                csvname = savedir + "/" + "clean_" + savename
                if os.path.exists(csvname + ".csv"):
                    os.remove(csvname + ".csv")
                writer = csv.writer(open(csvname + ".csv", "a", newline=""))
                filesize = os.stat(csvname + ".csv").st_size

                if filesize < 1:
                    writer.writerow(
                        [
                            "T",
                            "Z",
                            "Y",
                            "X",
                            "Score",
                            "Size",
                            "Confidence",
                        ]
                    )
                for line in event_count:
                    if line not in event_data:
                        event_data.append(line)
                    writer.writerows(event_data)
                    event_data = []


def headlessvolumecall(
    key_categories: dict,
    event_threshold: float,
    nms_space: int,
    nms_time: int,
    csvdir: str,
    savedir: str,
):
    for (event_name, event_label) in key_categories.items():
        if event_label > 0:
            event_locations = []
            size_locations = []
            score_locations = []
            event_locations = []
            confidence_locations = []
            event_locations_dict = {}
            event_locations_size_dict = {}
            csvnames = list(Path(csvdir).glob("*.csv"))
            for csvname in csvnames:
                event_locations = []
                size_locations = []
                score_locations = []
                event_locations = []
                confidence_locations = []
                event_locations_dict = {}
                event_locations_size_dict = {}
                savename = Path(csvname).stem
                print(savename)
                dataset = pd.read_csv(csvname, delimiter=",")
                # Data is written as T, Y, X, Score, Size, Confidence
                T = dataset[dataset.keys()[0]][0:]
                Z = dataset[dataset.keys()[1]][0:]
                Y = dataset[dataset.keys()[2]][0:]
                X = dataset[dataset.keys()[3]][0:]
                Score = dataset[dataset.keys()[4]][0:]
                Size = dataset[dataset.keys()[5]][0:]
                Confidence = dataset[dataset.keys()[6]][0:]
                listtime = T.tolist()
                listy = Y.tolist()
                listz = Z.tolist()
                listx = X.tolist()
                listsize = Size.tolist()

                listscore = Score.tolist()
                listconfidence = Confidence.tolist()

                for i in range(len(listtime)):

                    tcenter = int(listtime[i])
                    ycenter = listy[i]
                    zcenter = listz[i]
                    xcenter = listx[i]
                    size = listsize[i]
                    score = listscore[i]
                    confidence = listconfidence[i]
                    if score > event_threshold:
                        event_locations.append(
                            [int(tcenter), int(zcenter), int(ycenter), int(xcenter)]
                        )

                        if int(tcenter) in event_locations_dict.keys():
                            current_list = event_locations_dict[int(tcenter)]
                            current_list.append([int(zcenter), int(ycenter), int(xcenter)])
                            event_locations_dict[int(tcenter)] = current_list
                            event_locations_size_dict[
                                (int(tcenter), int(zcenter), int(ycenter), int(xcenter))
                            ] = [size, score]
                        else:
                            current_list = []
                            current_list.append([int(zcenter), int(ycenter), int(xcenter)])
                            event_locations_dict[int(tcenter)] = current_list
                            event_locations_size_dict[
                                int(tcenter), int(zcenter), int(ycenter), int(xcenter)
                            ] = [size, score]

                        size_locations.append(size)
                        score_locations.append(score)
                        confidence_locations.append(confidence)

                event_locations_size_dict = cluster_points(
                    event_locations_dict,
                    event_locations_size_dict,
                    nms_space,
                    nms_time,
                )
                event_locations_clean = []
                dict_locations = event_locations_size_dict.keys()
                tlocations = []
                zlocations = []
                ylocations = []
                xlocations = []
                scores = []
                radiuses = []
                confidences = []
                for location, sizescore in event_locations_size_dict.items():
                    tlocations.append(float(location[0]))
                    zlocations.append(float(location[1]))
                    ylocations.append(float(location[2]))
                    xlocations.append(float(location[3]))
                    scores.append(float(sizescore[1]))
                    radiuses.append(float(sizescore[0]))
                    confidences.append(1)
                for location in dict_locations:
                    event_locations_clean.append(location)

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
                event_count = sorted(
                    event_count, key=lambda x: x[0], reverse=False
                )

                event_data = []
                csvname = savedir + "/" + "clean_" + savename
                if os.path.exists(csvname + ".csv"):
                    os.remove(csvname + ".csv")
                writer = csv.writer(open(csvname + ".csv", "a", newline=""))
                filesize = os.stat(csvname + ".csv").st_size

                if filesize < 1:
                    writer.writerow(
                        [
                            "T",
                            "Z",
                            "Y",
                            "X",
                            "Score",
                            "Size",
                            "Confidence",
                        ]
                    )
                for line in event_count:
                    if line not in event_data:
                        event_data.append(line)
                    writer.writerows(event_data)
                    event_data = []