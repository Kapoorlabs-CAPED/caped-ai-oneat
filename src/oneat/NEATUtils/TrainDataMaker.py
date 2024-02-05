import napari
import os

from tifffile import imread
import random
import pandas as pd


class TrainDataMaker:
    def __init__(
        self,
        source_dir,
        save_dir=None,
        create_2dt=False,
        class_names=["Normal", "Mitosis", "Apoptosis"],
    ):

        self.source_dir = source_dir
        self.save_dir = save_dir if save_dir else source_dir
        self.create_2dt = create_2dt
        self.class_names = class_names
        self.acceptable_formats = [".tif", ".TIFF", ".TIF", ".png"]
        self._click_maker()

    def _click_maker(self):
        from qtpy.QtWidgets import QComboBox, QPushButton

        self.viewer = napari.Viewer()
        Imageids = []
        Boxname = "ImageIDBox"
        X = os.listdir(self.source_dir)
        imageidbox = QComboBox()
        imageidbox.addItem(Boxname)
        tracksavebutton = QPushButton("Save Clicks")
        for imagename in X:
            if any(imagename.endswith(f) for f in self.acceptable_formats):
                Imageids.append(os.path.join(self.source_dir, imagename))
        for i in range(0, len(Imageids)):
            imageidbox.addItem(str(Imageids[i]))

        self.viewer.window.add_dock_widget(imageidbox, name="Image", area="bottom")
        self.viewer.window.add_dock_widget(
            tracksavebutton, name="Save Clicks", area="bottom"
        )
        imageidbox.currentIndexChanged.connect(
            lambda trackid=imageidbox: EventViewer(
                self.viewer,
                imageidbox.currentText(),
                os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
                self.save_dir,
                save=False,
                newimage=True,
                create_2dt=self.create_2dt,
                class_names=self.class_names,
            )
        )

        tracksavebutton.clicked.connect(
            lambda trackid=tracksavebutton: EventViewer(
                self.viewer,
                imageidbox.currentText(),
                os.path.basename(os.path.splitext(imageidbox.currentText())[0]),
                self.save_dir,
                save=True,
                newimage=False,
                create_2dt=self.create_2dt,
                class_names=self.class_names,
            )
        )

        napari.run()


class EventViewer:
    def __init__(
        self,
        viewer: napari.Viewer,
        imagename,
        Name,
        csv_dir,
        save=False,
        newimage=True,
        create_2dt=False,
        class_names=[],
    ):

        self.save = save
        self.newimage = newimage
        self.viewer = viewer
        print("reading image")
        self.imagename = imagename
        self.image = imread(imagename)
        print("image read")
        self.Name = Name
        self.ndim = len(self.image.shape)
        self.csv_dir = csv_dir
        self.create_2dt = create_2dt
        self.class_names = class_names
        self.class_colors = {
            class_name: self.generate_random_color() for class_name in self.class_names
        }
        if not self.create_2dt:
            assert (
                self.ndim == 4
            ), f"Input image should be 4 dimensional, found {self.ndim}, try Training_data_maker for 2D + time images"
        else:
            assert (
                self.ndim == 3
            ), f"Input image should be 3 dimensional,found {self.ndim}, try contacting KapoorLabs for custom image analysis development"

        self._click()

    def generate_random_color(self):
        return f"#{random.randint(0, 0xFFFFFF):06x}"

    def _click(self):

        if self.save is True:
            for class_name in self.class_names:
                class_data = self.viewer.layers[class_name].data
                if not self.create_2dt:
                    class_df = pd.DataFrame(
                        class_data, index=None, columns=["T", "Z", "Y", "X"]
                    )
                else:
                    class_df = pd.DataFrame(
                        class_data, index=None, columns=["T", "Y", "X"]
                    )
                class_df.to_csv(
                    self.csv_dir + "/ONEAT" + class_name + self.Name + ".csv",
                    index=False,
                    mode="w",
                )

        if self.newimage is True:
            for layer in list(self.viewer.layers):

                self.viewer.layers.remove(layer)

        if self.save is False:
            self.viewer.add_image(self.image, name=self.Name)

            for class_name in self.class_names:
                face_color = self.class_colors[class_name]
                self.viewer.add_points(
                    name=class_name, face_color=face_color, ndim=self.ndim
                )
                self.viewer.layers[class_name].mode = "add"
