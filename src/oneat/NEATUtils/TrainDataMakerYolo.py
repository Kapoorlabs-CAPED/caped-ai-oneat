import napari
import os
import cv2
from tifffile import imread
from qtpy.QtWidgets import QComboBox, QPushButton


class TrainDataMakerYolo:
    def __init__(self, source_dir, save_dir=None, class_names=["Object"]):
        self.source_dir = source_dir
        self.save_dir = save_dir if save_dir else source_dir
        self.class_names = class_names
        self.acceptable_formats = [".tif", ".TIFF", ".TIF", ".png", ".jpg"]
        self._box_maker()

    def _box_maker(self):
        self.viewer = napari.Viewer()
        Imageids = []
        imageidbox = QComboBox()
        imageidbox.addItem("Select Image")
        savebutton = QPushButton("Save Annotations")

        # Get all images from source directory
        for imagename in os.listdir(self.source_dir):
            if any(imagename.endswith(f) for f in self.acceptable_formats):
                Imageids.append(os.path.join(self.source_dir, imagename))

        for img_path in Imageids:
            imageidbox.addItem(str(img_path))

        # Add widgets to Napari UI
        self.viewer.window.add_dock_widget(imageidbox, name="Image", area="bottom")
        self.viewer.window.add_dock_widget(savebutton, name="Save", area="bottom")

        # Connect functions
        imageidbox.currentIndexChanged.connect(
            lambda: EventViewer(
                self.viewer,
                imageidbox.currentText(),
                os.path.splitext(os.path.basename(imageidbox.currentText()))[0],
                self.save_dir,
                save=False,
                newimage=True,
                class_names=self.class_names,
            )
        )

        savebutton.clicked.connect(
            lambda: EventViewer(
                self.viewer,
                imageidbox.currentText(),
                os.path.splitext(os.path.basename(imageidbox.currentText()))[0],
                self.save_dir,
                save=True,
                newimage=False,
                class_names=self.class_names,
            )
        )

        napari.run()


class EventViewer:
    def __init__(
        self,
        viewer,
        imagename,
        Name,
        save_dir,
        save=False,
        newimage=True,
        class_names=[],
    ):
        self.viewer = viewer
        self.imagename = imagename
        self.image = imread(imagename)
        self.Name = Name
        self.save_dir = save_dir
        self.class_names = class_names
        self.save = save
        self.newimage = newimage

        self.images_dir = os.path.join(self.save_dir, "images")
        self.labels_dir = os.path.join(self.save_dir, "labels")

        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        assert len(self.image.shape) == 2, "Image must be 2D!"

        self._box()

    def _box(self):
        if self.save:
            for class_name in self.class_names:
                layer = self.viewer.layers.get(class_name)
                if layer is not None:
                    self._save_annotations(class_name, layer.data)

            # Save image
            img_save_path = os.path.join(self.images_dir, f"{self.Name}.png")
            cv2.imwrite(img_save_path, self.image)
            print(f"Saved Image: {img_save_path}")

        if self.newimage:
            for layer in list(self.viewer.layers):
                self.viewer.layers.remove(layer)

        if not self.save:
            self.viewer.add_image(self.image, name=self.Name)
            for class_name in self.class_names:
                self.viewer.add_shapes(
                    name=class_name,
                    shape_type="rectangle",
                    edge_color="red",
                    face_color="transparent",
                )
                self.viewer.layers[class_name].mode = "add"

    def _save_annotations(self, class_name, data):
        """Save bounding box annotations in YOLO format."""
        label_file = os.path.join(self.labels_dir, f"{self.Name}.txt")
        H, W = self.image.shape[:2]

        with open(label_file, "w") as f:
            for box in data:
                x_min, y_min, x_max, y_max = box[0][0], box[0][1], box[2][0], box[2][1]

                # Convert to YOLO format (normalized)
                x_center = ((x_min + x_max) / 2) / W
                y_center = ((y_min + y_max) / 2) / H
                width = (x_max - x_min) / W
                height = (y_max - y_min) / H

                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        print(f"Saved Labels: {label_file}")
