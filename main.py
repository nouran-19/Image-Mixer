from UI.ui_dark_blue import Ui_MainWindow
import sys
from PyQt6.QtCore import QElapsedTimer, QCoreApplication
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QSlider
from PyQt6.QtGui import QFont, QMouseEvent
import cv2
from PyQt6 import QtCore
import numpy as np
import pyqtgraph
from functools import partial
from PyQt6.QtCore import QPointF
import logging
import traceback
from image import Image

logging.basicConfig(
    level=logging.DEBUG,
    filename="log.log",
    filemode="w",
    format="%(asctime)s - %(message)s",
)
logging.debug("debug")
logging.info("info")
logging.warning("warning")
logging.error("error")
logging.critical("critical")


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.img1 = Image()
        self.img2 = Image()
        self.img3 = Image()
        self.img4 = Image()
        self.output1 = Image()
        self.output2 = Image()

        self.showCmbxs = [
            self.ui.image1Cmbx,
            self.ui.image2Cmbx,
            self.ui.image3Cmbx,
            self.ui.image4Cmbx,
        ]
        self.all_images = [
            self.ui.imageOneOrigin,
            self.ui.imageOneOrigin_2,
            self.ui.imageTwoOrigin,
            self.ui.imageTwoOrigin_2,
            self.ui.imageThreeOrigin,
            self.ui.imageThreeOrigin_2,
            self.ui.imageFourOrigin,
            self.ui.imageFourOrigin_2,
        ]

        self.images_input = [
            self.ui.imageOneOrigin,
            self.ui.imageTwoOrigin,
            self.ui.imageThreeOrigin,
            self.ui.imageFourOrigin,
        ]
        self.fourier_image_views = [
            self.ui.imageOneOrigin_2,
            self.ui.imageTwoOrigin_2,
            self.ui.imageThreeOrigin_2,
            self.ui.imageFourOrigin_2,
        ]
        self.images_output = [self.ui.outputOne, self.ui.outputTwo]

        self.checkboxes = [
            self.ui.checkBox,
            self.ui.checkBox_2,
            self.ui.checkBox_3,
            self.ui.checkBox_4,
        ]

        self.inner = True

        self.use_ROI = False
        self.remove_rois()

        self.output_dict = {"Output 1": self.output1, "Output 2": self.output2}

        self.mixer_radiobtns = [self.ui.radioButton_mp, self.ui.radioButton_ri]
        self.image_objects = [self.img1, self.img2, self.img3, self.img4]

        self.labels_img_phase = [
            self.ui.label_img_phase_1,
            self.ui.label_img_phase_2,
            self.ui.label_img_phase_3,
            self.ui.label_img_phase_4,
        ]
        self.labels_real_mag = [
            self.ui.label_real_mag_1,
            self.ui.label_real_mag_2,
            self.ui.label_real_mag_3,
            self.ui.label_real_mag_4,
        ]

        self.weights_real_mag = []
        self.weights_img_phase = []

        self.ui.radioButton_mp.setChecked(True)
        self.changeFinished = False

        self.slider_group1 = [
            self.ui.slider1_1,
            self.ui.slider2_1,
            self.ui.slider3_1,
            self.ui.slider4_1,
        ]
        self.slider_group2 = [
            self.ui.slider1_2,
            self.ui.slider2_2,
            self.ui.slider3_2,
            self.ui.slider4_2,
        ]

        for slider in self.slider_group1 + self.slider_group2:
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(25)

        self.resetbtns = [
            self.ui.ResetBtn,
            self.ui.ResetBtn_2,
            self.ui.ResetBtn_3,
            self.ui.ResetBtn_4,
        ]

        self.reset = False
        self.statusBar = self.statusBar()
        bold_font = QFont()
        bold_font.setBold(True)
        bold_font.setPointSize(20)
        self.statusBar.setFont(bold_font)
        self.status_timer = QElapsedTimer()

        self.connect_signals_slots()

    def connect_signals_slots(self):
        # Connect mouse click event for loading images
        for i, image_view in enumerate(self.images_input):
            image_view.getView().scene().sigMouseClicked.connect(
                partial(self.mainWindow_load_img, i + 1)
            )
        # Connect combo box activation to selecting components
        for index, bx in enumerate(self.showCmbxs):
            bx.activated.connect(partial(self.selectComponents, index + 1))

        # Connect slider value changes to updating slider groups and mixing images
        for i, slider in enumerate(self.slider_group1):
            slider.valueChanged.connect(
                lambda value, idx=i: self.update_slider_groups(idx, self.slider_group1)
            )
        for slider in self.slider_group1 + self.slider_group2:
            slider.sliderReleased.connect(self.mix_images)

        for i, slider in enumerate(self.slider_group2):
            slider.valueChanged.connect(
                lambda value, idx=i: self.update_slider_groups(idx, self.slider_group2)
            )

        # Connect mouse click event on Fourier transform views
        for index, image_view in enumerate(self.fourier_image_views):
            image_view.getView().scene().sigMouseClicked.connect(
                partial(self.view_clicked, index)
            )

        # Connect mouse drag event for adjusting brightness
        for index, image_view in enumerate(self.images_input):
            view_box = image_view.getView()
            view_box.setMouseEnabled(
                x=False, y=False
            )  # Disable default mouse interactions
            view_box.mouseDragEvent = partial(self.mouse_drag_event_brightness, index)

        # Connect radio button clicks to updating labels
        for radiobtn in self.mixer_radiobtns:
            radiobtn.clicked.connect(self.update_labels)

        # Connect ROI region changed signal to updating ROI
        for roi_tuple in self.ui.rois:
            roi_tuple[0].sigRegionChanged.connect(partial(self.updateROI, roi_tuple))

        # Connect reset button clicks to resetting images
        for index, btn in enumerate(self.resetbtns):
            btn.clicked.connect(partial(self.reset_img, index))

        # Connect checkbox toggles to linking checkboxes
        for index, check_btn in enumerate(self.checkboxes):
            check_btn.toggled.connect(self.link_checkbtns)

        # Connect ROI checkbox toggle to toggling ROIs
        self.ui.checkBox_2.toggled.connect(self.toggle_rois)

    def link_checkbtns(self) -> None:
        """
        This method is triggered when a checkbox state changes.
        It ensures that when one checkbox is checked or unchecked, all other checkboxes are
        synchronized to have the same state.
        """
        logging.debug(f"from link_checkbtns function, use_ROI is {self.use_ROI}")
        cbtn = self.sender()
        for btn in self.checkboxes:
            btn.setChecked(cbtn.isChecked())

    def toggle_rois(self) -> None:
        """
        This method toggles the state of the `use_ROI` attribute.
        if ROIs are enabled, it calls the `add_rois` method to add ROIs to the images,
        otherwise, it calls the `remove_rois` method to remove existing ROIs from the images.
        """
        logging.debug(f"from toggle_ROIs function, use_ROI is {self.use_ROI}")
        self.use_ROI = not self.use_ROI
        if self.use_ROI:
            self.add_rois()
        else:
            self.remove_rois()

    def reset_img(self, image_index: int) -> None:
        """
        This method resets the specified image to its original state.
         It sets the `reset` attribute to True, unchecks the checkbox for ROIs,
        shows the original image, and removes any lookup tables applied to the image.
        """
        self.reset = True
        self.ui.checkBox_2.setChecked(False)  # reset roi
        self.showImg(self.images_input[image_index], self.image_objects[image_index])
        self.images_input[image_index].getImageItem().lut = None

    def view_clicked(self, img_view_index, evt):
        """
        Handle click events on an image view.

        Args:
            img_view_index (int): The index of the image view in the `fourier_image_views` list.
            evt (QEvent): The click event.
        """
        logging.debug(f"evt: {evt} index {img_view_index}")
        view = self.fourier_image_views[img_view_index].getView()

        # Map the event position to the scene's coordinates
        pos_in_scene = view.mapToScene(evt.pos())

        roi_rect = self.ui.rois[img_view_index][0].mapRectToScene(
            self.ui.rois[img_view_index][0].boundingRect()
        )

        self.inner = roi_rect.contains(pos_in_scene)
        self.mix_images()
        self.updateROI((self.ui.rois[0]))

    def update_slider_groups(self, changed_index: int, group: list[QSlider]) -> None:
        """
        Update slider groups based on changes in one slider.

        This method recalculates the values of sliders in a group when one slider is
        changed, ensuring that the total sum of slider values remains constant.

        """

        total_value = 0
        for i, slider in enumerate(group):
            if i != changed_index:
                total_value += slider.value()

        # Avoid division by zero
        if total_value == 0:
            equal_value = 100 // len(group)
            for i, slider in enumerate(group):
                slider.setValue(equal_value)
        else:
            remaining_value = 100 - group[changed_index].value()
            for i, slider in enumerate(group):
                if i != changed_index:
                    slider_value = int(remaining_value * (slider.value() / total_value))
                    slider.setValue(slider_value)

        self.set_weights()

    def set_weights(self) -> None:
        """
        Set weights for real magnitude and imaginary phase components.

        """
        # Clear existing values
        self.weights_real_mag = []
        self.weights_img_phase = []

        for slider in self.slider_group1:
            self.weights_real_mag.append(slider.value())
        for slider in self.slider_group2:
            self.weights_img_phase.append(slider.value())

    def update_labels(self) -> None:
        """
        Update labels based on selected radio buttons.
        """
        for label_img_phase, label_real_mag in zip(
            self.labels_img_phase, self.labels_real_mag
        ):
            if self.ui.radioButton_ri.isChecked():
                label_img_phase.setText("img")
                label_real_mag.setText("real")
            else:
                label_img_phase.setText("phase")
                label_real_mag.setText("mag")

    def mouse_drag_event_brightness(self, index: int, ev: QMouseEvent) -> None:
        """
        Handle mouse drag events for adjusting brightness and contrast.

        """
        if ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            delta_brightness = ev.lastPos().y() - ev.pos().y()
            self.image_objects[index].brightness += delta_brightness * 0.04
            brightness_lut = np.linspace(0, 255, 256, dtype=np.uint8)
            brightness_lut = np.clip(
                brightness_lut * self.image_objects[index].brightness, 0, 255
            ).astype(np.uint8)

            delta_contrast = ev.lastPos().x() - ev.pos().x()
            self.image_objects[index].contrast += delta_contrast * 0.01
            contrast_lut = np.linspace(
                128 - 128 * self.image_objects[index].contrast,
                128 + 128 * self.image_objects[index].contrast,
                256,
                dtype=np.uint8,
            )
            contrast_lut = np.clip(contrast_lut, 0, 255).astype(np.uint8)

            img_item = self.images_input[index].getImageItem()
            if delta_brightness > delta_contrast:
                img_item.setLookupTable(brightness_lut, update=True)
            else:
                img_item.setLookupTable(contrast_lut, update=True)

            img_item.updateImage()
            self.selectComponents(index + 1)

    def mainWindow_load_img(self, Indx: int) -> None:
        """
        This method opens a file dialog to select an image file, then loads the image
        into the specified index of the application.

        """
        self.imagePath, self.imageFormat = QFileDialog.getOpenFileName(
            None, f"Load Image {Indx}"
        )
        self.loading_image_and_componants(
            self.image_objects[Indx - 1],
            self.images_input[Indx - 1],
            self.imagePath,
            Indx,
        )

    def loading_image_and_componants(
        self, imageInst: Image, widget: "pyqtgraph.ImageView", path: str, index: int
    ) -> None:
        """
        This method loads the specified image into the given ImageView widget,
        then selects components based on the index
        """
        try:
            # logging.debug(f"showimg called from mainWindow_load_img")
            self.showImg(widget=widget, imageInst=imageInst, imagePath=path)
            self.selectComponents(index)

        except ValueError:
            pass

    def showImg(
        self,
        widget: "pyqtgraph.ImageView",
        imageInst: Image = None,
        widgetData: np.ndarray = None,
        imagePath: str = None,
    ) -> None:
        """
        This method clears the ImageView widget, then loads and displays the specified
        image in the widget. The image can be provided either as an instance of the
        Image class (`imageInst`), as raw image data (`widgetData`), or as a file path
        (`imagePath`).
        """
        widget.clear()
        # If imagePath is provided, load the image into imageInst
        if imagePath is not None:
            imageInst.loadImage(path=imagePath)
        # Check if widgetData is provided
        if widgetData is not None:
            logging.debug(f"shape of showimg data {widgetData.shape}")
            widget.getImageItem().setImage(widgetData)
        else:
            logging.debug(f"shape of showimg data {imageInst.imgData.shape}")
            widget.getImageItem().setImage(imageInst.imgData)

    def selectComponents(self, img: int) -> None:
        """
        This method selects components of the specified image and displays them
        based on the selected mode
        """
        image_object = self.image_objects[img - 1]
        image_views = self.all_images[
            2 * img - 1
        ]  # 1->1, 2->3, 3->5 4->7 (see all images list)

        if image_object is not None:
            mode = self.showCmbxs[img - 1].currentText()
            if self.use_ROI and (self.reset == False):
                self.showComponent(mode, image_views, image_object, croppedFourier=True)
            else:
                self.showComponent(mode, image_views, image_object)

    def showComponent(
        self,
        mode: str,
        widget: "pyqtgraph.ImageView",
        image_inst: Image = None,
        croppedFourier=False,
        shifted: bool = True,
    ):
        """
        This method displays a specific component of the image, such as magnitude,
        phase, real component, or imaginary component, in the specified ImageView widget.
        """
        display_data = None

        if mode == "Magnitude":
            display_data = 20 * np.log(Image.magnitude(image_inst.imgFourierShifted))

        elif mode == "Phase":
            display_data = Image.phase(image_inst.imgFourierShifted)

        elif mode == "Real Component":
            display_data = Image.realComponent(image_inst.imgFourierShifted)

        else:
            display_data = Image.imaginaryComponent(image_inst.imgFourierShifted)

        if display_data is not None:
            logging.debug(f"showimg called from showComponent")
            self.showImg(widget=widget, widgetData=display_data)

    def get_roi_mask(self, roi_tuple: tuple) -> np.ndarray:
        """
        This method extracts the mask of the region of interest (ROI) from the Fourier
        transform based on the provided ROI tuple.
        """
        image_index = roi_tuple[1]
        roi_slices = roi_tuple[0].getArraySlice(
            self.image_objects[image_index].imgFourier,
            self.fourier_image_views[image_index].getImageItem(),
        )
        self.mask = np.zeros_like(self.image_objects[image_index].imgFourier).astype(
            bool
        )
        self.mask[
            roi_slices[0][0].start : roi_slices[0][0].stop,
            roi_slices[0][1].start : roi_slices[0][1].stop,
        ] = 1

        return self.mask

    def remove_rois(self):
        for index, widget in enumerate(self.ui.images[4:8]):
            self.ui.rois[index][0].setSize(QPointF(0, 0))
            self.ui.rois[index][0].setPos(QPointF(0, 0))

    def add_rois(self) -> None:
        """
        This method removes regions of interest (ROIs) from the images by setting
        their size and position to zero.
        """
        for index, widget in enumerate(self.ui.images[4:8]):
            self.ui.rois[index][0].setSize(QPointF(10, 10))
            self.ui.rois[index][0].setPos(QPointF(0, 0))

    def updateROI(self, roi_tuple: tuple) -> None:
        """
        This method updates the regions of interest (ROIs) in the images based on the
        provided ROI tuple. It forms cropped arrays for all ROIs, synchronizes the ROIs,
        updates the components, and mixes the images.
        """
        if not self.use_ROI:
            return

        logging.info(f"update is evoked by ROI {roi_tuple[1]}")

        final_arrays_to_mixer = []

        if self.use_ROI:
            final_arrays_to_mixer = self.form_cropped_arrays_for_all_ROIs(
                final_arrays_to_mixer
            )

        for index, img in enumerate(self.image_objects):
            img.cropped_data_fourier = final_arrays_to_mixer[index]

        self.sync_ROIs(roi_tuple)

        for index, img in enumerate(self.image_objects):
            self.selectComponents(index + 1)

        self.mix_images()

    def form_cropped_arrays_for_all_ROIs(
        self, final_arrays_to_mixer: list[np.ndarray]
    ) -> list[np.ndarray]:
        """
        Form cropped arrays for all regions of interest (ROIs).

        """
        for roi_tuple in self.ui.rois:
            self.get_roi_mask(roi_tuple)

            if self.inner:
                array_to_be_sent_to_mixer = (
                    self.mask * self.image_objects[roi_tuple[1]].imgFourierShifted
                )
            else:  # outer
                array_to_be_sent_to_mixer = (
                    ~self.mask * self.image_objects[roi_tuple[1]].imgFourierShifted
                )

            final_arrays_to_mixer.append(array_to_be_sent_to_mixer)

        for i, obj in enumerate(self.image_objects):
            obj.cropped_data_fourier = final_arrays_to_mixer[i]
        # logging.debug(f"TRACK PADDING AND CROPPING")
        # logging.debug(f"cropped array shape {cropped_array.shape}")
        # logging.debug(f"padded array shape {padded_array.shape}")

        return final_arrays_to_mixer

    def sync_ROIs(self, current_roi: tuple) -> None:  # movement
        """
        Synchronize the positions and sizes of all regions of interest (ROIs).

        """
        current_roi_size = current_roi[0].size()
        current_roi_pos = current_roi[0].pos()

        for other_roi_tuple in self.ui.rois:
            if other_roi_tuple != current_roi:
                other_roi = other_roi_tuple[0]
                other_roi.setPos(current_roi_pos, update=False)
                other_roi.setSize(current_roi_size, update=False)

    def show_message_in_status_bar(self, message: str, duration: int = 5000) -> None:
        """
        Display a message in the status bar for a specified duration.

        """
        self.statusBar.showMessage(message)
        self.status_timer.restart()

        while self.status_timer.elapsed() < duration:
            # Keep the event loop running to allow the message to be displayed
            QCoreApplication.processEvents()

        self.statusBar.clearMessage()

    def mix(
        self,
        weights_real_mag: list[float],
        weights_img_phase: list[float],
        use_ROI: bool,
        image_objects: list[Image],
    ) -> np.ndarray:
        """
        This method combines multiple images based on the specified weights for
        their real-magnitude and imaginary-phase components. It returns the mixed
        image.

        """
        lists_to_sum_on = [np.zeros(image_objects[0].imgShape) for _ in range(4)]

        logging.debug(f"the shape going into the mixer{image_objects[0].imgShape}")

        for i, img_obj in enumerate(image_objects):

            logging.debug(
                f"cropped data fourier (should be shifted) {img_obj.cropped_data_fourier}\n\n"
            )

            if img_obj.imgData is not None:
                # Whether to use the full Fourier or the padded array of ROI
                if use_ROI:
                    to_be_mixed = np.fft.ifftshift(img_obj.cropped_data_fourier)
                else:
                    to_be_mixed = img_obj.imgFourier

                if mainWindow.ui.radioButton_mp.isChecked():
                    # Magnitude
                    lists_to_sum_on[0] += (
                        weights_real_mag[i] / 100 * Image.magnitude(to_be_mixed)
                    )
                    # Phase
                    lists_to_sum_on[1] += (
                        weights_img_phase[i] / 100 * Image.phase(to_be_mixed)
                    )
                else:
                    # Real
                    lists_to_sum_on[2] += (
                        weights_real_mag[i] / 100 * Image.realComponent(to_be_mixed)
                    )
                    # Imaginary
                    lists_to_sum_on[3] += (
                        weights_img_phase[i]
                        / 100
                        * Image.imaginaryComponent(to_be_mixed)
                    )

                logging.debug(f"mags sum shifted {lists_to_sum_on[0]}\n\n")

        if mainWindow.ui.radioButton_mp.isChecked():
            to_be_sent_to_ifft = lists_to_sum_on[0] * np.exp(1j * lists_to_sum_on[1])
        else:
            to_be_sent_to_ifft = lists_to_sum_on[2] + 1j * lists_to_sum_on[3]

        inverse_img = Image.inverseFourier(to_be_sent_to_ifft)

        return inverse_img

    def mix_images(self) -> None:
        """
        Mix images based on selected weights and options.
        """

        output = self.output_dict[self.ui.mixerOutput.currentText()]
        try:
            mixed_image = self.mix(
                self.weights_real_mag,
                self.weights_img_phase,
                self.use_ROI,
                self.image_objects,
            )

            if mixed_image is not None:
                cv2.imwrite("test1.jpg", mixed_image)
                op_img = cv2.imread("test1.jpg")
                self.showImg(
                    widgetData=op_img,
                    widget=self.images_output[not (output == self.output1)],
                )

        except Exception as e:
            traceback.print_exc()  # Print the traceback to diagnose the issue
            return None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    mainWindow = MyMainWindow()
    mainWindow.setWindowTitle("Image Mixer")
    mainWindow.show()
    sys.exit(app.exec())
