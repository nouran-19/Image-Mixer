import logging
import numpy as np
import cv2


class Image:
    """
    A class for handling image processing operations.

    Attributes:
        imgData (numpy.ndarray): The image data.
        imgFourier (numpy.ndarray): The Fourier transform of the image data.
        imgFourierShifted (numpy.ndarray): The shifted Fourier transform of the image data.
        imgFourierInv (numpy.ndarray): The inverse Fourier transform of the image data.
        imgShape (tuple): The shape of the image data.
        cropped_data_fourier (numpy.ndarray): Cropped Fourier transform data.
        brightness (float): The brightness factor for the image.
        contrast (float): The contrast factor for the image.
    """

    def __init__(self):
        self.imgData = None
        self.imgFourier = None
        self.imgFourierShifted = None
        self.imgFourierInv = None
        self.imgShape = None

        self.cropped_data_fourier = None

        self.brightness = 1.0
        self.contrast = 1.0

    def loadImage(
        self,
        path: str = None,
        data: np.ndarray = None,
        imgShape: tuple = None,
    ):
        """
         Load an image from a file path or numpy array.

        Args:
             path (str, optional): The file path of the image. Defaults to None.
             data (numpy.ndarray, optional): The image data as a numpy array. Defaults to None.
             imgShape (tuple, optional): The shape of the image data. Defaults to None.
        """
        if data is not None:
            # logging.debug(f"loadimage given data: shape{data.shape}")
            self.imgData = data
            self.imgShape = imgShape

        else:

            if path:
                self.imgData = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                self.imgShape = self.imgData.shape
                # logging.info(f"the loaded image shape given path is {self.imgShape}")
            else:
                return

        resized = cv2.resize(self.imgData, (313, 165))
        self.imgData = resized.T
        self.imgShape = (313, 165)
        # logging.info(f"the image shape after resizing inside loadimage {self.imgShape}")

        try:
            self.imgFourier = np.fft.fft2(self.imgData)
            # logging.debug(f"shape of fourier transposed{self.imgFourier.shape}")
            # logging.debug(f"image shape {self.imgShape}")

        except Exception as e:
            # logging.exception(f"exception {e}", exc_info=True)
            pass

        self.imgFourierShifted = np.fft.fftshift(self.imgFourier)
        logging.debug(f" unshifted fourier load img{self.imgFourier} \n\n")
        logging.debug(f" shifted fourier load img{self.imgFourierShifted} \n\n")

    @staticmethod
    def inverseFourier(array: np.ndarray) -> np.ndarray:
        """
        Compute the inverse Fourier transform of an array.

        """
        return np.real(np.fft.ifft2(array))

    @staticmethod
    def realComponent(array: np.ndarray) -> np.ndarray:
        """
        Compute the real component of a complex array.
        """
        return np.real(array)

    @staticmethod
    def imaginaryComponent(array: np.ndarray) -> np.ndarray:
        """
        Compute the imaginary component of a complex array.

        """
        return np.imag(array)

    @staticmethod
    def magnitude(array: np.ndarray) -> np.ndarray:
        """
        Compute the magnitude of a complex array.

        """
        return np.abs(array)

    @staticmethod
    def phase(array: np.ndarray) -> np.ndarray:
        """
        Compute the phase of a complex array.

        """
        return np.angle(array)
