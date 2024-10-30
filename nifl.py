from pydicom import Dataset, FileDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
import datetime
import numpy as np
from PIL import Image
import os

def jpeg_to_dicom(jpeg_file, dicom_file):
    # Step 1: Load the JPEG image
    image = Image.open(jpeg_file)
    image = image.convert('L')  # Convert to grayscale if needed
    image_array = np.array(image)

    # Step 2: Create a DICOM dataset
    dataset = Dataset()

    # Step 3: Populate necessary DICOM metadata
    dataset.PatientName = "Test^Patient"
    dataset.PatientID = "123456"
    dataset.Modality = "OT"  # Other
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.SOPInstanceUID = generate_uid()
    dataset.SOPClassUID = "SecondaryCaptureImageStorage"
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.Rows, dataset.Columns = image_array.shape
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.ImageType = ["ORIGINAL", "PRIMARY", "OTHER"]
    dataset.InstanceCreationDate = datetime.datetime.now().strftime('%Y%m%d')
    dataset.InstanceCreationTime = datetime.datetime.now().strftime('%H%M%S')
    dataset.PixelData = image_array.tobytes()

    # Create the FileDataset instance (wraps the dataset)
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = dataset.SOPClassUID
    file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dicom_dataset = FileDataset(dicom_file, dataset, file_meta=file_meta, preamble=b"\0" * 128)
    dicom_dataset.is_little_endian = True
    dicom_dataset.is_implicit_VR = False

    # Step 4: Save the DICOM file
    dicom_dataset.save_as(dicom_file)

# Example usage
jpeg_file = "dataset1/test/Normal/Normal- (81).jpg"
dicom_file = "output_path_to_save_image.dcm"
jpeg_to_dicom(jpeg_file, dicom_file)

print(f"JPEG image converted and saved as DICOM at {dicom_file}")
