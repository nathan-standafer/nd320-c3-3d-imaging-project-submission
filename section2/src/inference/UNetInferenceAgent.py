"""
Contains class that runs inferencing
"""
import torch
import numpy as np

from networks.RecursiveUNet import UNet

from utils.utils import med_reshape

import torch.nn.functional as F

class UNetInferenceAgent:
    """
    Stores model and parameters and some methods to handle inferencing
    """
    def __init__(self, parameter_file_path='', model=None, device="cpu", patch_size=64):

        self.model = model
        self.patch_size = patch_size
        self.device = device

        if model is None:
            self.model = UNet(num_classes=3)

        if parameter_file_path:
            self.model.load_state_dict(torch.load(parameter_file_path, map_location=self.device))

        self.model.to(device)

    def single_volume_inference_unpadded(self, volume):
        """
        Runs inference on a single volume of arbitrary patch size,
        padding it to the conformant size first

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        print("single_volume_inference_unpadded, shape: {}".format(volume.shape))
        print("single_volume_inference_unpadded, max: {}".format(np.max(volume)))
        return self.single_volume_inference(volume)

    def single_volume_inference(self, volume):
        """
        Runs inference on a single volume of conformant patch size

        Arguments:
            volume {Numpy array} -- 3D array representing the volume

        Returns:
            3D NumPy array with prediction mask
        """
        self.model.eval()
        
        #print("volume.shape: ", volume.shape)   #(33, 64, 64)
        # Assuming volume is a numpy array of shape [X,Y,Z] and we need to slice X axis
        slices = []

        return_value = np.zeros(volume.shape)

        # TASK: Write code that will create mask for each slice across the X (0th) dimension. After 
        # that, put all slices into a 3D Numpy array. You can verify if your method is 
        # correct by running it on one of the volumes in your training set and comparing 
        # with the label in 3D Slicer.
        # ##### my code #####
        volume_reshaped = med_reshape(volume, (volume.shape[0], self.patch_size, self.patch_size))
        return_value = np.zeros(volume_reshaped.shape)

        for i in range(volume_reshaped.shape[0]):
            slice_input = volume_reshaped[i]

            #print("slice_input.shape: ", slice_input.shape)

            slice_input_tensor = torch.from_numpy(slice_input).unsqueeze(0).unsqueeze(0).type(torch.cuda.FloatTensor)
            #print("slice_input_tensor.shape: ", slice_input_tensor.shape)

            slice_prediction = self.model(slice_input_tensor)
            #print("slice_prediction.shape: ", slice_prediction.shape)

            
            #using the softmax produces image data that most matches the label data, but inversed.  The array * -1 + 1 operation fixes this.
            slice_prediction_array = F.softmax(slice_prediction, dim=1).cpu().detach().numpy()[0][0]       
            #print("inference image average: {}".format(np.mean(slice_prediction_array)))
            slice_prediction_array = slice_prediction_array * -1 + 1

            #now, change it into a binary mask containing either 1's or 0's
            slice_prediction_array[slice_prediction_array > 0.5] = 1
            slice_prediction_array[slice_prediction_array <= 0.5] = 0

            #print("inference image average post softmax: {}".format(np.mean(slice_prediction_array)))

            return_value[i] = slice_prediction_array

        return return_value
