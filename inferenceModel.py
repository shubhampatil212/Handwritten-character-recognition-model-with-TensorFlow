import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

class HandwrittenWordRecognizer(OnnxInferenceModel):
    def __init__(self, character_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.character_list = character_list

    def recognize(self, image: np.ndarray):
        resized_image = cv2.resize(image, self.input_shapes[0][1:3][::-1])
        image_batch = np.expand_dims(resized_image, axis=0).astype(np.float32)
        predictions = self.model.run(self.output_names, {self.input_names[0]: image_batch})[0]
        recognized_text = ctc_decoder(predictions, self.character_list)[0]
        return recognized_text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs

    config = BaseModelConfigs.load("D:\Codes\Internship\configs.yaml")
    recognizer = HandwrittenWordRecognizer(model_path=config.model_path, character_list=config.vocab)

    validation_data = pd.read_csv("D:\Codes\Internship\drive-download-20240727T174604Z-001\val.csv").values.tolist()

    cumulative_cer = []
    for img_path, ground_truth in tqdm(validation_data):
        input_image = cv2.imread(img_path.replace("\\", "/"))
        predicted_text = recognizer.recognize(input_image)
        error_rate = get_cer(predicted_text, ground_truth)
        print(f"Image: {img_path}, Ground Truth: {ground_truth}, Prediction: {predicted_text}, CER: {error_rate}")
        cumulative_cer.append(error_rate)

        display_image = cv2.resize(input_image, (input_image.shape[1] * 4, input_image.shape[0] * 4))
        cv2.imshow("Input Image", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Average CER: {np.mean(cumulative_cer)}")