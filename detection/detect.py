from detection.ml.detection import ModelConfig


def auto_detection_coco(model_path, image_url, class_names):
    model_config = ModelConfig(model_path)
    return model_config.detect_object(image_url, class_names)
