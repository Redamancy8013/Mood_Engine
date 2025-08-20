from visual.emotic import det

def Load():
    """
    Load the model and its parameters.
    """
    model, device, half, imgsz, stride, opt = det.load()
    return model, device, half, imgsz, stride, opt

def RunInference(url, model, device, half, imgsz, stride, opt):
    """
    Run inference on the given image URL using the loaded model.

    Args:
        url (str): The URL of the image to process.
        model: The loaded model.
        device: The device to run the model on.
        half (bool): Whether to use half precision.
        imgsz (int): The size of the input image.
        stride (int): The stride of the model.
        opt: Additional options for inference.

    Returns:
        tuple: Predicted category and content.
    """
    # print(f"Running inference on the image: {url}")
    pred_cat, pred_cont = det.run_inference(url, model, device, half, imgsz, stride, opt)
    return pred_cat, pred_cont
