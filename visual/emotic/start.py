from det import run_inference, load

if __name__ == "__main__":
    
    model, device, half, imgsz, stride, opt = load()

    url = '/root/emotic/visual/emotic/temp/happy.png'

    pred_cat, pred_cont = run_inference(url, model, device, half, imgsz, stride, opt)

    print(f"Predicted Category: {pred_cat}, Predicted Content: {pred_cont}")