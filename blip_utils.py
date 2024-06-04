from lavis.models import load_model_and_preprocess

models = {}


def load_default_blip2_model(device, mode='eval'):
    global models
    if device in models:
        model, vis_processors = models[device]
        return model, vis_processors[mode]
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_image_text_matching", 
                                                         model_type="pretrain", is_eval=True, 
                                                         device=device)
    models[device] = (model, vis_processors)
    return model, vis_processors[mode]


def clean_memory():
    global models
    models.clear()