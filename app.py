import os
import cv2
import numpy as np
import gradio as gr
from inference import run_inference


# points color and marker
colors = [(255, 0, 0), (0, 255, 0)]
markers = [1, 5]

# image examples
# in each list, the first element is image path,
# the second is id (used for original_image State),
# the third is an empty list (used for selected_points State)
image_examples = [
    [os.path.join(os.path.dirname(__file__), "./images/groceries.jpg"), 0, []],
    [os.path.join(os.path.dirname(__file__), "./images/truck.jpg"), 1, []],
]

with gr.Blocks(title="Segment Anything") as demo:
    with gr.Row():
        gr.Markdown(
            '''# Segment Anything!🚀
            The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. More information can be found in [**Official Project**](https://segment-anything.com/).
            '''
        )
        with gr.Row():
            # select model
            model_type = gr.Dropdown(["vit_h"], value='vit_h', label="Select Model")
            # select device
            device = gr.Dropdown(["cpu", "cuda"], value='cuda', label="Select Device")

    # SAM parameters
    with gr.Accordion(label='Parameters', open=False):
        with gr.Row():
            points_per_side = gr.Number(value=32, label="points_per_side", precision=0,
                                        info='''The number of points to be sampled along one side of the image. The total 
                                        number of points is points_per_side**2.''')
            pred_iou_thresh = gr.Slider(value=0.88, minimum=0, maximum=1.0, step=0.01, label="pred_iou_thresh",
                                        info='''A filtering threshold in [0,1], using the model's predicted mask quality.''')
            stability_score_thresh = gr.Slider(value=0.95, minimum=0, maximum=1.0, step=0.01, label="stability_score_thresh",
                                               info='''A filtering threshold in [0,1], using the stability of the mask under 
                                               changes to the cutoff used to binarize the model's mask predictions.''')
            min_mask_region_area = gr.Number(value=0, label="min_mask_region_area", precision=0,
                                             info='''If >0, postprocessing will be applied to remove disconnected regions 
                                             and holes in masks with area smaller than min_mask_region_area.''')
        with gr.Row():
            stability_score_offset = gr.Number(value=1, label="stability_score_offset",
                                               info='''The amount to shift the cutoff when calculated the stability score.''')
            box_nms_thresh = gr.Slider(value=0.7, minimum=0, maximum=1.0, step=0.01, label="box_nms_thresh",
                                       info='''The box IoU cutoff used by non-maximal ression to filter duplicate masks.''')
            crop_n_layers = gr.Number(value=0, label="crop_n_layers", precision=0,
                                      info='''If >0, mask prediction will be run again on crops of the image. 
                                      Sets the number of layers to run, where each layer has 2**i_layer number of image crops.''')
            crop_nms_thresh = gr.Slider(value=0.7, minimum=0, maximum=1.0, step=0.01, label="crop_nms_thresh",
                                        info='''The box IoU cutoff used by non-maximal suppression to filter duplicate 
                                        masks between different crops.''')

    # Segment image
    with gr.Tab(label='Image'):
        with gr.Row(equal_height=True):
            with gr.Column():
                # input image
                original_image = gr.State(value=None)   # store original image without points, default None
                input_image = gr.Image(type="numpy")
                # point prompt
                with gr.Column():
                    selected_points = gr.State([])      # store points
                    with gr.Row():
                        gr.Markdown('You can click on the image to select points prompt. Default: foreground_point.')
                        undo_button = gr.Button('Undo point')
                    radio = gr.Radio(['foreground_point', 'background_point'], label='point labels')
                # run button
                button = gr.Button("Submit")
            # show the image with mask
            with gr.Tab(label='Image+Mask'):
                output_image = gr.Image(type='numpy')
            # show only mask
            with gr.Tab(label='Mask'):
                output_mask = gr.Image(type='numpy')
        def process_example(img, ori_img, sel_p):
            return ori_img, []

        example = gr.Examples(
            examples=image_examples,
            inputs=[input_image, original_image, selected_points],
            outputs=[original_image, selected_points],
	        fn=process_example,
	        run_on_click=True
        )

    # once user upload an image, the original image is stored in `original_image`
    def store_img(img):
        return img, []  # when new image is uploaded, `selected_points` should be empty
    input_image.upload(
        store_img,
        [input_image],
        [original_image, selected_points]
    )

    # user click the image to get points, and show the points on the image
    def get_point(img, sel_pix, point_type, evt: gr.SelectData):
        if point_type == 'foreground_point':
            sel_pix.append((evt.index, 1))   # append the foreground_point
        elif point_type == 'background_point':
            sel_pix.append((evt.index, 0))    # append the background_point
        else:
            sel_pix.append((evt.index, 1))    # default foreground_point
        # draw points
        for point, label in sel_pix:
            cv2.drawMarker(img, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
        if img[..., 0][0, 0] == img[..., 2][0, 0]:  # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img if isinstance(img, np.ndarray) else np.array(img)
    input_image.select(
        get_point,
        [input_image, selected_points, radio],
        [input_image],
    )

    # undo the selected point
    def undo_points(orig_img, sel_pix):
        if isinstance(orig_img, int):   # if orig_img is int, the image if select from examples
            temp = cv2.imread(image_examples[orig_img][0])
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        else:
            temp = orig_img.copy()
        # draw points
        if len(sel_pix) != 0:
            sel_pix.pop()
            for point, label in sel_pix:
                cv2.drawMarker(temp, point, colors[label], markerType=markers[label], markerSize=20, thickness=5)
        if temp[..., 0][0, 0] == temp[..., 2][0, 0]:  # BGR to RGB
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        return temp if isinstance(temp, np.ndarray) else np.array(temp)
    undo_button.click(
        undo_points,
        [original_image, selected_points],
        [input_image]
    )

    # button image
    button.click(run_inference, inputs=[device, model_type, points_per_side, pred_iou_thresh, stability_score_thresh,
                                    min_mask_region_area, stability_score_offset, box_nms_thresh, crop_n_layers,
                                    crop_nms_thresh, original_image, selected_points],
                 outputs=[output_image, output_mask])



demo.queue().launch(server_name='0.0.0.0', server_port=7860, debug=True)



