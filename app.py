"""
CT-FM Demonstration App
Original Reference: https://huggingface.co/BAAI/SegVol

This Streamlit app demonstrates the CT-FM (Computed Tomography Feature Matching) algorithm.
It allows users to select or upload CT scans, specify a reference point on one of the scans,
and find the corresponding matching point on the other scan using the CT-FM algorithm.


The app is divided into several functions:
- init_states: Initializes the session state variables used in the app.
- reset_scan: Resets the data, reset flag, and selected point for a given scan index.
- clear_file: Clears the selected file, data, and resets the scan for a given scan index.
- render_scan_selection: Renders the scan selection UI (radio button and file uploader/selectbox).
- render_image_controls: Renders the image controls (slider for axial view selection).
- render_axial_view: Renders the axial view of the selected scan and allows point selection.
- render_action_buttons: Renders the action buttons (Clear and Run).
- main: The main function that puts together the UI components and handles the app flow.

The app uses the following external functions and modules:
- make_fig: Creates a plot figure of the axial view image.
- load_scan: Loads the selected CT scan file.
- run: Runs the CT-FM algorithm on the selected scans and points.
"""

import os
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
st.set_page_config(layout="wide")
from PIL import Image, ImageDraw
from utils import make_fig
from data import load_scan
from inference import run
from functools import partial

def init_states(count=1):
    """
    Initializes the session state variables used in the app.
    
    Args:
        count (int): Number of scans to initialize states for (default: 1).
    """
    states = ["data", "reset", "point", "selection"]
    for i in range(count):
        for state in states:
            if f"{state}_{i}" not in st.session_state:
                st.session_state[f"{state}_{i}"] = None
 
    if "running" not in st.session_state:
        st.session_state.running = False

    if "finished" not in st.session_state:
        st.session_state.finished = False

# Get all .nii.gz files in the assets/scans directory
scans_dir = "assets/scans"
case_list = [os.path.join(scans_dir, f) for f in os.listdir(scans_dir) if f.endswith(".nii.gz")]

def reset_scan(idx):
    """
    Resets the data, reset flag, and selected point for a given scan index.
    
    Args:
        idx (int): Index of the scan to reset.
    """
    setattr(st.session_state, f"data_{idx}", None)
    setattr(st.session_state, f"reset_{idx}", True)
    setattr(st.session_state, f"point_{idx}", None)

def clear_file(idx):
    """
    Clears the selected file, data, and resets the scan for a given scan index.
    
    Args:
        idx (int): Index of the scan to clear.
    """
    setattr(st.session_state, f"selection_{idx}", None)
    reset_scan(idx)

def render_scan_selection(idx):
    """
    Renders the scan selection UI (radio button and file uploader/selectbox).
    
    Args:
        idx (int): Index of the scan selection UI to render.
    """
    scan_type = st.radio(
        "Select Source",
        ["Preloaded", "Upload"],
        on_change=partial(clear_file, idx),
        key=f"scan_type_{idx}",
        disabled=st.session_state.finished
    )

    if scan_type == "Preloaded":
        uploaded_file = st.selectbox(
            "Select a preloaded case",
            case_list,
            index=None,
            placeholder="Click here to see options",
            on_change=partial(reset_scan, idx),
            key=f"selectionbox_{idx}",
            disabled=st.session_state.finished
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload your own scan (nii.gz)",
            type='nii.gz',
            on_change=partial(reset_scan, idx),
            key=f"uploader_{idx}"
        )

    setattr(st.session_state, f"selection_{idx}", uploaded_file)


def render_image_controls(image_3D, image_idx, selected_index=None):
    """
    Renders the image controls (slider for axial view selection).
    
    Args:
        image_3D (numpy.ndarray): 3D image data.
        image_idx (int): Index of the image.
        
    Returns:
        int: Selected axial view index.
    """
    selected_index = st.slider(
        'Axial view',
        0,
        image_3D.shape[0] - 1,
        image_3D.shape[0] // 2 if selected_index is None else selected_index,
        key=f'slider_{image_idx}',
        disabled=st.session_state.running
    )
    return selected_index


def render_axial_view(selected_index, image_3D, image_idx, patch_size=None):
    """
    Renders the axial view of the selected scan and allows point selection.
    
    Args:
        selected_index (int): Index of the selected axial view.
        image_3D (numpy.ndarray): 3D image data.
        image_idx (int): Index of the image.
    """
    image_array = image_3D[selected_index]
    image_z = make_fig(image_array, None, st.session_state[f"point_{image_idx}"], selected_index, f'view_{image_idx}', patch_size=patch_size)
    adapted_width = 600

    if not(st.session_state.finished):
        size = image_z.size
        adapted_height = int(adapted_width * size[1] / size[0])
        value = streamlit_image_coordinates(image_z, width=adapted_width)
        if value is not None:
            point_coord = (selected_index, value['y']/adapted_height * size[1], value['x']/adapted_width * size[0])
            st.session_state[f"point_{image_idx}"] = point_coord
            st.rerun()
    else:
        # rectangle = st.session_state[f"data_{image_idx}"]["bbox"]
        # if selected_index >= rectangle[0][0] and selected_index <= rectangle[1][0]:
        #     draw = ImageDraw.Draw(image_z)
        #     rectangle_coords = [(rectangle[0][2], rectangle[0][1]), (rectangle[1][2], rectangle[1][1])]
        #     draw.rectangle(rectangle_coords, outline='#2909F1', width=3)
        st.image(image_z, width=adapted_width)
        
def render_action_buttons(count):
    """
    Renders the action buttons (Clear and Run).
    
    Args:
        count (int): Number of scans.
    """
    col1, col2 = st.columns(2)
    with col1:
        # Trigger clear button if any selection with a point is made
        if st.button(
            "Clear",
            use_container_width=True,
            disabled=not(any(
                getattr(st.session_state, f"selection_{idx}") is not None and 
                getattr(st.session_state, f"point_{idx}") is not None 
                for idx in range(count)
            )) or st.session_state.finished
        ):
            for idx in range(count):
                setattr(st.session_state, f"point_{idx}", None)
            st.rerun()
    with col2:
        # Trigger run button if all selections are made and only one of them has a point
        run_button_name = 'Run' if not st.session_state.running else 'Running'
        if st.button(
            run_button_name,
            type="primary",
            use_container_width=True,
            disabled=any(
                getattr(st.session_state, f"data_{idx}") is None 
                for idx in range(count)
            ) or sum(
                getattr(st.session_state, f"point_{idx}") is not None 
                for idx in range(count)
            ) != 1 or st.session_state.running or st.session_state.finished
        ):
            st.session_state.running = True
            st.rerun()

def main():
    """
    The main function that puts together the UI components and handles the app flow.
    """
    img_count = 2
    patch_size = (16, 64, 64)
    col_gap = "large"
    st.title("CT-FM Demonstration App")
    init_states(count=img_count)
    scan_cols = st.columns(img_count, gap=col_gap)

    for idx, scan_col in enumerate(scan_cols):
        with scan_col:
            render_scan_selection(idx)

    for idx in range(img_count):
        if (
            getattr(st.session_state, f"selection_{idx}") is not None and 
            getattr(st.session_state, f"reset_{idx}")
        ) or (
            getattr(st.session_state, f"data_{idx}") is None and 
            getattr(st.session_state, f"selection_{idx}") is not None
        ):
            setattr(st.session_state, f"data_{idx}", load_scan(getattr(st.session_state, f"selection_{idx}")))
            setattr(st.session_state, f"reset_{idx}", False)
            setattr(st.session_state, f"selector_{idx}", None)

    st.divider()
    if st.session_state.finished:
        st.write("Results")
        st.write("You can scroll through the reference and matched patches below")
        st.divider()

    view_cols = st.columns(img_count, gap=col_gap)

    for idx, view_col in enumerate(view_cols):
        with view_col:
            if (
                getattr(st.session_state, f"selection_{idx}") is None or 
                getattr(st.session_state, f"data_{idx}") is None
            ):
                st.write(f"Please select a scan to load for image {idx+1}")
            else:
                image_3D = getattr(st.session_state, f"data_{idx}")['image'][0].numpy()
                selected_index = render_image_controls(image_3D, image_idx=idx, selected_index=st.session_state[f'point_{idx}'][0] if 
                                                                                    st.session_state.finished else None)
                render_axial_view(image_3D=image_3D, selected_index=selected_index, image_idx=idx, patch_size=patch_size)

                if not st.session_state.finished:
                    st.write(f"Selected point: {st.session_state[f'point_{idx}']}")

    st.divider()
    render_action_buttons(img_count)

    if st.session_state.running:
        st.session_state.running = False
        with st.status("Running...", expanded=False):
            run(img_count, patch_size=patch_size)
        st.rerun()

if __name__ == "__main__":
    main()