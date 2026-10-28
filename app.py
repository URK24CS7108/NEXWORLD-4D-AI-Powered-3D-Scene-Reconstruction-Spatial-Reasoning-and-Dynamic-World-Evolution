import time

from pathlib import Path

import cv2

import numpy as np

import pandas as pd

import streamlit as st

from PIL import Image

from config import APP_NAME, MODEL_NAME, DEFAULT_CONFIDENCE, DEFAULT_HORIZON_SECONDS, MAX_HISTORY, OUTPUT_DIR

from modules.detection import Detector

from modules.tracking import CentroidTracker

from modules.prediction import predict_track

from modules.reconstruction import make_scene_figure

from modules.spatial_reasoning import relationships

from modules.depth_estimation import relative_depth_preview

from modules.scene_analysis import analyze_scene

from utils.data_export import detections_to_csv

st.set_page_config(page_title=APP_NAME, page_icon="🌐", layout="wide", initial_sidebar_state="expanded")

BASE=Path(__file__).resolve().parent

css=BASE/"assets"/"styles.css"

if css.exists():

    st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>",unsafe_allow_html=True)

st.markdown("""

<div class="nx-header">

  <div class="nx-title">NEXWORLD 4D</div>

  <div class="nx-subtitle">AI-Powered 3D Scene Reconstruction, Spatial Reasoning and Dynamic World Evolution</div>

  <div class="nx-workflow">SEE&nbsp; → &nbsp;UNDERSTAND&nbsp; → &nbsp;RECONSTRUCT&nbsp; → &nbsp;PREDICT</div>

</div>

""",unsafe_allow_html=True)

def init_state():

    defaults={

        "detector":None,"tracker":CentroidTracker(),"history":[],"last_frame":None,

        "last_annotated":None,"last_detections":[],"last_predictions":{},

        "frame_index":0,"paused":False,"source_mode":"Upload image / video",

        "camera_running":False,"camera_frame":None,"status":"Ready",

    }

    for k,v in defaults.items():

        if k not in st.session_state: st.session_state[k]=v

init_state()

with st.sidebar:

    st.markdown("## ◈ CONTROL DECK")

    source_mode=st.radio("Input source",["Upload image / video","Webcam (snapshot mode)"],index=0)

    conf=st.slider("Detection confidence",0.10,0.90,float(DEFAULT_CONFIDENCE),0.05)

    horizon=st.slider("Prediction horizon (seconds)",1,10,int(DEFAULT_HORIZON_SECONDS))

    show_boxes=st.toggle("Show detection boxes",True)

    show_trails=st.toggle("Show movement trails",True)

    show_3d=st.toggle("Enable 3D scene view",True)

    run_analysis=st.button("▶ PROCESS INPUT",use_container_width=True)

    reset=st.button("↺ CLEAR TRACKING HISTORY",use_container_width=True)

    st.caption("CPU mode is supported. First model use may download YOLO weights.")

    st.markdown("---")

    st.markdown("**MODULE STATUS**")

    for name, state in [

        ("Object Detection","Loaded on demand" if st.session_state.detector else "Standby"),

        ("Object Tracking","Ready"),

        ("Depth Estimation","Image-structure proxy"),

        ("3D Reconstruction","Illustrative image-plane"),

        ("Spatial Reasoning","Pixel-space only"),

        ("Future Prediction","Constant velocity"),

    ]:

        st.write(("🟢 " if state in ("Ready","Constant velocity","Pixel-space only") else "🟡 ")+name)

        st.caption(state)

if reset:

    st.session_state.tracker=CentroidTracker()

    st.session_state.history=[]

    st.session_state.last_detections=[]

    st.session_state.last_predictions={}

    st.session_state.status="Tracking history cleared"

uploaded=None

if source_mode=="Upload image / video":

    uploaded=st.file_uploader("Upload an image or video",type=["jpg","jpeg","png","bmp","mp4","avi","mov","mkv"])

else:

    st.info("Webcam snapshot mode uses OpenCV to capture one frame when you click Capture. Browser camera access may require OS permission.")

    if st.button("📷 Capture webcam frame"):

        cap=cv2.VideoCapture(0)

        ok, frame=cap.read() if cap.isOpened() else (False,None)

        cap.release()

        if ok:

            st.session_state.camera_frame=frame

            st.session_state.status="Webcam frame captured"

        else:

            st.error("Could not open webcam. Close other camera apps and check Windows camera permissions.")

    if st.session_state.camera_frame is not None:

        st.image(cv2.cvtColor(st.session_state.camera_frame,cv2.COLOR_BGR2RGB),caption="Captured webcam frame",use_container_width=True)

def load_detector():

    if st.session_state.detector is None:

        with st.spinner(f"Loading {MODEL_NAME}. First run may download model weights..."):

            st.session_state.detector=Detector(MODEL_NAME)

    return st.session_state.detector

def process_frame(frame, confidence, horizon):

    detector=load_detector()

    detections=detector.detect(frame,confidence)

    now=time.time()

    tracked=st.session_state.tracker.update(detections,now)

    annotated=frame.copy()

    predictions={}

    # Refresh trajectories from track state.

    for d in tracked:

        tid=d["track_id"]

        tr=st.session_state.tracker.tracks.get(tid)

        if tr:

            p=predict_track(tr,horizon)

            if p.get("available"): predictions[tid]=p

    if show_boxes:

        for d in tracked:

            x1,y1,x2,y2=map(int,d["xyxy"])

            color=(255,210,40)

            cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)

            label=f'{d["class_name"]} #{d["track_id"]} {d["confidence"]:.2f}'

            cv2.rectangle(annotated,(x1,max(0,y1-24)),(x1+min(280,len(label)*9+10),y1),color,-1)

            cv2.putText(annotated,label,(x1+4,max(16,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.48,(4,14,25),1,cv2.LINE_AA)

    if show_trails:

        for tid,tr in st.session_state.tracker.tracks.items():

            hist=list(tr["history"])

            if len(hist)>1:

                pts=np.array([[int(p[1]),int(p[2])] for p in hist],dtype=np.int32).reshape((-1,1,2))

                cv2.polylines(annotated,[pts],False,(80,245,150),2)

    for tid,p in predictions.items():

        pts=p["points"]

        xy=[(int(p["current"]["x"]),int(p["current"]["y"]))]+[(int(q["x"]),int(q["y"])) for q in pts]

        for a,b in zip(xy,xy[1:]):

            cv2.line(annotated,a,b,(80,245,150),2,cv2.LINE_AA)

        if xy: cv2.circle(annotated,xy[-1],6,(80,245,150),-1)

    st.session_state.last_frame=frame.copy()

    st.session_state.last_annotated=annotated

    st.session_state.last_detections=tracked

    st.session_state.last_predictions=predictions

    st.session_state.history.append({

        "time":now,"frame":frame.copy(),"annotated":annotated.copy(),

        "detections":tracked.copy(),"predictions":predictions.copy()

    })

    st.session_state.history=st.session_state.history[-MAX_HISTORY:]

    st.session_state.frame_index+=1

    st.session_state.status="Processing complete"

    return tracked, predictions, annotated

if run_analysis:

    try:

        if source_mode=="Webcam (snapshot mode)":

            frame=st.session_state.camera_frame

            if frame is None: st.warning("Capture a webcam frame first.")

            else: process_frame(frame,conf,horizon)

        elif uploaded is None:

            st.warning("Upload an image or video first.")

        else:

            suffix=Path(uploaded.name).suffix.lower()

            raw=uploaded.getvalue()

            if suffix in {".jpg",".jpeg",".png",".bmp"}:

                arr=np.frombuffer(raw,np.uint8); frame=cv2.imdecode(arr,cv2.IMREAD_COLOR)

                if frame is None: st.error("Image could not be decoded.")

                else: process_frame(frame,conf,horizon)

            else:

                temp=BASE/"outputs"; temp.mkdir(exist_ok=True)

                temp_path=temp/("uploaded_video"+suffix)

                temp_path.write_bytes(raw)

                cap=cv2.VideoCapture(str(temp_path))

                if not cap.isOpened():

                    st.error("Video could not be opened. Try MP4 (H.264) or AVI.")

                else:

                    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

                    fps=cap.get(cv2.CAP_PROP_FPS)

                    fps=fps if fps and fps>0 else 15

                    # Process a bounded sample for responsiveness; increase stride if video is long.

                    stride=max(1,int(total/90)) if total else 3

                    bar=st.progress(0)

                    status=st.empty()

                    i=0; done=0

                    while True:

                        ok,frame=cap.read()

                        if not ok: break

                        if i % stride == 0:

                            process_frame(frame,conf,horizon); done+=1

                            if total: bar.progress(min(1.0,i/max(1,total)))

                            status.caption(f"Processed frame {i+1}"+(f" / {total}" if total else ""))

                        i+=1

                    cap.release()

                    bar.empty(); status.empty()

                    st.success(f"Video processing finished: {done} sampled frames processed at approximately {fps:.1f} source FPS.")

    except Exception as e:

        st.session_state.status="Error"

        st.error(f"Processing failed: {e}")

        st.info("Check that dependencies installed correctly and that the model can be downloaded on first run.")

current=st.session_state.last_annotated

tracks=st.session_state.tracker.tracks

preds=st.session_state.last_predictions

detections=st.session_state.last_detections

# Header metrics

counts={}

for d in detections: counts[d["class_name"]]=counts.get(d["class_name"],0)+1

people=sum(1 for d in detections if d["class_name"]=="person")

vehicle_classes = {
    "car", "truck", "bus", "motorcycle", "bicycle",
    "motorbike", "van", "scooter", "pickup", "pickup truck",
}
vehicles = sum(
    1 for d in detections
    if str(d.get("class_name", "")).lower().strip() in vehicle_classes
)

moving=sum(1 for tr in tracks.values() if len(tr["history"])>=2 and ((tr["history"][-1][1]-tr["history"][-2][1])**2+(tr["history"][-1][2]-tr["history"][-2][2])**2)**0.5>3)

m1,m2,m3,m4,m5=st.columns(5)

m1.metric("OBJECTS",len(detections))

m2.metric("PEOPLE",people)

m3.metric("VEHICLES",vehicles)

m4.metric("MOVING TRACKS",moving)

m5.metric("HISTORY FRAMES",len(st.session_state.history))

left,right=st.columns([1.45,1],gap="medium")

with left:

    st.markdown('<div class="panel"><div class="panel-title">A · LIVE CAMERA / INPUT VIEW</div></div>',unsafe_allow_html=True)

    if current is not None:

        st.image(cv2.cvtColor(current,cv2.COLOR_BGR2RGB),use_container_width=True)

        st.caption(f"Status: {st.session_state.status} · Tracked IDs: {len(tracks)} · Detections: {len(detections)}")

        st.download_button("⬇ Download annotated frame",data=cv2.imencode(".png",current)[1].tobytes(),file_name="nexworld_scene.png",mime="image/png")

    else:

        st.markdown('<div class="panel">Awaiting image, video, or webcam capture. Process an input to activate live analytics.</div>',unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">C · 3D WORLD REPRESENTATION</div></div>',unsafe_allow_html=True)

    if show_3d and st.session_state.last_frame is not None:

        fig=make_scene_figure(tracks,st.session_state.last_frame.shape,preds,show_trails)

        st.plotly_chart(fig,use_container_width=True)

        st.caption("This is an interactive image-plane scene layout with illustrative Z offsets, not metric 3D reconstruction.")

    else:

        st.info("Enable 3D view and process an input. Scene coordinates are image-space unless calibrated depth is available.")

with right:

    st.markdown('<div class="panel"><div class="panel-title">B · DETECTED OBJECTS</div></div>',unsafe_allow_html=True)

    if counts:

        cols=st.columns(3)

        for i,(name,count) in enumerate(sorted(counts.items(),key=lambda x:-x[1])):

            with cols[i%3]:

                st.metric(name.upper(),count)

    else: st.caption("No detections yet. Results appear after processing.")

    st.markdown('<div class="panel"><div class="panel-title">E · SCENE ANALYSIS</div></div>',unsafe_allow_html=True)

    if st.session_state.last_frame is not None:

        analysis=analyze_scene(st.session_state.last_frame,tracks)

        st.write("**Image quality:**",analysis["quality"])

        st.write("**Brightness:**",analysis["brightness"],"· **Sharpness score:**",analysis["sharpness_score"])

        st.write("**People:**",analysis["people"],"· **Vehicles:**",analysis["vehicles"],"· **Moving tracks:**",analysis["moving"])

        st.caption(analysis["note"])

    else: st.caption("Scene analysis will appear after processing.")

    st.markdown('<div class="panel"><div class="panel-title">H · AI INSIGHTS</div></div>',unsafe_allow_html=True)

    if detections:

        st.write(f"Observed **{len(detections)} current detections** across {len(counts)} object classes.")

        if moving: st.write(f"Motion heuristic: **{moving} tracked object(s)** changed image position between recent observations.")

        else: st.write("No clear recent image-space movement was detected in the available track history.")

        if preds:

            tid,p=next(iter(preds.items()))

            tr=tracks.get(tid,{})

            st.write(f"Track **#{tid} ({tr.get('class_name','object')})** has a projected image-space displacement of about **{p['speed_px_s']*horizon:.0f} pixels** over {horizon}s if its recent velocity remains constant.")

            st.caption(p["note"])

        st.warning("These are observations and simple motion extrapolations, not intent recognition or a safety assessment.")

    else: st.caption("Insights are generated from actual processed detections and tracking history.")

st.markdown('<div class="panel"><div class="panel-title">D · WORLD UNDERSTANDING / SPATIAL RELATIONSHIPS</div></div>',unsafe_allow_html=True)

if st.session_state.last_frame is not None and tracks:

    h,w=st.session_state.last_frame.shape[:2]

    rel=relationships(tracks,w,h)

    if rel: st.dataframe(pd.DataFrame(rel),use_container_width=True,hide_index=True)

    else: st.caption("Need at least two tracked objects to calculate image-plane proximity.")

else: st.caption("Spatial relationships appear when two or more objects are tracked.")

st.markdown('<div class="panel"><div class="panel-title">F · WORLD EVOLUTION TIMELINE</div></div>',unsafe_allow_html=True)

history = st.session_state.history

if len(history) > 1:

    # A Streamlit slider requires max_value > min_value, so show it only

    # when there are at least two history frames.

    ix = st.slider(

        "Inspect recorded frame / prediction snapshot",

        min_value=0,

        max_value=len(history) - 1,

        value=len(history) - 1,

        key="history_frame_slider",

    )

    snap = history[ix]

elif len(history) == 1:

    # A single image creates one history entry; display it without a slider.

    ix = 0

    snap = history[0]

    st.caption("1 observation recorded. Process a video or more snapshots to enable frame selection.")

else:

    snap = None

if snap is not None:

    c1, c2 = st.columns(2)

    with c1:

        st.image(

            cv2.cvtColor(snap["frame"], cv2.COLOR_BGR2RGB),

            caption=f"PAST / NOW · stored observation #{ix + 1}",

            use_container_width=True,

        )

    with c2:

        st.image(

            cv2.cvtColor(snap["annotated"], cv2.COLOR_BGR2RGB),

            caption="Detection overlay at that observation",

            use_container_width=True,

        )

    st.caption(

        "Past and current panels are stored observed frames. Future positions below "

        "are model extrapolations, not future camera images."

    )

else:

    st.info("No history frames available yet. Upload and process an image or video to begin.")

st.markdown('<div class="panel"><div class="panel-title">G · FUTURE MOTION PREDICTION</div></div>',unsafe_allow_html=True)

if preds:

    rows=[]

    for tid,p in preds.items():

        tr=tracks.get(tid,{})

        if p.get("available"):

            rows.append({

                "Track":f'#{tid}',"Class":tr.get("class_name","unknown"),

                "Speed (px/s)":round(p["speed_px_s"],1),

                f"Predicted X at +{horizon}s":round(p["future"]["x"],1),

                f"Predicted Y at +{horizon}s":round(p["future"]["y"],1),

                "Stability indicator":f'{p["stability_indicator"]:.0%} (heuristic)'

            })

    if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

    else: st.caption("Prediction needs multiple observations for each track.")

else: st.caption("No predictions available yet. Video input or repeated webcam snapshots provide the trajectory history.")

st.markdown('<div class="panel"><div class="panel-title">I · RGB / DEPTH-PROXY / SCENE MAP</div></div>',unsafe_allow_html=True)

if st.session_state.last_frame is not None:

    c1,c2,c3=st.columns(3)

    with c1: st.image(cv2.cvtColor(st.session_state.last_frame,cv2.COLOR_BGR2RGB),caption="RGB · actual input",use_container_width=True)

    with c2: st.image(cv2.cvtColor(relative_depth_preview(st.session_state.last_frame),cv2.COLOR_BGR2RGB),caption="Image-structure proxy · NOT metric depth",use_container_width=True)

    with c3:

        if show_3d:

            mini=make_scene_figure(tracks,st.session_state.last_frame.shape,preds,show_trails)

            st.plotly_chart(mini,use_container_width=True,config={"displayModeBar":False})

        else: st.caption("3D visualization disabled.")

else:

    st.caption("RGB, proxy visualization, and scene representation become available after processing an input.")

st.markdown("---")

colx,coly=st.columns([1,2])

with colx:

    if st.button("📸 Save scene snapshot"):

        if current is not None:

            OUTPUT_DIR.mkdir(exist_ok=True)

            path=OUTPUT_DIR/"nexworld_snapshot.png"

            cv2.imwrite(str(path),current)

            st.success(f"Saved to {path}")

        else: st.warning("Process an input first.")

with coly:

    csv=detections_to_csv(detections)

    st.download_button("⬇ Export current detections as CSV",data=csv,file_name="nexworld_detections.csv",mime="text/csv")

st.caption("NEXWORLD 4D · Local processing dashboard. Object classes depend on the pretrained model; metric distance and true 3D geometry require calibration/depth data.")
