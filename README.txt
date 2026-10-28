NEXWORLD 4D updated files

1. Replace your project's NEXWORLD-4D/app.py with app.py from this package.
2. Replace NEXWORLD-4D/modules/detection.py with modules/detection.py.
3. Keep config.py, tracking.py, prediction.py, reconstruction.py, spatial_reasoning.py,
   depth_estimation.py, scene_analysis.py, utils/, and assets/ unchanged.
4. In the dashboard, set detection confidence to 0.10–0.20 and process the image again.
5. Run: python -m streamlit run app.py

Note: The update uses a larger YOLO inference image size and test-time augmentation. This can
improve some difficult detections but uses more CPU/time and cannot guarantee that a vehicle
in a stylized illustration will be recognized by the general-purpose pretrained model.
