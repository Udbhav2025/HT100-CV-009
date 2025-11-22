from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

class VehicleCounter:
    def __init__(self, model_path='yolov8n.pt', skip_frames=2):
        """Initialize YOLOv8 model and tracking variables"""
        self.model = YOLO(model_path)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.track_history = defaultdict(lambda: [])
        self.counted_ids = set()
        self.total_count = 0
        self.skip_frames = skip_frames  # Process every Nth frame
        
    def detect_and_count(self, source, line_position=0.5):
        """
        Detect and count vehicles crossing a line
        
        Args:
            source: Video file path, camera index (0 for webcam), or image
            line_position: Position of counting line (0.0 to 1.0, default 0.5 for center)
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Calculate counting line position
        line_y = int(frame_height * line_position)
        
        print(f"Processing video... Press 'q' to quit")
        print(f"Video dimensions: {frame_width}x{frame_height} @ {fps}fps")
        print(f"Performance mode: Processing every {self.skip_frames} frame(s)")
        
        frame_count = 0
        last_results = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process only every Nth frame for better performance
            if frame_count % self.skip_frames == 0:
                # Run YOLOv8 tracking with optimized settings
                results = self.model.track(
                    frame, 
                    persist=True, 
                    classes=self.vehicle_classes, 
                    verbose=False,
                    imgsz=640,  # Smaller image size for faster processing
                    conf=0.3,   # Lower confidence threshold
                    iou=0.5     # IOU threshold for NMS
                )
                last_results = results
            else:
                # Skip processing, use last results for display
                if last_results is None:
                    continue
                results = last_results
            
            # Draw counting line
            cv2.line(frame, (0, line_y), (frame_width, line_y), (0, 255, 255), 2)
            cv2.putText(frame, "COUNTING LINE", (10, line_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Process detections
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    center_y = (y1 + y2) // 2
                    center_x = (x1 + x2) // 2
                    
                    # Store track history
                    track = self.track_history[track_id]
                    track.append((center_x, center_y))
                    if len(track) > 30:
                        track.pop(0)
                    
                    # Check if vehicle crossed the line
                    if len(track) > 1 and track_id not in self.counted_ids:
                        prev_y = track[-2][1]
                        curr_y = track[-1][1]
                        
                        # Count if crossed from top to bottom or bottom to top
                        if (prev_y < line_y <= curr_y) or (prev_y > line_y >= curr_y):
                            self.counted_ids.add(track_id)
                            self.total_count += 1
                    
                    # Draw bounding box and label
                    class_names = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
                    label = f"{class_names.get(cls, 'Vehicle')} ID:{track_id}"
                    color = (0, 255, 0) if track_id in self.counted_ids else (255, 0, 0)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw tracking line
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], False, color, 2)
            
            # Display count
            cv2.putText(frame, f"Total Count: {self.total_count}", (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Resize frame if too large for screen
            screen_height = 720
            if frame_height > screen_height:
                scale = screen_height / frame_height
                new_width = int(frame_width * scale)
                frame = cv2.resize(frame, (new_width, screen_height))
            
            # Show frame
            cv2.imshow('Vehicle Counter', frame)
            
            # Control playback speed - use minimal delay for real-time feel
            delay = 1  # Minimal delay for faster playback
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nFinal count: {self.total_count} vehicles")
        return self.total_count

if __name__ == "__main__":
    # Initialize counter with performance settings
    # skip_frames: Process every Nth frame (higher = faster, but may miss some vehicles)
    # 1 = process every frame (slow but accurate)
    # 2 = process every 2nd frame (2x faster)
    # 3 = process every 3rd frame (3x faster)
    counter = VehicleCounter('yolov8n.pt', skip_frames=15 )
    
    # Count vehicles in video - replace with your video path
    counter.detect_and_count('videos/2103099-uhd_3840_2160_30fps.mp4', line_position=0.9)
    
    # For even faster processing, use skip_frames=3 or 4:
    # counter = VehicleCounter('yolov8n.pt', skip_frames=3)
    
    # Examples:
    # counter.detect_and_count('videos/highway_traffic.mp4', line_position=0.5)
    # counter.detect_and_count('C:/Users/YourName/Videos/traffic.mp4', line_position=0.5)
