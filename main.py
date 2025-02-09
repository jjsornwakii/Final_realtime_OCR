import os
import cv2
import torch
import time
import queue
import threading
from datetime import datetime
from autoTransform.transform import process_auto_transform
from splitImage.split import process_split_image
from readLicense.read import process_read_license
from ultralytics import YOLO

import warnings
warnings.filterwarnings("ignore")

# กำหนดค่าเริ่มต้น
YOLO_WIDTH = 640
YOLO_HEIGHT = 640
OCR_SIZE = 224
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
FPS = 30

stop_event = threading.Event()

def capture_frame(cap, frame_queue):
    frame_count = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("ไม่สามารถอ่านภาพจากกล้องได้")
            stop_event.set()
            break
        
        if frame_count % 5 == 0:
            if not frame_queue.full():
                frame_queue.put(frame)
        
        frame_count += 1

def process_frame(frame_queue, model, last_ocr_time, trigger_zone, model_path, font_path, ocr_results):
    object_in_zone = False
    current_plate = None  # เก็บป้ายทะเบียนปัจจุบันที่อยู่ใน zone
    processed_plate = False  # flag สำหรับตรวจสอบว่าได้ประมวลผลป้ายนี้แล้วหรือไม่
    
    while not stop_event.is_set():
        if not frame_queue.empty():
            original_frame = frame_queue.get()
            
            # สร้าง copy สำหรับ YOLO model
            yolo_frame = cv2.resize(original_frame.copy(), (YOLO_WIDTH, YOLO_HEIGHT))
            
            # วาด Trigger Zone
            height, width = original_frame.shape[:2]
            scale_x = width / YOLO_WIDTH
            scale_y = height / YOLO_HEIGHT
            
            actual_trigger_zone = (
                (int(trigger_zone[0][0] * scale_x), int(trigger_zone[0][1] * scale_y)),
                (int(trigger_zone[1][0] * scale_x), int(trigger_zone[1][1] * scale_y))
            )
            
            cv2.rectangle(original_frame, actual_trigger_zone[0], actual_trigger_zone[1], (255, 0, 0), 2)

            # ตรวจจับวัตถุด้วย YOLO
            results = model.predict(yolo_frame, verbose=False)[0]
            
            # flag สำหรับตรวจสอบว่าพบป้ายทะเบียนใน frame นี้หรือไม่
            found_plate_in_frame = False

            if len(results.boxes) > 0:
                for box in results.boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf)
                    cls = int(box.cls)

                    if conf < model.conf:
                        continue
                    
                    xmin, ymin, xmax, ymax = coords
                    
                    # แปลงพิกัดกลับไปยังขนาดจริง
                    real_xmin = int(xmin * scale_x)
                    real_ymin = int(ymin * scale_y)
                    real_xmax = int(xmax * scale_x)
                    real_ymax = int(ymax * scale_y)

                    if cls == 1:  # licenseplate เท่านั้น
                        real_cx = (real_xmin + real_xmax) // 2
                        real_cy = (real_ymin + real_ymax) // 2

                        # วาดกรอบและข้อความ
                        cv2.rectangle(original_frame, (real_xmin, real_ymin), (real_xmax, real_ymax), (0, 255, 0), 2)
                        label = f"licenseplate: {conf:.2%}"
                        cv2.putText(original_frame, label, (real_xmin, real_ymin - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5 * max(scale_x, scale_y), (0, 255, 0), 2)

                        # ตรวจสอบว่าอยู่ใน Trigger Zone
                        if (actual_trigger_zone[0][0] < real_cx < actual_trigger_zone[1][0] and 
                            actual_trigger_zone[0][1] < real_cy < actual_trigger_zone[1][1]):
                            found_plate_in_frame = True
                            
                            if not object_in_zone:
                                object_in_zone = True
                                trigger_start_time = time.time()
                                processed_plate = False
                            elif time.time() - trigger_start_time >= 1 and not processed_plate:
                                current_time = time.time()
                                if current_time - last_ocr_time >= 2:
                                    print("Starting OCR process.")

                                    # ตัดภาพป้ายทะเบียนและ resize
                                    plate_img = original_frame[real_ymin:real_ymax, real_xmin:real_xmax]
                                    plate_img = cv2.resize(plate_img, (OCR_SIZE, OCR_SIZE))

                                    try:
                                        transformed_img = process_auto_transform(plate_img)
                                        top_img, _ = process_split_image(transformed_img)
                                        results, confidence = process_read_license(top_img, model_path, font_path)

                                        text = ''.join([char for char, _, _ in results])
                                        print(f"\nDetected: {text}")
                                        print(f"Confidence: {confidence:.2%}")

                                        if len(text.strip()) > 0:  # ถ้าอ่านตัวอักษรได้
                                            current_plate = text
                                            processed_plate = True
                                            ocr_results.append((text, confidence))
                                        else:  # ถ้าอ่านตัวอักษรไม่ได้
                                            print("ไม่พบตัวอักษร - จะลองอ่านใหม่")
                                            processed_plate = False  # ให้สามารถ OCR ซ้ำได้
                                            current_plate = None

                                    except Exception as e:
                                        print(f"OCR Error: {str(e)}")
                                        processed_plate = False  # กรณี error ให้ลองใหม่ได้
                                        current_plate = None

            # ถ้าไม่พบป้ายทะเบียนใน frame และก่อนหน้านี้มีป้ายอยู่ใน zone
            if not found_plate_in_frame and object_in_zone:
                object_in_zone = False
                if current_plate:
                    print(f"\nป้ายทะเบียน {current_plate} ออกจาก Trigger Zone แล้ว")
                    current_plate = None
                    processed_plate = False
                    ocr_results.clear()  # Clear the best detected results

            # แสดงผล
            display_frame = cv2.resize(original_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            
            # แสดงสถานะการตรวจจับบนหน้าจอ
            if current_plate:
                status_text = f"Current plate: {current_plate}"
                cv2.putText(display_frame, status_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("CCTV Stream", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("กำลังหยุดการทำงาน...")
                stop_event.set()
                break

def main():
    try:
        model_path = "readLicense/EfficientNet_model_2.pth"
        yolo_model_path = "detectCar/yolov8/best_v8_2.pt"
        font_path = "AnantasonReno-SemiExpanded-Italic.otf"
        capture_dir = "captured_plates"
        os.makedirs(capture_dir, exist_ok=True)

        model = YOLO(yolo_model_path)
        model.conf = 0.5
        model.max_det = 2

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("ไม่สามารถเปิดกล้องได้")

        trigger_zone = ((200, 400), (500, 600))
        last_ocr_time = 0
        ocr_results = []

        stop_event.clear()
        frame_queue = queue.Queue(maxsize=10)

        capture_thread = threading.Thread(target=capture_frame, args=(cap, frame_queue), daemon=True)
        process_thread = threading.Thread(target=process_frame, args=(frame_queue, model, last_ocr_time, trigger_zone, model_path, font_path, ocr_results), daemon=True)

        capture_thread.start()
        process_thread.start()

        while not stop_event.is_set():
            time.sleep(0.1)

        print("รอให้ threads จบการทำงาน...")
        capture_thread.join(timeout=2)
        process_thread.join(timeout=2)

    except Exception as e:
        print(f"\nเกิดข้อผิดพลาด: {str(e)}")
        stop_event.set()
    finally:
        stop_event.set()
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()