import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import math


class YOLOPlateNumberExtractor:
    def __init__(self, model_path, conf=0.25, device="cpu"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def rotate_image(self, image, angle_degrees):
        if abs(angle_degrees) < 1:
            return image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
        abs_cos = abs(rot_mat[0, 0])
        abs_sin = abs(rot_mat[0, 1])
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)
        rot_mat[0, 2] += new_w / 2 - center[0]
        rot_mat[1, 2] += new_h / 2 - center[1]
        return cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    def estimate_rotation_from_coords(self, coords):
        if len(coords) < 2:
            return 0.0
        xs = [((x1 + x2) / 2) for (x1, y1, x2, y2) in coords]
        ys = [((y1 + y2) / 2) for (x1, y1, x2, y2) in coords]
        z = np.polyfit(xs, ys, 1)
        angle_rad = math.atan(z[0])
        return np.degrees(angle_rad)

    def get_boxes(self, result):
        return result.obb if hasattr(result, 'obb') and result.obb is not None else result.boxes

    def separate_detections(self, detections):
        digits, letters, others = [], [], []
        for cid, name, coords in detections:
            if name.isdigit():
                digits.append((cid, name, coords))
            elif name.isalpha() and len(name) == 1:
                letters.append((cid, name, coords))
            else:
                others.append(name)
        return digits, letters, others

    # --- UAE Hybrid Area Code Extraction ---
    def extract_area_and_number_uae(self, row1, row2, row1_boxes=None):

        # make this function clss part not method?????
        def is_digit_area_code(lst):
            return len(lst) in [1, 2] and all(x.isdigit() for x in lst)

        # 1. Check for single alpha area code (row1)
        for i, val in enumerate(row1):
            if val.isalpha() and len(val) == 1:
                area_code = val
                number_part = ''.join(row1[:i] + row1[i + 1:] + row2)
                return area_code, number_part
        # 1b. Check for single alpha area code (row2)
        for i, val in enumerate(row2):
            if val.isalpha() and len(val) == 1:
                area_code = val
                number_part = ''.join(row1 + row2[:i] + row2[i + 1:])
                return area_code, number_part

        # 2. Single/double digit area code in row2
        if is_digit_area_code(row2):
            area_code = ''.join(row2)
            number_part = ''.join(row1)
            return area_code, number_part
        # 3. Single/double digit area code in row1
        if is_digit_area_code(row1):
            area_code = ''.join(row1)
            number_part = ''.join(row2)
            return area_code, number_part

        # 4. Gap threshold in row1 (use box centers)
        if row1_boxes is not None and len(row1_boxes) == len(row1) and len(row1) > 2:
            x_centers = [(x1 + x2) / 2 for (x1, y1, x2, y2) in row1_boxes]
            sorted_row1_centers = sorted(zip(row1, x_centers), key=lambda z: z[1])
            sorted_row1 = [rc[0] for rc in sorted_row1_centers]
            sorted_centers = [rc[1] for rc in sorted_row1_centers]
            gaps = [sorted_centers[i + 1] - sorted_centers[i] for i in range(len(sorted_centers) - 1)]
            if gaps:
                max_gap = max(gaps)
                avg_gap = np.mean(gaps)
                if max_gap > 1.5 * avg_gap:
                    split = gaps.index(max_gap) + 1
                    area_candidate = sorted_row1[:split]
                    rest_candidate = sorted_row1[split:]
                    if is_digit_area_code(area_candidate):
                        area_code = ''.join(area_candidate)
                        number_part = ''.join(rest_candidate + row2)
                        return area_code, number_part
        # 5. Default
        return "", ''.join(row1 + row2)

    # ---------------------------------------

    @staticmethod
    def process_plate_image_from_array(image_array, plate_model, ocr_extractor):
        """
        Given a numpy array, runs plate detection, crops plate, runs OCR, and returns output info.
        """
        orig_img = image_array.copy()
        # Plate detection and cropping (rest is same as before)
        results = plate_model.predict(orig_img, conf=0.2, device="cpu", task='obb')[0]
        boxes = results.obb
        if not hasattr(boxes, 'cls') or boxes.cls is None or boxes.cls.numel() == 0:
            print("❌ No plate detected in the image!")
            return None
        # ... rest of your logic, identical to process_plate_image, but using orig_img directly
        classes = boxes.cls.cpu().numpy().astype(int)
        xywhr = boxes.xywhr.cpu().numpy()
        plate_class_idx = None
        for k, v in plate_model.names.items():
            if v == "plate":
                plate_class_idx = k
                break
        crop = None
        for i in range(len(classes)):
            if int(classes[i]) == plate_class_idx:
                x, y, w, h, a = xywhr[i]
                angle_deg = math.degrees(float(a)) if abs(a) < 3.2 else float(a)
                rect = ((x, y), (w, h), angle_deg)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                x_min, y_min = np.min(box, axis=0)
                x_max, y_max = np.max(box, axis=0)
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(orig_img.shape[1], x_max), min(orig_img.shape[0], y_max)
                crop = orig_img[int(y_min):int(y_max), int(x_min):int(x_max)]
                if crop.size > 0 and crop.shape[0] > 5 and crop.shape[1] > 5:
                    break
        if crop is None or crop.size == 0:
            print("❌ Plate detected but crop failed!")
            return None

        # OCR on the cropped plate image
        ocr_result = ocr_extractor.extract_number_from_array(crop)
        img, bboxes, obb_xywhr = ocr_result[5:8]
        final_number, area, number, others, angle = ocr_result[:5]

        # You can return everything you want here
        return img, final_number, area, number, others, angle, bboxes, obb_xywhr

    def visualize(self, image, final_number, area_code, number_part, others, angle, boxes, obb_xywhr=None):
        img = image.copy()

        if obb_xywhr is not None:
            for i in range(len(obb_xywhr)):
                x, y, w, h, a = obb_xywhr[i]
                angle_deg = math.degrees(float(a))  # If angle is in radians
                rect = ((float(x), float(y)), (float(w), float(h)), angle_deg)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                cx, cy = int(x), int(y)
                cv2.putText(img, f"{angle_deg:.1f}°", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            for (x1, y1, x2, y2) in boxes:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        title = f"Angle: {angle:.1f}°"
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()

        # Show the original image (without bounding boxes)
        orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 6))
        plt.imshow(orig_rgb)
        plt.title(f"Original Image - Angle: {angle:.1f}°")
        plt.axis('off')
        plt.show()

        if area_code or number_part or others:
            summary = f"Number: {number_part}"
            plate = f"{number_part}"
            if area_code:
                summary = f"Area: {area_code} | " + summary
                plate = f"{area_code}  " + plate
            if others:
                filtered_others = [x for x in others if not (x.startswith("AreaCode:") and x.endswith(area_code))]
                if filtered_others:
                    summary += f" | Other: {', '.join(filtered_others)}"
                    plate += f" {', '.join(filtered_others)}"
            # print(summary)
            print("***************************************************")
            print(plate)
            print("***************************************************")
            print(summary)

    def process_all_cropped_arrays(self, plate_crops, visualize=True):
        """
        Process a list of (fname, crop_idx, plate_img) tuples.
        """
        for fname, crop_idx, plate_img in plate_crops:
            print(f"\nProcessing {fname} - Crop #{crop_idx}")
            final_number, area, number, others, angle, upright_img, bboxes, obb_xywhr = self.extract_number_from_array(
                plate_img)
            if visualize:
                self.visualize(upright_img, final_number, area, number, others, angle, bboxes, obb_xywhr)

    def extract_number_from_array(self, image_array):
        # Same as extract_number_from_image, but takes a numpy array
        orig_img = image_array.copy()
        r1 = self.model.predict(orig_img, conf=self.conf, device=self.device, task='obb')[0]
        boxes1 = self.get_boxes(r1)

        if boxes1 is None or boxes1.cls is None or boxes1.cls.numel() == 0:
            print("❌ No initial detections.")
            return "", "", "", [], 0.0, orig_img, [], None

        coords1 = boxes1.xyxy.cpu().numpy()
        angle = self.estimate_rotation_from_coords(coords1)
        print(f"📌 Rotation Angle: {angle:.1f}°")

        if abs(angle) >= 40:
            upright_img = self.rotate_image(orig_img, angle)
            r2 = self.model.predict(upright_img, conf=self.conf, device=self.device, task='obb')[0]
        else:
            upright_img = orig_img.copy()
            r2 = r1

        boxes = self.get_boxes(r2)
        if boxes is None or boxes.cls is None or boxes.cls.numel() == 0:
            print("❌ No detections after deskewing.")
            return "", "", "", [], angle, upright_img, [], None

        class_ids = boxes.cls.cpu().numpy().astype(int)
        coords = boxes.xyxy.cpu().numpy()
        names = [self.model.names[c] for c in class_ids]
        detections = list(zip(class_ids, names, coords))

        digits, letters, others = self.separate_detections(detections)
        all_valid = digits + letters
        if not all_valid:
            return "", "", "", others, angle, upright_img, [], None

        # reading from left to right using x coord
        all_valid = sorted(all_valid, key=lambda x: (x[2][0] + x[2][2]) / 2)

        # ---- Assign to rows (row split logic) ----
        row1, row2 = [], []
        last_row1_box_top = None  # Keeps track of the "top" (y1) of the last char added to row1
        row1_boxes = []
        for i, (_, name, (x1, y1, x2, y2)) in enumerate(all_valid):
            curr_top = y1
            curr_height = y2 - y1
            threshold = curr_height / 2
            if last_row1_box_top is None:
                row1.append(name)
                row1_boxes.append((x1, y1, x2, y2))
                last_row1_box_top = curr_top
            else:
                diff = abs(curr_top - last_row1_box_top)
                if diff > threshold:
                    row2.append(name)
                else:
                    row1.append(name)
                    row1_boxes.append((x1, y1, x2, y2))
                    last_row1_box_top = curr_top

        print(f"🔷 Row 1: {row1}")
        print(f"🟖️ Row 2: {row2}")

        area_code, number_part = self.extract_area_and_number_uae(row1, row2, row1_boxes)
        if area_code:
            others.append(f"AreaCode:{area_code}")
        print(f"🧪 Others: {others}")

        obb_xywhr = boxes.xywhr.cpu().numpy() if hasattr(boxes, 'xywhr') else None

        bboxes = [c[2] for c in all_valid]

        return (
            f"{area_code} {number_part}".strip(),
            area_code, number_part, others, angle, upright_img,
            bboxes, obb_xywhr
        )

    # utility method later on we can separate from this clss
    @staticmethod
    def crop_all_plates_from_folder(car_images_dir, plate_model):
        all_plate_crops = []  # Each entry: (img_name, crop_idx, crop_image)
        for fname in os.listdir(car_images_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(car_images_dir, fname)
                orig_img = cv2.imread(img_path)
                results = plate_model.predict(orig_img, conf=0.2, device="cpu", task='obb')[0]
                boxes = results.obb
                if not hasattr(boxes, 'cls') or boxes.cls is None:
                    continue
                classes = boxes.cls.cpu().numpy().astype(int)
                xywhr = boxes.xywhr.cpu().numpy()
                plate_class_idx = None
                for k, v in plate_model.names.items():
                    if v == "plate":
                        plate_class_idx = k
                        break
                if plate_class_idx is None:
                    print("No 'plate' class found in model!")
                    continue
                crop_count = 0
                for i in range(len(classes)):
                    if int(classes[i]) == plate_class_idx:
                        x, y, w, h, a = xywhr[i]
                        angle_deg = math.degrees(float(a)) if abs(a) < 3.2 else float(a)
                        rect = ((x, y), (w, h), angle_deg)
                        box = cv2.boxPoints(rect)
                        box = np.intp(box)
                        x_min, y_min = np.min(box, axis=0)
                        x_max, y_max = np.max(box, axis=0)
                        x_min, y_min = max(0, x_min), max(0, y_min)
                        x_max, y_max = min(orig_img.shape[1], x_max), min(orig_img.shape[0], y_max)
                        crop = orig_img[int(y_min):int(y_max), int(x_min):int(x_max)]
                        if crop.size > 0 and crop.shape[0] > 5 and crop.shape[1] > 5:
                            crop_count += 1
                            # Instead of saving, append to list
                            all_plate_crops.append((fname, crop_count, crop))
        return all_plate_crops

    @staticmethod
    def process_plate_image(img_path, plate_model, ocr_extractor):
        """
        Given an image path, runs plate detection, crops plate, runs OCR, and returns output info.
        """
        orig_img = cv2.imread(img_path)
        if orig_img is None:
            print("❌ Could not read image! Check the path.")
            return None

        # Plate detection and cropping
        results = plate_model.predict(orig_img, conf=0.2, device="cpu", task='obb')[0]
        boxes = results.obb
        if not hasattr(boxes, 'cls') or boxes.cls is None or boxes.cls.numel() == 0:
            print("❌ No plate detected in the image!")
            return None

        # Find first plate bbox and crop
        classes = boxes.cls.cpu().numpy().astype(int)
        xywhr = boxes.xywhr.cpu().numpy()
        plate_class_idx = None
        for k, v in plate_model.names.items():
            if v == "plate":
                plate_class_idx = k
                break
        crop = None
        for i in range(len(classes)):
            if int(classes[i]) == plate_class_idx:
                x, y, w, h, a = xywhr[i]
                angle_deg = math.degrees(float(a)) if abs(a) < 3.2 else float(a)
                rect = ((x, y), (w, h), angle_deg)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                x_min, y_min = np.min(box, axis=0)
                x_max, y_max = np.max(box, axis=0)
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(orig_img.shape[1], x_max), min(orig_img.shape[0], y_max)
                crop = orig_img[int(y_min):int(y_max), int(x_min):int(x_max)]
                if crop.size > 0 and crop.shape[0] > 5 and crop.shape[1] > 5:
                    break
        if crop is None or crop.size == 0:
            print("❌ Plate detected but crop failed(plate is not cropped)!")
            return None

        # OCR on the cropped plate image
        ocr_result = ocr_extractor.extract_number_from_array(crop)
        img, bboxes, obb_xywhr = ocr_result[5:8]
        final_number, area, number, others, angle = ocr_result[:5]

        # You can return everything you want here
        return img, final_number, area, number, others, angle, bboxes, obb_xywhr

