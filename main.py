import os
import cv2
import torch
import numpy as np
import logging
import configparser
import time 
from datetime import datetime
import asyncio
from ultralytics import YOLO
from LPRNet.LPRNet import build_lprnet
from notifications import NotificationManager


CHARS = ['А', 'В', 'Е', 'К', 'М', 'Н', 'О', 'Р', 'С', 'Т', 'У', 'Х',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']

class PlateRecognizer:
    def __init__(self, config):
        self.device = torch.device(config.get('General', 'device'))
        self.lprnet = build_lprnet(8, phase=False, class_num=len(CHARS), dropout_rate=0).to(self.device)
        self.lprnet.load_state_dict(torch.load(config.get('LPRNet', 'model_path'), map_location=self.device))
        self.lprnet.eval()

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        img = cv2.resize(img, (94, 24)); img = img.astype('float32') - 127.5; img *= 0.0078125
        return torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(self.device)

    def _decode(self, preds: np.ndarray) -> str:
        pred = preds[0, :, :]; preds_label = list(np.argmax(pred[:, j], axis=0) for j in range(pred.shape[1]))
        no_repeat_blank_label = []; pre_c = preds_label[0]
        if pre_c != len(CHARS) - 1: no_repeat_blank_label.append(pre_c)
        for c in preds_label[1:]:
            if (c != pre_c) and (c != len(CHARS) - 1): no_repeat_blank_label.append(c)
            pre_c = c
        return "".join(CHARS[c] for c in no_repeat_blank_label)

    def recognize(self, plate_image: np.ndarray) -> str:
        if plate_image is None or plate_image.size == 0: return ""
        with torch.no_grad(): preds = self.lprnet(self._preprocess(plate_image))
        return self._decode(preds.cpu().numpy())


def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"{datetime.now().strftime('%Y-%m-%d')}.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(log_dir, log_filename)), logging.StreamHandler()])

def load_lists(white_path: str, black_path: str) -> tuple[set, set]:
    try:
        with open(white_path, 'r') as f: whitelist = {line.strip().upper() for line in f}
    except FileNotFoundError: whitelist = set()
    try:
        with open(black_path, 'r') as f: blacklist = {line.strip().upper() for line in f}
    except FileNotFoundError: blacklist = set()
    return whitelist, blacklist

async def run_pipeline():
    config = configparser.ConfigParser()
    config.read('def.conf')

    setup_logging(config.get('General', 'log_dir'))
    whitelist, blacklist = load_lists(config.get('General', 'whitelist_path'), config.get('General', 'blacklist_path'))
    recently_seen_plates = {}
    cooldown_period = config.getint('General', 'notification_cooldown_seconds', fallback=60)

    try:
        detector = YOLO(config.get('YOLO', 'model_path'))
        recognizer = PlateRecognizer(config)
        notifier = NotificationManager(config)
        await notifier.start_session()
    except Exception as e:
        logging.error(f"Initialization error: {e}")
        return

    video_source = config.get('General', 'video_source')
    cap = cv2.VideoCapture(0 if video_source == '0' else video_source)
    if not cap.isOpened():
        logging.error(f"Can't open video: {video_source}")
        return

    logging.info(f"Running. Press 'q' to exit.")
    
    while cap.isOpened():
        ret, frame = cap.read()

        results = detector.predict(frame, conf=config.getfloat('YOLO', 'confidence_threshold'), verbose=False)
        current_time = time.time()
        
        notification_tasks = []

        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = frame[y1:y2, x1:x2]
                plate_text = recognizer.recognize(plate_crop)

                if plate_text and len(plate_text) >= 6:
                    last_seen_time = recently_seen_plates.get(plate_text)
                    if last_seen_time and (current_time - last_seen_time) < cooldown_period:
                        continue
                    
                    recently_seen_plates[plate_text] = current_time
                    logging.info(f"License plate: {plate_text}")
                    status = "unknown"
                    if plate_text in whitelist: status = "white"
                    elif plate_text in blacklist: status = "black"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    #cv2.putText(frame, f"{plate_text} [{status}]", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    task_api = notifier.send_api_request(plate_text, status)
                    task_telegram = notifier.send_telegram_photo(plate_text, status, frame)
                    notification_tasks.extend([task_api, task_telegram])

        if notification_tasks:
            await asyncio.gather(*notification_tasks)

        cv2.imshow("LPR", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    await notifier.close_session()
    logging.info("Stop")

if __name__ == '__main__':
    try:
        asyncio.run(run_pipeline())
    except KeyboardInterrupt:
        print("Program stopped by user")