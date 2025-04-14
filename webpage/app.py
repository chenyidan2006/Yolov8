from flask import Flask, render_template, request, jsonify, Response
import cv2
import os
import time
from ultralytics import YOLO

app = Flask(__name__)

# 加载 YOLO 模型
model = YOLO('yolov8n.pt')
print("YOLO 模型加载成功")

# 全局变量，用于摄像头实时检测
camera = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # 保存上传的文件
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # 读取图片
    frame = cv2.imread(file_path)
    if frame is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # 转换为 RGB 格式
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO 推理
    start_time = time.time()
    results = model(frame, imgsz=640, conf=0.25)[0]
    end_time = time.time()
    print(f"YOLO 推理耗时: {end_time - start_time:.2f} 秒")

    # 提取预测结果
    labels = results.names  # 标签名称
    boxes = results.boxes  # 检测框
    label_counts = {}  # 统计每个标签的数量
    predictions = []
    for box in boxes:
        label = labels[int(box.cls)]  # 获取标签名称
        conf = float(box.conf)  # 将 Tensor 转换为浮点数
        predictions.append({'label': label, 'confidence': conf})

        # 统计标签数量
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    # 绘制检测框
    img = results.plot(line_width=1)

    # 保存检测后的图片
    detected_path = os.path.join('uploads', f"detected_{file.filename}")
    cv2.imwrite(detected_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # 返回检测结果和图片路径
    return jsonify({
        'original_image': f"/uploads/{file.filename}",
        'detected_image': f"/uploads/detected_{file.filename}",
        'processing_time': f"{end_time - start_time:.2f} 秒",
        'predictions': predictions,
        'label_counts': label_counts
    })

@app.route('/video', methods=['POST'])
def video():
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # 保存上传的视频文件
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # 视频处理逻辑
    cap = cv2.VideoCapture(file_path)
    label_counts = {}
    predictions = []
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # YOLO 推理
        results = model(frame, imgsz=640, conf=0.25)[0]
        boxes = results.boxes
        labels = results.names

        for box in boxes:
            label = labels[int(box.cls)]
            conf = float(box.conf)
            predictions.append({'label': label, 'confidence': conf})

            # 统计标签数量
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

    cap.release()

    # 返回视频路径和检测结果
    return jsonify({
        'video_path': f"/uploads/{file.filename}",
        'predictions': predictions,
        'label_counts': label_counts
    })

@app.route('/camera_feed')
def camera_feed():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)  # 重新初始化摄像头

    def generate_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break

            # YOLO 推理
            results = model(frame, imgsz=640, conf=0.25)[0]
            img = results.plot(line_width=1)

            # 转换为 JPEG 格式
            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera
    if camera is not None:
        camera.release()  # 释放摄像头资源
        camera = None  # 确保摄像头对象被清空
    return jsonify({'message': 'Camera stopped successfully'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return Response(open(os.path.join('uploads', filename), 'rb').read(), mimetype='image/jpeg')

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)