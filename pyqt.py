from PySide6 import QtWidgets, QtCore, QtGui
import cv2, os, time
from threading import Thread

# 不然每次YOLO处理都会输出调试信息
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO 
from PySide6.QtCore import Qt

class MWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # 设置界面
        self.setupUI()

        # 绑定按钮事件
        self.videoBtn.clicked.connect(self.openVideoFile)
        self.camBtn.clicked.connect(self.startCamera)
        self.stopBtn.clicked.connect(self.stop)
        self.imageBtn.clicked.connect(self.openImageFile)  # 绑定检测照片按钮

        # 定义定时器，用于控制显示视频的帧率
        self.timer_camera = QtCore.QTimer()
        self.timer_camera.timeout.connect(self.show_camera)

        # 加载 YOLO 模型
        self.model = YOLO('yolov8n.pt')
        print("YOLO 模型加载成功")

        # 要处理的视频帧图片队列
        self.frameToAnalyze = []

        # 启动处理视频帧独立线程
        Thread(target=self.frameAnalyzeThreadFunc, daemon=True).start()

    def setupUI(self):

        self.resize(1200, 800)

        self.setWindowTitle('对人的目标检测')

        # central Widget
        centralWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(centralWidget)

        # central Widget 里面的 主 layout
        mainLayout = QtWidgets.QVBoxLayout(centralWidget)

        # 界面的上半部分 : 图形展示部分
        topLayout = QtWidgets.QHBoxLayout()
        self.label_ori_video = QtWidgets.QLabel(self)
        self.label_treated = QtWidgets.QLabel(self)
        self.label_ori_video.setMinimumSize(520,400)
        self.label_treated.setMinimumSize(520,400)
        self.label_ori_video.setStyleSheet('border:1px solid #D7E2F9;')
        self.label_treated.setStyleSheet('border:1px solid #D7E2F9;')

        topLayout.addWidget(self.label_ori_video)
        topLayout.addWidget(self.label_treated)

        mainLayout.addLayout(topLayout)

        # 界面下半部分： 输出框 和 按钮
        groupBox = QtWidgets.QGroupBox(self)

        bottomLayout =  QtWidgets.QHBoxLayout(groupBox)
        self.textLog = QtWidgets.QTextBrowser()
        bottomLayout.addWidget(self.textLog)

        mainLayout.addWidget(groupBox)

        btnLayout = QtWidgets.QVBoxLayout()
        self.videoBtn = QtWidgets.QPushButton('🎞️视频文件')
        self.camBtn   = QtWidgets.QPushButton('📹摄像头')
        self.stopBtn  = QtWidgets.QPushButton('🛑停止')
        self.imageBtn = QtWidgets.QPushButton('🖼️图片文件')  # 添加检测照片按钮
        btnLayout.addWidget(self.videoBtn)
        btnLayout.addWidget(self.camBtn)
        btnLayout.addWidget(self.stopBtn)
        btnLayout.addWidget(self.imageBtn)  # 添加检测照片按钮
        bottomLayout.addLayout(btnLayout)


    def startCamera(self):
        # 如果摄像头对象已存在且未释放，先释放资源
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.cap.release()

        # 重新初始化摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("摄像头无法打开")
            return
        else:
            print("摄像头已成功打开")

        if not self.timer_camera.isActive():
            self.timer_camera.start(50)  # 启动定时器
            print("定时器启动成功")
        else:
            print("定时器已经启动")

    def show_camera(self):
        ret, frame = self.cap.read()
        if not ret:
            print("未能从摄像头读取帧")
            return
        else:
            print("成功读取摄像头帧")

        # 调整帧大小并转换为 RGB
        frame = cv2.resize(frame, (520, 400))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0],
                              QtGui.QImage.Format_RGB888)
        self.label_ori_video.setPixmap(QtGui.QPixmap.fromImage(qImage))

        # 如果当前没有处理任务
        if not self.frameToAnalyze:
            self.frameToAnalyze.append(frame)

    def frameAnalyzeThreadFunc(self):
        while True:
            if not self.frameToAnalyze:
                time.sleep(0.01)  # 短时间休眠
                continue

            frame = self.frameToAnalyze.pop(0)

            # YOLO 推理
            start_time = time.time()
            results = self.model(frame, imgsz=640, conf=0.25)[0]
            end_time = time.time()
            print(f"YOLO 推理耗时: {end_time - start_time:.2f} 秒")

            # 提取预测结果
            labels = results.names  # 标签名称
            boxes = results.boxes  # 检测框
            label_counts = {}  # 统计每个标签的数量

            log_text = f"YOLO 推理耗时: {end_time - start_time:.2f} 秒\n"
            for box in boxes:
                label = labels[int(box.cls)]  # 获取标签名称
                conf = float(box.conf)  # 将 Tensor 转换为浮点数
                log_text += f"标签: {label}, 置信度: {conf:.2f}\n"

                # 统计标签数量
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

            log_text += "标签统计:\n"
            for label, count in label_counts.items():
                log_text += f"{label}: {count}\n"

            # 更新到 QTextBrowser
            self.textLog.append(log_text)

            # 绘制检测框
            img = results.plot(line_width=1)
            if img is None:
                print("绘制检测框失败")
                continue

            # 转换为 QImage
            qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)

            # 在主线程中更新 GUI
            QtCore.QMetaObject.invokeMethod(self.label_treated, "setPixmap",
                                            Qt.QueuedConnection,
                                            QtCore.Q_ARG(QtGui.QPixmap, QtGui.QPixmap.fromImage(qImage)))

    def stop(self):
        self.timer_camera.stop()  # 关闭定时器
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()  # 释放摄像头资源
            self.cap = None  # 确保摄像头对象被清空
        self.label_ori_video.clear()  # 清空视频显示区域        
        self.label_treated.clear()  # 清空视频显示区域
        print("程序已停止")

    def openVideoFile(self):
        # 打开文件选择对话框
        file_dialog = QtWidgets.QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mkv)")

        if file_path:
            print(f"选择的视频文件: {file_path}")
            # 打开视频文件
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                print("无法打开视频文件")
                return
            else:
                print("视频文件已成功打开")

            # 启动定时器以显示视频帧
            if not self.timer_camera.isActive():
                self.timer_camera.start(50)
                print("定时器启动成功")

    def openImageFile(self):
        # 打开文件选择对话框
        file_dialog = QtWidgets.QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, "选择图片文件", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)")

        if file_path:
            print(f"选择的图片文件: {file_path}")
            # 读取图片
            frame = cv2.imread(file_path)
            if frame is None:
                print("无法读取图片文件")
                return

            # 转换为 RGB 格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # YOLO 推理
            start_time = time.time()
            results = self.model(frame, imgsz=640, conf=0.25)[0]
            end_time = time.time()
            print(f"YOLO 推理耗时: {end_time - start_time:.2f} 秒")

            # 提取预测结果
            labels = results.names  # 标签名称
            boxes = results.boxes  # 检测框
            label_counts = {}  # 统计每个标签的数量

            log_text = f"YOLO 推理耗时: {end_time - start_time:.2f} 秒\n"
            for box in boxes:
                label = labels[int(box.cls)]  # 获取标签名称
                conf = float(box.conf)  # 将 Tensor 转换为浮点数
                log_text += f"标签: {label}, 置信度: {conf:.2f}\n"

                # 统计标签数量
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1

            log_text += "标签统计:\n"
            for label, count in label_counts.items():
                log_text += f"{label}: {count}\n"

            # 更新到 QTextBrowser
            self.textLog.append(log_text)

            # 绘制检测框
            img = results.plot(line_width=1)
            if img is None:
                print("绘制检测框失败")
                return

            # 转换为 QImage 并显示在 label_treated
            qImage = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
            self.label_treated.setPixmap(QtGui.QPixmap.fromImage(qImage))

            # 显示原始图片在 label_ori_video
            qImage_original = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            self.label_ori_video.setPixmap(QtGui.QPixmap.fromImage(qImage_original))

app = QtWidgets.QApplication()
window = MWindow()
window.show()
app.exec()