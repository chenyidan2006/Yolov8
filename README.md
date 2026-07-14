graph TB
    %% 样式定义
    classDef node_sensor fill:#F9F,stroke:#333,stroke-width:2px;
    classDef node_core fill:#85C1E9,stroke:#333,stroke-width:2px,rx:10px;
    classDef node_hardware fill:#F5B041,stroke:#333,stroke-width:2px;
    classDef topic fill:#EAFAF1,stroke:#27AE60,stroke-width:1px,stroke-dasharray: 5 5;

    %% 传感器与输入节点
    Sub_Cam_RGB[/"摄像头 / YOLOv5 检测线程"<br/>(Camera & Inference)/]:::node_sensor
    Sub_Cam_Depth[/"深度相机 / 深度感知线程"<br/>(Depth Camera)/]:::node_sensor
    Sub_Odom[/"车轮编码器"<br/>(Odometry Source)/]:::node_sensor

    %% ROS 核心控制节点
    Node_DetectLine[["detect_line3.py"<br/>(车道线识别与滑动窗口拟合)]]:::node_core
    Node_Vth2ros[["vth2ros.py"<br/>(底层 PID 视觉伺服)]]:::node_core
    Node_Velpub[["velpub51.py"<br/>(上层多源状态机决策)]]:::node_core

    %% 硬件驱动与物理执行
    Node_Base[["ai_racecar.py"<br/>(下位机电机与舵机驱动)]]:::node_hardware

    %% 话题流向连接 - 视觉巡线部分
    Node_DetectLine -->|发布 / Float64| Topic_VthError["/vtherror_topic<br/>(横向跟踪偏差)"]:::topic
    Topic_VthError --> Node_Vth2ros

    %% 话题流向连接 - YOLO与深度传感器部分
    Sub_Cam_RGB -->|发布 / Int32| Topic_YoloClass["/yolo_class_id<br/>(标识牌类别ID)"]:::topic
    Sub_Cam_RGB -->|发布 / Int32| Topic_YoloArea["/yolo_area<br/>(标志物像素面积)"]:::topic
    Sub_Cam_RGB -->|发布 / Point| Topic_YoloCenter["/yolo_center<br/>(目标中心坐标)"]:::topic
    Sub_Cam_Depth -->|发布 / Image| Topic_Depth["/camera/depth/image<br/>(深度图像流)"]:::topic
    Sub_Odom -->|发布 / Odometry| Topic_Odom["/odom<br/>(里程计反馈)"]:::topic

    Topic_YoloClass --> Node_Velpub
    Topic_YoloArea --> Node_Velpub
    Topic_YoloCenter --> Node_Velpub
    Topic_Depth --> Node_Velpub
    Topic_Odom --> Node_Velpub

    %% 核心控制流切换与重映射 (过渡话题 -> 最终话题)
    Node_Vth2ros -->|发布 / Twist| Topic_CmdVel1["/cmd_vel_1<br/>(基础巡线速度指令)"]:::topic
    Topic_CmdVel1 --> Node_Velpub

    Node_Velpub -->|发布 / Twist| Topic_CmdVel["/cmd_vel<br/>(决策修正后控制指令)"]:::topic
    Topic_CmdVel --> Node_Base

    %% 反向控制反馈与辅助话题
    Node_Velpub -.->|发布 / Bool| Topic_Limit["/limit_flag_topic<br/>(限速触发状态)"]:::topic
    Node_Velpub -.->|发布 / Int32| Topic_Reverse["/reverse_flag_topic<br/>(环岛方向等转向状态)"]:::topic
    Node_Velpub -.->|发布 / Bool| Topic_ForceLeft["/force_follow_left_topic<br/>(强制巡左线指令)"]:::topic
    Node_Velpub -.->|发布 / Bool| Topic_ForceRight["/force_follow_right_topic<br/>(强制巡右线指令)"]:::topic

    Topic_Limit -.-> Node_DetectLine
    Topic_Reverse -.-> Node_DetectLine
    Topic_ForceLeft -.-> Node_DetectLine
    Topic_ForceRight -.-> Node_DetectLine

    %% 状态干预说明
    note1["<b>状态机干预逻辑 (velpub51.py):</b><br/>1. 环岛状态: 自主打方向盲行<br/>2. 限速状态: 初始线速度折半<br/>3. 避让/红灯: 强制线速度角速度清零"]:::topic
    Node_Velpub -.-> note1
