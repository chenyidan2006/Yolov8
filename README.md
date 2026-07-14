graph TB
    %% 1. 传感器与输入节点定义 (使用 GitHub 完美兼容的标准矩形)
    Sub_Cam_RGB["摄像头 / YOLOv5 检测线程 (Camera & Inference)"]
    Sub_Cam_Depth["深度相机 / 深度感知线程 (Depth Camera)"]
    Sub_Odom["车轮编码器 (Odometry Source)"]

    %% 2. ROS 核心控制节点定义 (用圆括号 ( ) 实现圆角矩形)
    Node_DetectLine("detect_line3.py (车道线识别与滑动窗口拟合)")
    Node_Vth2ros("vth2ros.py (底层 PID 视觉伺服)")
    Node_Velpub("velpub51.py (上层多源状态机决策)")

    %% 3. 硬件驱动与物理执行节点定义 (使用标准直角矩形)
    Node_Base["ai_racecar.py (下位机电机与舵机驱动)"]

    %% 4. 话题数据流向连接 - 视觉巡线链路
    Node_DetectLine -->|发布 / Float64| Topic_VthError["/vtherror_topic (横向跟踪偏差)"]
    Topic_VthError --> Node_Vth2ros

    %% 话题数据流向连接 - YOLO与深度传感器链路
    Sub_Cam_RGB -->|发布 / Int32| Topic_YoloClass["/yolo_class_id (标识牌类别ID)"]
    Sub_Cam_RGB -->|发布 / Int32| Topic_YoloArea["/yolo_area (标志物像素面积)"]
    Sub_Cam_RGB -->|发布 / Point| Topic_YoloCenter["/yolo_center (目标中心坐标)"]
    Sub_Cam_Depth -->|发布 / Image| Topic_Depth["/camera/depth/image (深度图像流)"]
    Sub_Odom -->|发布 / Odometry| Topic_Odom["/odom (里程计反馈)"]

    Topic_YoloClass --> Node_Velpub
    Topic_YoloArea --> Node_Velpub
    Topic_YoloCenter --> Node_Velpub
    Topic_Depth --> Node_Velpub
    Topic_Odom --> Node_Velpub

    %% 核心控制流切换与重映射 (过渡话题 -> 最终话题)
    Node_Vth2ros -->|发布 / Twist| Topic_CmdVel1["/cmd_vel_1 (基础巡线速度指令)"]
    Topic_CmdVel1 --> Node_Velpub

    Node_Velpub -->|发布 / Twist| Topic_CmdVel["/cmd_vel (决策修正后控制指令)"]
    Topic_CmdVel --> Node_Base

    %% 反向控制反馈与辅助状态话题 (使用虚线 .->)
    Node_Velpub -.->|发布 / Bool| Topic_Limit["/limit_flag_topic (限速触发状态)"]
    Node_Velpub -.->|发布 / Int32| Topic_Reverse["/reverse_flag_topic (环岛方向等转向状态)"]
    Node_Velpub -.->|发布 / Bool| Topic_ForceLeft["/force_follow_left_topic (强制巡左线指令)"]
    Node_Velpub -.->|发布 / Bool| Topic_ForceRight["/force_follow_right_topic (强制巡右线指令)"]

    Topic_Limit -.-> Node_DetectLine
    Topic_Reverse -.-> Node_DetectLine
    Topic_ForceLeft -.-> Node_DetectLine
    Topic_ForceRight -.-> Node_DetectLine

    %% 5. GitHub 专用的内联 CSS 样式定义 (避免使用 classDef 导致崩溃)
    style Sub_Cam_RGB fill:#F9F,stroke:#333,stroke-width:2px;
    style Sub_Cam_Depth fill:#F9F,stroke:#333,stroke-width:2px;
    style Sub_Odom fill:#F9F,stroke:#333,stroke-width:2px;

    style Node_DetectLine fill:#85C1E9,stroke:#333,stroke-width:2px;
    style Node_Vth2ros fill:#85C1E9,stroke:#333,stroke-width:2px;
    style Node_Velpub fill:#85C1E9,stroke:#333,stroke-width:2px;

    style Node_Base fill:#F5B041,stroke:#333,stroke-width:2px;

    style Topic_VthError fill:#EAFAF1,stroke:#27AE60,stroke-width:1px,stroke-dasharray: 5 5;
    style Topic_YoloClass fill:#EAFAF1,stroke:#27AE60,stroke-width:1px,stroke-dasharray: 5 5;
    style Topic_YoloArea fill:#EAFAF1,stroke:#27AE60,stroke-width:1px,stroke-dasharray: 5 5;
    style Topic_YoloCenter fill:#EAFAF1,stroke:#27AE60,stroke-width:1px,stroke-dasharray: 5 5;
    style Topic_Depth fill:#EAFAF1,stroke:#27AE60,stroke-width:1px,stroke-dasharray: 5 5;
    style Topic_Odom fill:#EAFAF1,stroke:#27AE60,stroke-width:1px,stroke-dasharray: 5 5;
    style Topic_CmdVel1 fill:#EAFAF1,stroke:#27AE60,stroke-width:1px,stroke-dasharray: 5 5;
    style Topic_CmdVel fill:#EAFAF1,stroke:#27AE60,stroke-width:1px,stroke-dasharray: 5 5;
    style Topic_Limit fill:#EAFAF1,stroke:#27AE60,stroke-width:1px,stroke-dasharray: 5 5;
    style Topic_Reverse fill:#EAFAF1,stroke:#27AE60,stroke-width:1px,stroke-dasharray: 5 5;
    style Topic_ForceLeft fill:#EAFAF1,stroke:#27AE60,stroke-width:1px,stroke-dasharray: 5 5;
    style Topic_ForceRight fill:#EAFAF1,stroke:#27AE60,stroke-width:1px,stroke-dasharray: 5 5;
