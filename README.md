graph TB
    %% 1. 传感器与输入节点定义
    Sub_Cam_RGB[\"摄像头 / YOLOv5 检测线程<br/>(Camera & Inference)"/]
    Sub_Cam_Depth[\"深度相机 / 深度感知线程<br/>(Depth Camera)"/]
    Sub_Odom[\"车轮编码器<br/>(Odometry Source)"/]

    %% 2. ROS 核心控制节点定义 (用 ( ) 代替 rx:10px 实现圆角)
    Node_DetectLine("detect_line3.py<br/>(车道线识别与滑动窗口拟合)")
    Node_Vth2ros("vth2ros.py<br/>(底层 PID 视觉伺服)")
    Node_Velpub("velpub51.py<br/>(上层多源状态机决策)")

    %% 3. 硬件驱动与物理执行节点定义
    Node_Base["ai_racecar.py<br/>(下位机电机与舵机驱动)"]

    %% 4. 话题数据流向连接 - 视觉巡线链路
    Node_DetectLine -->|发布 / Float64| Topic_VthError["/vtherror_topic<br/>(横向跟踪偏差)"]
    Topic_VthError --> Node_Vth2ros

    %% 话题数据流向连接 - YOLO与深度传感器链路
    Sub_Cam_RGB -->|发布 / Int32| Topic_YoloClass["/yolo_class_id<br/>(标识牌类别ID)"]
    Sub_Cam_RGB -->|发布 / Int32| Topic_YoloArea["/yolo_area<br/>(标志物像素面积)"]
    Sub_Cam_RGB -->|发布 / Point| Topic_YoloCenter["/yolo_center<br/>(目标中心坐标)"]
    Sub_Cam_Depth -->|发布 / Image| Topic_Depth["/camera/depth/image<br/>(深度图像流)"]
    Sub_Odom -->|发布 / Odometry| Topic_Odom["/odom<br/>(里程计反馈)"]

    Topic_YoloClass --> Node_Velpub
    Topic_YoloArea --> Node_Velpub
    Topic_YoloCenter --> Node_Velpub
    Topic_Depth --> Node_Velpub
    Topic_Odom --> Node_Velpub

    %% 核心控制流切换与重映射 (过渡话题 -> 最终话题)
    Node_Vth2ros -->|发布 / Twist| Topic_CmdVel1["/cmd_vel_1<br/>(基础巡线速度指令)"]
    Topic_CmdVel1 --> Node_Velpub

    Node_Velpub -->|发布 / Twist| Topic_CmdVel["/cmd_vel<br/>(决策修正后控制指令)"]
    Topic_CmdVel --> Node_Base

    %% 反向控制反馈与辅助状态话题 (使用虚线 .->)
    Node_Velpub -.->|发布 / Bool| Topic_Limit["/limit_flag_topic<br/>(限速触发状态)"]
    Node_Velpub -.->|发布 / Int32| Topic_Reverse["/reverse_flag_topic<br/>(环岛方向等转向状态)"]
    Node_Velpub -.->|发布 / Bool| Topic_ForceLeft["/force_follow_left_topic<br/>(强制巡左线指令)"]
    Node_Velpub -.->|发布 / Bool| Topic_ForceRight["/force_follow_right_topic<br/>(强制巡右线指令)"]

    Topic_Limit -.-> Node_DetectLine
    Topic_Reverse -.-> Node_DetectLine
    Topic_ForceLeft -.-> Node_DetectLine
    Topic_ForceRight -.-> Node_DetectLine

    %% 5. 样式类定义 (符合官方规范的独立分行写法)
    classDef node_sensor fill:#F9F,stroke:#333,stroke-width:2px;
    classDef node_core fill:#85C1E9,stroke:#333,stroke-width:2px;
    classDef node_hardware fill:#F5B041,stroke:#333,stroke-width:2px;
    classDef topic fill:#EAFAF1,stroke:#27AE60,stroke-width:1px,stroke-dasharray: 5 5;

    %% 6. 样式绑定
    class Sub_Cam_RGB,Sub_Cam_Depth,Sub_Odom node_sensor;
    class Node_DetectLine,Node_Vth2ros,Node_Velpub node_core;
    class Node_Base node_hardware;
    class Topic_VthError,Topic_YoloClass,Topic_YoloArea,Topic_YoloCenter,Topic_Depth,Topic_Odom,Topic_CmdVel1,Topic_CmdVel,Topic_Limit,Topic_Reverse,Topic_ForceLeft,Topic_ForceRight topic;
