```mermaid
graph LR
    %% 节点样式定义
    classDef sensor fill:#f9f,stroke:#333,stroke-width:2px;
    classDef nodeClass fill:#dfd,stroke:#333,stroke-width:2px;
    classDef topicClass fill:#bbf,stroke:#333,stroke-width:1px;
    classDef fileClass fill:#ffb,stroke:#333,stroke-width:1px;

    %% 1. 感知输入层
    Cam[车载摄像头 Video Stream]:::sensor
    Odom[车轮编码器/IMU]:::sensor

    %% 2. 核心处理层
    DL_Node((detect_line 节点 <br> detect_line3.py)):::nodeClass
    YOLO_Node((YOLOv5 检测线程)):::nodeClass
    VR_Node((veltalker 节点 <br> vth2ros.py)):::nodeClass
    VP_Node((veltalker 节点 <br> velpub51.py)):::nodeClass

    %% 3. ROS话题与交互介质
    Topic_vth[/ /vtherror_topic <br> Float64 /]:::topicClass
    Topic_vel1[/ /cmd_vel_1 <br> Twist /]:::topicClass
    Topic_odom[/ /odom <br> Odometry /]:::topicClass
    File_sign[( yolosign.txt <br> 文本交互 /)]:::fileClass
    Topic_yolo[/ yolo_class_id等话题 <br> Int32 /]:::topicClass

    %% 4. 执行层
    Base_Node((base_control 节点 <br> ai_racecar.py)):::nodeClass
    Topic_vel[/ /cmd_vel <br> Twist /]:::topicClass

    %% 信号连接线
    Cam -->|图像帧| DL_Node
    Cam -->|图像帧| YOLO_Node

    %% 车道线跟踪链路
    DL_Node -->|计算横向偏差| Topic_vth
    Topic_vth -->|订阅偏差| VR_Node
    VR_Node -->|发布基础巡线速度| Topic_vel1
    Topic_vel1 -->|控制流输入| VP_Node

    %% 交通标志链路
    YOLO_Node -->|持久化写入| File_sign
    YOLO_Node -->|异步发布| Topic_yolo
    File_sign -.->|Timer定时器每0.1s轮询| VP_Node
    Topic_yolo -->|订阅标志ID/面积| VP_Node

    %% 里程计链路
    Odom -->|发布位姿| Topic_odom
    Topic_odom -->|订阅位置实现精确动作| VP_Node

    %% 最终输出
    VP_Node -->|状态机决策动作| Topic_vel
    Topic_vel -->|驱动电机与舵机| Base_Node
