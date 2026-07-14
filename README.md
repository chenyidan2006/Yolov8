graph TB
    Sub_Cam_RGB["摄像头与YOLOv5检测"]
    Sub_Cam_Depth["深度相机"]
    Sub_Odom["车轮编码器"]

    Node_DetectLine["detect_line3 车道线识别"]
    Node_Vth2ros["vth2ros 底层PID控制"]
    Node_Velpub["velpub51 决策控制状态机"]

    Node_Base["ai_racecar 底盘驱动"]

    Node_DetectLine --> Node_Vth2ros

    Sub_Cam_RGB --> Node_Velpub
    Sub_Cam_Depth --> Node_Velpub
    Sub_Odom --> Node_Velpub

    Node_Vth2ros --> Node_Velpub
    Node_Velpub --> Node_Base

    Node_Velpub -.-> Node_DetectLine
