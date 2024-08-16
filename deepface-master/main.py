import cv2
from deepface import DeepFace
import numpy as np

TF_ENABLE_ONEDNN_OPTS=0

#初始化视频写入器
def init_video_writer(cap, output_video):
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

#分析视频帧的情绪
def analyze_emotion(frame):
    try:
        result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)  # 改成了False 不强制进行人脸检测？
        if result and len(result) > 0:
            emotion = result[0]["dominant_emotion"]
            region = result[0]["region"]
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            return emotion, (x, y), (x + w, y + h)
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
    return None, None, None #三个参数怎么样？

# 定义情绪分数统计函数
def update_emotion_score(emotion, emotion_score):
    if emotion in ['surprise', 'happy']:
        emotion_score['total'] += 10
    elif emotion in ['sad', 'angry', 'disgust', 'fear']:
        emotion_score['total'] -= 1  
    return emotion_score

# 初始化情绪分数字典
# emotion_scores = {'total': 0}  
# 初始化情绪分数字典，存储两个摄像头的分数
emotion_scores = {'camera1': 0, 'camera2': 0}

# 定义一个函数创建一个窗口包含两个视频流和一个按钮
def create_window(frame1, frame2):
    # 调整视频帧大小为相同大小
    frame1 = cv2.resize(frame1, (640, 480))
    frame2 = cv2.resize(frame2, (640, 480))

    # 拼接两个视频帧
    combined_frame = np.hstack((frame1, frame2))
    cv2.imshow('Eyes of Howdy', combined_frame)

   
    # 创建窗口
    #cv2.namedWindow('Eyes of Howdy')

def save_image(frame1,frame2):
    cv2.imwrite('saved_image1.jpg', frame1)
    cv2.imwrite('saved_image2.jpg', frame2)
    print("Images saved.")

# 鼠标回调函数，用于按钮点击事件
def handle_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        if 10 <= x <= 130 and 10 <= y <= 50:  # 假设按钮区域
            save_image(param[0], param[1])      
    '''
    # 在窗口上创建按钮
    cv2.resizeWindow('Eyes of Howdy', 960, 720)  # 根据需要调整大小
    cv2.imshow('Eyes of Howdy', combined_frame)

    # 定义按钮回调函数
    def save_image():
        cv2.imwrite('saved_image1.jpg', frame1)
        cv2.imwrite('saved_image2.jpg', frame2)
        print("Images saved.")

    # 创建一个按钮并绑定回调函数
    button = cv2.createButton('Save Image', save_image, 10, 10, 120)

    # 显示图像和按钮
    cv2.imshow('Eyes of Howdy', combined_frame)
    cv2.setMouseCallback('Eyes of Howdy', handle_mouse)

    # 鼠标回调函数，用于按钮点击事件
    def handle_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:  # 鼠标左键释放事件
            if button.isPressed():
                save_image()
    '''
    

# 将主程序代码块包裹起来，避免在导入模块时执行主程序代码

if __name__ == "__main__":
    # 创建两个VideoCapture对象，分别对应两个摄像头
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)

    output_video1 = 'emotion1.mp4'
    output_video2 = 'emotion2.mp4'

    # 检查摄像头是否成功打开
    if not cap1.isOpened() or not cap2.isOpened():
        print("无法打开摄像头")
        exit()

    # 初始化视频写入器
    out1 = init_video_writer(cap1, output_video1)
    out2 = init_video_writer(cap2, output_video2)

    emodict = dict()
    frame_cnt = 0

    # 创建窗口
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    create_window(frame1, frame2)


    while True:
        # 从摄像头读取一帧图像
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # 如果正确读取帧，ret为True
        if not ret1 and ret2:
            print("无法接受帧，请退出")
            break
        # 分析情绪并显示结果
        emotion1, (x1, y1), (x2, y2) = analyze_emotion(frame1)
        emotion2, (x3, y3), (x4, y4) = analyze_emotion(frame2)

       
        if emotion1:
            if emotion1 not in emodict:
                emodict[emotion1] = 0
            emodict[emotion1] += 1
            emotion_scores['camera1'] = update_emotion_score(emotion1, {'total': emotion_scores['camera1']})['total']
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame1, emotion1, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            #cv2.putText(frame1, f"Score: {emotion_scores['camera1']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            #emotion_scores = update_emotion_score(emotion1, emotion_scores)

        if emotion2:
            if emotion2 not in emodict:
                emodict[emotion2] = 0
            emodict[emotion2] += 1
            emotion_scores['camera2'] = update_emotion_score(emotion2, {'total': emotion_scores['camera2']})['total']
            cv2.rectangle(frame2, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cv2.putText(frame2, emotion2, (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            #cv2.putText(frame2, f"Score: {emotion_scores['camera2']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            #emotion_scores = update_emotion_score(emotion2, emotion_scores)

        # 显示图像
        '''
        cv2.imshow('Video 1', frame1)
        cv2.imshow('Video 2', frame2)
        '''

        # 显示两个视频流
        create_window(frame1, frame2)
        '''
        # 根据情绪更新分数
        if emotion1 in ['surprise', 'happy']:
            emotion_score += 1
        elif emotion1 in ['sad', 'angry', 'disgust', 'fear']:
            emotion_score -= 1
        elif emotion1 in ['neutral']:
            pass

        if emotion2 in ['surprise', 'happy']:
            emotion_score += 100
        elif emotion2 in ['sad', 'angry', 'disgust', 'fear']:
            emotion_score -= 0.5
        elif emotion2 in ['neutral']:
            pass
        '''
    
        # 显示分数
        '''
        cv2.putText(frame1, f"emotion score: {emotion_scores['total']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Video 1', frame1)
        cv2.putText(frame2, f"emotion score: {emotion_scores['total']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Video 2', frame2)
        '''
        cv2.putText(frame1, f"Score: {emotion_scores['camera1']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Video 1', frame1)
        cv2.putText(frame2, f"Score: {emotion_scores['camera2']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Video 2', frame2)

        # 写入视频文件
        out1.write(frame1)
        out2.write(frame2)

        frame_cnt += 1
        print(f"Processing frame {frame_cnt}")

         # 设置鼠标回调
        cv2.setMouseCallback('Eyes of Howdy', handle_mouse, param=[frame1, frame2])

        # 按 'q' 退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源并关闭窗口
    cap1.release()
    cap2.release()
    out1.release()
    out2.release()
    cv2.destroyAllWindows()

# 打印最终分数
# print(f"Final emotion score: {emotion_scores['total']}")
# 打印两个摄像头的最终分数和平均分
camera1_final_score = emotion_scores['camera1']
camera2_final_score = emotion_scores['camera2']
average_score = (camera1_final_score + camera2_final_score) / 2
print(f"Final emotion score for Camera 1: {camera1_final_score}")
print(f"Final emotion score for Camera 2: {camera2_final_score}")
print(f"Average emotion score: {average_score}")