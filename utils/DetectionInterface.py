# Ultralytics YOLO 🚀, AGPL-3.0 license
 
import argparse
import cv2
import numpy as np
import onnxruntime as ort
import os
 
# 类外定义类别映射关系，使用字典格式
CLASS_NAMES = {
    0: 'person'
}
 
 
class DetectionInterface:
    """目标检测模型类，用于处理单张图片或视频推理和可视化。"""
    def __init__(self, onnx_model = r"models/yolo12n.onnx", input_path= r'demo/1.jpg', output= r'results', confidence_thres = 0.5, iou_thres = 0.45):
        """
        初始化 Detection 类的实例。
        参数：
            onnx_model: ONNX 模型的路径。
            input_path: 输入图像或视频的路径。
            output: 输出结果的保存路径或文件名。
            confidence_thres: 用于过滤检测结果的置信度阈值。
            iou_thres: 非极大值抑制（NMS）的 IoU（交并比）阈值。
        """
        self.onnx_model = onnx_model
        self.input_path = input_path
        self.output_path = output
        self.confidence_thres =confidence_thres 
        self.iou_thres = iou_thres
        # 判断输入是图像还是视频
        self.is_video = self.check_if_video(input_path)
        # 加载类别名称
        self.classes = CLASS_NAMES
        # 为每个类别生成一个颜色调色板
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.ort_session =  ort.InferenceSession(
            self.onnx_model, 
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"],
        )
        print("Initialized Detection class with ONNX model: ", self.onnx_model)
        self.input_width = self.ort_session.get_inputs()[0].shape[2]
        self.input_height = self.ort_session.get_inputs()[0].shape[3]
        
        # 确保输出目录存在
        self._prepare_output_path()
    
    def check_if_video(self, path):
        """
        检查输入路径是否为视频文件
        参数：
            path: 输入文件路径
        返回：
            bool: 如果是视频则返回True，否则返回False
        """
        # 如果路径为None，返回False
        if path is None:
            return False
            
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        _, ext = os.path.splitext(path)
        return ext.lower() in video_extensions
    
    def preprocess_frame(self, frame):
        """
        对单帧图像进行预处理，以便进行推理。
        参数：
            frame: 输入的视频帧
        返回：
            image_data: 经过预处理的图像数据，准备进行推理。
        """
        # 获取输入图像的高度和宽度
        self.img_height, self.img_width = frame.shape[:2]
        # 将图像颜色空间从 BGR 转换为 RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 保持宽高比，进行 letterbox 填充, 使用模型要求的输入尺寸
        img, self.ratio, (self.dw, self.dh) = self.letterbox(img, new_shape=(self.input_width, self.input_height))
        # 通过除以 255.0 来归一化图像数据
        image_data = np.array(img) / 255.0
        # 将图像的通道维度移到第一维
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道优先
        # 扩展图像数据的维度，以匹配模型输入的形状
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        # 返回预处理后的图像数据
        return image_data

    def preprocess(self):
        """
        对输入图像进行预处理，以便进行推理。
        返回：
            image_data: 经过预处理的图像数据，准备进行推理。
        """
        # 使用 OpenCV 读取输入图像
        self.img = cv2.imread(self.input_path)
        return self.preprocess_frame(self.img)
    
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        """
        将图像进行 letterbox 填充，保持纵横比不变，并缩放到指定尺寸。
        """
        shape = img.shape[:2]  # 当前图像的宽高
        # print(f"Original image shape: {shape}")
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 选择宽高中最小的缩放比
        if not scaleup:  # 仅缩小，不放大
            r = min(r, 1.0)
        # 缩放后的未填充尺寸
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        # 计算需要的填充
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算填充的尺寸
        dw /= 2  # padding 均分
        dh /= 2
        # 缩放图像
        if shape[::-1] != new_unpad:  # 如果当前图像尺寸不等于 new_unpad，则缩放
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        # 为图像添加边框以达到目标尺寸
        top, bottom = int(round(dh)), int(round(dh))
        left, right = int(round(dw)), int(round(dw))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        # print(f"Final letterboxed image shape: {img.shape}")
        return img, (r, r), (dw, dh)
    
    def _prepare_output_path(self):
        """
        准备输出路径，判断是文件还是目录
        """
        # 检查输出路径是否为文件
        _, ext = os.path.splitext(self.output_path)
        self.output_is_file = ext != ''
        
        if self.output_is_file:
            # 如果是文件，确保其父目录存在
            output_dir = os.path.dirname(self.output_path)
            if output_dir:  # 如果有父目录
                os.makedirs(output_dir, exist_ok=True)
        else:
            # 如果是目录，确保目录存在
            os.makedirs(self.output_path, exist_ok=True)
    
    def postprocess(self, input_image, output):
        """
        对模型输出进行后处理，以提取边界框、分数和类别 ID。
        参数：
            input_image (numpy.ndarray): 输入图像。
            output (numpy.ndarray): 模型的输出。
        返回：
            numpy.ndarray: 包含检测结果的输入图像。
        """
        # 转置并压缩输出，以匹配预期形状
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        boxes, scores, class_ids = [], [], []
        # 计算缩放比例和填充
        ratio = self.img_width / self.input_width, self.img_height / self.input_height
        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = np.argmax(classes_scores)
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                # 将框调整到原始图像尺寸，考虑缩放和填充
                x -= self.dw  # 移除填充
                y -= self.dh
                x /= self.ratio[0]  # 缩放回原图
                y /= self.ratio[1]
                w /= self.ratio[0]
                h /= self.ratio[1]
                left = int(x - w / 2)
                top = int(y - h / 2)
                width = int(w)
                height = int(h)
                boxes.append([left, top, width, height])
                scores.append(max_score)
                class_ids.append(class_id)
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        for i in indices:
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            self.draw_detections(input_image, box, score, class_id)
        return input_image
    
    def draw_detections(self, img, box, score, class_id):
        """
        在输入图像上绘制检测到的边界框和标签。
        参数：
            img: 用于绘制检测结果的输入图像。
            box: 检测到的边界框。
            score: 对应的检测分数。
            class_id: 检测到的目标类别 ID。
        
        返回：
            None
        """
        # 提取边界框的坐标
        x1, y1, w, h = box
        # 获取类别对应的颜色
        color = self.color_palette[class_id]
        # 在图像上绘制边界框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
        # 创建包含类别名和分数的标签文本
        label = f"{self.classes[class_id]}: {score:.2f}"
        # 计算标签文本的尺寸
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # 计算标签文本的位置
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
        # 绘制填充的矩形作为标签文本的背景
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
        # 在图像上绘制标签文本
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    
    def detect_frame(self, frame):
        """
        对单帧图像进行目标检测
        参数：
            frame: 输入的视频帧
        返回：
            processed_frame: 处理后的带有检测结果的帧
        """
        img_data = self.preprocess_frame(frame)
        outputs = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: img_data})
        return self.postprocess(frame, outputs)

    def detect_video(self):
        """
        处理视频并进行目标检测
        返回：
            output_path: 处理后的视频保存路径
        """
        # 打开视频文件
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            print(f"无法打开视频: {self.input_path}")
            return None
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 确定输出视频路径
        if self.output_is_file:
            # 检查输出文件扩展名是否与输入文件类型匹配
            _, input_ext = os.path.splitext(self.input_path)
            _, output_ext = os.path.splitext(self.output_path)
            if not self._is_compatible_extension(input_ext, output_ext):
                print(f"错误：输入文件为视频({input_ext})，但输出文件扩展名({output_ext})不兼容")
                return None
            output_video_path = self.output_path
        else:
            # 创建输出视频文件名
            base_name = os.path.basename(self.input_path)
            name, ext = os.path.splitext(base_name)
            output_video_path = os.path.join(self.output_path, f"detect_{name}{ext}")
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可以根据需要更改编码器
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        # frame_count = 0
        print(f"开始处理视频: {self.input_path}")
        print(f"总帧数: {total_frames}, FPS: {fps}")
        # import time
        # start_time = time.time()  # 记录开始时间
        # print(f"时间开始：{start_time}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 处理当前帧
            processed_frame = self.detect_frame(frame)
            # 写入处理后的帧
            out.write(processed_frame)
            # frame_count += 1
            # if frame_count % 10 == 0:  # 每10帧打印一次进度
            #     print(f"已处理 {frame_count}/{total_frames} 帧 ({frame_count/total_frames*100:.1f}%)")
        
        # 释放资源
        cap.release()
        out.release()
        
        # print(f"\n总耗时: {time.time() - start_time:.2f}秒")

        print(f"视频处理完成，已保存至: {output_video_path}")
        return output_video_path

    def _is_compatible_extension(self, input_ext, output_ext):
        """
        检查输入和输出文件扩展名是否兼容
        """
        input_ext = input_ext.lower()
        output_ext = output_ext.lower()
        
        # 视频扩展名列表
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        # 图像扩展名列表
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # 如果输入是视频，输出也应该是视频
        if input_ext in video_extensions:
            return output_ext in video_extensions
        # 如果输入是图像，输出也应该是图像
        elif input_ext in image_extensions:
            return output_ext in image_extensions
        
        # 默认情况下，认为不兼容
        return False

    def detect(self):
        """
        执行推理并保存结果
        返回：
            output_path: 处理后的图像或视频的保存路径
        """

        if self.is_video:
            return self.detect_video()
        else:
            # 处理图像
            img_data = self.preprocess()
            outputs = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: img_data})
            processed_img = self.postprocess(self.img, outputs)
            
            # 确定输出图像路径
            if self.output_is_file:
                # 检查输出文件扩展名是否与输入文件类型匹配
                _, input_ext = os.path.splitext(self.input_path)
                _, output_ext = os.path.splitext(self.output_path)
                if not self._is_compatible_extension(input_ext, output_ext):
                    print(f"错误：输入文件为图像({input_ext})，但输出文件扩展名({output_ext})不兼容")
                    return None
                output_image_path = self.output_path
            else:
                # 保存处理后的图像
                base_name = os.path.basename(self.input_path)
                name, ext = os.path.splitext(base_name)
                output_image_path = os.path.join(self.output_path, f"detect_{name}{ext}")
            
            cv2.imwrite(output_image_path, processed_img)

            #打印经过的时间
            

            print(f"图像处理完成，已保存至: {output_image_path}")
            return processed_img

    def main(self):
        # 使用 ONNX 模型创建推理会话，自动选择CPU或GPU
        print("YOLO12 🚀 目标检测 ONNXRuntime")
        print("模型名称：", self.onnx_model)
        print(f"模型输入尺寸：宽度 = {self.input_width}, 高度 = {self.input_height}")
        print(f"输入类型: {'视频' if self.is_video else '图像'}")
        print(f"输出路径: {self.output_path}")
        
        # 执行检测并返回保存路径
        return self.detect()
 
 
 
if __name__ == "__main__":
    # 创建参数解析器以处理命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=r"models\yolo12n.onnx", help="输入你的 ONNX 模型路径。")
    parser.add_argument("--input", type=str, default=r"D:\WorkSpace\fwwb_commit\outputSmoked3.mp4", help="输入图像或视频的路径。")
    parser.add_argument("--output", type=str, default=r"result", help="输出结果的保存路径。")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU 阈值")
    args = parser.parse_args()
 
    # 使用指定的参数创建 DetectionInterface 类的实例
    detection = DetectionInterface(args.model, args.input, args.output, args.conf_thres, args.iou_thres)
 
    # 执行目标检测并获取输出路径
    output_path = detection.main()
    print(f"处理完成，结果已保存至: {output_path}")
    
