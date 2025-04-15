import cv2
import os
import numpy as np
import onnxruntime as ort


class FusionInterface:
    """视频融合接口类，用于处理RGB和IR视频的对齐、融合和检测。"""
    def __init__(self, mode='fuse', rgb_path=None, ir_path=None, output='results', 
                 ir_ahead=0, fusefunc='densefuse', confidence_thres=0.25, iou_thres=0.45, rknn=False):
        """
        初始化 FusionInterface 类的实例。
        参数：
            mode: 处理模式，'fuse'或'fuse_and_detect'
            rgb_path: RGB视频的路径
            ir_path: IR视频的路径
            output: 输出结果的保存路径
            ir_ahead: IR视频早于RGB视频的秒数，可为负数
            fusefunc: 融合函数名称，可选值为'densefuse'或'mfeif'
            confidence_thres: 检测置信度阈值（仅在fuse_and_detect模式下有效）
            iou_thres: IOU阈值（仅在fuse_and_detect模式下有效）
        """
        self.mode = mode
        self.rgb_path = rgb_path
        self.ir_path = ir_path
        self.output_path = output
        self.ir_ahead = ir_ahead if ir_ahead is not None else 0
        self.fusefunc_name = fusefunc
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.rknn = rknn
        if self.rknn:
            from .DetectionInterfaceRKNN import RKNNDetection
            self.detection_interface = RKNNDetection(input_path = None, output_path = self.output_path, confidence_thres = self.confidence_thres, iou_thres = self.iou_thres)
        else:
            from .DetectionInterface import DetectionInterface
            self.detection_interface = DetectionInterface(input_path = None, output = self.output_path, confidence_thres = self.confidence_thres, iou_thres = self.iou_thres)
        # 判断输出路径是文件还是目录
        self.output_is_file = False
        if self.output_path:
            _, ext = os.path.splitext(self.output_path)
            self.output_is_file = ext != ''
        
        # 确保输出目录存在
        if self.output_is_file:
            # 如果是文件，确保其父目录存在
            output_dir = os.path.dirname(self.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
        else:
            # 如果是目录，确保目录存在
            os.makedirs(self.output_path, exist_ok=True)
        
        # 根据fusefunc_name选择融合函数
        if self.fusefunc_name == 'densefuse':
            self.fusion_func = self.densefuse_onnx
            # 加载DenseFuse模型
            self.encoder_session = ort.InferenceSession(
                r"./models/model_gray_encoder.onnx",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
            )
            self.decoder_session = ort.InferenceSession(
                r"./models/model_gray_decoder.onnx",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
            )
        elif self.fusefunc_name == 'mfeif':
            self.fusion_func = self.mfeif_onnx
            # 加载mfeif模型
            self.mfeif_session = ort.InferenceSession(
                r"./models/mfeif.onnx",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
            )
        
        # 如果是fuse_and_detect模式，初始化检测接口
        if self.mode == 'fuse_and_detect':
            # 在这里导入DetectionInterface
            from utils.DetectionInterface import DetectionInterface
            self.detection_interface = DetectionInterface(
                input_path=None,  # 将在处理每一帧时动态设置
                output=self.output_path,
                confidence_thres=self.confidence_thres,
                iou_thres=self.iou_thres
            )
    
    def densefuse_onnx(self, rgb_frame, ir_frame):
        """
        使用DenseFuse模型融合RGB和IR帧
        参数：
            rgb_frame: RGB帧
            ir_frame: IR帧
        返回：
            融合后的帧
        """
        # 获取输入输出名称
        encoder_input_name = self.encoder_session.get_inputs()[0].name
        encoder_output_name = self.encoder_session.get_outputs()[0].name
        decoder_input_name = self.decoder_session.get_inputs()[0].name
        decoder_output_name = self.decoder_session.get_outputs()[0].name
    
        # 转换为灰度图像
        if len(rgb_frame.shape) == 3:
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        if len(ir_frame.shape) == 3:
            ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
    
        # 使用OpenCV和NumPy替代torchvision的transforms
        # 归一化图像数据 (0-255 -> 0-1)
        ir_norm = ir_frame.astype(np.float32) / 255.0
        vis_norm = rgb_frame.astype(np.float32) / 255.0
        
        # 添加通道维度 (H,W) -> (1,H,W)
        ir_tensor = np.expand_dims(ir_norm, axis=0)
        vis_tensor = np.expand_dims(vis_norm, axis=0)
        
        # 添加批次维度 (1,H,W) -> (1,1,H,W)
        ir_tensor = np.expand_dims(ir_tensor, axis=0)
        vis_tensor = np.expand_dims(vis_tensor, axis=0)
        
        # 使用编码器
        ir_features = self.encoder_session.run([encoder_output_name], {encoder_input_name: ir_tensor})[0]
        vis_features = self.encoder_session.run([encoder_output_name], {encoder_input_name: vis_tensor})[0]
        # 融合特征
        fusion_features = (ir_features + vis_features) / 2
        # 使用解码器
        fusion_image = self.decoder_session.run([decoder_output_name], {decoder_input_name: fusion_features})[0]
        # 处理融合图像
        fusion_image = fusion_image[0]  # 移除批次维度
        fusion_image = np.transpose(fusion_image, (1, 2, 0))  # CHW -> HWC
        fusion_image = (fusion_image * 255).clip(0, 255).astype(np.uint8)
        # 如果是单通道图像，移除通道维度
        if fusion_image.shape[2] == 1:
            fusion_image = fusion_image[:, :, 0]
        # 转换回三通道图像用于视频保存
        if len(fusion_image.shape) == 2:
            fusion_image = cv2.cvtColor(fusion_image, cv2.COLOR_GRAY2BGR)
        return fusion_image
    
    def mfeif_onnx(self, vi_input, ir_input):
        """
        使用mfeif模型融合可见光和红外图像
        参数:
            vi_input: 可见光图像(numpy数组)
            ir_input: 红外图像(numpy数组)
        返回:
            融合后的图像(numpy数组，BGR格式)
        """
        session = self.mfeif_session
        input_names = [input.name for input in session.get_inputs()]
        output_names = [output.name for output in session.get_outputs()]
        
        # 复制输入图像
        vi_image = vi_input.copy()
        ir_image = ir_input.copy()
        
        # 转换为灰度图
        if len(vi_image.shape) == 3:
            vi_image = cv2.cvtColor(vi_image, cv2.COLOR_BGR2GRAY)
        if len(ir_image.shape) == 3:
            ir_image = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
        
        # 图像预处理
        # 归一化
        ir_tensor = ir_image.astype(np.float32) / 255.0
        vi_tensor = vi_image.astype(np.float32) / 255.0
        
        # 转换为CHW格式
        if len(ir_tensor.shape) == 2:
            ir_tensor = np.expand_dims(ir_tensor, axis=0)
        else:
            ir_tensor = np.transpose(ir_tensor, (2, 0, 1))
            
        if len(vi_tensor.shape) == 2:
            vi_tensor = np.expand_dims(vi_tensor, axis=0)
        else:
            vi_tensor = np.transpose(vi_tensor, (2, 0, 1))
        
        # 添加批次维度
        ir_tensor = np.expand_dims(ir_tensor, axis=0)
        vi_tensor = np.expand_dims(vi_tensor, axis=0)
        
        # 准备输入
        ort_inputs = {
            input_names[0]: ir_tensor,
            input_names[1]: vi_tensor
        }
        
        # 运行推理
        ort_outputs = session.run(output_names, ort_inputs)
        fused = ort_outputs[0]
        
        # 处理融合图像
        fused = fused[0]  # 移除批次维度
        
        # 如果是单通道图像
        if fused.shape[0] == 1:
            # 移除通道维度
            fused_image = fused[0]
        else:
            # 转换为HWC格式
            fused_image = np.transpose(fused, (1, 2, 0))
        
        # 反归一化并转换为uint8
        fused_image = (fused_image * 255.0).clip(0, 255).astype(np.uint8)
        
        # 转换回三通道图像
        if len(fused_image.shape) == 2:
            fused_image = cv2.cvtColor(fused_image, cv2.COLOR_GRAY2BGR)
        
        return fused_image
    
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
    
    def fuse(self):
        """
        执行视频对齐和融合，如果是fuse_and_detect模式还会进行检测
        返回：
            output_path: 处理后的视频保存路径
        """
        # 打开视频文件
        ir_cap = cv2.VideoCapture(self.ir_path)
        rgb_cap = cv2.VideoCapture(self.rgb_path)
        
        if not ir_cap.isOpened() or not rgb_cap.isOpened():
            print(f"无法打开视频文件: {self.ir_path} 或 {self.rgb_path}")
            return None
        
        # 获取视频信息
        rgb_fps = 25
        ir_fps = 30
        
        # 计算总帧数
        rgb_frame_count = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ir_frame_count = int(ir_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 设置输出视频的宽高为处理后的尺寸
        output_width = 640  # 裁剪后的宽度 (672-32)
        output_height = 418  # 裁剪后的高度 (485-66)
        
        # 创建输出视频文件名
        base_name = os.path.basename(self.rgb_path)
        name, ext = os.path.splitext(base_name)
        
        # 根据输出路径是文件还是目录来确定输出视频路径
        if self.output_is_file:
            # 检查输出文件扩展名是否与输入文件类型匹配
            _, output_ext = os.path.splitext(self.output_path)
            if not self._is_compatible_extension(ext, output_ext):
                print(f"错误：输入文件为视频({ext})，但输出文件扩展名({output_ext})不兼容")
                return None
            output_video_path = self.output_path
        else:
            output_video_path = os.path.join(self.output_path, f"fused_{name}{ext}")
        
        # 创建输出视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, rgb_fps, (output_width, output_height))
        
        # 预读取所有IR帧及其时间戳
        ir_frames = []
        ir_timestamps = []
        
        print("正在读取IR视频帧...")
        frame_idx = 0
        while True:
            ret, frame = ir_cap.read()
            if not ret:
                break
            
            # 显示IR帧读取进度
            print(f"\r读取IR帧进度: {frame_idx + 1}/{ir_frame_count} ({(frame_idx + 1)/ir_frame_count*100:.1f}%)", end="")
            
            # 计算实际时间戳
            actual_timestamp = frame_idx * (1.0 / ir_fps)
            ir_frames.append(frame)
            ir_timestamps.append(actual_timestamp)
            frame_idx += 1
        
        print(f"\nIR视频共有 {len(ir_frames)} 帧")
        
        # 处理RGB视频的每一帧
        print("\n正在处理RGB视频帧...")
        rgb_frame_idx = 0
        successful_fusions = 0  # 记录成功融合的帧数
        time_threshold = 0.05  # 时间差阈值，单位为秒

        import time
        start_time = time.time()  # 记录开始时间
        print("开始时间:", start_time)

        while True:
            ret, rgb_frame = rgb_cap.read()
            if not ret:
                break
                
            # 显示处理进度
            print(f"\r处理进度: {rgb_frame_idx + 1}/{rgb_frame_count} ({(rgb_frame_idx + 1)/rgb_frame_count*100:.1f}%), 已成功融合: {successful_fusions}帧", end="")
            
            # 计算RGB帧的时间戳（考虑时间差）
            rgb_timestamp = rgb_frame_idx * (1.0 / rgb_fps) - self.ir_ahead
            
            # 直接计算预估的IR帧索引
            estimated_ir_idx = int(rgb_timestamp * ir_fps)
            
            # 如果预估索引已经超出IR帧范围，则跳过当前RGB帧
            if estimated_ir_idx >= len(ir_timestamps) or estimated_ir_idx < 0:
                print(f"\n跳过RGB帧 {rgb_frame_idx+1}/{rgb_frame_count}, IR帧索引超出范围")
                rgb_frame_idx += 1
                continue
            
            # 在预估位置附近查找最佳匹配（搜索范围设为前后5帧）
            search_start = max(0, estimated_ir_idx - 5)
            search_end = min(len(ir_timestamps), estimated_ir_idx + 5)
            
            # 确保搜索范围有效
            if search_start >= len(ir_timestamps) or search_end <= search_start:
                print(f"\n跳过RGB帧 {rgb_frame_idx+1}/{rgb_frame_count}, 无效的搜索范围")
                rgb_frame_idx += 1
                continue
                
            # 在局部范围内找最佳匹配
            closest_ir_idx = search_start
            min_time_diff = abs(ir_timestamps[search_start] - rgb_timestamp)
            
            for i in range(search_start + 1, search_end):
                time_diff = abs(ir_timestamps[i] - rgb_timestamp)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_ir_idx = i
            
            # 如果找到时间差在阈值内的IR帧，则进行融合
            if min_time_diff <= time_threshold:
                # 获取对应的IR帧
                ir_frame = ir_frames[closest_ir_idx]
                
                # 对RGB帧进行调整大小和裁剪
                rgb_resized = cv2.resize(rgb_frame, (704, 419), cv2.INTER_AREA)
                rgb_cropped = rgb_resized[:, 32:672]
                
                # 对IR帧进行裁剪
                ir_cropped = ir_frame[66:485, :]
                
                # 确保IR帧与RGB帧大小一致
                if ir_cropped.shape[1] != rgb_cropped.shape[1] or ir_cropped.shape[0] != rgb_cropped.shape[0]:
                    ir_cropped = cv2.resize(ir_cropped, (rgb_cropped.shape[1], rgb_cropped.shape[0]))
                
                # 将处理后的帧送入融合函数
                fused_frame = self.fusion_func(rgb_cropped, ir_cropped)
                
                # 确保融合后的帧尺寸正好是640x418
                if fused_frame.shape[1] != 640 or fused_frame.shape[0] != 418:
                    fused_frame = cv2.resize(fused_frame, (640, 418), cv2.INTER_AREA)
                
                # 如果是fuse_and_detect模式，对融合后的帧进行检测
                if self.mode == 'fuse_and_detect':
                    # 直接使用detect_frame方法处理融合后的帧，避免保存和重新加载
                    if self.rknn:
                        detected_frame = self.detection_interface.detect_frame(fused_frame)
                    else:
                        detected_frame = self.detection_interface.detect_frame(fused_frame)
                    
                    # 写入检测后的帧
                    out.write(detected_frame)
                else:
                    # 直接写入融合后的帧
                    out.write(fused_frame)
                
                successful_fusions += 1
            else:
                print(f"\n跳过RGB帧 {rgb_frame_idx+1}/{rgb_frame_count}, 未找到匹配的IR帧 (最小时间差: {min_time_diff:.4f}秒)")
            
            rgb_frame_idx += 1
        
        # 计算并打印总耗时
        print(f"\n总耗时: {time.time() - start_time:.2f}秒")

        # 释放资源
        ir_cap.release()
        rgb_cap.release()
        out.release()
        
        print(f"\n处理完成，共处理 {rgb_frame_count} 帧，成功融合 {successful_fusions} 帧 ({successful_fusions/rgb_frame_count*100:.1f}%)，输出视频保存至 {output_video_path}")
        return output_video_path
    
    def main(self):
        """
        主函数，执行融合和检测
        返回：
            output_path: 处理后的视频保存路径
        """
        print(f"视频融合 🚀 模式: {self.mode}")
        print(f"RGB视频: {self.rgb_path}")
        print(f"IR视频: {self.ir_path}")
        print(f"输出路径: {self.output_path}")
        print(f"IR提前时间: {self.ir_ahead}秒")
        print(f"融合函数: {self.fusefunc_name}")
        
        if self.mode == 'fuse_and_detect':
            print(f"检测置信度阈值: {self.confidence_thres}")
            print(f"IOU阈值: {self.iou_thres}")
        
        # 执行融合并返回保存路径
        return self.fuse()

if __name__ == "__main__":
    # 添加项目根目录到 Python 路径，以便能够正确导入模块
    import sys
    import os
    # 获取当前文件所在目录的父目录（项目根目录）
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 将项目根目录添加到 Python 路径
    sys.path.append(root_dir)
    
    # 在这里导入DetectionInterface用于测试
    from utils.DetectionInterface import DetectionInterface
    
    # 测试代码
    fusion = FusionInterface(
        mode='fuse_and_detect',
        rgb_path=r"demo\output_rgb_smoked3.mp4",
        ir_path=r"demo\output_tr_smoked3.mp4",
        output=r"results",
        ir_ahead=-4.55,
        fusefunc='densefuse'
    )
    output_path = fusion.main()
    print(f"处理完成，结果已保存至: {output_path}")