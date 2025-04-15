import cv2
import numpy as np
import os
from tqdm import tqdm
import onnxruntime as ort


def dehaze(input, output = 'results/hhh.mp4', target_size=(640, 480)):
    """
    去雾处理函数，支持处理图片和视频

    参数：
    input -- 输入文件路径（支持视频/图片）
    output -- 输出文件保存路径（支持绝对路径、相对路径、纯文件名）
    target_size -- 输出分辨率，默认(640, 480)

    返回：
    True表示成功，False表示失败
    """

    # 初始化ONNX会话
    def _init_session():
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            return ort.InferenceSession('models/dehazer.onnx', providers=providers)
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return None

    # 帧处理函数
    def _process_frame(frame, session):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data_hazy = (frame_rgb / 255.0).astype(np.float32)
        data_hazy = np.transpose(data_hazy, (2, 0, 1))
        data_hazy = np.expand_dims(data_hazy, axis=0)

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        clean_image = session.run([output_name], {input_name: data_hazy})[0]
        clean_image = clean_image.squeeze(0).transpose(1, 2, 0)
        clean_image = np.clip(clean_image * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(clean_image, cv2.COLOR_RGB2BGR)

    # 判断文件类型
    def _is_image(path):
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        return os.path.splitext(path)[1].lower() in image_exts

    # ------------------------ 路径处理逻辑 ------------------------ #
    input_filename = os.path.basename(input)
    input_basename, input_ext = os.path.splitext(input_filename)

    output_basename = os.path.basename(output)
    output_has_extension = '.' in output_basename

    # 处理不同输出路径场景
    if os.path.isabs(output):
        # 绝对路径场景
        if output_has_extension:
            output_path = output
        else:
            output_dir = output
            output_filename = f"{input_basename}_dehazed{input_ext}"
            output_path = os.path.join(output_dir, output_filename)
    else:
        # 相对路径场景
        if output_has_extension:
            output_path = os.path.join(os.getcwd(), output)
        else:
            output_dir = os.path.join(os.getcwd(), output)
            output_filename = f"{input_basename}_dehazed{input_ext}"
            output_path = os.path.join(output_dir, output_filename)

    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ------------------------ 主逻辑 ------------------------ #
    # 验证输入路径
    if not os.path.exists(input):
        print(f"输入文件不存在：{input}")
        return False

    # 初始化模型
    session = _init_session()
    if session is None:
        return False

    try:
        # import time
        # start_time = time.time()  # 记录开始时间

        # 处理图片
        if _is_image(input):
            img = cv2.imread(input)
            if img is None:
                print(f"无法读取图片文件：{input}")
                return False

            processed = _process_frame(cv2.resize(img, target_size), session)
            cv2.imwrite(output_path, processed)
            print(f"图片处理完成：{output_path}")
            return True

        # 处理视频
        else:
            cap = cv2.VideoCapture(input)
            if not cap.isOpened():
                print(f"无法打开视频文件：{input}")
                return False

            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                target_size
            )

            pbar = tqdm(total=total, desc="Processing")
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    processed = _process_frame(cv2.resize(frame, target_size), session)
                    writer.write(processed)
                    pbar.update(1)
            finally:
                cap.release()
                writer.release()
                pbar.close()

            print(f"视频处理完成：{output_path}")

            # print(f"处理时间：{time.time() - start_time:.4f} 秒")
            
            return True

    except Exception as e:
        print(f"处理过程中发生错误：{str(e)}")
        return False


# 使用示例
if __name__ == '__main__':
    # 示例调用
    #dehaze("/home/rophing/Datasets/waibao1/output_rgb_smoked1.mp4", "/home/rophing/Datasets/waibao1/output.mp4")  # 绝对路径+文件名
    #dehaze("/home/rophing/Datasets/waibao1/output_rgb_smoked1.mp4", "output.mp4")  # 纯文件名
    dehaze("demo/original_video.mp4")  # 绝对路径目录

