import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='处理图像的工具')
    
    # 添加模式参数，支持全名和缩写
    parser.add_argument('--mode', '-m', type=str, required=True, 
                        choices=['smoke', 's', 'fuse', 'f', 'detect', 'd', 'fuse_and_detect', 'fd'],
                        help='处理模式: smoke(s), fuse(f), detect(d), fuse_and_detect(fd)')
    parser.add_argument('--input', '-i', type=str, required=True, nargs='+', default=r"D:\WorkSpace\fwwb_yolo\frames\output_frames\3874.jpg",
                        help='输入文件路径，smoke/detect模式需要1个参数，fuse/fuse_and_detect模式需要2个参数')
    parser.add_argument('--output', '-o', type=str, default="results",
                        help='输出文件路径（可选）')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='检测置信度阈值（仅在detect或fuse_and_detect模式下有效）')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IOU阈值（仅在detect或fuse_and_detect模式下有效）')
    parser.add_argument('--ir_ahead', '-a', type=float, 
                        help='ir视频早于rgb视频的秒数，可为负数')
    parser.add_argument('--fusefunc', '-f', type=str, default='densefuse',
                        choices=['densefuse', 'mfeif'],
                        help='双模态视频融合函数，可选值为densefuse, mfeif')
                            # 添加RKNN参数选项
    parser.add_argument('--rknn', '-r', action='store_true', default=False,
                        help='使用RKNN模型进行检测')
    args = parser.parse_args()
    # 将缩写模式转换为全名
    mode_mapping = {'s': 'smoke', 'f': 'fuse', 'd': 'detect', 'fd': 'fuse_and_detect'}
    if args.mode in mode_mapping:
        args.mode = mode_mapping[args.mode]
    # 验证输入参数数量
    if args.mode in ['smoke', 'detect'] and len(args.input) != 1:
        parser.error(f"{args.mode}模式需要提供1个输入文件路径")
    elif args.mode in ['fuse', 'fuse_and_detect'] and len(args.input) != 2:
        parser.error(f"{args.mode}模式需要提供2个输入文件路径")
    # 验证输入文件是否存在
    for input_path in args.input:
        if not os.path.exists(input_path):
            parser.error(f"输入文件不存在: {input_path}")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    print(f"模式: {args.mode}")
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"融合函数: {args.fusefunc}")
    print(f"ir视频早于rgb视频的秒数: {args.ir_ahead}")
    
    # 如果是检测相关模式，打印检测参数
    if args.mode in ['detect', 'fuse_and_detect']:
        print(f"检测置信度阈值: {args.conf}")
        print(f"IOU阈值: {args.iou}")    
    

    if args.mode == 'detect':
        if args.rknn:
            from utils.DetectionInterfaceRKNN import RKNNDetection
            detection_interface = RKNNDetection(input_path = args.input[0], output_path = args.output, confidence_thres = args.conf, iou_thres = args.iou)
            detection_interface.detect()
        else:
            from utils.DetectionInterface import DetectionInterface
            detection_interface = DetectionInterface(input_path = args.input[0], output = args.output, confidence_thres = args.conf, iou_thres = args.iou)
            detection_interface.detect()
    elif args.mode == 'fuse' or args.mode == 'fuse_and_detect':
        from utils.FusionInterface import FusionInterface
        fusion_interface = FusionInterface(mode = args.mode, rgb_path = args.input[0], ir_path = args.input[1], output = args.output, ir_ahead = args.ir_ahead, 
        fusefunc = args.fusefunc, confidence_thres = args.conf, iou_thres = args.iou,rknn=args.rknn)
        fusion_interface.fuse()
    elif args.mode == 'smoke':
        from utils.dehaze import dehaze
        dehaze(args.input[0], args.output)