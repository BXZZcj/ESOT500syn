import cv2
import os
import argparse
import glob


def create_video_from_images(input_dir, output_path, fps):
    """
    从一个包含有序命名图片的文件夹创建一个视频。

    参数:
    - input_dir (str): 包含PNG图片的文件夹路径。
    - output_path (str): 输出视频文件的路径 (例如 'output.mp4')。
    - fps (int): 输出视频的帧率 (每秒帧数)。
    """
    # --- 1. 验证输入路径 ---
    if not os.path.isdir(input_dir):
        print(f"错误：输入文件夹 '{input_dir}' 不存在。")
        return


    # --- 2. 查找并排序所有PNG图片 ---
    # 使用glob查找所有.png文件，然后排序以确保帧顺序正确
    image_pattern = os.path.join(input_dir, '*.png')
    image_files = sorted(glob.glob(image_pattern))


    if not image_files:
        print(f"错误：在文件夹 '{input_dir}' 中未找到任何 .png 图片。")
        return


    print(f"找到了 {len(image_files)} 张图片，将开始合成视频...")


    # --- 3. 获取图片尺寸以初始化视频写入器 ---
    try:
        first_frame = cv2.imread(image_files[0])
        height, width, layers = first_frame.shape
        size = (width, height)
    except Exception as e:
        print(f"错误：无法读取第一张图片 '{image_files[0]}' 以获取尺寸。错误信息: {e}")
        return


    # --- 4. 初始化视频写入器 (VideoWriter) ---
    # 定义视频编码器，'mp4v' 适用于 .mp4 文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    # 创建VideoWriter对象
    try:
        out = cv2.VideoWriter(output_path, fourcc, fps, size)
    except Exception as e:
        print(f"错误：无法初始化视频写入器。请检查输出路径 '{output_path}' 是否有效。错误信息: {e}")
        return


    # --- 5. 循环读取图片并写入视频 ---
    for i, filename in enumerate(image_files):
        # 读取一帧
        frame = cv2.imread(filename)
        # 写入视频
        out.write(frame)
        
        # 打印进度
        if (i + 1) % 50 == 0:
            print(f"  ...已处理 {i + 1}/{len(image_files)} 帧")


    # --- 6. 释放资源 ---
    out.release()
    cv2.destroyAllWindows()


    print("-" * 30)
    print(f"视频合成成功！")
    print(f"文件已保存至: {os.path.abspath(output_path)}")
    print("-" * 30)



if __name__ == "__main__":
    # --- 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(
        description="将一个文件夹中的PNG序列图像转换为一个视频文件。",
        formatter_class=argparse.RawTextHelpFormatter  # 保持帮助信息格式
    )
    
    parser.add_argument(
        '-i', '--input_dir', 
        type=str, 
        default="/home/chujie/Data/ESOT500syn/test/images/archithor_ablation_test_no_robot", 
        help="包含PNG图片的输入文件夹路径。\n图片应按序列命名，例如 00000.png, 00001.png, ..."
    )
    
    parser.add_argument(
        '-o', '--output_path', 
        type=str, 
        default="/home/chujie/Data/ESOT500syn/test/video.mp4", 
        help="输出视频文件的完整路径，包括文件名和后缀。\n例如: /path/to/your/video.mp4"
    )
    
    parser.add_argument(
        '-f', '--fps', 
        type=int, 
        default=33,  # 注意原代码用33.333但int类型会截断，改为33
        help="输出视频的帧率 (FPS)。\n默认值为 30。"
    )


    args = parser.parse_args()


    # 调用主函数
    create_video_from_images(args.input_dir, args.output_path, args.fps)
