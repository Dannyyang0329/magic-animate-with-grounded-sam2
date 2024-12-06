import ffmpeg
import argparse

def get_video_resolution(input_file):
    """使用 ffprobe 獲取影片解析度"""
    probe = ffmpeg.probe(input_file)
    video_stream = next(
        (stream for stream in probe["streams"] if stream["codec_type"] == "video"),
        None,
    )
    if video_stream is None:
        raise ValueError("找不到影片流！")
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    return width, height

def video_edit(input_file, output_file):
    # 獲取輸入影片的解析度
    width, height = get_video_resolution(input_file)

    # 確保裁切區域為正方形，選擇較小的邊長
    square_size = min(width, height)
    crop_x = f"(in_w-{square_size})/2"  # 水平居中
    crop_y = f"(in_h-{square_size})/2"  # 垂直居中

    # 使用 ffmpeg 裁切、縮放、轉換
    ffmpeg.input(input_file).filter(
        "crop", out_w=square_size, out_h=square_size, x=crop_x, y=crop_y
    ).filter(
        "scale", 512, 512  # 縮放到 512x512
    ).output(
        output_file,
        vcodec="libx264",  # 編碼器 H.264
        r=25,  # 幀率 25 fps
        format="mp4",  # 輸出格式
    ).run()

    
    #video_bitrate="294k",  # 比特率 294 kbps

    print("處理完成！輸出檔案為：", output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="輸入影片檔案，例如 input.webm")
    parser.add_argument("--output", type=str, required=True, help="輸出影片檔案，例如 output.mp4")

    args = parser.parse_args()

    # 確保輸出檔案的副檔名為 .mp4
    if not args.output.endswith(".mp4"):
        print("警告：輸出的檔案必須是 .mp4 格式，已自動修正為 .mp4")
        args.output += ".mp4"

    video_edit(args.input, args.output)
