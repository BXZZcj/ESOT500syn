import cv2
import os
import argparse
import glob


def create_video_from_images(input_dir, output_path, fps):
    """
    create a video from a folder containing sequentially named images.

    parameters:
    - input_dir (str): the path to the folder containing the PNG images.
    - output_path (str): the path to the output video file (e.g. 'output.mp4').
    - fps (int): the frame rate of the output video (frames per second).
    """
    # --- 1. validate input path ---
    if not os.path.isdir(input_dir):
        print(f"error: input folder '{input_dir}' does not exist.")
        return


    # --- 2. find and sort all PNG images ---
    # use glob to find all .png files, then sort to ensure correct frame order
    image_pattern = os.path.join(input_dir, '*.png')
    image_files = sorted(glob.glob(image_pattern))


    if not image_files:
        print(f"error: no .png images found in folder '{input_dir}'.")
        return


    print(f"found {len(image_files)} images, starting to create video...")


    # --- 3. get image size to initialize video writer ---
    try:
        first_frame = cv2.imread(image_files[0])
        height, width, layers = first_frame.shape
        size = (width, height)
    except Exception as e:
        print(f"error: cannot read the first image '{image_files[0]}' to get the size. error: {e}")
        return


    # --- 4. initialize video writer (VideoWriter) ---
    # define video encoder, 'mp4v' is suitable for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    
    # create VideoWriter object
    try:
        out = cv2.VideoWriter(output_path, fourcc, fps, size)
    except Exception as e:
        print(f"error: cannot initialize video writer. please check if the output path '{output_path}' is valid. error: {e}")
        return


    # --- 5. loop through images and write to video ---
    for i, filename in enumerate(image_files):
        # read a frame
        frame = cv2.imread(filename)
        # write to video
        out.write(frame)
        
        # print progress
        if (i + 1) % 50 == 0:
            print(f"  ...processed {i + 1}/{len(image_files)} frames")


    # --- 6. release resources ---
    out.release()
    cv2.destroyAllWindows()


    print("-" * 30)
    print(f"video created successfully!")
    print(f"file saved to: {os.path.abspath(output_path)}")
    print("-" * 30)



if __name__ == "__main__":
    # --- set command line argument parsing ---
    parser = argparse.ArgumentParser(
        description="convert a folder of PNG sequence images into a video file.",
        formatter_class=argparse.RawTextHelpFormatter  # keep help information format
    )
    
    parser.add_argument(
        '-i', '--input_dir', 
        type=str, 
        default="/home/chujie/Data/ESOT500syn/test/images/archithor_ablation_test_no_robot", 
        help="the path to the input folder containing the PNG images.\nimages should be named sequentially, e.g. 00000.png, 00001.png, ..."
    )
    
    parser.add_argument(
        '-o', '--output_path', 
        type=str, 
        default="/home/chujie/Data/ESOT500syn/test/video.mp4", 
        help="the full path to the output video file, including the file name and extension.\nfor example: /path/to/your/video.mp4"
    )
    
    parser.add_argument(
        '-f', '--fps', 
        type=int, 
        default=33,  # note that the original code uses 33.333 but int type will truncate it, so we use 33
        help="the frame rate (FPS) of the output video.\ndefault value is 30."
    )


    args = parser.parse_args()


    # call the main function
    create_video_from_images(args.input_dir, args.output_path, args.fps)
