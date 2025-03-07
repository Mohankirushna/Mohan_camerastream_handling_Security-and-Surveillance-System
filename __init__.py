
# ---------------------- create a video with just a image --------------------------------------
# from moviepy.editor import ImageSequenceClip
# import os

# image_files = [r'c:\Users\kachr\Downloads\image.png']*90
# fps = 24
# clip = ImageSequenceClip(image_files, fps=fps)
# clip = clip.set_duration(10)
# output_video = "tests/assets/output_video.mp4"
# clip.write_videofile(output_video, codec='libx264')
# -----------------------------------------------------------------------------------------------

from tests import Integration_test
# from tests import scene_understanding_test
# from tests import logger_test


Integration_test.main()
# scene_understanding_test.test_clip_model() # 605MB model
# scene_understanding_test.test_blip_model() # 990MB model
# scene_understanding_test.test_ollava()

# print("-"*50, "Running test for YOLO image processing", "-"*50)
# image_normalizer.test_normalize_image_for_yolo()
# image_normalizer.test_denoise_and_sharpen()
# image_normalizer.test_preprocess_image_for_yolo()
# print("All tests passed.")

# print("-"*50, "Running test for YOLO object detection", "-"*50)
