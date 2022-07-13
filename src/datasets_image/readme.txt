# Notes
# The currently included data generator is designed for the  FlyingThings3D RGB images (cleanpass) downloaded as WebP lossy format
# https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html#downloads

# To use the image generator included in DFlat without modification, download the "flyingthings3d_frames_cleanpass_webp" and the
# "flyingthings3d__object_index" from the above link and place here. To avoid needing to change relative path strings in the image_layer, 
# ensure that the datapath is like the following example:
# " datasets_image/flyingthings3d__frames_cleanpass_webp/frames_cleanpass_webp/* "
# " datasets_image/flyingthings3d__object_index/object_index/* "

# The COCO dataset is also useful as a library of common objects in images rather than synthetic scenes: Download at 
# https://cocodataset.org/#download
