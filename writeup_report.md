**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 1st through and 5th code cells of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text](https://github.com/tlapinsk/CarND-Vehicle-Detection/blob/master/example_images/car_noncar.png?raw=true "Car and Non-Car")

I then utilized the code provided by Udacity to create a couple example HOG pictures. See below for an example:

![alt text](https://github.com/tlapinsk/CarND-Vehicle-Detection/blob/master/example_images/HOG.png?raw=true "HOG")

I then explored extracting features in code cell 6 and explored different color spaces / parameters in code cell 7. In particular, I tried `HLS` and `YCrCb` because many other students had success and recommended these on the forums. 

I settled on the following parameters:

```# Define parameters for feature extraction
color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 6 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off```

2. Explain how you settled on your final choice of HOG parameters.

I tried many different variations of HOG parameters before settling on my final choice. In particular, I played around with `HLS` and `YCrCb` by running the test video. I even ran a full test on the final video with both color spaces, and `HLS` always outperformed `YCrCb`.

I also attempted changing the `orient`, `pix_per_cell`, `spatial_size`, and `hist_bins`. I found that these helped determine how quickly the video was processed and settled on the following parameters. As a note, it took ~1-2 hours each time when processing the video.

```# Define parameters for feature extraction
color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 6 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off```

3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in code cell 7 using LinearSVC as recommended in the lessons. It took ~45 seconds to train SVC and I achieved ~98%-99% each time. Although, I found that `HLS` and 98% performed stronger than `YCrCb` at 99%. This may be due to overfitting.

### Sliding Window Search

1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I performed a sliding window search in code cell 8. In my final pipeline, I utilized the following parameters:

``` xy_window = [[80, 80], [96, 96], [128, 128]]
    x_start_stop = [[200, None], [200, None], [412, 1280]]
    y_start_stop = [[390, 540], [400, 600], [400, 640]]```

2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To optimize performance of pipeline, I decided to ultimately landed on using `HLS` 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. That gave this result:

![alt text](https://github.com/tlapinsk/CarND-Vehicle-Detection/blob/master/example_images/classify.png?raw=true "Example 1")

![alt text](https://github.com/tlapinsk/CarND-Vehicle-Detection/blob/master/example_images/classify2.png?raw=true "Example 2")
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In code cells 12 and 13, you can see example code for how I detected false positives. I utilized Udacity's recommendation of creating a heat map and then thresholding that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. It is a fair assumption that each blog corresponds to a vehicle, so I constructed bounding boxes to cover the area of each blog detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

![alt text](https://github.com/tlapinsk/CarND-Advanced-Lane-Lines/blob/master/example_images/undistort_chessboard.png?raw=true "Undistorted chessboard")

---

###Discussion

1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

When working through the initial stages, I found the pipeline performance to be slow. Even as it stands in my final implementation, this is not nearly close to being a real time system. It takes at least 1.5 hours to process only 50 seconds of video - I can't imagine trying to utilize this pipeline in real time. 

Resources for slow pipeline:
- https://discussions.udacity.com/t/ways-to-improve-processing-time/237941/32

I also found the final output to be imperfect. Even watching other students videos, there are clearly many false positives and the bounding boxes are never perfect. If I were to take this project further, I would definitely research how to more accuractly detect cars. My final solution was to ignore a large part of the video, which is definitely not suitable for a true self-driving car.

Resources for false positives:
- https://discussions.udacity.com/t/continuous-false-positives/387288/2
- https://discussions.udacity.com/t/my-accuracy-is-pretty-high-but-seems-like-i-have-too-many-false-positives/363759/10
- https://discussions.udacity.com/t/false-negatives/327725/4
- https://discussions.udacity.com/t/false-positives/242618/4
- https://discussions.udacity.com/t/way-too-many-false-positives/302929/8
- https://discussions.udacity.com/t/false-positives-for-shadows/310077
- https://discussions.udacity.com/t/kitti-images-work-for-training-but-not-detection/243918/4?u=tim.lapinskas

I implemented a bounding box averaging system using deque. Check out [this post](https://discussions.udacity.com/t/wobbly-box-during-video-detection/231487/20?u=tim.lapinskas) to see why I chose to try this out. I found that it increased the number of false positives, so I decided to scrap it in my final video. It would be interesting to attempt implementing this via a class as suggested [here](https://discussions.udacity.com/t/wobbly-box-during-video-detection/231487/4?u=tim.lapinskas).

I am very curious to see how deep learning can be used to detect cars. Based on a couple other students input, it is very doable. Check out these repos to see how Udacity students utilized deep learning for this project: [here](https://github.com/xslittlegrass/CarND-Vehicle-Detection/blob/master/vehicle%20detection.ipynb) and [here](https://github.com/subodh-malgonde/vehicle-detection/blob/master/Vehicle_Detection.ipynb).

[This Medium article](https://medium.com/towards-data-science/vehicle-detection-and-tracking-44b851d70508) also is a great writeup for utilizing CNNs for vehicle detection.

Miscellaneous resources:
- https://discussions.udacity.com/t/better-explanation-of-scale-parameter-in-hog-sub-sampling/381895
- https://docs.python.org/2/library/collections.html