# **Finding Lane Lines on the Road**

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

The pipleline goes like this:

1. Store the dimensions of the image (for use in for setting the masked region of interest)
2. Use the `grayscale()` helper function (which calls `cv2.cvtColor()`) to create a new image converted

![grayscale_image]: (https://github.com/gpjonesii/carnd-term1-project1/blob/master/image_tmp/gray-solidYellowCurve.jpg?raw=true)  "Grayscale"

3. Use the `gaussian_blur()` helper function (which calls `cv2.GaussianBlur()`), passing the grayscale image as input, returning a blurred image.

4. Use the `canny()` helper function (which calls `cv2.Canny()`), passing the blurred image as input, returning an image containing only detected edges.

5. Create an image containing a mask, using predefined vertices (partially computed from image dimensions), the canny edges image, and leveraging the helper function `region_of_interest()`

6.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image:

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ...

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
