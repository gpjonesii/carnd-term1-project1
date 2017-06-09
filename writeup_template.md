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

![grayscale_image][./image_tmp/gray-solidYellowCurve.jpg]

3. Use the `gaussian_blur()` helper function (which calls `cv2.GaussianBlur()`), passing the grayscale image as input, returning a blurred image.

4. Use the `canny()` helper function (which calls `cv2.Canny()`), passing the blurred image as input, returning an image containing only detected edges.

5. Create an image containing a mask, using predefined vertices (partially computed from image dimensions), the canny edges image, and leveraging the helper function `region_of_interest()`, which returns an image containing edges within only the masked region.

6. Pass the masked edges image to the `hough_lines()` helper function. I used the following parameters: `masked_edges_image, rho=1, theta=np.pi / 180, threshold=15, min_line_len=40, max_line_gap=15`

`hough_lines()` finds lines in the masked edge image (using `cv2.HoughLinesP()`) then passes the lines to the `draw_lines()` helper function, which draws the lines on an empty image.

7. Pass the "lines_image" and the original image to the `weighted_img()` helper function, and then return as a result of `image_pipeline()`

#### Modifying the `draw_lines()` function

1. First, I created a couple of lambas that I could use to determine the slope of a line and find an x value, given a slope and another point:
```
slope = lambda x1,y1,x2,y2: (y1-y2)/(x1-x2)
getx  = lambda  b,y1,x2,y2: ((y1-y2)/b)+x2
```
2. Next up, I separated the lines into two separate numpy.ndarrays representing the left lane line and right lane line.

3. Then, I find the mean for all of the points in all of the lines of both arrays. The thinking here is that an average of all the points would be a very close proximate of the actual lane lines:
```
rmean = np.array(rightlines).mean(axis=0).astype(int)
lmean = np.array(leftlines).mean(axis=0).astype(int)
```
(casting to int, because `cv2.line()` doesn't accept float values)

4. Now, I actually draw lines, but first, I compute the slope of the `lmean` and `rmean` lines, then compute new end points using my `getx()` lambda and y dimension of the image. (`image.shape[0]`).
```
x1,y1,x2,y2 = lmean[0]
b = slope(x1,y1,x2,y2)
x1 = int(getx(b,ysize-1,x2,y2))
x2 = int(getx(b,325,x1,ysize-1))
cv2.line(img, (x1, ysize-1), (x2, 325), [0, 0, 255], thickness)
```

### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ...

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
