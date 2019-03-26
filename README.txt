General notes:
Both the input images and the output images (1a.png, 1b.png, 1c.pmg, 2.png, 3.png, 4.png) are all in the folder project_images.
OpenCV and OpenCV-contrib 3.4.2.16 were used while developing this project.

Part 1:
I had to use built-in OpenCV methods as the methods I wrote for assignment 2 were unreliable (feature detection, SIFT) with the project images or non-functional (feature matching). Much of the code is adapted from OpenCV tutorial/documentation code for this part, but it is functional.


Part 2:
This part is coded by hand. Each method from this part has a slightly different method signature from the suggested signatures from the handout:
- project(x, y, H)
- computerInlierCount(H, matches, img1_pts, img2_pts, inlierThreshold)
- RANSAC(matches, kp1, kp2, numIterations, inlierThreshold, img1, img2)

The reason why computeInlierCount needs lists of points from both images is to project the points from img1 and then compute the Euclidian distance with the points from img2 to count the inliers. There is also a utility method get_inliers that is mechanically identical to computerInlierCount, except that instead of counting inliers, it adds the inliers to a list and returns them at the end.

RANSAC requires lists of keypoints kp1 and kp2 for multiple reasons: 
- Get the points for computerInlierCount to use; 
- Get the points from the randomly selected matches;
- Get the points of the inliers to then get the refined homography.
The parameters img1 and img2 are there for OpenCV to draw the matches between the two images, with the resulting matched image being returned at the end (along with the refined homography and its inverse).

In stitch, projecting only the top right and bottom right corners of img2 are enough to calculate the size of the image.