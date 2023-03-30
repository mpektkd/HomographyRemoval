## HomographyRemoval
NTUA

We are asked to remove the homography:projective transformation from the image,
which is given to us. In particular, following the Metric Rectification via Orthogonal Lines algorithm, 
which is described in [1], we aim to remove the distortion due to the projection
H transformation.

### Algorithm
  - We apply the Canny algorithm to find edges in the image(cv2.Canny())
  - As input the edges of the previous step, we apply the probabilistic Hough algorithm
     transform to locate straight lines(cv2.HoughLinesP()), returning the coordinates of the two
     points for each line and selection of suitable vertical pairs.
  - After converting the coordinates of the points for each line into homogeneous ones,
    we apply an outer product and arrive at the homogeneous coordinates of the lines
  - We consider f = 1 and from the equation for all pairs l, m we arrive at a system of the form
    Ax = b.
  - With the resulting vector c, we construct C<sub>∞</sub> , and by correspondence we find K, v.
  - As a homography we consider H = H<sub>a</sub> ∗ H<sub>p</sub>, that is, we consider that it has no similarity, as it is
    impossible to calculate with K, v and apply the inverse transformation.   

### Conclusions:
  - It is clear that the result is not perfect, as the similarity component is missing in
    calculation of H, which cannot be calculated. Nevertheless, we observe a very good one
    approximation of the actual image slightly rotated.
  - To implement reverse homography, the function cv2.warpPerspective() was used.
  - We notice gaps in the image, which did not exist in the previous one. An attempt
    interpretation of this fact is that, when applying reverse homography the positions
    of pixels were shifted and distorted to exceed the boundaries of the image, fact
    which could be expected as the image dimensions differ with the real world.

 ### Bibliography
[1]: Hartley - Zisserman, Multiple View Geometry in Computer Vision, 2nd edition, Cambridge University Press, 2000.
