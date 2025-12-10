Small POC of Marker Tracking without ML algorithms using Computer Vision and well defined geometry rules

The steps are simple:
1) Mask each RBGW color separately
2) Find and filter contours in those masks
	- Filter for minimum area
	- Filter for "squariness"
3) Check contour centroids spatial correlations
	- White contours must have at least one of each RGB contour inside
	- RGB contours must be positioned at the right vertices of a pythagorean triangle


This is just a demonstration in Python and i'm actually reinventing the wheel here. There are industry proven ready-made techs. I suggest checking ArUco (Augmented Reality University of Cordoba).

