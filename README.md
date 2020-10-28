# BTechProject
The project is on developing an automated attendance system based on face dtection and recognition techniques. Conventional methods involve taking of attendance 
by professor/administrator manually. This is error-prone due to proxy attendance, human error etc. The proposed system aims at solving these problems. The steps 
involved in this approach are:
1. Capture image/video streams via CCTV cameras in lecture halls 
2. Involves detection of the faces of multiple students from the photograph
3. Detected faces are queried, matched with faces from student database 
For Face Detection, HAAR Cascade and HOG methods have been used. For Face Recognition, Principal Components Analysis, Local Binary Patterns Histograms and Siamese
Network have been used.
Preprocessing involving Canny Edge Detection, Sharp and inverse, and Sobel Edge Detection had also been performed, but it was not of much significance to the
project. 
Codes for all the processes involved can be found in this folder.

