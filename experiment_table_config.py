import numpy as np
npa = lambda k: np.array(k)

# cartesian coords of the end of the arm above each of the markers.
# measured by moving the arm above each marker and read the cartesian coords.
# $ python javis_ui_2021.py

marker_coordinates_in_robot_cartesian_frame = mcrcf = \
{
    0:npa([8,936,-130]),
    1:npa([8,372,-130]),
    3:npa([-622,918,-130]),
    2:npa([-581,328,-130]),
}
