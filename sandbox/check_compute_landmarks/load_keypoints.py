import numpy as np

# https://github.com/tranluan/Nonlinear_Face_3DMM/blob/master/3DMM_definition/3DMM_keypoints.dat
fd = open('./3DMM_keypoints.dat')
landmarks_indices1 = np.fromfile(file=fd, dtype=np.int32).tolist()


landmarks_indices2 = [21873, 22149, 21653, 21036, 43236, 44918, 46166, 47135, 47914, 48695, 49667, 50924
                         , 52613, 33678, 33005, 32469, 32709, 38695, 39392, 39782, 39987, 40154, 40893, 41059
                         , 41267, 41661, 42367, 8161, 8177, 8187, 8192, 6515, 7243, 8204, 9163, 9883
                         , 2215, 3886, 4920, 5828, 4801, 3640, 10455, 11353, 12383, 14066, 12653, 11492
                         , 5522, 6025, 7495, 8215, 8935, 10395, 10795, 9555, 8836, 8236, 7636, 6915
                         , 5909, 7384, 8223, 9064, 10537, 8829, 8229, 7629]
print('landmark 1: {0}'.format(len(landmarks_indices1)))
print('landmark 2: {0}'.format(len(landmarks_indices2)))

landmarks_indices1.sort()
landmarks_indices2.sort()

for id1, id2 in zip(landmarks_indices1, landmarks_indices2):
    print(id1, id2, id1 - id2)
