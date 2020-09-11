import os

import genData as genData
import data_utils as data_utils


if __name__ == '__main__':
	root = os.getcwd()

	os.chdir(root + "/dev")
	print("Switched to directory dev")

	num_images = 10

	genData.thermal_noise_sequence(num_images)
	print("Generated images.")

	data_utils.train_test_split('../data', '../data/')
	print("Setup test-train split.")

	print("Finished.")