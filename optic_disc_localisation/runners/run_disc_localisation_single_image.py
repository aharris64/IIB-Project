from pathlib import Path
import os

from optic_disc_localisation.runners.run_disc_localisation import detect_disc

image_name = "normal_0001_EDD.jpg"

disc_localisation_path = Path(__file__).parents[1]
image_path = os.path.join(disc_localisation_path, "input_images", image_name)

intermediate_save_path = os.path.join(disc_localisation_path, "outputs", "intermediate_output_images")
final_save_path = os.path.join(disc_localisation_path, "outputs", "testing_output_images")

score = detect_disc(image_path, 
                    save_final=True,
                    save_final_path=final_save_path,
                    save_intermediate=True,
                    save_intermediate_path=intermediate_save_path)

print("Score: ", score)