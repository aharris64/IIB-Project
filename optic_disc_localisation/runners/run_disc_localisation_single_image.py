"""Ad-hoc manual smoke test: runs disc localisation on one image from input_images/
and prints the result, saving both intermediate step images and the final overlay —
useful for eyeballing the pipeline on a single case rather than a full dataset run."""

from pathlib import Path
import os

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from optic_disc_localisation.runners.run_disc_localisation import detect_disc

image_name = "normal_2274_RFM.png"  # change to try a different image from input_images/

disc_localisation_path = Path(__file__).parents[1]  # optic_disc_localisation/ (one level up from runners/)
image_path = os.path.join(disc_localisation_path, "input_images", image_name)

intermediate_save_path = os.path.join(disc_localisation_path, "outputs", "intermediate_output_images")
final_save_path = os.path.join(disc_localisation_path, "outputs", "testing_output_images")

result = detect_disc(image_path, 
                    save_final=True,
                    save_final_path=final_save_path,
                    save_intermediate=True,
                    save_intermediate_path=intermediate_save_path)

print(result)