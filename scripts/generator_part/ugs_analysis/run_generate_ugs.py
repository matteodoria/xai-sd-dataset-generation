import os
import tensorflow as tf
from PIL import Image
from tqdm import tqdm

from Models.stable_diffusion import MyStableDiffusion
from scripts.tools.run_generate_dataset import load_info


# ---------------------------------------------------------
# EXPERIMENT CONFIGURATION
# ---------------------------------------------------------

DATASET = "cifar10"
EXP = "xAI"

ENC_EPOCH = 31
DIF_EPOCH = 10
INFERENCE_STEPS = 20

UGS_VALUES = [0.0, 0.5, 1.0, 2.0, 4.0, 7.5]

N_IMAGES_PER_CLASS = 100

# Same seed for every class and every UGS:
# this ensures that the same initial noise is reused
# across experimental conditions.
SEED = 1234


# ---------------------------------------------------------
# GPU SETUP
# ---------------------------------------------------------

gpus = tf.config.experimental.list_physical_devices("GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print("Num GPUs Available:", len(gpus))


# ---------------------------------------------------------
# LOAD CIFAR-10 INFORMATION
# ---------------------------------------------------------

res, class_names = load_info(DATASET)

num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)


# ---------------------------------------------------------
# LOAD GENERATOR
# ---------------------------------------------------------

enc_weights_path = (
    f"Checkpoints/DDPM/Exp_{EXP}/{DATASET}/"
    f"MyEmbedding/epoch{ENC_EPOCH}.hdf5"
)

dif_weights_path = (
    f"Checkpoints/DDPM/Exp_{EXP}/{DATASET}/"
    f"DiffusionFt/epoch{DIF_EPOCH}.hdf5"
)

print("\nLoading generator...")

generator = MyStableDiffusion(
    res=res,
    num_classes=num_classes,
    original_diffusion=True,
    original_text_encoder=False,
    enc_weight_path=enc_weights_path,
    diff_weight_path=dif_weights_path,
)

print("Generator loaded successfully.")


# ---------------------------------------------------------
# OUTPUT DIRECTORY
# ---------------------------------------------------------

base_save_path = (
    f"Data/Synthetic/Exp_{EXP}/{DATASET}/"
    f"UGS_analysis_Enc{ENC_EPOCH}_Dif{DIF_EPOCH}_"
    f"Is{INFERENCE_STEPS}"
)

os.makedirs(base_save_path, exist_ok=True)


# ---------------------------------------------------------
# GENERATION
# ---------------------------------------------------------

for ugs in UGS_VALUES:

    print("\n" + "=" * 60)
    print(f"UGS = {ugs}")
    print("=" * 60)

    ugs_path = os.path.join(base_save_path, f"ugs_{ugs}")

    for class_id, class_name in enumerate(class_names):

        print(f"\nGenerating class {class_id}: {class_name}")

        # One-hot encoding of the current class
        class_vector = tf.one_hot(class_id, depth=num_classes)

        # Repeat the same class conditioning 100 times
        labels = tf.repeat(
            tf.expand_dims(class_vector, axis=0),
            repeats=N_IMAGES_PER_CLASS,
            axis=0,
        )

        
        # using the same seed for every class and every UGS means
        # that the same set of initial random noises is used.
        images = generator.gen(
            inputs=labels,
            num_images=N_IMAGES_PER_CLASS,
            num_inference_steps=INFERENCE_STEPS,
            ugs=ugs,
            seed=SEED,
        )

        # Directory for this class
        class_path = os.path.join(
            ugs_path,
            f"class_{class_id:02d}_{class_name}"
        )

        os.makedirs(class_path, exist_ok=True)

        # Save images
        for image_id, image in enumerate(tqdm(images)):

            pil_image = Image.fromarray(
                image.astype("uint8"),
                "RGB"
            )

            image_path = os.path.join(
                class_path,
                f"img_{image_id:03d}.png"
            )

            pil_image.save(image_path)


print("\nGeneration completed.")
print("Images saved in:")
print(base_save_path)