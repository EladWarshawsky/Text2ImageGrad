import os
os.environ["OPENAI_API_KEY"] = ''

import numpy as np
import matplotlib.pyplot as plt
import textgrad as tg
from PIL import Image
import io
from textgrad.loss import ImageQALoss

# Step 1: Generate the meshgrid
def generate_meshgrid(size=150):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    return np.stack([X.ravel(), Y.ravel()], axis=1)

# Function to evaluate the RGB image from R, G, B channel formulas
def evaluate_rgb_formula(r_formula, g_formula, b_formula, meshgrid_coords, size=150):
    X = meshgrid_coords[:, 0]
    Y = meshgrid_coords[:, 1]
    
    try:
        R = eval(r_formula.value, {"__builtins__": None}, {"X": X, "Y": Y, "np": np, "sin": np.sin, "cos": np.cos})
        G = eval(g_formula.value, {"__builtins__": None}, {"X": X, "Y": Y, "np": np, "sin": np.sin, "cos": np.cos})
        B = eval(b_formula.value, {"__builtins__": None}, {"X": X, "Y": Y, "np": np, "sin": np.sin, "cos": np.cos})
    except Exception as e:
        print(f"Error in formula execution: {e}")
        return None
    
    rgb_image = np.stack([R, G, B], axis=-1).reshape((size, size, 3)).astype(np.uint8)
    return rgb_image

# Convert generated image to bytes for LLM processing
def image_to_bytes(image_array):
    img = Image.fromarray(image_array)
    byte_io = io.BytesIO()
    img.save(byte_io, "PNG")
    return byte_io.getvalue()

# Initialize TextGrad with separate RGB formula variables
r_formula = tg.Variable("np.clip(255 * (0.5 + 0.5 * np.sin(X + Y)), 0, 255)", requires_grad=False, role_description="R channel formula")
g_formula = tg.Variable("np.clip(255 * (0.5 + 0.5 * np.cos(X - Y)), 0, 255)", requires_grad=False, role_description="G channel formula")
b_formula = tg.Variable("np.clip(255 * (0.5 + 0.5 * np.sin(2 * X) * np.cos(2 * Y)), 0, 255)", requires_grad=False, role_description="B channel formula")

# Set up ImageQALoss with gpt-4o for image evaluation
evaluation_instruction = "Does this image resemble the description provided?"
loss_fn = ImageQALoss(evaluation_instruction=evaluation_instruction, engine="gpt-4o")

# Generate meshgrid
meshgrid_coords = generate_meshgrid(size=150)

# Define desired prompt and expected response as Variables
desired_image_prompt = tg.Variable("a pony in a field", requires_grad=False, role_description="target prompt")
expected_response = tg.Variable("Expected description related to a pony in a field", requires_grad=False, role_description="Expected response format")

# Refinement Loop
for i in range(1000):
    # Generate the current image based on RGB channel formulas
    current_image = evaluate_rgb_formula(r_formula, g_formula, b_formula, meshgrid_coords)
    if current_image is None:
        break
    
    # Convert the image to bytes for ImageQALoss
    image_bytes = image_to_bytes(current_image)
    image_variable = tg.Variable(image_bytes, requires_grad=False, role_description="Generated image")
    
    # Step 3: Calculate similarity score using gpt-4o with ImageQALoss
    loss_variable = loss_fn(question=desired_image_prompt, image=image_variable, response=expected_response)
    loss_score = loss_variable.value  # Access the score using `.value`
    print(f"Iteration {i+1}: Feedback Score -> {loss_score}")

    # Use feedback to guide formula adjustments (manually or through heuristics if needed)
    print(f"Iteration {i+1}: Refined R Formula -> {r_formula.value}")
    print(f"Iteration {i+1}: Refined G Formula -> {g_formula.value}")
    print(f"Iteration {i+1}: Refined B Formula -> {b_formula.value}")
    
    # Display current image
    plt.imshow(current_image)
    plt.title(f"Iteration {i+1}: Generated Image")
    plt.axis("off")
    plt.show()
