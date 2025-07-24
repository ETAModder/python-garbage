import random
import struct

# Read the image file
with open('img.png', 'rb') as f:
    image = f.read()

# Preprocess the image
def preprocess_image(image):
    pixel_values = [pixel for pixel in image]
    return [val / 255.0 for val in pixel_values]

flattened_image = preprocess_image(image)

# Define the neural network with rewards
class SimpleNN:
    def __init__(self, input_size):
        self.weights = [random.random() for _ in range(input_size)]
    
    def forward(self, x):
        return sum(x[i] * self.weights[i] for i in range(len(x))) / len(x)  # Average the output to avoid large values
    
    def backward(self, x, error, learning_rate, reward):
        if not (error is None or error != error):  # Check if error is NaN
            for i in range(len(x)):
                self.weights[i] -= learning_rate * error * x[i] * reward  # Update based on reward

nn = SimpleNN(input_size=len(flattened_image))

# Train the neural network with positive and negative rewards
epochs = 250000
learning_rate = 0.01
stats_interval = 1000

for epoch in range(epochs):
    output = nn.forward(flattened_image)
    error = output - sum(flattened_image) / len(flattened_image)  # Simplified error calculation
    reward = 1.0 - abs(error)  # Positive reward for accuracy
    penalty = -abs(error) if abs(error) > 0.1 else 0  # Negative reward (penalty) for inaccuracy
    nn.backward(flattened_image, error, learning_rate, reward + penalty)
    
    if epoch % stats_interval == 0:
        accuracy = 1.0 - abs(error)
        print(f"Epoch {epoch}: Error = {error}, Reward = {reward}, Penalty = {penalty}, Accuracy = {accuracy}")

# Reconstruct the image
output_image = [int(min(max(val * 255, 0), 255)) for val in nn.weights]

# Create a valid PNG file
width = int(len(output_image) ** 0.5)
height = width
output_image = output_image[:width * height]  # Ensure it's a square image

def create_png(image_data, width, height):
    def png_chunk(chunk_type, data):
        chunk = struct.pack("!I", len(data)) + chunk_type + data
        return chunk + struct.pack("!I", (zlib.crc32(chunk_type + data) & 0xffffffff))

    import zlib

    png = b'\x89PNG\r\n\x1a\n'
    png += png_chunk(b'IHDR', struct.pack("!2I5B", width, height, 8, 0, 0, 0, 0))
    raw_data = b''.join(b'\x00' + bytes(image_data[y*width:(y+1)*width]) for y in range(height))
    png += png_chunk(b'IDAT', zlib.compress(raw_data, 9))
    png += png_chunk(b'IEND', b'')
    return png

output_png = create_png(output_image, width, height)

with open('reconstructed_image.png', 'wb') as f:
    f.write(output_png)
