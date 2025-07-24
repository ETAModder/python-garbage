import tkinter as tk
import random
import math

# Define feelings and their corresponding effects on movement and behavior
emotions = {
    'happy': {'move': (2, 2), 'color': 'yellow'},
    'joyful': {'move': (3, 3), 'color': 'orange'},
    'mad': {'move': (-2, -2), 'color': 'red'},
    'very_angry': {'move': (-3, -3), 'color': 'darkred'},
    'sad': {'move': (-1, -1), 'color': 'blue'},
    'weeping': {'move': (-1, -2), 'color': 'darkblue'},
    'content': {'move': (1, 1), 'color': 'green'},
    'neutral': {'move': (0, 0), 'color': 'grey'},
    'pain': {'move': (-1, 1), 'color': 'purple'},
    'discomfort': {'move': (-1, 1), 'color': 'brown'}
}

# Simple neural network
class NeuralNetwork:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]

    def sigmoid(self, x):
        x = min(max(x, -100), 100)  # Clamp the input to the sigmoid function to avoid NaN values
        return 1 / (1 + math.exp(-x))

    def forward(self, inputs):
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs))
        return self.sigmoid(weighted_sum)

    def choose_emotion(self, inputs):
        outputs = {}
        for emotion, attributes in emotions.items():
            move = attributes['move']
            output = self.forward([inputs[0], inputs[1], move[0], move[1], inputs[2], inputs[3]])
            outputs[emotion] = output
        return max(outputs, key=outputs.get)

# Pixel movement and behavior
class Pixel:
    def __init__(self, canvas, x, y, speed_scale, walls, auto_emotion_var, food):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.shape = canvas.create_rectangle(self.x, self.y, self.x + 10, self.y + 10, fill="grey")
        self.nn = NeuralNetwork(num_inputs=8)  # x, y, emotion (2), wall (2), food (2)
        self.emotion = (0, 0)
        self.color = 'grey'
        self.speed_scale = speed_scale
        self.walls = walls
        self.auto_emotion_var = auto_emotion_var
        self.food = food
        self.energy = 100
        self.health = 100

    def move_and_act(self):
        speed = self.speed_scale.get()
        closest_food = self.get_closest_food()
        inputs = [self.x, self.y, *self.emotion, self.get_closest_wall_distance(), self.get_closest_wall_distance(), *closest_food]
        output = self.nn.forward(inputs)
        dx = int(output * (self.emotion[0] or 1) * speed)
        dy = int(output * (self.emotion[1] or 1) * speed)

        new_x = self.x + dx
        new_y = self.y + dy

        # Debugging: print the current state and movement decision
        print(f"Energy: {self.energy}, Health: {self.health}, Position: ({self.x}, {self.y}), Movement: ({dx}, {dy})")

        # Boundary checks
        if new_x < 0 or new_x > self.canvas.winfo_width() - 10:
            new_x = self.x
        if new_y < 0 or new_y > self.canvas.winfo_height() - 10:
            new_y = self.y

        # Wall collision check
        if not self.check_collision(new_x, new_y):
            self.canvas.move(self.shape, new_x - self.x, new_y - self.y)
            self.x = new_x
            self.y = new_y

        self.update_stats()

        if self.auto_emotion_var.get():
            self.auto_change_emotion()

        self.canvas.itemconfig(self.shape, fill=self.color)
        self.canvas.after(100, self.move_and_act)

    def trigger_emotion(self, emotion):
        if emotion in emotions:
            self.emotion = emotions[emotion]['move']
            self.color = emotions[emotion]['color']
            print(f"Emotion triggered: {emotion}, Move: {self.emotion}, Color: {self.color}")
        if emotion == 'pain':
            self.speed_scale.set(max(1, self.speed_scale.get() // 2))  # Reduce speed temporarily

    def auto_change_emotion(self):
        inputs = [self.x, self.y, self.get_closest_wall_distance(), self.get_closest_wall_distance()]
        emotion = self.nn.choose_emotion(inputs)
        self.trigger_emotion(emotion)

    def check_collision(self, x, y):
        for wall in self.walls:
            if x < wall[0] + 10 and x + 10 > wall[0] and y < wall[1] + 10 and y + 10 > wall[1]:
                return True
        return False

    def get_closest_wall_distance(self):
        closest_dist = float('inf')
        for wall in self.walls:
            dist = math.sqrt((self.x - wall[0])**2 + (self.y - wall[1])**2)
            if dist < closest_dist:
                closest_dist = dist
        if closest_dist == float('inf'):  # Handle infinite distance by using a large finite number
            closest_dist = 1000  # Use a large number to simulate a far distance
        return closest_dist

    def get_closest_food(self):
        closest_food = (0, 0)
        closest_dist = float('inf')
        for food_item in self.food:
            food_x, food_y = self.canvas.coords(food_item)[:2]
            dist = math.sqrt((self.x - food_x)**2 + (self.y - food_y)**2)
            if dist < closest_dist:
                closest_dist = dist
                closest_food = (food_x, food_y)
        return closest_food

    def update_stats(self):
        self.energy -= 1
        if self.energy <= 0:
            self.health -= 1

        for food_item in self.food:
            # Improved collision detection for food
            food_x, food_y = self.canvas.coords(food_item)[:2]
            if abs(self.x - food_x) < 10 and abs(self.y - food_y) < 10:
                self.food.remove(food_item)
                self.energy = min(100, self.energy + 50)
                self.health = min(100, self.health + 25)
                self.canvas.delete(food_item)

        if self.energy < 10:
            self.trigger_emotion('discomfort')
        if self.health < 50:
            self.trigger_emotion('pain')

        self.canvas.itemconfig(self.energy_label, text=f"Energy: {self.energy}")
        self.canvas.itemconfig(self.health_label, text=f"Health: {self.health}")

# Generate walls
def generate_walls(canvas, num_walls=5):
    walls = []
    for _ in range(num_walls):
        x = random.randint(0, canvas.winfo_width() - 10)
        y = random.randint(0, canvas.winfo_height() - 10)
        canvas.create_rectangle(x, y, x + 10, y + 10, fill="black")
        walls.append((x, y))
    return walls

# Generate food
def generate_food(canvas, num_food=5):
    food = []
    for _ in range(num_food):
        x = random.randint(0, canvas.winfo_width() - 10)
        y = random.randint(0, canvas.winfo_height() - 10)
        food_item = canvas.create_rectangle(x, y, x + 10, y + 10, fill="green")
        food.append(food_item)
    return food

# Initialize walls and food after main loop starts
def initialize_environment():
    global walls, food
    walls = generate_walls(canvas, num_walls=10)
    food = generate_food(canvas, num_food=5)
    pixel.walls = walls
    pixel.food = food

# Tkinter setup
root = tk.Tk()
root.title("Neural Network Pixel")

canvas = tk.Canvas(root, width=400, height=400)
canvas.pack()

speed_scale = tk.Scale(root, from_=1, to=10, orient=tk.HORIZONTAL, label="Speed")
speed_scale.pack()

auto_emotion_var = tk.IntVar()
auto_emotion_checkbox = tk.Checkbutton(root, text="Auto Change Emotion", variable=auto_emotion_var)
auto_emotion_checkbox.pack()

walls = []
food = []

pixel = Pixel(canvas, 200, 200, speed_scale, walls, auto_emotion_var, food)

# Display energy and health stats
pixel.energy_label = canvas.create_text(10, 10, anchor=tk.NW, text=f"Energy: {pixel.energy}")
pixel.health_label = canvas.create_text(10, 30, anchor=tk.NW, text=f"Health: {pixel.health}")

# Continue creating buttons to trigger emotions
button_frame = tk.Frame(root)
button_frame.pack()

for emotion in emotions.keys():
    button = tk.Button(button_frame, text=emotion.capitalize(), command=lambda e=emotion: pixel.trigger_emotion(e))
    button.pack(side=tk.LEFT)

root.after(100, initialize_environment)  # Initialize walls and food after the main loop starts
pixel.move_and_act()
root.mainloop()
