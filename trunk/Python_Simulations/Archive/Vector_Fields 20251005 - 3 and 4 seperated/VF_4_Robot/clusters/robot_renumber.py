import numpy as np
import matplotlib.pyplot as plt
import random

# Step 1: Generate 10 random points
num_points = 10
points = np.random.rand(num_points, 2)

# Step 2: Calculate the average Xc, Yc
Xc, Yc = np.mean(points, axis=0)

# Step 3: Pick a point at random and label it as point 1
point_1_index = random.randint(0, num_points-1)
point_1 = points[point_1_index]

# Function to calculate cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Function to calculate the cross product in 2D
def cross_product_2d(vec1, vec2):
    return vec1[0] * vec2[1] - vec1[1] * vec2[0]

# Step 4 & 5: Calculate cosine similarity for each point with point 1
cosine_similarities = []
vector_from_center_to_point_1 = np.array([Xc, Yc]) - point_1
for i, point in enumerate(points):
    vector_to_point_1 = point - point_1
    similarity = cosine_similarity(vector_to_point_1, vector_from_center_to_point_1)
    direction = cross_product_2d(vector_from_center_to_point_1, vector_to_point_1)
    direction_str = "ccw" if direction > 0 else "cw" if direction < 0 else "col"
    cosine_similarities.append((i, point, similarity, direction_str))

# Step 6: Separate points by direction
collinear = [p for p in cosine_similarities if p[3] == "col"]
ccw = [p for p in cosine_similarities if p[3] == "ccw"]
cw = [p for p in cosine_similarities if p[3] == "cw"]

# Sort ccw points in ascending order and cw points in descending order
ccw.sort(key=lambda x: x[2], reverse=True)
cw.sort(key=lambda x: x[2], reverse=False)

# Combine sorted lists
sorted_points = collinear + cw + ccw

# Renumber points based on sorted order and make the initial point 1 as the new point 0
new_indices = {}
new_index = 1
for p in sorted_points:
    old_idx = p[0]
    if old_idx == point_1_index:
        new_indices[old_idx] = 0
    else:
        new_indices[old_idx] = new_index
        new_index += 1

# Step 7: Output the array of points with their cosine similarity scores and new indices
output = np.array([(new_indices[p[0]], p[1][0], p[1][1], np.round(p[2], 5), p[3]) for p in sorted_points])

# Print the output
print(output)

# Plot the points
plt.scatter(points[:, 0], points[:, 1], label='Points')
plt.scatter([Xc], [Yc], color='red', label='Center (Xc, Yc)')
plt.scatter([point_1[0]], [point_1[1]], color='green', label='Point 1')

# Annotate points with their new index
for i, point in enumerate(points):
    plt.annotate(str(new_indices[i]), (point[0], point[1]))

# Draw lines
for i, point, similarity, _ in sorted_points:
    plt.plot([point[0], point_1[0]], [point[1], point_1[1]], 'gray', linestyle='--', alpha=0.5)

# plt.plot([Xc, point_1[0]], [Yc, point_1[1]], 'blue', label='Line from center to Point 1')

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Points and Cosine Similarity')
plt.show()