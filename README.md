# IADAI201-1000065--Aarish_Panda
For creating my pose detection project, I started by conducting research on how to approach the task. After gaining a solid understanding, I wrote my Python code using Visual Studio Code (VS Code). I installed the required extensions for Jupyter Notebook and Python in VS Code and then used the command prompt to download essential Python libraries such as numpy, pandas, opencv-python, tensorflow, and keras.

Next, I downloaded 12 videos from YouTube, representing 3 different gestures: clap, walk, and run (4 videos per gesture: 3 for training and 1 for testing). After downloading the videos, I converted them into AVI format for easy processing. From these videos, I created my dataset, normalized it, split the data for training and testing, trained the model, and finally tested it with an accuracy of 100%. 

Here's how the code works:

1. Capturing frames and extracting pose landmarks:

We use OpenCV to open the video files and MediaPipe to detect pose landmarks.
As each video frame is processed, the detected landmarks (such as key body joints) are saved into a CSV file along with the frame number and a label (e.g., clap, walk, or run). MediaPipe Pose detects 33 landmarks (e.g., elbows, knees), and each landmark's x, y, z coordinates, and visibility score are stored.

cap = cv2.VideoCapture('clap1.avi')
...
csv_writer.writerow(headers)

2. Normalizing the data:

Once the raw landmark data is collected, we normalize it by referencing the left hip's position to ensure that the model focuses on relative body movements, ignoring the person's absolute position.

for i in range(33):
    df[f'x_{i}'] -= df['x_23']  # Normalize by left hip x-coordinate

3. Splitting the dataset:

The normalized data is split into training and testing sets using the train_test_split() method. This allows the model to learn from one portion of the data and test its accuracy on the other portion.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

4. Training the model:

I built a neural network using TensorFlow/Keras, which consists of multiple layers:
Two dense layers with 64 and 32 neurons respectively, both using the ReLU activation function.
The output layer uses the softmax activation function to predict one of the 3 gestures (clap, walk, run).
After compiling the model, I trained it on the landmark data and achieved 100% accuracy on the test data.

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(set(y)), activation='softmax')
])

5. Saving the model:

After training, the model is saved to a file, making it reusable for later testing or deployment.

model.save(model_path)

6. Testing the model:

In the final step, we load the trained model and use it to make predictions on new video data. For each frame in the test video, we detect the pose landmarks, prepare them for the model, and predict the gesture (clap, walk, or run). The predicted gesture is displayed on the video screen in real-time.

predicted_label = np.argmax(prediction)
label_text = label_map[predicted_label]
cv2.putText(image, f"Pose: {label_text}", (10, 40), ...)

7. Terminating the program:

A new window comes out and shows the output. The new window which was open or which was the testing runs can be closeed by pressing the Q button to stop the video.
In summary, after setting up the environment and downloading the necessary videos, I created a dataset of pose landmarks from different gestures. Then, I trained a neural network model, tested it on unseen video data, and achieved 100% accuracy in recognizing the gestures. The project is completed by simply pressing Q to terminate the video display.


Output:

For walk: ![Screenshot 2024-11-11 122345](https://github.com/user-attachments/assets/8c6c60be-9336-4db5-84ce-bacff8e80cb1)


For Running: ![Screenshot 2024-11-11 122707](https://github.com/user-attachments/assets/3a4741c7-52ae-4ca4-85e1-381c5369668b)

Link to the github repositary: https://github.com/Aarish96/IADAI201-1000065-Aarish_Panda



The future scope of this pose detection project offers several exciting directions for improvement and application. Here are 5-6 detailed points on how it can be extended and enhanced:

Adding More Gestures and Activities:

Currently, the model is trained on three gestures: clap, walk, and run. In the future, the dataset can be expanded to include more complex activities like jumping, sitting, standing up, or even yoga poses and dance moves. By adding more diverse gestures, the system could be used in various fields like sports coaching, fitness tracking, and rehabilitation therapy.
Real-time Pose Feedback and Correction:

The current system only detects poses, but future developments could involve giving real-time feedback on the correctness of a user’s pose. For example, in fitness training or physical therapy, the system could analyze if the user is performing an exercise correctly and provide corrective suggestions or warnings if a pose is incorrect, helping reduce injuries and improve form.
Integration with Wearable Devices:

Integrating this pose detection system with wearable devices like smartwatches or AR (Augmented Reality) glasses could offer even more detailed analysis. Wearables could provide additional sensor data such as heart rate, motion, or orientation, which could be combined with visual pose data to offer a more holistic analysis of human activity, such as tracking fatigue levels or overall performance metrics.
Enhanced Model Accuracy with 3D Pose Estimation:

Future versions could incorporate more advanced 3D pose estimation using depth sensors or stereo cameras. This would allow the model to better understand the full three-dimensional movements of a person, leading to more precise tracking of activities like jumping, bending, or other movements where depth information is critical. This would improve accuracy in more dynamic environments.
Application in Augmented Reality (AR) and Virtual Reality (VR):

Pose detection could be further developed for immersive experiences in AR and VR. For example, users could control virtual avatars in real time by performing gestures. In gaming, this would allow players to interact with the virtual world through body movements, making the experience more engaging and interactive. Similarly, virtual fitness or yoga classes could be enhanced by real-time tracking and correction of participants' poses in a virtual environment.
Multi-person Pose Detection and Interaction:

Currently, the model focuses on detecting the pose of a single individual. Expanding the system to handle multi-person pose detection can enable applications in team sports analysis, social interaction in virtual environments, or group fitness classes. The system could track multiple users, analyze their interactions, and even detect group activities or collaboration patterns, which would be useful in fields like sports training or even crowd behavior analysis in security systems.
In conclusion, this pose detection system has the potential to evolve from a gesture recognition tool into a sophisticated platform that can be used in industries ranging from fitness and sports to healthcare, gaming, and entertainment. With the advancements in AI, sensor technologies, and real-time processing, the possibilities are vast for future development.







