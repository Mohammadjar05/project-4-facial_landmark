# project-4-facial_landmark
this repo created by MOHAMMAD JARADAT
print("HELLO")
# Facial Landmark Detection and Person Recognition

## About the Project

This project is a simple facial recognition system built using Python. The main idea is to detect a face in an image, find important points on the face, and use these points as features to recognize the person.

Instead of using the whole image directly, the project uses **68 facial landmarks** detected by Dlib. Each landmark has an `x` and `y` coordinate, which gives us a total of **136 features** for every face.

For this project, the dataset contains three people:

* **Ahmad → Class 0**
* **Sara → Class 1**
* **Ali → Class 2**

Each person has 10 images, giving a total of **30 images/samples** in the dataset.

---

## How the Project Works

The project can be divided into a few main steps:

**Images → Face Detection → 68 Facial Landmarks → 136 Features → Dataset → Machine Learning Model → Prediction**

The system first prepares the facial landmark dataset. After that, a machine learning model is trained using the extracted features. Finally, a new image can be uploaded and the system tries to identify the person.

---

## 1. Libraries

The project uses several Python libraries:

* **Dlib** – for face detection and detecting the 68 facial landmarks.
* **OpenCV** – for reading and processing images and drawing the landmarks.
* **NumPy** – for working with numerical data and coordinates.
* **Pandas** – for creating and managing the dataset.
* **Matplotlib** – for displaying images and results.
* **Scikit-learn** – for splitting the data, scaling the features, training the Perceptron, and evaluating the model.

The required libraries are installed at the beginning of the notebook and then imported into the project.

---

## 2. Connecting Google Drive

The project was developed using **Google Colab**, so Google Drive is mounted to access the dataset and save the generated files.

The dataset is organized into separate folders for each person:

```text
facial_dataset/
│
├── Ahmad/
│   ├── ahmad1.jpg
│   ├── ahmad2.jpg
│   └── ...
│
├── Sara/
│   ├── sara1.jpg
│   ├── sara2.jpg
│   └── ...
│
└── Ali/
    ├── ali1.jpg
    ├── ali2.jpg
    └── ...
```

The code checks that the dataset exists and then displays the folders and number of images for each person.

---

## 3. Loading the Dlib Facial Landmark Model

The project downloads the Dlib file:

```text
shape_predictor_68_face_landmarks.dat
```

This model is used to locate the **68 specific points on a human face**.

The code creates two important Dlib objects:

```python
detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(
    "/content/shape_predictor_68_face_landmarks.dat"
)
```

The `detector` is responsible for finding the face, while the `predictor` finds the 68 facial landmarks inside the detected face.

---

## 4. Class Mapping

Machine learning models work with numerical class labels, so each person is assigned a number:

| Person | Class |
| ------ | ----: |
| Ahmad  |     0 |
| Sara   |     1 |
| Ali    |     2 |

This mapping is used throughout the project to connect the numerical prediction back to the person's name.

---

## 5. Detecting the Face

For every image, OpenCV first reads the image and converts it to grayscale.

Dlib then searches for faces in the image.

If more than one face is detected, the project chooses the **largest detected face**, assuming that it is the main face in the image.

```python
face = max(
    faces,
    key=lambda rect: rect.width() * rect.height()
)
```

If no face is found, that image is skipped and counted as a failed image.

---

## 6. Detecting the 68 Facial Landmarks

Once the face has been detected, Dlib finds 68 points around different parts of the face.

These landmarks describe areas such as:

* Jawline
* Eyebrows
* Eyes
* Nose
* Mouth

For example, each landmark contains:

```text
(x, y)
```

where `x` represents the horizontal position and `y` represents the vertical position of the point.

The project displays these points on the original image so that we can visually check whether the landmarks were detected correctly.

---

## 7. Creating 136 Features

The 68 landmarks are converted into numerical features.

For every landmark, the code stores both its `x` and `y` coordinates:

```python
for i in range(68):

    x = landmarks.part(i).x
    y = landmarks.part(i).y

    features.append(x)
    features.append(y)
```

Since there are 68 landmarks and each one has two coordinates:

```text
68 × 2 = 136 features
```

So, one face is represented by **136 numerical values**.

For example:

```text
x1, y1, x2, y2, x3, y3, ... , x68, y68
```

The class label is then added as the final column.

This means every row in the final dataset contains:

```text
136 facial features + 1 class label = 137 columns
```

The generated dataset contains **30 samples and 137 columns**.

---

## 8. Creating the Dataset

The program goes through all the images in the three folders.

For each image, it:

1. Reads the image.
2. Converts it to grayscale.
3. Detects the face.
4. Selects the largest face.
5. Detects the 68 landmarks.
6. Extracts the 136 `x,y` coordinates.
7. Adds the person's class label.
8. Saves the processed image with the landmarks drawn on it.

The successful samples are stored in a list and then converted into a Pandas DataFrame.

The final dataset has:

```text
Samples: 30
Features: 136
Classes: 3
Columns: 137
```

There are 10 samples for each class.

---

## 9. Saving the Dataset as CSV

After creating the DataFrame, the project saves it as:

```text
facial_landmarks.csv
```

The CSV file contains columns like:

```text
x1, y1, x2, y2, x3, y3, ..., x68, y68, Class
```

The notebook also loads the CSV again afterward to make sure that it was saved and can be read correctly.

---

# Machine Learning Part

After creating the facial landmark dataset, the second part of the project uses the data to train a machine learning model.

## 10. Separating Features and Labels

The `Class` column is the target that the model needs to predict.

Therefore, the dataset is separated into:

* **X** → the 136 facial landmark features
* **y** → the person's class

```python
X = df.drop("Class", axis=1)
y = df["Class"]
```

The shapes are:

```text
X = (30, 136)
y = (30,)
```

This means there are 30 faces, and every face has 136 features.

---

## 11. Splitting the Dataset

The dataset is divided into training and testing data.

The project uses:

```python
test_size=0.20
```

This means:

* **80% → Training**
* **20% → Testing**

With 30 samples, this results in:

```text
Training samples: 24
Testing samples: 6
```

The `stratify=y` option is used so that the three classes remain represented in the training and testing sets.

---

## 12. Feature Scaling

Before training the model, the features are scaled using `StandardScaler`.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Scaling puts the feature values on a more consistent scale, which helps the machine learning algorithm work with the data more effectively.

The scaler is fitted only on the training data and then applied to the test data.

---

## 13. Training the Perceptron

The machine learning model used in this project is a **Perceptron**.

```python
perceptron = Perceptron(
    max_iter=2000,
    eta0=0.01,
    random_state=42,
    tol=1e-3
)
```

The Perceptron learns patterns from the 136 facial landmark features and tries to separate the three classes.

The model is then trained using:

```python
perceptron.fit(
    X_train_scaled,
    y_train
)
```

After training, the model can make predictions for faces that it has not seen before.

---

## 14. Model Evaluation

The trained model is tested using the six test samples.

The project compares:

```text
Predicted classes
```

with:

```text
Actual classes
```

The accuracy obtained in this run was:

```text
50%
```

This means the model correctly classified 50% of the test samples in this particular train/test split.

Because the dataset is very small, this result should not be treated as a reliable measure of how the system would perform on a much larger dataset.

---

## 15. Confusion Matrix

The project also generates a confusion matrix to see how the predictions are distributed between the three classes.

The confusion matrix from this run was:

```text
[[0 2 0]
 [0 2 0]
 [0 1 1]]
```

The rows represent the actual classes, while the columns represent the predicted classes.

A confusion matrix is useful because it shows not only how many predictions were correct, but also which people were being confused with each other.

The project also generates a classification report containing metrics such as precision, recall, and F1-score.

---

# Testing a New Image

## 16. Uploading a New Face

The final part allows the user to upload an image.

The uploaded image goes through almost the same process as the training images:

```text
New Image
    ↓
Face Detection
    ↓
68 Facial Landmarks
    ↓
136 Features
    ↓
Feature Scaling
    ↓
Perceptron
    ↓
Predicted Person
```

The 136 extracted features are converted into the correct shape before being passed to the model.

---

## 17. Recognizing the Person

The trained Perceptron predicts a class number.

For example:

```text
Predicted class: 1
```

The class mapping is then used to convert that number into a person's name:

```text
Class 0 → Ahmad
Class 1 → Sara
Class 2 → Ali
```

In the example run included in the notebook, the uploaded face was predicted as:

```text
Predicted person: Sara
```

---

# Eye Detection Using EAR

The project also includes an additional feature for determining whether the person's eyes are open or closed.

It uses something called the **Eye Aspect Ratio (EAR)**.

The 68 facial landmarks include six points around each eye. The project uses these points to calculate the ratio between the vertical and horizontal distances of the eye.

The formula implemented in the project is:

```text
EAR = (vertical_1 + vertical_2) / (2 × horizontal)
```

The project uses:

```python
EAR_THRESHOLD = 0.23
```

If:

```text
EAR > 0.23
```

the eye is considered **Open**.

Otherwise, it is considered **Closed**.

For the example image, the calculated average EAR was approximately:

```text
0.25
```

and the detected eye status was:

```text
Open
```

---

# Final Output

At the end, the project displays the processed image with:

* The detected face rectangle
* The 68 facial landmarks
* The predicted person's name
* The predicted class
* The EAR value
* The eye status
* The model accuracy

This gives a simple visual summary of what the system detected and predicted.

---

# Project Structure

A possible organization for the project is:

```text
Facial-Landmark-Recognition/
│
├── facial_landmarks.ipynb
├── facial_landmarks.csv
├── README.md
│
├── dataset/
│   ├── Ahmad/
│   ├── Sara/
│   └── Ali/
│
└── annotated_images/
    ├── Ahmad/
    ├── Sara/
    └── Ali/
```

---

# Technologies Used

* Python
* Google Colab
* Dlib
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

# Conclusion

This project demonstrates a basic approach to facial recognition using facial landmarks instead of directly using raw image pixels.

The important idea is to turn a face into a set of numerical values. In this case, the 68 facial landmarks give us 136 coordinate-based features. These features are stored in a CSV file and used to train a Perceptron classification model.

The project also shows how the same facial landmarks can be used for another task, such as detecting whether the eyes are open or closed using the Eye Aspect Ratio.

The current dataset is relatively small, with only 30 samples, so the recognition accuracy can vary depending on the images and train/test split. A larger and more varied dataset would be useful for improving the reliability of the model.
