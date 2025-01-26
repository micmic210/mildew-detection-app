# Mildew Detection in Cherry Leaves

## Project Overview
This project aims to address the challenges faced by **Farmy & Foods** in detecting powdery mildew on cherry leaves. Powdery mildew is a fungal disease that compromises the quality of cherry crops. Currently, the detection process is manual and time-consuming, taking approximately 30 minutes per tree. This manual method is neither scalable nor efficient, given the large number of cherry trees across multiple farms.

The goal of this project is to develop a **Machine Learning (ML)-powered dashboard** to:

1. **Visually differentiate** between healthy cherry leaves and those affected by powdery mildew.
2. **Predict the health status** of a cherry leaf based on an uploaded image.

By automating this process, the company can significantly reduce time and labor costs, while improving the scalability and accuracy of mildew detection. If successful, this system can be replicated across other crops to detect pests and diseases. 

---

## Dataset Content

The dataset contains images of cherry leaves categorized into two classes:
- **Healthy leaves**
- **Leaves with powdery mildew**

The images were captured from Farmy & Foods’ cherry crops and are publicly available on Kaggle.

- **Dataset Source**: [Cherry Leaves Dataset](https://www.kaggle.com/codeinstitute/cherry-leaves)
- **Dataset Size**: 4208 images

---

## Business Requirements

The client has outlined two primary business requirements for this project:

1. **Visual Analysis**: Conduct a study to visually differentiate healthy cherry leaves from those with powdery mildew.
2. **Classification and Prediction**: Develop a binary classification model to predict whether a given cherry leaf is healthy or affected by powdery mildew.

---

## Hypotheses and Validation

### Hypotheses
1. **Visual Differences**: Healthy leaves have a uniform texture and color, while leaves with powdery mildew exhibit visible discoloration and fungal patterns.
2. **Predictive Feasibility**: A supervised ML model can reliably classify images of cherry leaves into healthy or mildew-affected categories.

### Validation Steps
- **For Hypothesis 1**: Use data visualization techniques to display the average and variability images for healthy and mildew-affected leaves. Compare the differences visually.
- **For Hypothesis 2**: Train a binary classification ML model and evaluate its performance using metrics such as accuracy and confusion matrix.

---

## The rationale to map the business requirements to the Data Visualisations and ML tasks

### **Business Requirement 1**: Visual Analysis
- Display the mean and standard deviation images for healthy and mildew-affected leaves.
- Showcase differences between the average healthy and mildew-affected leaves.
- Create an image montage to illustrate the dataset's diversity.

### **Business Requirement 2**: Classification and Prediction
- Develop a binary classifier to predict whether a leaf is healthy or has mildew.
- Build a user-friendly interface for uploading images and receiving predictions in real time.

---

## ML Business Case

### **Mildew Detection Model**
- **Objective**: Predict if a cherry leaf is healthy or has powdery mildew.
- **Model Type**: Supervised binary classification.
- **Success Metrics**:
  - Accuracy ≥ 90%
  - Confusion Matrix: High recall for mildew-affected leaves.
- **Model Output**: Probability of the leaf being healthy or mildew-affected, along with a classification label.

**Heuristics**:
Currently, the manual inspection process relies on human expertise, which is prone to errors and inefficiencies. An ML-based solution can provide faster, more consistent results, minimizing human error and operational costs.

**Training Data**:
The training data is derived from the cherry leaves dataset, consisting of labeled images for healthy and mildew-affected categories.

---

## CRISP-DM Framework

1. **Business Understanding**:
   - **Problem**: Manual inspection of cherry leaves is inefficient and unscalable.
   - **Goal**: Develop a scalable ML solution to automate leaf classification and provide actionable farm insights.

2. **Data Understanding and Preparation**:
   - **Dataset**: 4208 labeled images of healthy and mildew-affected cherry leaves.
   - **Preprocessing**:
     - Resize images to 50x50 pixels for consistency.
     - Normalize pixel values and apply data augmentation (e.g., rotation, flipping) to improve model robustness.

3. **Modeling**:
   - Design and train a convolutional neural network (CNN) using TensorFlow/Keras.
   - Optimize hyperparameters to enhance model accuracy and generalization.

4. **Evaluation**:
   - Assess the model using metrics such as precision, recall, F1-score, and accuracy.
   - Focus on high recall to minimize false negatives in mildew detection.

5. **Deployment**:
   - Integrate the trained model into a Streamlit dashboard for real-time predictions.
   - Ensure the dashboard is user-friendly and accessible across devices.

---

## Dashboard Design
### Page 1: **Quick Project Summary**
- **Content**:
  - Overview of powdery mildew and its impact on cherry crops.
  - Description of the manual detection process and its inefficiencies.
  - Project goals and how the solution addresses the client’s needs.

### Page 2: **Leaf Visualizer**
- **Business Requirement**: Address visual differentiation of healthy vs. mildew-affected leaves.
- **Features**:
  - Checkbox 1: Display mean and standard deviation images for both categories.
  - Checkbox 2: Show differences between average healthy and mildew-affected leaves.
  - Checkbox 3: Display an image montage of the dataset.

### Page 3: **Mildew Detector**
- **Business Requirement**: Predict leaf health status.
- **Features**:
  - Upload widget for cherry leaf images.
  - Display predictions with associated probabilities.
  - Generate a table summarizing image names and predictions.
  - Download button for saving prediction results.

### Page 4: **Project Hypotheses and Validation**
- **Content**:
  - List hypotheses and their validation steps.
  - Include visualizations and results supporting each hypothesis.

### Page 5: **ML Performance Metrics**
- **Content**:
  - Label frequencies for train, validation, and test datasets.
  - Model training history (accuracy and loss curves).
  - Model evaluation metrics (e.g., confusion matrix, precision, recall).

---

## User Stories

1. Intuitive Navigation
   - **Priority**: Must-Have
   - **User Story**:
     As a client, I want an intuitive dashboard with clear navigation so that I can easily access data, predictions, and insights.
   - **Acceptance Criteria**:
     - A navigation bar is present and allows switching between all pages.
     - All navigation links are clearly labeled and functional.
     - The user can access any page in no more than two clicks.

2. Visual Differentiation
   - **Priority**: Must-Have
   - **User Story**:
     As a client, I want to observe average and variability images of healthy and mildew-infected cherry leaves so that I can visually differentiate between the two categories.
   - **Acceptance Criteria**:
     - The dashboard displays average and variability images for healthy and infected leaves.
     - The user can toggle between these visualizations using checkboxes or buttons.
     - The visualizations are clear and labeled with appropriate captions.

3. Image Montage
   - **Priority**: Nice-to-Have
   - **User Story**:
     As a client, I want to view montages of healthy and infected leaves so that I can compare them more easily.
   - **Acceptance Criteria**:
     - The user can select “Healthy” or “Infected” leaves to create a montage.
     - The montage displays at least 9 images per category in a grid format.
     - There is a button to dynamically generate a new montage.

4. Real-Time Predictions
   - **Priority**: Must-Have
   - **User Story**:
     As a client, I want to upload images of cherry leaves and receive predictions about their health status (healthy/infected) in real-time.
   - **Acceptance Criteria**:
     - A file uploader is available and supports single and multiple image uploads.
     - The system predicts the health status of each uploaded image with at least 90% accuracy.
     - Predictions are displayed on the dashboard with confidence scores.

5. Infection Rate Summary
   - **Priority**: Nice-to-Have
   - **User Story**:
     As a client, I want to see a summary of the infection rate (percentage of healthy vs. infected leaves) based on the uploaded images so that I can quickly understand the overall situation.
   - **Acceptance Criteria**:
     - A pie chart or bar chart displays the percentage of healthy vs. infected leaves.
     - The chart updates dynamically based on the uploaded images.
     - The chart is labeled clearly and easy to interpret.

6. Hypotheses and Validation
   - **Priority**: Must-Have
   - **User Story**:
     As a stakeholder, I want to understand the hypotheses tested and their validation process so that I can trust the results of the project.
   - **Acceptance Criteria**:
     - A dedicated page presents the project’s hypotheses and explains them simply.
     - Validation steps are described clearly, including visuals and results.
     - The page includes insights on how the visual differentiation supports the ML model’s development.

7. Model Performance Metrics
   - **Priority**: Must-Have
   - **User Story**:
     As a stakeholder, I want to see detailed performance metrics so that I can evaluate how reliable the model is.
   - **Acceptance Criteria**:
     - The dashboard displays metrics such as precision, recall, F1-score, and confusion matrix.
     - The metrics are easy to understand, with clear labels and explanations.
     - The dashboard includes an overall accuracy metric and explains its significance.

8. Scalability and Future Applications
   - **Priority**: Nice-to-Have
   - **User Story**:
     As a stakeholder, I want to see potential applications of this solution for other crops and diseases so that I can plan for scalability.
   - **Acceptance Criteria**:
     - A section in the README discusses opportunities for applying the model to other crops or pests.
     - The section outlines potential integrations with IoT devices or drone-based image capture.
     - Lessons learned from this project that support scalability are highlighted.


## Unfixed Bugs

- The model’s performance may vary under non-standard conditions (e.g., unusual lighting or damaged leaves)

---

## Deployment

### Heroku


- The App live link is: `https://YOUR_APP_NAME.herokuapp.com/`
- The project was deployed to Heroku following these simplified steps:

1. Log in to Heroku and create an app.
2. Link the app to the GitHub repository containing the project code.
3. Select the branch to deploy and click "Deploy Branch."
4. Once the deployment completes, click "Open App" to access the live app.
5. Ensure that deployment files, such as `Procfile` and `requirements.txt`, are correctly configured.
6. Use a `.slugignore` file to exclude unnecessary large files if the slug size exceeds limits.

### Repository Structure
- **app_pages/**: Streamlit app pages.
- **src/**: Auxiliary scripts (e.g., data preprocessing, model evaluation).
- **notebooks/**: Jupyter notebooks for data analysis and model training.
- **Procfile, requirements.txt, runtime.txt, setup.sh**: Files for Heroku deployment.

---

## Technologies Used
- **Python Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, TensorFlow/Keras, Streamlit.
- **Tools**: Jupyter Notebook, Heroku for deployment.

---

## Future Work
- Extend the system to detect other crop diseases.
- Incorporate real-time image capture from drones for automated data collection.
- Integrate the system with IoT devices for automated spraying of antifungal compounds.

---

## Main Data Analysis and Machine Learning Libraries

- Here, you should list the libraries used in the project and provide an example(s) of how you used these libraries.

---

## Credits

- The deployment steps were adapted from [Heroku Documentation](https://devcenter.heroku.com/).
- Data preprocessing techniques were inspired by [TensorFlow tutorials](https://www.tensorflow.org/tutorials).
- Model evaluation approaches referenced [Scikit-Learn Documentation](https://scikit-learn.org/stable/).
- Icons used in the dashboard are from [Font Awesome](https://fontawesome.com/).
- Visualization techniques were guided by examples from [Matplotlib Documentation](https://matplotlib.org/stable/index.html).

---

### Content

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

---

### Media

---


## Acknowledgements
- **Farmy & Foods** for providing the dataset and project inspiration.
- Code Institute for guidance and support in building this project.
- Kaggle for hosting the cherry leaves dataset and enabling access to quality data.
- The contributors of TensorFlow and Scikit-Learn for their excellent documentation and tutorials.
- Community forums and online resources for addressing technical challenges and sharing best practices.

---

